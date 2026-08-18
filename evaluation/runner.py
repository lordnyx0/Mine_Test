# coding=utf-8
"""
runner.py — Orchestration: generate every (model, item) response, then report.

Generation and reporting are independent: `report_only` regenerates every
report from the JSONL already on disk without loading a model.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

from .benchmark import category_counts, label_for, load_benchmark
from .config import EvalConfig
from .generation import Generator
from .models import (
    LoadedModel,
    ModelLoadError,
    load_model,
    load_tokenizer,
    resolve_device,
    resolve_model_specs,
)
from .progress import ProgressReporter
from .reports import (
    build_summaries,
    write_csv_report,
    write_markdown_report,
    write_pairwise_dataset,
)
from .storage import (
    JsonlLogger,
    completed_ids,
    ensure_dir,
    responses_path,
    write_json,
)
from .types import EvalItem, ModelSpec

METADATA_FILENAME = "run_metadata.json"


@dataclass(slots=True)
class RunResult:
    """What the CLI reports back to the user."""

    specs: list[ModelSpec]
    items: list[EvalItem]
    generated: int = 0
    skipped: int = 0
    failed: int = 0
    skipped_models: list[str] = field(default_factory=list)
    report_paths: dict[str, str] = field(default_factory=dict)


def _print_plan(cfg: EvalConfig, specs: Sequence[ModelSpec], items: Sequence[EvalItem]) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print(" Qwen3Loop Checkpoint Evaluation")
    print(sep)
    print(f"  Benchmark        : {cfg.benchmark_path} ({len(items)} items)")
    print(f"  Results dir      : {cfg.run.results_dir}")
    print(f"  Device           : {resolve_device(cfg.run.device)}")
    print(f"  Quantization     : {'4-bit ' + cfg.quantization.bnb_4bit_quant_type if cfg.quantization.enabled else 'disabled'}")
    print(f"  Decoding         : greedy (temp={cfg.generation.temperature}, top_p={cfg.generation.top_p}, seed={cfg.run.seed})")
    print(f"  Resume           : {'on' if cfg.run.resume else 'off'}")
    print(f"\n  Models ({len(specs)}):")
    for spec in specs:
        print(f"    - {spec.describe()}")
    print("\n  Categories:")
    for cat, count in category_counts(items).items():
        print(f"    - {label_for(cat):<24} {count:>3} items")
    print(f"{sep}\n")


def _write_metadata(cfg: EvalConfig, specs: Sequence[ModelSpec], items: Sequence[EvalItem]) -> None:
    write_json(
        os.path.join(cfg.run.results_dir, METADATA_FILENAME),
        {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "benchmark_path": cfg.benchmark_path,
            "eval_config_path": cfg.source_path,
            "item_count": len(items),
            "categories": category_counts(items),
            "base_model_id": cfg.model.base_model_id,
            "num_loops": cfg.model.num_loops,
            "seed": cfg.run.seed,
            "temperature": cfg.generation.temperature,
            "top_p": cfg.generation.top_p,
            "do_sample": cfg.generation.do_sample,
            "enable_thinking": cfg.model.enable_thinking,
            "quantization": {
                "enabled": cfg.quantization.enabled,
                "quant_type": cfg.quantization.bnb_4bit_quant_type,
                "compute_dtype": cfg.quantization.bnb_4bit_compute_dtype,
            },
            "device": resolve_device(cfg.run.device),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "models": [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "num_loops": s.num_loops,
                    "adapter_path": s.adapter_path,
                    "step": s.step,
                }
                for s in specs
            ],
        },
    )


def _evaluate_model(
    spec: ModelSpec,
    cfg: EvalConfig,
    items: Sequence[EvalItem],
    tokenizer: object,
    generator: Generator,
    progress: ProgressReporter,
    result: RunResult,
) -> None:
    """Generate all pending responses for one model variant."""
    path = responses_path(cfg.run.results_dir, spec.slug)
    ensure_dir(os.path.dirname(path))

    done = completed_ids(path) if cfg.run.resume else set()
    pending = [item for item in items if item.id not in done]
    result.skipped += len(items) - len(pending)

    if not pending:
        progress.note(f"{spec.name}: all {len(items)} items already present — skipped")
        return

    if not cfg.run.resume and os.path.exists(path):
        os.remove(path)

    try:
        loaded: LoadedModel = load_model(spec, cfg, tokenizer)
    except ModelLoadError as exc:
        progress.warn(str(exc))
        result.skipped_models.append(spec.name)
        return

    task = progress.start_task(spec.name, len(pending))
    try:
        with JsonlLogger(path) as logger:
            for item in pending:
                record = generator.generate(loaded, item)
                logger.write(record)
                if record.error:
                    result.failed += 1
                    progress.warn(f"{spec.name} / {item.id}: {record.error}")
                else:
                    result.generated += 1
                progress.advance(task)
    finally:
        progress.finish_task(task)
        loaded.release()


def run(
    cfg: EvalConfig,
    *,
    report_only: bool = False,
    only_models: Sequence[str] | None = None,
) -> RunResult:
    """Execute the full evaluation and write every report."""
    items = load_benchmark(cfg.benchmark_path, cfg.run.categories or None)
    if cfg.run.limit:
        items = items[: cfg.run.limit]

    specs = resolve_model_specs(cfg)
    if only_models:
        wanted = set(only_models)
        specs = [s for s in specs if s.name in wanted]
        unknown = wanted - {s.name for s in specs}
        if unknown:
            print(f"[WARN] Unknown model names ignored: {', '.join(sorted(unknown))}")
    if not specs:
        raise SystemExit("[ERROR] No models to evaluate. Check checkpoints.* in your eval config.")

    result = RunResult(specs=specs, items=items)
    ensure_dir(cfg.run.results_dir)

    if not report_only:
        _print_plan(cfg, specs, items)
        _write_metadata(cfg, specs, items)

        tokenizer = load_tokenizer(cfg)
        generator = Generator(cfg)
        total = len(specs) * len(items)
        with ProgressReporter(total=total) as progress:
            for spec in specs:
                _evaluate_model(spec, cfg, items, tokenizer, generator, progress, result)

    # --- reports -----------------------------------------------------------
    summaries = build_summaries(cfg.run.results_dir, specs, items)
    if not summaries:
        print("[WARN] No results on disk — nothing to report.")
        return result

    if cfg.reports.csv:
        result.report_paths["csv"] = write_csv_report(cfg.run.results_dir, summaries, items)
    if cfg.reports.markdown:
        result.report_paths["markdown"] = write_markdown_report(
            cfg.run.results_dir,
            summaries,
            items,
            max_response_chars=cfg.reports.max_response_chars,
            baseline=cfg.reports.pairwise.baseline,
            benchmark_path=cfg.benchmark_path,
        )
    if cfg.reports.pairwise.enabled:
        paths = write_pairwise_dataset(
            cfg.run.results_dir, summaries, items, cfg.reports.pairwise
        )
        if paths:
            result.report_paths["pairwise"], result.report_paths["pairwise_key"] = paths

    _print_summary(result, summaries)
    return result


def _print_summary(result: RunResult, summaries: Sequence[object]) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print(" Evaluation complete")
    print(sep)
    print(f"  Responses generated : {result.generated}")
    print(f"  Reused (resume)     : {result.skipped}")
    print(f"  Failed items        : {result.failed}")
    if result.skipped_models:
        print(f"  Models skipped      : {', '.join(result.skipped_models)}")
    print(f"  Models reported     : {len(summaries)}")
    if result.report_paths:
        print("\n  Reports:")
        for name, path in result.report_paths.items():
            print(f"    - {name:<13} {path}")
    print(f"{sep}\n")
