# coding=utf-8
"""
config.py — Typed view over eval_config.yaml.

Keeps every tunable in one file and gives the rest of the package attribute
access with defaults, so a partial config never crashes the run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


def _get(cfg: dict[str, Any], path: str, default: Any) -> Any:
    """Dotted-path lookup with a default (`_get(cfg, "run.seed", 42)`)."""
    node: Any = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return default if node is None else node


@dataclass(slots=True)
class RunConfig:
    results_dir: str = "results"
    seed: int = 42
    resume: bool = True
    device: str = "auto"
    limit: int | None = None
    categories: tuple[str, ...] = ()


@dataclass(slots=True)
class ModelConfig:
    base_model_id: str = "Qwen/Qwen3-0.6B"
    num_loops: int = 2
    dtype: str = "bfloat16"
    enable_thinking: bool = False
    trust_remote_code: bool = False


@dataclass(slots=True)
class QuantConfig:
    enabled: bool = True
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


@dataclass(slots=True)
class GenerationConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    repetition_penalty: float = 1.0
    default_max_new_tokens: int = 256
    max_new_tokens_by_category: dict[str, int] = field(default_factory=dict)

    def max_new_tokens_for(self, category: str, override: int | None) -> int:
        if override is not None:
            return override
        return int(self.max_new_tokens_by_category.get(category, self.default_max_new_tokens))


@dataclass(slots=True)
class CheckpointConfig:
    root: str = "checkpoints/qwen3loop_qlora"
    auto_discover: bool = True
    include_base: bool = True
    include_untrained_loop: bool = True
    final_adapter_dirs: tuple[str, ...] = ("final_adapter", "final", "final_model")
    deduplicate_identical: bool = True
    entries: tuple[dict[str, Any], ...] = ()


@dataclass(slots=True)
class PairwiseConfig:
    enabled: bool = True
    mode: str = "vs_base"          # "vs_base" | "all_pairs"
    baseline: str = "base"
    seed: int = 1234
    max_response_chars: int = 4000
    skip_identical: bool = True


@dataclass(slots=True)
class ReportConfig:
    markdown: bool = True
    csv: bool = True
    max_response_chars: int = 1200
    pairwise: PairwiseConfig = field(default_factory=PairwiseConfig)


@dataclass(slots=True)
class EvalConfig:
    benchmark_path: str = "benchmarks/eval_benchmark.json"
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantConfig = field(default_factory=QuantConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    reports: ReportConfig = field(default_factory=ReportConfig)
    source_path: str = ""

    @property
    def results_dir(self) -> str:
        return self.run.results_dir


def load_eval_config(path: str) -> EvalConfig:
    """Load eval_config.yaml. A missing file yields the documented defaults."""
    raw: dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    elif path:
        print(f"[WARN] Eval config '{path}' not found — using built-in defaults.")

    pairwise = PairwiseConfig(
        enabled=bool(_get(raw, "reports.pairwise.enabled", True)),
        mode=str(_get(raw, "reports.pairwise.mode", "vs_base")),
        baseline=str(_get(raw, "reports.pairwise.baseline", "base")),
        seed=int(_get(raw, "reports.pairwise.seed", 1234)),
        max_response_chars=int(_get(raw, "reports.pairwise.max_response_chars", 4000)),
        skip_identical=bool(_get(raw, "reports.pairwise.skip_identical", True)),
    )

    return EvalConfig(
        benchmark_path=str(_get(raw, "benchmark.path", "benchmarks/eval_benchmark.json")),
        run=RunConfig(
            results_dir=str(_get(raw, "run.results_dir", "results")),
            seed=int(_get(raw, "run.seed", 42)),
            resume=bool(_get(raw, "run.resume", True)),
            device=str(_get(raw, "run.device", "auto")),
            limit=(int(_get(raw, "run.limit", 0)) or None),
            categories=tuple(_get(raw, "run.categories", ()) or ()),
        ),
        model=ModelConfig(
            base_model_id=str(_get(raw, "model.base_model_id", "Qwen/Qwen3-0.6B")),
            num_loops=int(_get(raw, "model.num_loops", 2)),
            dtype=str(_get(raw, "model.dtype", "bfloat16")),
            enable_thinking=bool(_get(raw, "model.enable_thinking", False)),
            trust_remote_code=bool(_get(raw, "model.trust_remote_code", False)),
        ),
        quantization=QuantConfig(
            enabled=bool(_get(raw, "quantization.enabled", True)),
            load_in_4bit=bool(_get(raw, "quantization.load_in_4bit", True)),
            bnb_4bit_quant_type=str(_get(raw, "quantization.bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(_get(raw, "quantization.bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=str(_get(raw, "quantization.bnb_4bit_compute_dtype", "bfloat16")),
        ),
        generation=GenerationConfig(
            temperature=float(_get(raw, "generation.temperature", 0.0)),
            top_p=float(_get(raw, "generation.top_p", 1.0)),
            do_sample=bool(_get(raw, "generation.do_sample", False)),
            repetition_penalty=float(_get(raw, "generation.repetition_penalty", 1.0)),
            default_max_new_tokens=int(_get(raw, "generation.default_max_new_tokens", 256)),
            max_new_tokens_by_category=dict(_get(raw, "generation.max_new_tokens_by_category", {})),
        ),
        checkpoints=CheckpointConfig(
            root=str(_get(raw, "checkpoints.root", "checkpoints/qwen3loop_qlora")),
            auto_discover=bool(_get(raw, "checkpoints.auto_discover", True)),
            include_base=bool(_get(raw, "checkpoints.include_base", True)),
            include_untrained_loop=bool(_get(raw, "checkpoints.include_untrained_loop", True)),
            final_adapter_dirs=tuple(_get(raw, "checkpoints.final_adapter_dirs", ("final_adapter", "final", "final_model"))),
            deduplicate_identical=bool(_get(raw, "checkpoints.deduplicate_identical", True)),
            entries=tuple(_get(raw, "checkpoints.list", ()) or ()),
        ),
        reports=ReportConfig(
            markdown=bool(_get(raw, "reports.markdown", True)),
            csv=bool(_get(raw, "reports.csv", True)),
            max_response_chars=int(_get(raw, "reports.max_response_chars", 1200)),
            pairwise=pairwise,
        ),
        source_path=path,
    )
