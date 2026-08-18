# coding=utf-8
"""
models.py — Checkpoint discovery and model loading.

Three model kinds are supported:
  base            — the untouched Qwen3-0.6B (num_loops = 1), the recovery target
  loop_untrained  — the looped architecture with no adapter (the damage baseline)
  adapter         — looped architecture + a QLoRA checkpoint

All three are loaded under the *same* quantization settings so the comparison
isolates the architectural/adapter effect rather than numeric precision.
"""

from __future__ import annotations

import gc
import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import EvalConfig
from .types import ModelSpec

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
_ADAPTER_FILES = ("adapter_config.json",)
_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin", "model.safetensors", "pytorch_model.bin")


class ModelLoadError(RuntimeError):
    """Raised when a model variant cannot be materialised."""


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def is_adapter_dir(path: str) -> bool:
    """True when `path` looks like a saved PEFT adapter or full model checkpoint."""
    if not os.path.isdir(path):
        return False
    has_adapter = any(os.path.exists(os.path.join(path, f)) for f in _ADAPTER_FILES)
    has_full = os.path.exists(os.path.join(path, "config.json")) and (
        os.path.exists(os.path.join(path, "model.safetensors"))
        or os.path.exists(os.path.join(path, "pytorch_model.bin"))
    )
    return has_adapter or has_full


def discover_checkpoints(root: str, final_dirs: tuple[str, ...]) -> list[ModelSpec]:
    """
    Scan `root` for `checkpoint-N` directories and any final-adapter directory.

    Returns specs sorted by training step, with the final adapter last.
    """
    search_roots = [root]
    # If the specified root does not exist, search adjacent subdirectories in parent
    parent = os.path.dirname(root)
    if not os.path.isdir(root) and parent and os.path.isdir(parent):
        for sub in sorted(os.listdir(parent)):
            candidate = os.path.join(parent, sub)
            if os.path.isdir(candidate) and candidate not in search_roots:
                search_roots.append(candidate)

    numbered: list[ModelSpec] = []
    finals: list[ModelSpec] = []

    for r in search_roots:
        if not os.path.isdir(r):
            continue
        for entry in sorted(os.listdir(r)):
            path = os.path.join(r, entry)
            if not is_adapter_dir(path):
                continue
            match = _CHECKPOINT_RE.match(entry)
            if match:
                numbered.append(
                    ModelSpec(name=entry, kind="adapter", num_loops=0,
                              adapter_path=path, step=int(match.group(1)))
                )
            elif entry in final_dirs:
                finals.append(
                    ModelSpec(name="final", kind="adapter", num_loops=0,
                              adapter_path=path, step=None)
                )

    numbered.sort(key=lambda s: s.step or 0)
    return numbered + finals[:1]


def adapter_fingerprint(path: str) -> str | None:
    """
    Content hash of an adapter's weight file, or None if there is none.

    Trainers routinely save the last checkpoint twice — once as `checkpoint-N`
    and again as `final_adapter` — and greedy decoding over identical weights
    produces byte-identical transcripts. Hashing lets the runner notice that and
    skip the duplicate work.
    """
    for filename in _WEIGHT_FILES:
        weights = os.path.join(path, filename)
        if not os.path.exists(weights):
            continue
        digest = hashlib.blake2b(digest_size=16)
        with open(weights, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return f"{os.path.getsize(weights)}:{digest.hexdigest()}"
    return None


def deduplicate_specs(specs: list[ModelSpec]) -> list[ModelSpec]:
    """
    Collapse adapter specs whose weights are byte-identical.

    The surviving entry keeps the `final` name when one of the twins is the
    final adapter — that is the more meaningful label for the end state — and
    inherits the training step from its numbered twin so the recovery curve
    still knows where it sits.
    """
    fingerprints: dict[str, list[int]] = {}
    for index, spec in enumerate(specs):
        if spec.kind != "adapter" or not spec.adapter_path:
            continue
        fingerprint = adapter_fingerprint(spec.adapter_path)
        if fingerprint:
            fingerprints.setdefault(fingerprint, []).append(index)

    drop: set[int] = set()
    replace: dict[int, ModelSpec] = {}
    for indices in fingerprints.values():
        if len(indices) < 2:
            continue
        group = [(i, specs[i]) for i in indices]
        keep_index, keep_spec = next(
            ((i, s) for i, s in group if s.name == "final"), group[0]
        )
        step = keep_spec.step or next((s.step for _, s in group if s.step is not None), None)
        if step != keep_spec.step:
            replace[keep_index] = ModelSpec(
                name=keep_spec.name,
                kind=keep_spec.kind,
                num_loops=keep_spec.num_loops,
                adapter_path=keep_spec.adapter_path,
                step=step,
            )
        dropped = [s.name for i, s in group if i != keep_index]
        drop.update(i for i, _ in group if i != keep_index)
        print(
            f"[INFO] '{keep_spec.name}' has identical weights to "
            f"{', '.join(repr(n) for n in dropped)} — evaluating it once."
        )

    return [replace.get(i, s) for i, s in enumerate(specs) if i not in drop]


def resolve_model_specs(cfg: EvalConfig) -> list[ModelSpec]:
    """
    Build the ordered list of models to evaluate.

    Explicit `checkpoints.list` entries win; `auto_discover` appends anything
    on disk that the list did not already name. Configured checkpoints that no
    longer exist (e.g. rotated away by `save_total_limit`) are reported and
    skipped rather than aborting the run.
    """
    ck = cfg.checkpoints
    num_loops = cfg.model.num_loops
    specs: list[ModelSpec] = []
    seen: set[str] = set()

    def add(spec: ModelSpec) -> None:
        if spec.name in seen:
            return
        seen.add(spec.name)
        specs.append(spec)

    if ck.include_base:
        add(ModelSpec(name="base", kind="base", num_loops=1))
    if ck.include_untrained_loop:
        add(ModelSpec(name="loop-untrained", kind="loop_untrained", num_loops=num_loops))

    missing: list[str] = []
    for entry in ck.entries:
        if isinstance(entry, str):
            entry = {"name": entry}
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        kind = str(entry.get("kind") or "adapter")
        if kind == "base":
            add(ModelSpec(name=name, kind="base", num_loops=1))
            continue
        if kind == "loop_untrained":
            add(ModelSpec(name=name, kind="loop_untrained", num_loops=num_loops))
            continue

        path = str(entry.get("path") or os.path.join(ck.root, name))
        if not is_adapter_dir(path):
            missing.append(f"{name} -> {path}")
            continue
        match = _CHECKPOINT_RE.match(os.path.basename(path.rstrip("/\\")))
        add(ModelSpec(name=name, kind="adapter", num_loops=num_loops,
                      adapter_path=path, step=int(match.group(1)) if match else None))

    if ck.auto_discover:
        for spec in discover_checkpoints(ck.root, ck.final_adapter_dirs):
            add(ModelSpec(name=spec.name, kind="adapter", num_loops=num_loops,
                          adapter_path=spec.adapter_path, step=spec.step))

    if missing:
        print("[WARN] Configured checkpoints not found on disk (skipped):")
        for m in missing:
            print(f"       - {m}")
        print("       (save_total_limit in config.yaml rotates older checkpoints away.)")

    if ck.deduplicate_identical:
        specs = deduplicate_specs(specs)

    return specs


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LoadedModel:
    """A model ready for generation, plus its tokenizer."""

    spec: ModelSpec
    model: Any
    tokenizer: Any
    device: str

    def release(self) -> None:
        """Drop references and reclaim VRAM before the next variant loads."""
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def resolve_device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ModelLoadError(f"Unknown torch dtype: {name}")
    return dtype


def build_quant_config(cfg: EvalConfig) -> BitsAndBytesConfig | None:
    q = cfg.quantization
    if not q.enabled or not q.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=q.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=q.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=_dtype(q.bnb_4bit_compute_dtype),
    )


def load_tokenizer(cfg: EvalConfig) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.base_model_id, trust_remote_code=cfg.model.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _from_pretrained(loader: Any, model_id: str, **kwargs: Any) -> Any:
    """
    Call `from_pretrained`, tolerating the `dtype`/`torch_dtype` rename in
    recent transformers releases.
    """
    try:
        return loader.from_pretrained(model_id, **kwargs)
    except TypeError as exc:
        if "dtype" not in kwargs or "dtype" not in str(exc):
            raise
        legacy = dict(kwargs)
        legacy["torch_dtype"] = legacy.pop("dtype")
        return loader.from_pretrained(model_id, **legacy)


def load_model(spec: ModelSpec, cfg: EvalConfig, tokenizer: Any) -> LoadedModel:
    """
    Materialise one model variant.

    Raises `ModelLoadError` with context; the runner catches it and continues
    with the remaining variants.
    """
    device = resolve_device(cfg.run.device)
    quant = build_quant_config(cfg)
    dtype = _dtype(cfg.model.dtype)

    kwargs: dict[str, Any] = {"dtype": dtype}
    if quant is not None:
        kwargs["quantization_config"] = quant
        kwargs["device_map"] = "auto"
    if cfg.model.trust_remote_code:
        kwargs["trust_remote_code"] = True

    try:
        if spec.kind == "base":
            model = _from_pretrained(AutoModelForCausalLM, cfg.model.base_model_id, **kwargs)
        else:
            from qwen3loop import Qwen3LoopForCausalLM  # local import: heavy + optional

            if spec.kind == "adapter":
                if not spec.adapter_path or not is_adapter_dir(spec.adapter_path):
                    raise ModelLoadError(f"Adapter directory missing: {spec.adapter_path}")
                is_peft = os.path.exists(os.path.join(spec.adapter_path, "adapter_config.json"))
                if is_peft:
                    model = _from_pretrained(
                        Qwen3LoopForCausalLM,
                        cfg.model.base_model_id,
                        num_loops=spec.num_loops,
                        **kwargs,
                    )
                    from peft import PeftModel  # local import for the same reason
                    model = PeftModel.from_pretrained(model, spec.adapter_path)
                else:
                    # Full fine-tune checkpoint! Load weights directly from checkpoint directory
                    model = _from_pretrained(
                        Qwen3LoopForCausalLM,
                        spec.adapter_path,
                        num_loops=spec.num_loops,
                        **kwargs,
                    )
            else:
                model = _from_pretrained(
                    Qwen3LoopForCausalLM,
                    cfg.model.base_model_id,
                    num_loops=spec.num_loops,
                    **kwargs,
                )

        if quant is None:
            model = model.to(device)
        model.eval()
        if getattr(model, "config", None) is not None:
            model.config.use_cache = True

        # Qwen3 ships sampling defaults in generation_config; under greedy
        # decoding transformers warns about them on every single call.
        gen_defaults = getattr(model, "generation_config", None)
        if gen_defaults is not None:
            gen_defaults.do_sample = False
            for attr in ("temperature", "top_p", "top_k"):
                if hasattr(gen_defaults, attr):
                    setattr(gen_defaults, attr, None)
    except ModelLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 — surfaced as a skipped variant
        raise ModelLoadError(f"Failed to load '{spec.name}': {exc}") from exc

    return LoadedModel(spec=spec, model=model, tokenizer=tokenizer, device=device)
