# coding=utf-8
"""
evaluation — Capability-recovery benchmark for Qwen3Loop checkpoints.

This package measures how much of the original Qwen3-0.6B behaviour each QLoRA
checkpoint recovers after the looped-decoder architectural change. It is
deliberately *not* a perplexity harness: it generates real responses to a
categorised prompt suite and reports them side by side.

Public entry point: `evaluation.runner.run`.
"""

from .types import (
    CheckResult,
    EvalItem,
    GenerationRecord,
    ItemScore,
    ModelSpec,
    Turn,
)

__all__ = [
    "CheckResult",
    "EvalItem",
    "GenerationRecord",
    "ItemScore",
    "ModelSpec",
    "Turn",
]
