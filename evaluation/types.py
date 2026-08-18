# coding=utf-8
"""
types.py — Core data structures shared by every stage of the evaluation.

Everything that crosses a module boundary is one of these frozen dataclasses,
so the pipeline stays easy to reason about and to extend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]
ModelKind = Literal["base", "loop_untrained", "adapter"]
ScoreType = Literal["checked", "heuristic", "error"]


# ---------------------------------------------------------------------------
# Benchmark items
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Turn:
    """One message of a pre-seeded conversation history."""

    role: Role
    content: str

    def as_message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class EvalItem:
    """
    A single benchmark question.

    Only `id`, `category` and `prompt` are required; everything else refines
    how the item is presented or graded.
    """

    id: str
    category: str
    prompt: str
    system: str | None = None
    history: tuple[Turn, ...] = ()
    expected: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    max_new_tokens: int | None = None
    variant_group: str | None = None
    reference: str | None = None

    @property
    def is_scorable(self) -> bool:
        """True when the item carries at least one automatic check."""
        return bool(self.expected)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A model variant to evaluate."""

    name: str
    kind: ModelKind
    num_loops: int
    adapter_path: str | None = None
    step: int | None = None

    @property
    def slug(self) -> str:
        """Filesystem-safe directory name under results/."""
        return self.name.replace("/", "_").replace("\\", "_")

    def describe(self) -> str:
        if self.kind == "base":
            return f"{self.name} (original Qwen3, num_loops=1)"
        if self.kind == "loop_untrained":
            return f"{self.name} (looped, num_loops={self.num_loops}, no adapter)"
        return f"{self.name} (looped, num_loops={self.num_loops}, adapter={self.adapter_path})"


# ---------------------------------------------------------------------------
# Generation output
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """One model's answer to one item — serialised verbatim to responses.jsonl."""

    checkpoint: str
    category: str
    question_id: str
    prompt: str
    response: str
    generation_time: float
    tokens_generated: int
    prompt_tokens: int
    temperature: float
    top_p: float = 1.0
    seed: int = 42
    truncated: bool = False
    timestamp: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Key order matches the documented record schema."""
        return {
            "checkpoint": self.checkpoint,
            "category": self.category,
            "question_id": self.question_id,
            "prompt": self.prompt,
            "response": self.response,
            "generation_time": round(self.generation_time, 4),
            "tokens_generated": self.tokens_generated,
            "prompt_tokens": self.prompt_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "truncated": self.truncated,
            "timestamp": self.timestamp,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GenerationRecord:
        return cls(
            checkpoint=raw["checkpoint"],
            category=raw.get("category", "unknown"),
            question_id=raw["question_id"],
            prompt=raw.get("prompt", ""),
            response=raw.get("response", ""),
            generation_time=float(raw.get("generation_time", 0.0)),
            tokens_generated=int(raw.get("tokens_generated", 0)),
            prompt_tokens=int(raw.get("prompt_tokens", 0)),
            temperature=float(raw.get("temperature", 0.0)),
            top_p=float(raw.get("top_p", 1.0)),
            seed=int(raw.get("seed", 42)),
            truncated=bool(raw.get("truncated", False)),
            timestamp=raw.get("timestamp", ""),
            error=raw.get("error"),
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of one automatic check on one response."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ItemScore:
    """
    Aggregated score for one (model, item) pair, in [0.0, 1.0].

    `score_type` records how the number was produced:
      - "checked"   — fraction of declared `expected` checks that passed
      - "heuristic" — no checks declared; degeneracy/length proxy only
      - "error"     — generation failed; scored 0.0
    """

    question_id: str
    category: str
    score: float
    score_type: ScoreType
    checks: tuple[CheckResult, ...] = ()

    @property
    def passed_all(self) -> bool:
        return all(c.passed for c in self.checks) if self.checks else self.score >= 1.0
