# coding=utf-8
"""
benchmark.py — Loading and validation of the benchmark prompt dataset.

The dataset lives entirely in data files (JSON or YAML). Adding a question or a
whole new category never requires touching Python code.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Iterable

import yaml

from .types import EvalItem, Turn

# Canonical category order. Categories found in the data file but missing here
# are appended alphabetically, so new categories work without a code change.
CATEGORY_ORDER: tuple[str, ...] = (
    "general_knowledge",
    "reasoning",
    "mathematics",
    "programming",
    "writing",
    "translation",
    "summarization",
    "instruction_following",
    "creativity",
    "context",
    "robustness",
)

CATEGORY_LABELS: dict[str, str] = {
    "general_knowledge": "General Knowledge",
    "reasoning": "Reasoning",
    "mathematics": "Mathematics",
    "programming": "Programming",
    "writing": "Writing",
    "translation": "Translation",
    "summarization": "Summarization",
    "instruction_following": "Instruction Following",
    "creativity": "Creativity",
    "context": "Context",
    "robustness": "Robustness",
}


class BenchmarkError(ValueError):
    """Raised when the benchmark file is malformed."""


def label_for(category: str) -> str:
    """Human-readable column/heading name for a category key."""
    return CATEGORY_LABELS.get(category, category.replace("_", " ").title())


def sort_categories(categories: Iterable[str]) -> list[str]:
    """Canonical order first, then anything new in alphabetical order."""
    known = [c for c in CATEGORY_ORDER if c in set(categories)]
    extra = sorted(set(categories) - set(CATEGORY_ORDER))
    return known + extra


def _read_raw(path: str) -> Any:
    if not os.path.exists(path):
        raise BenchmarkError(f"Benchmark file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        return json.load(f)


def _parse_history(raw: Any, item_id: str) -> tuple[Turn, ...]:
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise BenchmarkError(f"[{item_id}] 'history' must be a list of messages")
    turns: list[Turn] = []
    for i, msg in enumerate(raw):
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            raise BenchmarkError(f"[{item_id}] history[{i}] needs 'role' and 'content'")
        role = str(msg["role"])
        if role not in ("system", "user", "assistant"):
            raise BenchmarkError(f"[{item_id}] history[{i}] has invalid role '{role}'")
        turns.append(Turn(role=role, content=str(msg["content"])))  # type: ignore[arg-type]
    return tuple(turns)


def _parse_item(raw: Any, index: int) -> EvalItem:
    if not isinstance(raw, dict):
        raise BenchmarkError(f"Item #{index} is not an object")
    for required in ("id", "category", "prompt"):
        if not raw.get(required):
            raise BenchmarkError(f"Item #{index} is missing required field '{required}'")

    item_id = str(raw["id"])
    expected = raw.get("expected") or {}
    if not isinstance(expected, dict):
        raise BenchmarkError(f"[{item_id}] 'expected' must be an object")

    max_new = raw.get("max_new_tokens")
    if max_new is not None:
        max_new = int(max_new)
        if max_new <= 0:
            raise BenchmarkError(f"[{item_id}] 'max_new_tokens' must be positive")

    return EvalItem(
        id=item_id,
        category=str(raw["category"]),
        prompt=str(raw["prompt"]),
        system=str(raw["system"]) if raw.get("system") else None,
        history=_parse_history(raw.get("history"), item_id),
        expected=expected,
        tags=tuple(str(t) for t in (raw.get("tags") or ())),
        max_new_tokens=max_new,
        variant_group=str(raw["variant_group"]) if raw.get("variant_group") else None,
        reference=str(raw["reference"]) if raw.get("reference") else None,
    )


def load_benchmark(path: str, categories: Iterable[str] | None = None) -> list[EvalItem]:
    """
    Parse the benchmark file into `EvalItem`s.

    Accepts either a bare list of items or `{"items": [...]}` with optional
    sibling metadata. Duplicate ids are rejected — they would silently collide
    in the resume index and in every report.
    """
    raw = _read_raw(path)
    if isinstance(raw, dict):
        raw_items = raw.get("items")
        if raw_items is None:
            raise BenchmarkError(f"{path}: object form must contain an 'items' list")
    else:
        raw_items = raw
    if not isinstance(raw_items, list) or not raw_items:
        raise BenchmarkError(f"{path}: expected a non-empty list of items")

    items = [_parse_item(r, i) for i, r in enumerate(raw_items)]

    duplicates = [i for i, n in Counter(it.id for it in items).items() if n > 1]
    if duplicates:
        raise BenchmarkError(f"{path}: duplicate item ids: {', '.join(sorted(duplicates))}")

    if categories:
        wanted = set(categories)
        items = [it for it in items if it.category in wanted]
        if not items:
            raise BenchmarkError(f"No benchmark items left after filtering to {sorted(wanted)}")

    return items


def category_counts(items: Iterable[EvalItem]) -> dict[str, int]:
    """Item count per category, in canonical order."""
    counts = Counter(it.category for it in items)
    return {c: counts[c] for c in sort_categories(counts)}


def variant_groups(items: Iterable[EvalItem]) -> dict[str, list[EvalItem]]:
    """Robustness paraphrase groups, keyed by `variant_group`."""
    groups: dict[str, list[EvalItem]] = {}
    for item in items:
        if item.variant_group:
            groups.setdefault(item.variant_group, []).append(item)
    return {k: v for k, v in groups.items() if len(v) > 1}
