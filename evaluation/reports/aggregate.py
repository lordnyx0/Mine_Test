# coding=utf-8
"""
aggregate.py — Turns raw JSONL into the per-model summary every report consumes.

Reports read only from disk, so `--report-only` can regenerate all of them from
an earlier run without touching a GPU.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Sequence

from ..benchmark import variant_groups
from ..scoring import (
    category_means,
    consistency_by_group,
    distinct_ngram_ratio,
    overall_score,
    score_all,
)
from ..storage import load_records, responses_path
from ..types import EvalItem, GenerationRecord, ItemScore, ModelSpec


@dataclass(slots=True)
class ModelSummary:
    """Everything the reports need about one model variant."""

    name: str
    spec: ModelSpec | None
    records: list[GenerationRecord] = field(default_factory=list)
    scores: list[ItemScore] = field(default_factory=list)

    @property
    def records_by_id(self) -> dict[str, GenerationRecord]:
        return {r.question_id: r for r in self.records}

    @property
    def scores_by_id(self) -> dict[str, ItemScore]:
        return {s.question_id: s for s in self.scores}

    @property
    def category_scores(self) -> dict[str, float]:
        return category_means(self.scores)

    @property
    def overall(self) -> float:
        return overall_score(self.scores)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.records if r.error)

    @property
    def truncated_count(self) -> int:
        return sum(1 for r in self.records if r.truncated)

    def _ok_records(self) -> list[GenerationRecord]:
        return [r for r in self.records if not r.error]

    @property
    def mean_latency(self) -> float:
        ok = self._ok_records()
        return statistics.fmean(r.generation_time for r in ok) if ok else 0.0

    @property
    def median_latency(self) -> float:
        ok = self._ok_records()
        return statistics.median(r.generation_time for r in ok) if ok else 0.0

    @property
    def total_generation_time(self) -> float:
        return sum(r.generation_time for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_generated for r in self.records)

    @property
    def mean_tokens(self) -> float:
        ok = self._ok_records()
        return statistics.fmean(r.tokens_generated for r in ok) if ok else 0.0

    @property
    def mean_prompt_tokens(self) -> float:
        ok = self._ok_records()
        return statistics.fmean(r.prompt_tokens for r in ok) if ok else 0.0

    @property
    def tokens_per_second(self) -> float:
        elapsed = sum(r.generation_time for r in self._ok_records())
        return (sum(r.tokens_generated for r in self._ok_records()) / elapsed) if elapsed > 0 else 0.0

    @property
    def mean_distinct_ratio(self) -> float:
        """
        Mean distinct-4-gram ratio across responses.

        Looping a decoder without recovery training tends to produce text that
        repeats itself; this number drops well below the base model's when that
        happens, independently of whether the answer was correct.
        """
        ok = [r for r in self._ok_records() if r.response.strip()]
        return statistics.fmean(distinct_ngram_ratio(r.response) for r in ok) if ok else 0.0

    def consistency(self, items: Sequence[EvalItem]) -> dict[str, float]:
        return consistency_by_group(variant_groups(items), self.records_by_id)

    def mean_consistency(self, items: Sequence[EvalItem]) -> float | None:
        values = list(self.consistency(items).values())
        return statistics.fmean(values) if values else None


def build_summaries(
    results_root: str,
    specs: Sequence[ModelSpec],
    items: Sequence[EvalItem],
) -> list[ModelSummary]:
    """
    Load and score `results/<model>/responses.jsonl` for each spec.

    Models with no results on disk are dropped, so a partial run still reports.
    """
    summaries: list[ModelSummary] = []
    for spec in specs:
        records = load_records(responses_path(results_root, spec.slug))
        if not records:
            continue
        summaries.append(
            ModelSummary(
                name=spec.name,
                spec=spec,
                records=records,
                scores=score_all(items, records),
            )
        )
    return summaries
