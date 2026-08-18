# coding=utf-8
"""
csv_report.py — summary.csv: one row per model, one column per category.

Scores are percentages (0-100) of the automatic constraint checks defined in
the benchmark. Empty cells mean the model produced no scored item in that
category.
"""

from __future__ import annotations

import csv
import os
from typing import Sequence

from ..benchmark import label_for, sort_categories
from ..storage import ensure_dir
from ..types import EvalItem
from .aggregate import ModelSummary

CSV_FILENAME = "summary.csv"
LATENCY_CSV_FILENAME = "summary_latency.csv"


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value * 100:.1f}"


def write_csv_report(
    results_root: str,
    summaries: Sequence[ModelSummary],
    items: Sequence[EvalItem],
) -> str:
    """Write summary.csv (+ summary_latency.csv) and return the main path."""
    categories = sort_categories({item.category for item in items})
    path = os.path.join(results_root, CSV_FILENAME)
    ensure_dir(results_root)

    header = ["Checkpoint", *(label_for(c) for c in categories), "Overall"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for summary in summaries:
            cat_scores = summary.category_scores
            writer.writerow(
                [summary.name, *(_fmt(cat_scores.get(c)) for c in categories), _fmt(summary.overall)]
            )

    _write_latency_csv(results_root, summaries, items)
    return path


def _write_latency_csv(
    results_root: str,
    summaries: Sequence[ModelSummary],
    items: Sequence[EvalItem],
) -> str:
    """Companion file with the cost side of the comparison."""
    path = os.path.join(results_root, LATENCY_CSV_FILENAME)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Checkpoint",
                "Items",
                "Errors",
                "Truncated",
                "Mean Latency (s)",
                "Median Latency (s)",
                "Total Time (s)",
                "Mean Prompt Tokens",
                "Mean Generated Tokens",
                "Total Generated Tokens",
                "Tokens/s",
                "Mean Distinct 4-gram Ratio",
                "Mean Robustness Consistency",
            ]
        )
        for s in summaries:
            consistency = s.mean_consistency(items)
            writer.writerow(
                [
                    s.name,
                    len(s.records),
                    s.error_count,
                    s.truncated_count,
                    f"{s.mean_latency:.3f}",
                    f"{s.median_latency:.3f}",
                    f"{s.total_generation_time:.1f}",
                    f"{s.mean_prompt_tokens:.1f}",
                    f"{s.mean_tokens:.1f}",
                    s.total_tokens,
                    f"{s.tokens_per_second:.2f}",
                    f"{s.mean_distinct_ratio:.3f}",
                    "" if consistency is None else f"{consistency:.3f}",
                ]
            )
    return path
