# coding=utf-8
"""
markdown_report.py — summary.md: the human-readable comparison.

Layout: headline recovery table, per-category scoreboard, cost table, then
every question with all models' answers stacked underneath it for eyeballing.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Sequence

from ..benchmark import category_counts, label_for, sort_categories, variant_groups
from ..storage import write_text
from ..types import EvalItem
from .aggregate import ModelSummary

MARKDOWN_FILENAME = "summary.md"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out.extend("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    out.append("")
    return out


def _truncate(text: str, limit: int) -> str:
    body = (text or "").strip()
    if not body:
        return "(empty response)"
    if limit > 0 and len(body) > limit:
        return body[:limit].rstrip() + f"\n… [truncated, {len(body)} chars total]"
    return body


def _fence(text: str) -> str:
    """Pick a fence long enough to survive code blocks inside the response."""
    longest = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            longest = max(longest, len(stripped) - len(stripped.lstrip("`")))
    return "`" * max(3, longest + 1)


def _recovery_section(summaries: Sequence[ModelSummary], baseline: str) -> list[str]:
    base = next((s for s in summaries if s.name == baseline), None)
    lines = ["## Capability recovery", ""]
    if base is None:
        lines += [f"_Baseline `{baseline}` has no results; recovery deltas omitted._", ""]
        return lines + _table(
            ["Model", "Overall"], [[s.name, _pct(s.overall)] for s in summaries]
        )

    base_overall = base.overall
    rows = []
    for s in summaries:
        delta = s.overall - base_overall
        recovered = (s.overall / base_overall * 100) if base_overall > 0 else 0.0
        rows.append(
            [
                s.name,
                _pct(s.overall),
                f"{delta * 100:+.1f}",
                f"{recovered:.1f}%" if s.name != baseline else "—",
            ]
        )
    lines += [
        f"Overall score is the macro-average of category scores. "
        f"`% of baseline` compares each model against `{baseline}`.",
        "",
    ]
    return lines + _table(["Model", "Overall", "Δ vs baseline", "% of baseline"], rows)


def _scoreboard_section(
    summaries: Sequence[ModelSummary], items: Sequence[EvalItem]
) -> list[str]:
    categories = sort_categories({i.category for i in items})
    counts = category_counts(items)
    header = ["Category", "Items", *(s.name for s in summaries)]
    rows = []
    for cat in categories:
        rows.append(
            [
                label_for(cat),
                str(counts.get(cat, 0)),
                *(_pct(s.category_scores.get(cat)) for s in summaries),
            ]
        )
    rows.append(["**Overall**", str(len(items)), *(f"**{_pct(s.overall)}**" for s in summaries)])
    return ["## Scores by category", "", *_table(header, rows)]


def _cost_section(summaries: Sequence[ModelSummary]) -> list[str]:
    rows = [
        [
            s.name,
            str(len(s.records)),
            str(s.error_count),
            str(s.truncated_count),
            f"{s.mean_latency:.2f}",
            f"{s.median_latency:.2f}",
            f"{s.mean_prompt_tokens:.0f}",
            f"{s.mean_tokens:.0f}",
            f"{s.tokens_per_second:.1f}",
            f"{s.mean_distinct_ratio:.2f}",
        ]
        for s in summaries
    ]
    header = [
        "Model", "Items", "Errors", "Truncated",
        "Mean latency (s)", "Median latency (s)",
        "Mean prompt tok", "Mean gen tok", "Tokens/s", "Distinct 4-gram",
    ]
    return [
        "## Latency and token usage",
        "",
        "`Distinct 4-gram` is the mean ratio of unique to total 4-grams per "
        "response (1.0 = no repetition). A drop against the base model means the "
        "checkpoint is looping on itself, regardless of whether answers are correct.",
        "",
        *_table(header, rows),
    ]


def _consistency_section(
    summaries: Sequence[ModelSummary], items: Sequence[EvalItem]
) -> list[str]:
    groups = variant_groups(items)
    if not groups:
        return []
    header = ["Paraphrase group", *(s.name for s in summaries)]
    rows = []
    for group in sorted(groups):
        row = [group]
        for s in summaries:
            value = s.consistency(items).get(group)
            row.append("—" if value is None else f"{value:.2f}")
        rows.append(row)
    mean_row = ["**Mean**"]
    for s in summaries:
        value = s.mean_consistency(items)
        mean_row.append("—" if value is None else f"**{value:.2f}**")
    rows.append(mean_row)
    return [
        "## Robustness: answer consistency across paraphrases",
        "",
        "Mean pairwise word-overlap (Jaccard) between a model's answers to "
        "differently-worded versions of the same question. Higher means the "
        "model is less sensitive to phrasing.",
        "",
        *_table(header, rows),
    ]


def _responses_section(
    summaries: Sequence[ModelSummary],
    items: Sequence[EvalItem],
    max_chars: int,
) -> list[str]:
    lines = ["## Responses by question", ""]
    by_category: dict[str, list[EvalItem]] = {}
    for item in items:
        by_category.setdefault(item.category, []).append(item)

    for category in sort_categories(by_category):
        lines += [f"### {label_for(category)}", ""]
        for item in by_category[category]:
            lines += [f"#### `{item.id}`", ""]
            if item.system:
                lines += [f"**System:** {item.system}", ""]
            if item.history:
                lines += ["<details><summary>Conversation history</summary>", ""]
                lines += [f"- **{t.role}:** {t.content}" for t in item.history]
                lines += ["", "</details>", ""]
            lines += ["**Prompt:**", "", "> " + item.prompt.replace("\n", "\n> "), ""]
            if item.reference:
                lines += [f"**Reference answer:** {item.reference}", ""]

            for summary in summaries:
                record = summary.records_by_id.get(item.id)
                if record is None:
                    lines += [f"**{summary.name}** — _not evaluated_", ""]
                    continue
                score = summary.scores_by_id.get(item.id)
                meta = [
                    f"{record.generation_time:.2f}s",
                    f"{record.tokens_generated} tok",
                ]
                if score is not None:
                    meta.append(f"score {score.score * 100:.0f}% ({score.score_type})")
                if record.truncated:
                    meta.append("truncated")
                lines += [f"**{summary.name}** — {' · '.join(meta)}", ""]
                if record.error:
                    lines += [f"> ERROR: {record.error}", ""]
                    continue
                body = _truncate(record.response, max_chars)
                fence = _fence(body)
                lines += [fence, body, fence, ""]
                if score is not None and score.checks and score.score < 1.0:
                    failed = [c for c in score.checks if not c.passed]
                    if failed:
                        lines += [
                            "Failed checks: "
                            + ", ".join(f"`{c.name}` ({c.detail})" for c in failed),
                            "",
                        ]
            lines += ["---", ""]
    return lines


def write_markdown_report(
    results_root: str,
    summaries: Sequence[ModelSummary],
    items: Sequence[EvalItem],
    *,
    max_response_chars: int = 1200,
    baseline: str = "base",
    benchmark_path: str = "",
) -> str:
    """Render summary.md and return its path."""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Qwen3Loop checkpoint evaluation",
        "",
        f"_Generated {generated}_",
        "",
        f"- Benchmark: `{benchmark_path or 'n/a'}` — {len(items)} items across "
        f"{len(set(i.category for i in items))} categories",
        f"- Models evaluated: {len(summaries)}",
        "- Decoding: greedy (temperature 0, top_p 1), fixed seed, identical for every model",
        "",
        "> Scores below are automatic constraint checks (exact answers, formats, "
        "word counts, language). Open-ended writing and creativity items have no "
        "checkable answer and are scored only for non-degeneracy — use "
        "`pairwise.jsonl` with an LLM judge to rank their quality.",
        "",
    ]
    lines += _recovery_section(summaries, baseline)
    lines += _scoreboard_section(summaries, items)
    lines += _cost_section(summaries)
    lines += _consistency_section(summaries, items)
    lines += _responses_section(summaries, items, max_response_chars)

    path = os.path.join(results_root, MARKDOWN_FILENAME)
    write_text(path, "\n".join(lines))
    return path
