# coding=utf-8
"""
pairwise.py — Blinded A/B dataset for LLM-as-a-Judge evaluation.

Two files are produced:

  pairwise.jsonl      what the judge sees — prompt, answer_a, answer_b, and
                      nothing that identifies which model wrote which answer
  pairwise_key.json   the de-blinding map, pair_id -> {model_a, model_b}

Side assignment is drawn from a seeded RNG keyed on the pair id, so ordering is
random with respect to model identity yet reproducible across runs.
"""

from __future__ import annotations

import os
import random
from itertools import combinations
from typing import Sequence

from ..config import PairwiseConfig
from ..storage import write_json, write_jsonl
from ..types import EvalItem
from .aggregate import ModelSummary

PAIRWISE_FILENAME = "pairwise.jsonl"
PAIRWISE_KEY_FILENAME = "pairwise_key.json"
JUDGE_PROMPT_FILENAME = "pairwise_judge_prompt.txt"

JUDGE_PROMPT_TEMPLATE = """You are grading two anonymous AI assistants.

Read the user's request and both answers. Decide which answer is better on
correctness first, then instruction-following, then helpfulness and clarity.
Length is not quality. Ignore the order in which the answers are shown.

[REQUEST]
{prompt}

[ANSWER A]
{answer_a}

[ANSWER B]
{answer_b}

Reply with JSON only:
{{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}}
"""


def _clip(text: str, limit: int) -> str:
    body = text or ""
    if limit > 0 and len(body) > limit:
        return body[:limit].rstrip() + "\n… [truncated]"
    return body


def _model_pairs(
    summaries: Sequence[ModelSummary], cfg: PairwiseConfig
) -> list[tuple[ModelSummary, ModelSummary]]:
    if cfg.mode == "all_pairs":
        return list(combinations(summaries, 2))
    baseline = next((s for s in summaries if s.name == cfg.baseline), None)
    if baseline is None:
        print(
            f"[WARN] Pairwise baseline '{cfg.baseline}' has no results — "
            "falling back to all_pairs."
        )
        return list(combinations(summaries, 2))
    return [(baseline, s) for s in summaries if s is not baseline]


def write_pairwise_dataset(
    results_root: str,
    summaries: Sequence[ModelSummary],
    items: Sequence[EvalItem],
    cfg: PairwiseConfig,
) -> tuple[str, str] | None:
    """
    Build the blinded comparison set. Returns (dataset_path, key_path), or None
    when there is nothing to compare.
    """
    if len(summaries) < 2:
        print("[WARN] Pairwise export needs at least 2 models with results — skipped.")
        return None

    pairs = _model_pairs(summaries, cfg)
    if not pairs:
        return None

    rows: list[dict] = []
    key: dict[str, dict] = {}
    skipped_identical = 0

    for item in items:
        for left, right in pairs:
            rec_l = left.records_by_id.get(item.id)
            rec_r = right.records_by_id.get(item.id)
            if rec_l is None or rec_r is None or rec_l.error or rec_r.error:
                continue
            if cfg.skip_identical and rec_l.response.strip() == rec_r.response.strip():
                skipped_identical += 1
                continue

            pair_id = f"{item.id}::{left.name}__vs__{right.name}"
            # Seed per pair: side assignment is stable run-to-run but unrelated
            # to model identity or list order.
            rng = random.Random(f"{cfg.seed}:{pair_id}")
            flip = rng.random() < 0.5
            model_a, rec_a = (right, rec_r) if flip else (left, rec_l)
            model_b, rec_b = (left, rec_l) if flip else (right, rec_r)

            rows.append(
                {
                    "pair_id": pair_id,
                    "question_id": item.id,
                    "category": item.category,
                    "system": item.system,
                    "history": [t.as_message() for t in item.history],
                    "prompt": item.prompt,
                    "reference": item.reference,
                    "answer_a": _clip(rec_a.response, cfg.max_response_chars),
                    "answer_b": _clip(rec_b.response, cfg.max_response_chars),
                }
            )
            key[pair_id] = {
                "question_id": item.id,
                "category": item.category,
                "model_a": model_a.name,
                "model_b": model_b.name,
                "a_generation_time": round(rec_a.generation_time, 4),
                "b_generation_time": round(rec_b.generation_time, 4),
                "a_tokens_generated": rec_a.tokens_generated,
                "b_tokens_generated": rec_b.tokens_generated,
            }

    if not rows:
        print("[WARN] Pairwise export produced no comparisons.")
        return None

    dataset_path = os.path.join(results_root, PAIRWISE_FILENAME)
    key_path = os.path.join(results_root, PAIRWISE_KEY_FILENAME)
    write_jsonl(dataset_path, rows)
    write_json(
        key_path,
        {
            "mode": cfg.mode,
            "baseline": cfg.baseline,
            "seed": cfg.seed,
            "pairs": key,
        },
    )
    with open(os.path.join(results_root, JUDGE_PROMPT_FILENAME), "w", encoding="utf-8") as f:
        f.write(JUDGE_PROMPT_TEMPLATE)

    if skipped_identical:
        print(f"[INFO] Pairwise: skipped {skipped_identical} identical-answer pairs.")
    return dataset_path, key_path
