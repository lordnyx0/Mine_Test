# coding=utf-8
"""
scoring.py — Automatic, deterministic grading of responses.

What this does and does not measure
-----------------------------------
Each benchmark item may declare an `expected` block of *checkable constraints*
(a required substring, a word count, valid JSON, ...). The score for an item is
the fraction of its constraints that the response satisfies. That is an
objective, reproducible signal, and it is what the CSV reports.

It is deliberately not a judgement of prose quality. Open-ended items (writing,
creativity) carry no constraints and fall back to a degeneracy proxy — is the
answer non-empty, and does it avoid looping on itself? Genuine quality ranking
is what the pairwise LLM-as-a-Judge export exists for.

Adding a new constraint type means adding one function and one entry to
`CHECK_HANDLERS` — no other module changes.
"""

from __future__ import annotations

import json
import re
import statistics
import unicodedata
from typing import Any, Callable, Iterable, Sequence

from .types import CheckResult, EvalItem, GenerationRecord, ItemScore

# --- text utilities --------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_BULLET = re.compile(r"^\s*(?:[-*•–—]|•)\s+\S")
_NUMBERED = re.compile(r"^\s*\d+\s*[.)]\s+\S")
_SENTENCE_SPLIT = re.compile(r"[.!?]+(?:\s|$)")
_FENCED = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)```", re.DOTALL)
_CODE_HINT = re.compile(
    r"(=|\(|\)|\{|\}|;|:\s*$|->|=>|#include|\bdef\b|\breturn\b|\bfunction\b|"
    r"\bconst\b|\blet\b|\bvar\b|\bclass\b|\bimport\b|\bfor\b|\bif\b)"
)

_PT_MARKERS = frozenset(
    "que não uma com para mais como isso você são está então também muito "
    "seu sua dos das nós pelo pela quando porque já ser tem".split()
)
_EN_MARKERS = frozenset(
    "the and is to of in that it for you with this are was were their there "
    "have has been they which".split()
)


def normalize(text: str, *, strip_accents: bool = True) -> str:
    """Lowercase, drop punctuation and collapse whitespace (and optionally accents)."""
    out = text.strip().lower()
    if strip_accents:
        out = "".join(
            c for c in unicodedata.normalize("NFD", out) if unicodedata.category(c) != "Mn"
        )
    out = _PUNCT.sub(" ", out)
    return _WS.sub(" ", out).strip()


def words(text: str) -> list[str]:
    return [w for w in _WS.split(text.strip()) if w]


def word_count(text: str) -> int:
    return len(words(text))


def sentence_count(text: str) -> int:
    return len([s for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()])


def bullet_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if _BULLET.match(line))


def numbered_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if _NUMBERED.match(line))


def distinct_ngram_ratio(text: str, n: int = 4) -> float:
    """
    Ratio of unique n-grams to total n-grams — a cheap degeneracy detector.

    A model stuck in a repetition loop scores near 0; healthy prose sits well
    above 0.7. Texts shorter than `n` words are treated as non-degenerate.
    """
    toks = words(normalize(text))
    if len(toks) < n:
        return 1.0
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams)


def extract_json(text: str) -> Any:
    """Parse the response as JSON, tolerating a ```json fence or surrounding prose."""
    candidates: list[str] = []
    fenced = _FENCED.findall(text)
    candidates.extend(block.strip() for block in fenced)
    stripped = text.strip()
    candidates.append(stripped)
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = stripped.find(opener), stripped.rfind(closer)
        if 0 <= start < end:
            candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("no valid JSON found")


def _looks_like_code(line: str) -> bool:
    return bool(line[:1].isspace() or _CODE_HINT.search(line))


def is_code_only(text: str) -> bool:
    """True when the response is code with no explanatory prose around it."""
    body = text.strip()
    if not body:
        return False
    if "```" in body:
        # Acceptable only when the entire answer is one fenced block.
        return body.startswith("```") and body.endswith("```") and body.count("```") == 2
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if not lines:
        return False
    return sum(1 for ln in lines if _looks_like_code(ln)) / len(lines) >= 0.7


def detect_language(text: str) -> str | None:
    """Coarse pt/en discrimination via function-word counts."""
    toks = set(words(normalize(text, strip_accents=False)))
    pt = len(toks & _PT_MARKERS)
    en = len(toks & _EN_MARKERS)
    if pt == en:
        return None
    return "pt" if pt > en else "en"


# --- individual checks -----------------------------------------------------

CheckFn = Callable[[str, Any, EvalItem], CheckResult]


def _as_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _check_contains_any(response: str, value: Any, _item: EvalItem) -> CheckResult:
    needles = _as_list(value)
    haystack = normalize(response)
    hit = next((n for n in needles if normalize(n) in haystack), None)
    return CheckResult(
        "contains_any",
        hit is not None,
        f"matched '{hit}'" if hit else f"none of {needles} present",
    )


def _check_contains_all(response: str, value: Any, _item: EvalItem) -> CheckResult:
    needles = _as_list(value)
    haystack = normalize(response)
    missing = [n for n in needles if normalize(n) not in haystack]
    return CheckResult("contains_all", not missing, f"missing {missing}" if missing else "all present")


def _check_not_contains(response: str, value: Any, _item: EvalItem) -> CheckResult:
    needles = _as_list(value)
    haystack = normalize(response)
    found = [n for n in needles if normalize(n) in haystack]
    return CheckResult("not_contains", not found, f"found forbidden {found}" if found else "clean")


def _check_regex(response: str, value: Any, _item: EvalItem) -> CheckResult:
    patterns = _as_list(value)
    for pattern in patterns:
        if not re.search(pattern, response, re.IGNORECASE | re.MULTILINE):
            return CheckResult("regex", False, f"no match for /{pattern}/")
    return CheckResult("regex", True, "all patterns matched")


def _check_equals_normalized(response: str, value: Any, _item: EvalItem) -> CheckResult:
    expected = normalize(str(value))
    actual = normalize(response)
    return CheckResult("equals_normalized", actual == expected, f"got '{actual[:80]}'")


def _check_min_words(response: str, value: Any, _item: EvalItem) -> CheckResult:
    n = word_count(response)
    return CheckResult("min_words", n >= int(value), f"{n} words (min {value})")


def _check_max_words(response: str, value: Any, _item: EvalItem) -> CheckResult:
    n = word_count(response)
    return CheckResult("max_words", n <= int(value), f"{n} words (max {value})")


def _check_word_count(response: str, value: Any, item: EvalItem) -> CheckResult:
    target = int(value)
    tolerance = int(item.expected.get("word_count_tolerance", 5))
    n = word_count(response)
    ok = abs(n - target) <= tolerance
    return CheckResult("word_count", ok, f"{n} words (target {target} +/- {tolerance})")


def _check_bullet_count(response: str, value: Any, item: EvalItem) -> CheckResult:
    target = int(value)
    n = bullet_count(response)
    if n != target:  # some models number their bullets instead
        n = max(n, numbered_count(response))
    return CheckResult("bullet_count", n == target, f"{n} bullets (want {target})")


def _check_min_bullets(response: str, value: Any, _item: EvalItem) -> CheckResult:
    n = max(bullet_count(response), numbered_count(response))
    return CheckResult("min_bullets", n >= int(value), f"{n} bullets (min {value})")


def _check_sentence_count(response: str, value: Any, _item: EvalItem) -> CheckResult:
    n = sentence_count(response)
    return CheckResult("sentence_count", n == int(value), f"{n} sentences (want {value})")


def _check_max_sentences(response: str, value: Any, _item: EvalItem) -> CheckResult:
    n = sentence_count(response)
    return CheckResult("max_sentences", n <= int(value), f"{n} sentences (max {value})")


def _check_json_valid(response: str, value: Any, _item: EvalItem) -> CheckResult:
    if not value:
        return CheckResult("json_valid", True, "not required")
    try:
        extract_json(response)
        return CheckResult("json_valid", True, "parsed")
    except ValueError as exc:
        return CheckResult("json_valid", False, str(exc))


def _check_json_keys(response: str, value: Any, _item: EvalItem) -> CheckResult:
    keys = _as_list(value)
    try:
        payload = extract_json(response)
    except ValueError as exc:
        return CheckResult("json_keys", False, str(exc))
    if not isinstance(payload, dict):
        return CheckResult("json_keys", False, f"expected object, got {type(payload).__name__}")
    missing = [k for k in keys if k not in payload]
    return CheckResult("json_keys", not missing, f"missing {missing}" if missing else "all keys present")


def _check_code_only(response: str, value: Any, _item: EvalItem) -> CheckResult:
    if not value:
        return CheckResult("code_only", True, "not required")
    ok = is_code_only(response)
    return CheckResult("code_only", ok, "code only" if ok else "prose mixed with code")


def _check_contains_code(response: str, value: Any, _item: EvalItem) -> CheckResult:
    if not value:
        return CheckResult("contains_code", True, "not required")
    ok = "```" in response or any(_looks_like_code(ln) for ln in response.splitlines() if ln.strip())
    return CheckResult("contains_code", ok, "code present" if ok else "no code found")


def _check_uppercase(response: str, value: Any, _item: EvalItem) -> CheckResult:
    if not value:
        return CheckResult("uppercase", True, "not required")
    letters = [c for c in response if c.isalpha()]
    ok = bool(letters) and all(c.isupper() for c in letters)
    return CheckResult("uppercase", ok, "all caps" if ok else "contains lowercase letters")


def _check_language(response: str, value: Any, _item: EvalItem) -> CheckResult:
    want = str(value).lower()
    got = detect_language(response)
    ok = got == want
    return CheckResult("language", ok, f"detected '{got or 'undetermined'}' (want '{want}')")


def _check_min_distinct_ratio(response: str, value: Any, _item: EvalItem) -> CheckResult:
    ratio = distinct_ngram_ratio(response)
    return CheckResult("min_distinct_ratio", ratio >= float(value), f"distinct 4-gram ratio {ratio:.2f}")


CHECK_HANDLERS: dict[str, CheckFn] = {
    "contains_any": _check_contains_any,
    "contains_all": _check_contains_all,
    "not_contains": _check_not_contains,
    "regex": _check_regex,
    "equals_normalized": _check_equals_normalized,
    "min_words": _check_min_words,
    "max_words": _check_max_words,
    "word_count": _check_word_count,
    "bullet_count": _check_bullet_count,
    "min_bullets": _check_min_bullets,
    "sentence_count": _check_sentence_count,
    "max_sentences": _check_max_sentences,
    "json_valid": _check_json_valid,
    "json_keys": _check_json_keys,
    "code_only": _check_code_only,
    "contains_code": _check_contains_code,
    "uppercase": _check_uppercase,
    "language": _check_language,
    "min_distinct_ratio": _check_min_distinct_ratio,
}

# Keys that configure another check rather than being one themselves.
_MODIFIER_KEYS = frozenset({"word_count_tolerance"})


# --- item scoring ----------------------------------------------------------

def score_response(item: EvalItem, record: GenerationRecord) -> ItemScore:
    """Grade one response against its item's declared constraints."""
    if record.error:
        return ItemScore(item.id, item.category, 0.0, "error",
                         (CheckResult("generation", False, record.error),))

    response = record.response or ""

    if not item.is_scorable:
        return _heuristic_score(item, response)

    checks: list[CheckResult] = []
    for key, value in item.expected.items():
        if key in _MODIFIER_KEYS:
            continue
        handler = CHECK_HANDLERS.get(key)
        if handler is None:
            print(f"[WARN] [{item.id}] unknown check '{key}' — ignored")
            continue
        checks.append(handler(response, value, item))

    if not checks:
        return _heuristic_score(item, response)

    score = sum(1 for c in checks if c.passed) / len(checks)
    return ItemScore(item.id, item.category, score, "checked", tuple(checks))


def _heuristic_score(item: EvalItem, response: str) -> ItemScore:
    """Degeneracy proxy for open-ended items: non-empty, substantive, not looping."""
    ratio = distinct_ngram_ratio(response)
    n_words = word_count(response)
    checks = (
        CheckResult("non_empty", n_words > 0, f"{n_words} words"),
        CheckResult("substantive", n_words >= 10, f"{n_words} words (min 10)"),
        CheckResult("non_degenerate", ratio >= 0.35, f"distinct 4-gram ratio {ratio:.2f}"),
    )
    score = sum(1 for c in checks if c.passed) / len(checks)
    return ItemScore(item.id, item.category, score, "heuristic", checks)


def score_all(items: Sequence[EvalItem], records: Iterable[GenerationRecord]) -> list[ItemScore]:
    """Grade every record whose item is present in `items`."""
    by_id = {item.id: item for item in items}
    scores: list[ItemScore] = []
    for record in records:
        item = by_id.get(record.question_id)
        if item is None:
            continue  # stale record from an older benchmark revision
        scores.append(score_response(item, record))
    return scores


# --- aggregation -----------------------------------------------------------

def category_means(scores: Iterable[ItemScore]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for s in scores:
        buckets.setdefault(s.category, []).append(s.score)
    return {cat: statistics.fmean(vals) for cat, vals in buckets.items() if vals}


def overall_score(scores: Iterable[ItemScore]) -> float:
    """
    Macro-average across categories.

    Categories carry different item counts; averaging the per-category means
    keeps a large category from dominating the headline number.
    """
    means = category_means(scores)
    return statistics.fmean(means.values()) if means else 0.0


def consistency_by_group(
    groups: dict[str, list[EvalItem]],
    records_by_id: dict[str, GenerationRecord],
) -> dict[str, float]:
    """
    Robustness metric: agreement between a model's answers to paraphrases of the
    same question, as mean pairwise Jaccard overlap of normalized word sets.
    """
    result: dict[str, float] = {}
    for group, items in groups.items():
        texts = [
            records_by_id[i.id].response
            for i in items
            if i.id in records_by_id and not records_by_id[i.id].error
        ]
        if len(texts) < 2:
            continue
        sims: list[float] = []
        for a in range(len(texts)):
            for b in range(a + 1, len(texts)):
                sa, sb = set(words(normalize(texts[a]))), set(words(normalize(texts[b])))
                union = sa | sb
                sims.append(len(sa & sb) / len(union) if union else 1.0)
        if sims:
            result[group] = statistics.fmean(sims)
    return result
