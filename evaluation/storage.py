# coding=utf-8
"""
storage.py — JSONL persistence and resume support.

One `responses.jsonl` per model variant under `results/<checkpoint>/`. Records
are appended and flushed immediately so an interrupted run loses at most the
item that was in flight.
"""

from __future__ import annotations

import json
import os
from types import TracebackType
from typing import Any, Iterator

from .types import GenerationRecord

RESPONSES_FILENAME = "responses.jsonl"


def results_dir_for(results_root: str, checkpoint_slug: str) -> str:
    return os.path.join(results_root, checkpoint_slug)


def responses_path(results_root: str, checkpoint_slug: str) -> str:
    return os.path.join(results_dir_for(results_root, checkpoint_slug), RESPONSES_FILENAME)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def iter_records(path: str) -> Iterator[GenerationRecord]:
    """
    Stream records from a JSONL file.

    Malformed lines (a half-written final line after a hard kill) are skipped
    with a warning instead of aborting the run.
    """
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield GenerationRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                print(f"[WARN] {path}:{lineno} — skipping malformed record")


def load_records(path: str) -> list[GenerationRecord]:
    return list(iter_records(path))


def completed_ids(path: str, include_errors: bool = False) -> set[str]:
    """
    Question ids already present in `path`.

    Failed generations are re-attempted on resume unless `include_errors`.
    """
    return {
        rec.question_id
        for rec in iter_records(path)
        if include_errors or rec.error is None
    }


class JsonlLogger:
    """Append-only JSONL writer with immediate flush."""

    def __init__(self, path: str) -> None:
        self.path = path
        ensure_dir(os.path.dirname(path) or ".")
        self._handle: Any = None

    def __enter__(self) -> JsonlLogger:
        self._handle = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def write(self, record: GenerationRecord) -> None:
        if self._handle is None:
            raise RuntimeError("JsonlLogger used outside of its context manager")
        self._handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    """Overwrite `path` with `rows` (used by the pairwise generator)."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, payload: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
