# coding=utf-8
"""
progress.py — Progress reporting with a plain-text fallback.

Uses `rich` when available (it is already a project dependency) and degrades to
periodic prints otherwise, so the runner never has to care which is active.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any

try:  # pragma: no cover - presentation only
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    _RICH = True
except ImportError:  # pragma: no cover
    _RICH = False


class ProgressReporter:
    """
    One overall bar plus a per-model bar.

    `note`/`warn` route through the progress console so messages never tear the
    rendered bars.
    """

    def __init__(self, total: int) -> None:
        self.total = total
        self._progress: Any = None
        self._console: Any = None
        self._overall: Any = None
        self._counter = 0

    def __enter__(self) -> ProgressReporter:
        if _RICH:
            self._console = Console()
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=self._console,
                transient=False,
            )
            self._progress.start()
            self._overall = self._progress.add_task("overall", total=self.total)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

    # --- tasks -------------------------------------------------------------

    def start_task(self, name: str, total: int) -> Any:
        if self._progress is not None:
            return self._progress.add_task(name, total=total)
        print(f"\n[RUN] {name}: {total} items")
        return {"name": name, "total": total, "done": 0}

    def advance(self, task: Any) -> None:
        self._counter += 1
        if self._progress is not None:
            self._progress.advance(task)
            if self._overall is not None:
                self._progress.advance(self._overall)
            return
        task["done"] += 1
        if task["done"] % 10 == 0 or task["done"] == task["total"]:
            print(f"      {task['name']}: {task['done']}/{task['total']} "
                  f"(overall {self._counter}/{self.total})")

    def finish_task(self, task: Any) -> None:
        if self._progress is not None:
            self._progress.remove_task(task)

    # --- messages ----------------------------------------------------------

    def note(self, message: str) -> None:
        if self._console is not None:
            self._console.print(f"[dim][INFO][/dim] {message}")
        else:
            print(f"[INFO] {message}")

    def warn(self, message: str) -> None:
        if self._console is not None:
            self._console.print(f"[yellow][WARN][/yellow] {message}")
        else:
            print(f"[WARN] {message}")
