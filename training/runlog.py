"""Run bookkeeping that has to survive a restart: time, cost, the progress log.

Everything here lives in ``log.dir``, next to the TensorBoard files:

    run_state.json   training time and cost so far, and the result of the learning check
    progress.csv     one row every ``log.report_every`` steps
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = "run_state.json"
PROGRESS_FILE = "progress.csv"
PROGRESS_COLUMNS = ["utc_time", "step", "bit_acc", "psnr_db", "steps_averaged", "elapsed_hours", "cost_usd"]


class RunClock:
    """Wall-clock time train.py has spent on this run, summed over restarts, and its cost.

    The cost is ``price_per_hour`` times that time. It is an estimate of the
    GPU bill, not the bill: it leaves out time the pod was up while train.py
    was not running (setup, downloads, idling after the run) and all storage.
    Time since the last save is lost if the process is killed outright; the
    state is saved at every report line and every checkpoint.
    """

    def __init__(self, directory: str | Path, price_per_hour: float | None, fresh: bool = False):
        self.path = Path(directory) / STATE_FILE
        self.price_per_hour = price_per_hour
        state: dict = {}
        if fresh:
            self.path.unlink(missing_ok=True)
        elif self.path.is_file():
            try:
                state = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                print(f"warning: could not read {self.path} ({error}); time and cost restart from zero")
        self.elapsed_seconds = float(state.get("elapsed_seconds", 0.0))
        self.cost_usd = float(state.get("cost_usd", 0.0))
        # Time run without --price-per-hour. It is in elapsed_seconds but not in cost_usd.
        self.unpriced_seconds = float(state.get("unpriced_seconds", 0.0))
        self.learning_check: dict | None = state.get("learning_check")
        self._last = time.monotonic()

    def tick(self) -> None:
        now = time.monotonic()
        delta, self._last = now - self._last, now
        self.elapsed_seconds += delta
        if self.price_per_hour is None:
            self.unpriced_seconds += delta
        else:
            self.cost_usd += delta / 3600.0 * self.price_per_hour

    def save(self) -> None:
        self.tick()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "elapsed_seconds": self.elapsed_seconds,
            "cost_usd": self.cost_usd,
            "unpriced_seconds": self.unpriced_seconds,
            "learning_check": self.learning_check,
        }
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    def cost_text(self) -> str:
        if self.price_per_hour is None:
            return "cost n/a (no --price-per-hour)"
        return f"cost ${self.cost_usd:.2f}"


class ProgressLog:
    """Appends one row per report to ``progress.csv``."""

    def __init__(self, directory: str | Path, fresh: bool = False):
        self.path = Path(directory) / PROGRESS_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fresh:
            self.path.unlink(missing_ok=True)
        if not self.path.is_file():
            with open(self.path, "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(PROGRESS_COLUMNS)

    def write(self, step: int, bit_acc: float, psnr_db: float, steps_averaged: int, clock: RunClock) -> str:
        """Write the row and return the same information as one line of text."""
        priced = clock.price_per_hour is not None
        row = [
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            step,
            f"{bit_acc:.4f}",
            f"{psnr_db:.2f}",
            steps_averaged,
            f"{clock.elapsed_seconds / 3600.0:.4f}",
            f"{clock.cost_usd:.4f}" if priced else "",
        ]
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(row)
        return (
            f"[report] step {step:7d}  bit_acc {bit_acc:.3f}  psnr {psnr_db:.1f} dB  "
            f"(mean of last {steps_averaged} steps)  {clock.elapsed_seconds / 3600.0:.2f} h  {clock.cost_text()}"
        )
