"""Run bookkeeping that has to survive a restart: time, cost, the progress log.

Everything here lives in ``log.dir``, next to the TensorBoard files:

    run_state.json   training time and cost so far, and the results of the learning check
                     and the image-loss check
    progress.csv     one row every ``log.report_every`` steps
    spread.csv       one row per spread check (every sample sheet): how piled-up the mark is
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
# image_scale: the image-loss weight (0 to 1) the report's last step trained with. Last, so that
# readers of the earlier columns are not disturbed. visibility: the mean over the report window of the batch
# statistic of the masked S-CIELAB delta E; vis_weight: the visibility-loss weight at the last step. Both empty
# when loss.visibility is off.
PROGRESS_COLUMNS = [
    "utc_time", "step", "bit_acc", "psnr_db", "steps_averaged", "elapsed_hours", "cost_usd", "image_scale",
    "visibility", "vis_weight",
]


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
        self.image_loss_check: dict | None = state.get("image_loss_check")
        self.reading_check: dict | None = state.get("reading_check")
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
            "image_loss_check": self.image_loss_check,
            "reading_check": self.reading_check,
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
        if self.path.is_file():
            with open(self.path, "r", encoding="utf-8", newline="") as f:
                header = next(csv.reader(f), None)
            if header != PROGRESS_COLUMNS:
                # Written by an earlier version with other columns: keep it, start a new file.
                aside = self.path.with_name(f"progress_before_{time.strftime('%Y%m%d-%H%M%S')}.csv")
                os.replace(self.path, aside)
                print(f"note: {self.path} had other columns; kept as {aside.name}, starting a new one")
        if not self.path.is_file():
            with open(self.path, "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(PROGRESS_COLUMNS)

    def write(
        self,
        step: int,
        bit_acc: float,
        psnr_db: float,
        steps_averaged: int,
        clock: RunClock,
        image_scale: float | None = None,
        visibility: float | None = None,
        vis_weight: float | None = None,
    ) -> str:
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
            "" if image_scale is None else f"{image_scale:.4f}",
            "" if visibility is None else f"{visibility:.4f}",
            "" if vis_weight is None else f"{vis_weight:.6g}",
        ]
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(row)
        return (
            f"[report] step {step:7d}  bit_acc {bit_acc:.3f}  psnr {psnr_db:.1f} dB  "
            f"(mean of last {steps_averaged} steps)  "
            + ("" if image_scale is None else f"image-loss weight {image_scale:.3f}  ")
            + ("" if visibility is None else f"seen {visibility:.2f}  ")
            + ("" if vis_weight is None else f"vis weight {vis_weight:.3g}  ")
            + f"{clock.elapsed_seconds / 3600.0:.2f} h  {clock.cost_text()}"
        )


SPREAD_FILE = "spread.csv"
SPREAD_COLUMNS = [
    "utc_time",
    "local_time",
    "step",
    "images",
    "fixed_place",
    "strongest_spot",
    "square_share",
    "busy_share",
    "psnr_fixed_db",
    "bit_acc",
    "psnr_db",
    "visibility_p90",
    "visibility_mean",
    "mark_cycles",
    "colour_share",
]


def _maybe(value: float, digits: int) -> str:
    """A number for the CSV, or an empty cell for NaN (the measure was off)."""
    return "" if value is None or value != value else f"{value:.{digits}f}"


class SpreadLog:
    """One row per spread check in ``spread.csv``; optionally the same rows as a table in a Markdown file.

    ``markdown`` is ``{"file": path, "marker": name}``. The table replaces
    whatever stands between the lines ``<!-- name:start -->`` and
    ``<!-- name:end -->`` in that file, so the file has to hold both lines
    already. A file that cannot be updated costs a warning, never the run.
    """

    def __init__(self, directory: str | Path, markdown: dict | None = None, fresh: bool = False):
        self.path = Path(directory) / SPREAD_FILE
        self.markdown = markdown
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fresh:
            self.path.unlink(missing_ok=True)
        if self.path.is_file():
            with open(self.path, "r", encoding="utf-8", newline="") as f:
                header = next(csv.reader(f), None)
            if header != SPREAD_COLUMNS:
                # Written by an earlier version with other columns: keep it, start a new file.
                aside = self.path.with_name(f"spread_before_{time.strftime('%Y%m%d-%H%M%S')}.csv")
                os.replace(self.path, aside)
                print(f"note: {self.path} had other columns; kept as {aside.name}, starting a new one")
        if not self.path.is_file():
            with open(self.path, "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(SPREAD_COLUMNS)

    def rows(self) -> list[dict]:
        with open(self.path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def has_step(self, step: int) -> bool:
        return any(int(row["step"]) == step for row in self.rows())

    def write(
        self, step: int, report: dict, bit_acc: float | None, psnr_db: float | None, local_time: str | None = None
    ) -> str:
        """Write the row, refresh the Markdown table, and return the row as one line of text."""
        now = datetime.now(timezone.utc)
        row = [
            now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            local_time or now.astimezone().strftime("%Y-%m-%d %H:%M"),
            step,
            report["images"],
            f"{report['fixed_place']:.2f}",
            f"{report['strongest_spot']:.2f}",
            f"{report['square_share']:.4f}",
            f"{report['busy_share']:.4f}",
            f"{report['psnr']:.2f}",
            "" if bit_acc is None else f"{bit_acc:.4f}",
            "" if psnr_db is None else f"{psnr_db:.2f}",
            _maybe(report.get("visibility_p90"), 3),
            _maybe(report.get("visibility_mean"), 3),
            _maybe(report.get("mark_cycles"), 2),
            _maybe(report.get("colour_share"), 4),
        ]
        seen = report.get("visibility_p90", float("nan"))
        seen_text = (
            "" if seen != seen
            else f"visibility p90 {seen:.2f}  mark at {report['mark_cycles']:.1f} cycles per tile  "
            f"{report['colour_share'] * 100:.0f} percent in colour  "
        )
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(row)
        self.update_markdown()
        return (
            f"[spread] step {step:7d}  fixed place {report['fixed_place']:.2f}  "
            f"strongest spot {report['strongest_spot']:.2f}  "
            f"{report['square_share'] * 100:.1f} percent of the mark's energy in one 96 px square  "
            f"{report['busy_share'] * 100:.0f} percent on the busier half of the image  "
            f"{seen_text}psnr {report['psnr']:.1f} dB  ({report['images']} fixed images)"
        )

    def update_markdown(self) -> None:
        if not self.markdown:
            return
        try:
            path, marker = Path(self.markdown["file"]), self.markdown["marker"]
            start, end = f"<!-- {marker}:start -->", f"<!-- {marker}:end -->"
            text = path.read_text(encoding="utf-8")
            head, found_start, rest = text.partition(start)
            _, found_end, tail = rest.partition(end)
            if not found_start or not found_end:
                print(f"warning: {path} has no {start} ... {end} lines; the table there was not updated")
                return
            lines = [
                "| Time | Step | Fixed place | Strongest spot | Energy in one 96 px square | "
                "Energy on the busier half | PSNR, dB | Bit accuracy |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for row in self.rows():
                square = f"{float(row['square_share']) * 100:.1f} percent"
                busy = f"{float(row['busy_share']) * 100:.0f} percent"
                psnr = row["psnr_db"] or f"{row['psnr_fixed_db']} (fixed images)"
                lines.append(
                    f"| {row['local_time']} | {int(row['step']):,} | {row['fixed_place']} | {row['strongest_spot']} | "
                    f"{square} | {busy} | {psnr} | {row['bit_acc'] or 'n/a'} |"
                )
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(head + start + "\n" + "\n".join(lines) + "\n" + end + tail, encoding="utf-8", newline="\n")
            os.replace(temporary, path)
        except Exception as error:  # noqa: BLE001 - a document must never stop a training run
            print(f"warning: could not update the spread table ({type(error).__name__}: {error})")
