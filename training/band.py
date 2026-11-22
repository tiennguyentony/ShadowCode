"""The starting frequency band of the code, derived from camera_px_per_tile.

With ``message_init: gratings`` each bit starts as one grating at
(u, v) = grating_step * (a, b) cycles per tile. This module picks
``grating_step`` and the sizes that have to match it:

* ``message_grid`` (embedder): 16 * grating_step cells per tile, the smallest
  grid that holds the gratings below its Nyquist frequency.
* the texture branch window (reader): TILE_PX / grating_step input pixels,
  which is the window needed to tell neighbouring gratings apart.

``band.grating_step: auto`` picks the highest band the camera can still
resolve: the largest step whose highest starting frequency leaves at least
``band.min_camera_px_per_cycle`` camera pixels per cycle at the low end of
``camera_px_per_tile``. Higher bands are preferred because picture content is
weaker there. While ``camera_px_per_tile`` is unset, ``band.fallback_step`` is
used.

A step set by hand is checked the same way. Below the margin it is kept and
the shortfall is reported; at or past the sampling limit (2 camera pixels per
cycle) it is refused.

This only sets where training starts. The dense layer and the reader are free
to move the code elsewhere.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from camera import NYQUIST_PX_PER_CYCLE, camera_range
from constants import MESSAGE_BITS, TILE_PX

# Steps for which 16 * step divides TILE_PX.
ALLOWED_STEPS = (2, 3, 4, 6, 8, 12)
_MIN_TEXTURE_KERNEL = 24


def grating_frequencies(grid: int, step: int, bits: int = MESSAGE_BITS) -> list[tuple[int, int]]:
    """The (a, b) pairs used by the gratings init, lowest frequency first.

    Frequencies are step * (a, b) cycles per tile. One of each +-pair is
    kept, DC and the lowest few are skipped, and everything stays strictly
    below the Nyquist frequency of a ``grid``-cell map.
    """
    limit = (grid // 2 - 1) // step
    candidates = [
        (a, b)
        for a in range(0, limit + 1)
        for b in range(-limit, limit + 1)
        if (a > 0 or b > 0) and math.hypot(a, b) >= 1.5
    ]
    if len(candidates) < bits:
        raise ValueError(
            f"message_grid {grid} with grating_step {step} has only {len(candidates)} "
            f"usable frequencies for {bits} bits"
        )
    candidates.sort(key=lambda ab: (math.hypot(*ab), ab))
    return candidates[:bits]


def band_limits(step: int) -> tuple[float, float]:
    """Lowest and highest starting frequency for a step, in cycles per tile."""
    radii = [math.hypot(a, b) for a, b in grating_frequencies(16 * step, step)]
    return step * min(radii), step * max(radii)


@dataclass(frozen=True)
class Band:
    grating_step: int
    message_grid: int
    low_cycles: float          # lowest starting frequency, cycles per tile
    high_cycles: float         # highest starting frequency, cycles per tile
    texture_kernel: int
    texture_downsample: int
    source: str                # how grating_step was chosen
    # Camera pixels per cycle of the highest starting frequency at the low and
    # the high end of camera_px_per_tile; None while that is unset.
    px_per_cycle_low: float | None = None
    px_per_cycle_high: float | None = None
    margin: float | None = None        # band.min_camera_px_per_cycle

    @property
    def within_margin(self) -> bool | None:
        if self.px_per_cycle_low is None or self.margin is None:
            return None
        return self.px_per_cycle_low >= self.margin

    def describe(self) -> str:
        text = (
            f"starting band {self.low_cycles:.0f}-{self.high_cycles:.0f} cycles per tile "
            f"(grating_step {self.grating_step}, {self.source}); message_grid {self.message_grid}, "
            f"texture window {self.texture_kernel * self.texture_downsample} px"
        )
        if self.px_per_cycle_low is not None:
            text += (
                f"\nhighest starting frequency gets {self.px_per_cycle_low:.2f} to {self.px_per_cycle_high:.2f} "
                f"camera px per cycle over the camera range (sampling limit {NYQUIST_PX_PER_CYCLE:.0f}, "
                f"margin {self.margin:g})"
            )
            if not self.within_margin:
                text += ": BELOW THE MARGIN at the low end of the range"
        return text

    def as_dict(self) -> dict:
        return asdict(self)


def _texture_window(step: int) -> tuple[int, int]:
    """(kernel, downsample) with kernel * downsample = TILE_PX / step and the kernel kept small."""
    window = TILE_PX // step
    downsample = 1
    while window % (2 * downsample) == 0 and window // (2 * downsample) >= _MIN_TEXTURE_KERNEL:
        downsample *= 2
    return window // downsample, downsample


def _auto_step(cfg: dict) -> tuple[int, str]:
    band_cfg = cfg["band"]
    span = camera_range(cfg)
    if span is None:
        return int(band_cfg["fallback_step"]), "fallback, camera_px_per_tile is unset"
    camera = span[0]                   # the low end is where the camera resolves least
    highest_allowed = camera / float(band_cfg["min_camera_px_per_cycle"])
    usable = [
        s for s in ALLOWED_STEPS if s <= int(band_cfg["max_step"]) and band_limits(s)[1] <= highest_allowed
    ]
    if not usable:
        lowest = band_limits(ALLOWED_STEPS[0])[1]
        raise ValueError(
            f"camera_px_per_tile={camera} resolves at most {highest_allowed:.1f} cycles per tile at "
            f"{band_cfg['min_camera_px_per_cycle']} px per cycle, but the lowest band already reaches "
            f"{lowest:.1f}. Set band.grating_step by hand or lower band.min_camera_px_per_cycle."
        )
    return max(usable), f"derived from camera_px_per_tile={camera:g}"


def resolve_band(cfg: dict) -> Band:
    requested = cfg["band"]["grating_step"]
    if requested == "auto":
        step, source = _auto_step(cfg)
    else:
        step, source = int(requested), "set in the config"
    if step not in ALLOWED_STEPS:
        raise ValueError(f"grating_step must be one of {ALLOWED_STEPS}, got {step}")

    grid = cfg["model"]["embedder"]["message_grid"]
    grid = 16 * step if grid == "auto" else int(grid)
    low, high = (step * r for r in _radius_range(grid, step))

    kernel, downsample = _texture_window(step)
    texture = cfg["model"]["reader"].get("texture") or {}
    if texture.get("kernel", "auto") != "auto":
        kernel = int(texture["kernel"])
    if texture.get("downsample", "auto") != "auto":
        downsample = int(texture["downsample"])
    span = camera_range(cfg)
    if span is None:
        return Band(step, grid, low, high, kernel, downsample, source)
    at_low, at_high = span[0] / high, span[1] / high
    if at_low <= NYQUIST_PX_PER_CYCLE:
        raise ValueError(
            f"grating_step {step} starts as high as {high:.1f} cycles per tile, which a camera at "
            f"{span[0]:g} px per tile cannot sample ({at_low:.2f} px per cycle, the limit is "
            f"{NYQUIST_PX_PER_CYCLE:.0f}). Use a lower step or raise the low end of camera_px_per_tile."
        )
    margin = float(cfg["band"]["min_camera_px_per_cycle"])
    return Band(step, grid, low, high, kernel, downsample, source, at_low, at_high, margin)


def _radius_range(grid: int, step: int) -> tuple[float, float]:
    radii = [math.hypot(a, b) for a, b in grating_frequencies(grid, step)]
    return min(radii), max(radii)
