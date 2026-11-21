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
``band.min_camera_px_per_cycle`` camera pixels per cycle. Higher bands are
preferred because picture content is weaker there. While
``camera_px_per_tile`` is unset, ``band.fallback_step`` is used.

This only sets where training starts. The dense layer and the reader are free
to move the code elsewhere.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

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

    def describe(self) -> str:
        return (
            f"starting band {self.low_cycles:.0f}-{self.high_cycles:.0f} cycles per tile "
            f"(grating_step {self.grating_step}, {self.source}); message_grid {self.message_grid}, "
            f"texture window {self.texture_kernel * self.texture_downsample} px"
        )

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
    camera = cfg.get("camera_px_per_tile")
    if camera is None:
        return int(band_cfg["fallback_step"]), "fallback, camera_px_per_tile is unset"
    highest_allowed = float(camera) / float(band_cfg["min_camera_px_per_cycle"])
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
    return max(usable), f"derived from camera_px_per_tile={camera}"


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
    return Band(step, grid, low, high, kernel, downsample, source)


def _radius_range(grid: int, step: int) -> tuple[float, float]:
    radii = [math.hypot(a, b) for a, b in grating_frequencies(grid, step)]
    return min(radii), max(radii)
