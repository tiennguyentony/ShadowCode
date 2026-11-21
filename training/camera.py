"""The demo camera: what the config says about it, in one place.

``camera_px_per_tile`` is how many camera pixels one printed tile covers when
seen straight on. It is ``null`` (not measured), one number, or ``[low, high]``.
Training draws from the whole range; the starting band is checked against its
low end, where the camera resolves least.

``demo`` holds what the range was derived from: the printed tile size, the
camera stream size and the measurements. Those are not used in training. They
are checked against the range here and copied into contract.json.
"""

from __future__ import annotations

# A grating needs more than this many pixels per cycle to be sampled at all.
NYQUIST_PX_PER_CYCLE = 2.0


def camera_range(cfg: dict) -> tuple[float, float] | None:
    """(low, high) camera pixels per printed tile, or None while unmeasured."""
    value = cfg.get("camera_px_per_tile")
    if value is None:
        return None
    if isinstance(value, (int, float)):
        low = high = float(value)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        low, high = float(value[0]), float(value[1])
    else:
        raise ValueError(f"camera_px_per_tile must be null, a number or [low, high], got {value!r}")
    if not 0 < low <= high:
        raise ValueError(f"camera_px_per_tile needs 0 < low <= high, got {value!r}")
    return low, high


def frame_size(cfg: dict) -> tuple[int, int]:
    """(width, height) of the simulated frame. ``simulation.frame_px`` is one side or [width, height]."""
    value = cfg["simulation"]["frame_px"]
    if isinstance(value, (list, tuple)):
        width, height = value
        return int(width), int(height)
    return int(value), int(value)


def measurements(cfg: dict) -> list[dict]:
    """The measurements in ``demo.measured``, each with its px_per_tile worked out."""
    demo = cfg.get("demo") or {}
    tile_cm = float(demo.get("tile_mm", 0)) / 10.0
    out = []
    for item in demo.get("measured") or []:
        out.append({**item, "px_per_tile": round(float(item["px_per_cm"]) * tile_cm, 2)})
    return out


def check_demo(cfg: dict) -> None:
    """Stop if a measurement lies outside the range training covers."""
    span = camera_range(cfg)
    if span is None:
        return
    for m in measurements(cfg):
        if not span[0] <= m["px_per_tile"] <= span[1]:
            raise ValueError(
                f"measured {m['px_per_tile']} camera px per tile ({m.get('device', 'camera')} at "
                f"{m.get('distance_cm', '?')} cm) is outside camera_px_per_tile {list(span)}"
            )


def camera_summary(cfg: dict) -> dict | None:
    """The camera block of contract.json, or None when nothing is known."""
    span = camera_range(cfg)
    demo = cfg.get("demo") or {}
    if span is None and not demo:
        return None
    out: dict = {
        "trained_px_per_tile": [int(v) if v.is_integer() else v for v in span] if span else None,
        "px_per_tile_means": "camera pixels across one printed tile (one tile period), seen straight on",
    }
    if demo.get("tile_mm") is not None:
        out["tile_mm"] = demo["tile_mm"]
    stream = demo.get("stream_px")
    if stream:
        width, height = int(stream[0]), int(stream[1])
        out["stream_px"] = [width, height]
        out["stream_px_order"] = "[width, height]"
        out["orientation"] = "portrait" if height > width else "landscape" if width > height else "square"
        if span:
            out["tiles_across_stream_width"] = [round(width / span[1], 2), round(width / span[0], 2)]
    if demo.get("measured"):
        out["measured"] = measurements(cfg)
    return out
