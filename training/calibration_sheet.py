"""The calibration sheet: what visibility target does the user's own eye give?

The next run pushes the mark under a visibility target, the 95th percentile of the masked S-CIELAB
delta E of a tile (visibility.py). The number has to come from the eye that will look at the print.

    python calibration_sheet.py --config configs/runpod_labels_generated_fresh.yaml
        --checkpoint kept/labels_generated_fresh/<run>/checkpoint/step_00140000.pt --out runs/calibration
    python calibration_sheet.py --out runs/calibration --seen 3,7,12

The first form writes <out>/calibration.html (print it at 100 percent: each tile is 50 mm wide),
calibration_key.json and pairs/*.png. Every pair is a tile beside its clean original, left and right
shuffled, and every marked tile is scaled to a set masked visibility (LEVELS). The pairs are flat-colour
and busy labels marked by the model, and pure gratings on mid-grey in luminance and blue-yellow.

Look at each numbered pair from arm's length (about 50 cm) and write down the numbers of the pairs where
you see any difference. The second form takes those numbers and applies the design's rules
(docs/superpowers/specs/2026-11-24-eye-model-loss-design.md, section 4):

1. Target: the highest level with no flat-colour pair seen at it or below it. If a flat pair is seen at
   the lowest level, the run waits ("every level seen: the run waits" when every level was seen).
   Answers that skip a level (seen at 3, not at 5) are read the safe way: the target stays under the
   lowest level that was seen.
2. A busy-label pair seen at or below the target: the masking is too generous. Raise the masking floor
   (0.5, 0.6, ... 1.0) to the smallest one that puts that pair's masked p95, recomputed from its stored
   PNGs, above the target.
3. A 74-cycle grating seen at or below the target: the target drops to the highest level under the lowest
   such level.

The result goes to <out>/calibration_result.json. Exit code 1 when the run waits.
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import os
import random
import sys
from pathlib import Path

import torch

from constants import TILE_PX
from visibility import from_opponent, opponent, tile_visibility

HERE = Path(__file__).resolve().parent
LEVELS = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0)  # masked p95 delta E of the marked tiles
FLOORS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)  # masking floors tried when a busy pair is seen
ALPHA = 0.30
FLAT_LABELS, BUSY_LABELS = 3, 2
GRATINGS = (("luminance", 23), ("luminance", 74), ("blue-yellow", 23), ("blue-yellow", 74))
MAX_FACTOR = 4.0
TOLERANCE = 0.05
SEED = 0


@torch.no_grad()
def _p95(clean: torch.Tensor, marked: torch.Tensor, cfg: dict) -> float:
    return float(tile_visibility(clean, marked, cfg)["p95"][0])


def quantise(x: torch.Tensor) -> torch.Tensor:
    """x as an 8-bit PNG holds it (viz.save_image rounds the same way), back in [0, 1]."""
    return (x.clamp(0.0, 1.0) * 255.0 + 0.5).floor() / 255.0


@torch.no_grad()
def scale_factor(clean: torch.Tensor, residual: torch.Tensor, level: float, cfg: dict) -> tuple[float, torch.Tensor]:
    """The factor in [0, MAX_FACTOR] that makes ``(clean + factor * residual).clamp(0, 1)`` have a masked
    p95 delta E within 5 percent of ``level``, and that tile. ``clean`` and ``residual`` are one tile, (1,3,H,W).

    The marked tile is quantised to 8 bits inside the search, so the p95 is that of the tile as printed
    (the p95 is then a step function of the factor, and the search keeps the closest tile it saw). The
    clean tile is quantised too: it is saved the same way.

    Raises ValueError when even the largest factor falls short, which clamping to [0, 1] can cause.
    """
    assert clean.shape[0] == 1 and residual.shape == clean.shape, f"one tile at a time, got {tuple(clean.shape)}"

    clean = quantise(clean)

    def make(factor: float) -> torch.Tensor:
        return quantise(clean + factor * residual)

    top = _p95(clean, make(MAX_FACTOR), cfg)
    if top < level * (1.0 - TOLERANCE):
        raise ValueError(f"level {level:g} is out of reach: even a factor of {MAX_FACTOR:g} gives a p95 of {top:.3f}")
    low, high = 0.0, MAX_FACTOR
    best = (MAX_FACTOR, make(MAX_FACTOR), top)
    for _ in range(60):
        mid = 0.5 * (low + high)
        marked = make(mid)
        got = _p95(clean, marked, cfg)
        if abs(got - level) < abs(best[2] - level):
            best = (mid, marked, got)
        if abs(got - level) <= 0.01 * level:
            break
        low, high = (mid, high) if got < level else (low, mid)
    if abs(best[2] - level) > TOLERANCE * level:
        raise ValueError(f"level {level:g} could not be met: the closest tile has a p95 of {best[2]:.3f}")
    return best[0], best[1]


def scale_to_level(clean: torch.Tensor, residual: torch.Tensor, level: float, cfg: dict) -> torch.Tensor:
    """The marked tile ``quantise(clean + factor * residual)`` whose masked p95 delta E is within 5 percent of ``level``."""
    return scale_factor(clean, residual, level, cfg)[1]


def grating(channel: str, cycles: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(mid-grey tile, grating residual on it), both (1,3,H,W): ``cycles`` whole cycles across the tile, horizontal (cos along x)."""
    grey = torch.full((1, 3, TILE_PX, TILE_PX), 0.5)
    wave = torch.cos(2.0 * torch.pi * cycles * torch.arange(TILE_PX, dtype=torch.float32) / TILE_PX)
    wave = wave.expand(TILE_PX, TILE_PX)[None, None]
    if channel == "luminance":
        return grey, 0.1 * wave.expand(1, 3, TILE_PX, TILE_PX).contiguous()
    if channel == "blue-yellow":
        opp = opponent(grey)
        opp[:, 2:3] = opp[:, 2:3] + 0.05 * wave
        return grey, from_opponent(opp) - grey
    raise ValueError(f"unknown grating channel {channel!r}")


def decide(key: dict, seen, floor_p95=None) -> dict:
    """The target and masking floor from the numbers of the pairs the user saw a difference in.

    ``key`` is calibration_key.json; ``floor_p95(pair, floor)`` is the masked p95 of a pair's tile under
    ``floor`` (default: recomputed from the pair's PNGs). Returns {"target", "floor", "wait", "fired"}:
    target None when the run waits, floor None when it is unchanged, ``fired`` one line per rule.
    """
    pairs = {p["pair"]: p for p in key["pairs"]}
    unknown = sorted(set(seen) - set(pairs))
    if unknown:
        raise ValueError(f"no such pair in the key: {unknown}")
    seen_pairs = [pairs[n] for n in sorted(set(seen))]
    levels = sorted(key["levels"])
    fired: list[str] = []

    def below(level: float) -> float | None:
        under = [x for x in levels if x < level]
        return under[-1] if under else None

    def result(target, floor) -> dict:
        return {"target": target, "floor": floor, "wait": target is None, "fired": fired}

    flat_seen = [p["level"] for p in seen_pairs if p["kind"] == "flat label"]
    target = levels[-1]
    if flat_seen:
        every = {p["level"] for p in key["pairs"] if p["kind"] == "flat label"} <= set(flat_seen)
        target = below(min(flat_seen))
        if target is None:
            fired.append(
                "every level seen: the run waits" if every
                else f"a flat-colour pair was seen at the lowest level ({levels[0]:g}): the run waits"
            )
            return result(None, None)
        fired.append(f"flat-colour pairs seen from level {min(flat_seen):g}: target {target:g}")

    probe = floor_p95 or (lambda p, f: _recompute(key, p, f))
    floor = None
    for pair in (p for p in seen_pairs if p["kind"] == "busy label" and p["level"] <= target):
        fit = next((f for f in FLOORS if probe(pair, f) > target), None)
        if fit is None:
            fit = FLOORS[-1]
            fired.append(
                f"busy pair {pair['pair']} (level {pair['level']:g}) was seen and no floor up to {fit:g} "
                f"lifts it above {target:g}: floor {fit:g}"
            )
        else:
            fired.append(f"busy pair {pair['pair']} (level {pair['level']:g}) was seen at or below the target: masking floor {fit:g}")
        floor = fit if floor is None else max(floor, fit)

    fine = [p["level"] for p in seen_pairs if p["kind"] == "grating" and p["cycles"] == 74 and p["level"] <= target]
    if fine:
        lowered = below(min(fine))
        if lowered is None:
            fired.append(f"a 74-cycle grating was seen at the lowest level ({levels[0]:g}): the run waits")
            return result(None, floor)
        fired.append(f"a 74-cycle grating was seen at level {min(fine):g}, at or below the target: target {lowered:g}")
        target = lowered
    return result(target, floor)


def _read(path: Path) -> torch.Tensor:
    import numpy as np
    from PIL import Image

    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy()).permute(2, 0, 1).float().div(255.0)[None]


def _recompute(key: dict, pair: dict, floor: float) -> float:
    """The pair's masked p95 under masking floor ``floor``, from its stored PNGs."""
    from config import load_config

    directory = Path(key["directory"])
    left, right = (_read(directory / "pairs" / f"{pair['pair']:02d}_{side}.png") for side in ("left", "right"))
    marked, clean = (left, right) if pair["marked"] == "left" else (right, left)
    cfg = load_config(directory / key["config"])  # relative to the sheet's folder, or absolute
    loss = cfg.get("loss") or {}
    cfg["loss"] = {**loss, "visibility": {**(loss.get("visibility") or {}), "masking_floor": floor}}
    return _p95(clean, marked, cfg)


def _relative(path: Path, folder: Path) -> str:
    """``path`` relative to ``folder`` so a moved sheet still finds it; absolute when they share no drive."""
    try:
        return Path(os.path.relpath(path.resolve(), folder.resolve())).as_posix()
    except ValueError:
        return str(path.resolve())


def kind_text(desc: dict) -> str:
    return f"grating, {desc['channel']}, {desc['cycles']} cycles" if desc["kind"] == "grating" else desc["kind"]


def build(args: argparse.Namespace) -> int:
    from acceptance import IMAGES, PAGE_CSS, content_scores, fixed_bits
    from checkpoint import load_checkpoint
    from config import load_config
    from data import data_source, generated_tiles
    from labelgen import ACCEPTANCE_SEEDS
    from train import Trainer
    from viz import save_image

    cfg = load_config(args.config)
    if data_source(cfg) != "generated_labels":
        print("the calibration sheet takes its labels from labelgen.py: set data.source: generated_labels")
        return 2
    path = Path(args.checkpoint)
    if not path.is_file():
        print(f"no checkpoint at {path}")
        return 2
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg["loss"]["lpips_weight"] = 0.0  # no loss is computed here, so its pretrained network is not needed
    trainer = Trainer(eval_cfg, device)
    load_checkpoint(trainer, path)
    trainer.embedder.eval()
    print(f"checkpoint {path}, step {trainer.step}, device {device}")

    seeds = ACCEPTANCE_SEEDS[:IMAGES]
    tiles = generated_tiles(seeds)
    busy, flat = content_scores(tiles, cfg)
    flat_picks = flat.argsort(descending=True)[:FLAT_LABELS].tolist()
    busy_picks = [i for i in busy.argsort(descending=True).tolist() if i not in flat_picks][:BUSY_LABELS]
    labels = [(i, "flat label") for i in flat_picks] + [(i, "busy label") for i in busy_picks]
    bits = fixed_bits(len(labels), SEED)

    items, skipped = [], []  # items: (description, clean (1,3,H,W), marked)

    def add(desc: dict, clean: torch.Tensor, residual: torch.Tensor, level: float) -> None:
        try:
            factor, marked = scale_factor(clean.to(device), residual.to(device), level, cfg)
        except ValueError as error:
            skipped.append(f"{kind_text(desc)} {desc.get('source', '')} level {level:g}: {error}")
            return
        items.append(({**desc, "level": level, "factor": factor}, quantise(clean).cpu(), marked.cpu()))

    with torch.no_grad():
        for n, (index, kind) in enumerate(labels):
            tile = tiles[index][None].to(device)
            residual = trainer.embedder(tile, bits[n : n + 1].to(device), alpha=ALPHA)
            for level in LEVELS:
                add({"kind": kind, "source": f"seed {int(seeds[index])}"}, tile, residual, level)
        for channel, cycles in GRATINGS:
            grey, residual = grating(channel, cycles)
            for level in LEVELS:
                add({"kind": "grating", "channel": channel, "cycles": cycles}, grey, residual, level)

    rng = random.Random(SEED)
    rng.shuffle(items)
    out = Path(args.out)
    key_pairs, body = [], []
    for number, (desc, clean, marked) in enumerate(items, start=1):
        marked_left = rng.random() < 0.5
        left, right = (marked, clean) if marked_left else (clean, marked)
        save_image(left[0], out / "pairs" / f"{number:02d}_left.png")
        save_image(right[0], out / "pairs" / f"{number:02d}_right.png")
        body.append(
            f'<div class="pair"><span class="n">{number}</span>'
            f'<img src="pairs/{number:02d}_left.png" alt="pair {number}, left">'
            f'<img src="pairs/{number:02d}_right.png" alt="pair {number}, right"></div>'
        )
        key_pairs.append(
            {"pair": number, **desc, "marked": "left" if marked_left else "right", "p95": _p95(clean, marked, cfg)}
        )
    credit = cfg["data"].get("credit")
    footer = f"<footer>{html.escape(credit)}</footer>" if credit else ""
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="robots" content="noindex, nofollow">
<title>Calibration sheet</title>
<style>
{PAGE_CSS}
</style>
</head>
<body>
<h1>Calibration sheet</h1>
<p>Print at 100 percent, view from arm's length, about 50 cm, write down the numbers of the pairs where you can
see any difference. Each tile is 50 mm wide on paper. In every pair one tile carries a faint mark. The answers are
in calibration_key.json; do not open it first.</p>
{chr(10).join(body)}
{footer}
</body>
</html>
"""
    (out / "calibration.html").write_text(page, encoding="utf-8")
    key = {
        "config": _relative(Path(args.config), out), "checkpoint": str(path), "step": trainer.step, "alpha": ALPHA,
        "levels": list(LEVELS), "seed": SEED, "pairs": key_pairs,
        "skipped": skipped,
    }
    (out / "calibration_key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    print(f"{len(key_pairs)} pairs: {out / 'calibration.html'}")
    for line in skipped:
        print(f"skipped {line}")
    if skipped:
        print(f"{len(skipped)} pairs skipped: their level cannot be reached")
    return 0


def seen_mode(args: argparse.Namespace) -> int:
    out = Path(args.out)
    key = json.loads((out / "calibration_key.json").read_text(encoding="utf-8"))
    key["directory"] = str(out)  # wherever the sheet folder is now
    seen = [int(x) for x in args.seen.split(",") if x.strip()]
    d = decide(key, seen)
    for line in d["fired"]:
        print(line)
    if d["wait"]:
        print("no target: the run waits")
    else:
        print(f"target: p95 masked delta E {d['target']:g}")
        print(f"masking floor: {d['floor']:g}" if d["floor"] is not None else "masking floor: unchanged")
        if d["floor"] is not None:
            print(f"re-run the sheet with loss.visibility.masking_floor={d['floor']:g} and check the busy rows")
    (out / "calibration_result.json").write_text(json.dumps({"seen": seen, **d}, indent=2) + "\n", encoding="utf-8")
    return 1 if d["wait"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="training config (to make the sheet)")
    parser.add_argument("--checkpoint", help="the 140,000-step checkpoint (to make the sheet)")
    parser.add_argument("--out", default="runs/calibration")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda when there is one)")
    parser.add_argument("--seen", default=None, help="numbers of the pairs where a difference was seen, e.g. 3,7,12")
    args = parser.parse_args()
    if args.seen is not None:
        return seen_mode(args)
    if not args.config or not args.checkpoint:
        parser.error("making the sheet needs --config and --checkpoint")
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
