"""The phone check of fine colour detail: does the simulation lose as much as the phone does?

The eye model pushes the mark toward fine colour, where phone cameras are weakest (demosaicing, colour
noise reduction, the stream's compression). The simulation models blur, loss of resolution and JPEG
4:2:0 only. This page and script measure how much of a grating survives the phone, and compare it with
the simulation at the same camera pixels per tile
(docs/superpowers/specs/2026-11-24-eye-model-loss-design.md, section 5b).

    python phone_gratings.py make --out runs/phone_gratings
    python phone_gratings.py measure --frame frame-1080x1920-20261125-143000.png --px-per-tile 331.5

``make`` writes <out>/gratings.html and the image it shows, gratings.png. Print it at 100 percent: each
patch is a 50 mm tile. Columns are luminance, red-green and blue-yellow, rows 24, 38, 51 and 77 cycles
per tile; the strip under them is flat grey, and four black squares mark the corners.

On the phone open the web app's measure page, hold the sheet upright in the frame with all four black
squares in view (about 20 cm and about 28 cm), press Freeze and Save frame, and measure px per cm on a
ruler as before. ``measure`` takes that frame and px per cm x 5 (camera px per 50 mm tile). It finds the
squares, maps the frame onto the page, sets black and grey from the squares and the grey strip, and fits
each patch's grating in its opponent channel (visibility.py) on the frame's own pixels. It then puts the
page through the simulation at that resolution (flattened mode, full strength, tilt and the colour step
off: the colour step changes the whole image, which the black and grey levels undo, and is no loss of
detail), averaged over ``--draws`` random draws, and prints both and their ratio. The verdict says
"colour detail matches" unless the phone keeps under half of the simulation's grating somewhere.
"""

from __future__ import annotations

import argparse
import copy
import html
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from constants import TILE_PX
from visibility import from_opponent, opponent

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs" / "runpod_labels_generated_fresh.yaml"  # the newest pod run's settings

CHANNELS = ("luminance", "red-green", "blue-yellow")  # opponent channels O1, O2, O3: the page's columns
CYCLES = (24, 38, 51, 77)  # cycles per tile: the page's rows
# Opponent-space amplitude of each grating on mid-grey; every RGB value stays within about 0.25 to 0.65.
AMPLITUDE = {"luminance": 0.07, "red-green": 0.02, "blue-yellow": 0.06}
GREY = 0.5
TILE_MM = 50

# Page layout in page pixels, TILE_PX per tile. The page is three tiles wide, so it fits the short side
# of a 1080 px portrait stream up to about 350 camera px per tile.
BORDER_PX = 32  # white paper around everything
SQUARE_PX = 64  # the black corner squares, above and below the grid
GAP_PX = 16  # white between a square and the grid
GREY_PX = 192  # height of the flat grey strip under the gratings
EDGE_PX = 48  # left out at every edge of a patch when measuring, for blur from the next patch

UNDER = 0.5  # the phone keeping less than this share of the simulation's grating fails the check
DRAWS = 16  # one draw is no reference: blur and JPEG quality vary widely from draw to draw
SEED = 0


def page_size() -> tuple[int, int]:
    """(width, height) of the page image."""
    width = 2 * BORDER_PX + len(CHANNELS) * TILE_PX
    height = 2 * BORDER_PX + 2 * (SQUARE_PX + GAP_PX) + len(CYCLES) * TILE_PX + GREY_PX
    return width, height


def _grid_origin() -> tuple[int, int]:
    return BORDER_PX, BORDER_PX + SQUARE_PX + GAP_PX


def patch_box(channel: str, cycles: int) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of a grating patch on the page."""
    x0, y0 = _grid_origin()
    col, row = CHANNELS.index(channel), CYCLES.index(cycles)
    return x0 + col * TILE_PX, y0 + row * TILE_PX, x0 + (col + 1) * TILE_PX, y0 + (row + 1) * TILE_PX


def grey_box() -> tuple[int, int, int, int]:
    x0, y0 = _grid_origin()
    y0 += len(CYCLES) * TILE_PX
    return x0, y0, x0 + len(CHANNELS) * TILE_PX, y0 + GREY_PX


def square_boxes() -> list[tuple[int, int, int, int]]:
    """The four corner squares, top-left, top-right, bottom-right, bottom-left."""
    width, height = page_size()
    left, right = BORDER_PX, width - BORDER_PX - SQUARE_PX
    top, bottom = BORDER_PX, height - BORDER_PX - SQUARE_PX
    return [(x, y, x + SQUARE_PX, y + SQUARE_PX) for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))]


def _centre(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])


def _quantise(x: torch.Tensor) -> torch.Tensor:
    """As an 8-bit PNG holds it (viz.save_image rounds the same way)."""
    return (x.clamp(0.0, 1.0) * 255.0 + 0.5).floor() / 255.0


def grating(channel: str, cycles: int) -> torch.Tensor:
    """One patch, (1,3,TILE_PX,TILE_PX): mid-grey with ``cycles`` whole cycles across it in one opponent channel (cos along x)."""
    grey = torch.full((1, 3, TILE_PX, TILE_PX), GREY)
    wave = torch.cos(2.0 * math.pi * cycles * torch.arange(TILE_PX, dtype=torch.float64) / TILE_PX).float()
    opp = opponent(grey)
    c = CHANNELS.index(channel)
    opp[:, c] = opp[:, c] + AMPLITUDE[channel] * wave.expand(TILE_PX, TILE_PX)
    return from_opponent(opp)


def page() -> torch.Tensor:
    """The page image as printed, (1,3,H,W) in [0, 1], 8-bit levels."""
    width, height = page_size()
    image = torch.ones(1, 3, height, width)
    for channel in CHANNELS:
        for cycles in CYCLES:
            x0, y0, x1, y1 = patch_box(channel, cycles)
            image[..., y0:y1, x0:x1] = grating(channel, cycles)
    x0, y0, x1, y1 = grey_box()
    image[..., y0:y1, x0:x1] = GREY
    for x0, y0, x1, y1 in square_boxes():
        image[..., y0:y1, x0:x1] = 0.0
    return _quantise(image)


# -- the simulation at a given camera resolution -----------------------------


def camera_frame(page_image: torch.Tensor, cfg: dict, px_per_tile: float, seed: int) -> torch.Tensor:
    """One draw of the simulation of the page as the camera records it, at ``px_per_tile`` camera px per tile.

    Flattened mode at full strength with perspective (tilt and rotation) and the colour step off: blur
    (simulate._scene), then loss of resolution down to the camera, noise and JPEG (simulate._sensor) at
    camera resolution. The frame is left at camera resolution, as a phone frame is. The global random
    state is left as it was.
    """
    from simulate import _scene, _sensor, _sensor_draws

    sim_cfg = copy.deepcopy(cfg)
    sim = sim_cfg["simulation"]
    sim["enabled"] = True
    sim["effects"] = {**sim["effects"], "perspective": False, "colour": False}
    _, _, height, width = page_image.shape
    size = (max(16, round(height * px_per_tile / TILE_PX)), max(16, round(width * px_per_tile / TILE_PX)))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        frame, _, _ = _scene(page_image, sim_cfg, 1.0)
        sigma, quality = _sensor_draws(page_image.shape[0], sim_cfg, 1.0, page_image)
        frame = F.interpolate(frame, size=size, mode="bilinear", antialias=True, align_corners=False)
        return _sensor(frame, sigma, quality, sim["jpeg"]["chroma_subsampling"])


# -- measuring a frame -------------------------------------------------------


def read_frame(path: str | Path) -> torch.Tensor:
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB")).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).float().div(255.0)[None]


def find_squares(frame: torch.Tensor, px_per_tile: float) -> torch.Tensor:
    """Centres of the four black corner squares in the frame, (4,2) in frame pixels: top-left, top-right, bottom-right, bottom-left.

    Dark regions of about the expected size and square shape, clear of the frame's edge (a square cut by
    it has its centre in the wrong place); of those, the ones furthest toward each corner. Raises
    ValueError when there are not four.
    """
    import numpy as np
    from scipy import ndimage  # installed with lpips

    grey = frame[0].mean(dim=0).numpy()
    black, paper = np.percentile(grey, 0.2), np.percentile(grey, 95)
    dark = grey < black + 0.3 * (paper - black)
    labels, _ = ndimage.label(dark)
    side = SQUARE_PX * px_per_tile / TILE_PX
    found = []
    for index, where in enumerate(ndimage.find_objects(labels), start=1):
        if where is None:
            continue
        if where[0].start == 0 or where[1].start == 0 or where[0].stop == grey.shape[0] or where[1].stop == grey.shape[1]:
            continue
        h, w = where[0].stop - where[0].start, where[1].stop - where[1].start
        mask = labels[where] == index
        area = int(mask.sum())
        if not (0.4 * side**2 <= area <= 2.0 * side**2 and 0.6 * side <= min(h, w) and max(h, w) <= 1.6 * side):
            continue
        if area < 0.7 * h * w:
            continue
        ys, xs = np.nonzero(mask)
        found.append((xs.mean() + where[1].start + 0.5, ys.mean() + where[0].start + 0.5))
    if len(found) < 4:
        raise ValueError(
            f"found {len(found)} of the four black corner squares (expected about {side:.0f} px wide at "
            f"{px_per_tile:g} px per tile): keep all four in the frame and check --px-per-tile"
        )
    points = torch.tensor(found, dtype=torch.float64)
    total, diff = points.sum(dim=1), points[:, 0] - points[:, 1]
    picks = [int(total.argmin()), int(diff.argmax()), int(total.argmax()), int(diff.argmin())]
    if len(set(picks)) < 4:
        raise ValueError("the dark squares found do not form the four corners of the sheet")
    return points[picks]


def _inside(u: torch.Tensor, v: torch.Tensor, box, margin: float) -> torch.Tensor:
    x0, y0, x1, y1 = box
    return (u >= x0 + margin) & (u < x1 - margin) & (v >= y0 + margin) & (v < y1 - margin)


def measure_frame(frame: torch.Tensor, px_per_tile: float) -> dict[tuple[str, int], float]:
    """Each patch's grating amplitude in the frame, as a share of its amplitude on the page.

    The frame's pixels are mapped onto the page through the four squares. Black (the squares) and grey
    (the strip) set the levels of each RGB channel; then each patch's opponent channel is fitted, on the
    frame's own pixels with no resampling, with a mean, a slope in x and y, and the cosine and sine at the
    patch's frequency. The amplitude is the length of (cosine, sine).
    """
    from geometry import apply_homography, homography_from_points

    corners = find_squares(frame, px_per_tile)
    targets = torch.tensor([_centre(box) for box in square_boxes()], dtype=torch.float64)
    to_page = homography_from_points(corners[None], targets[None])
    _, _, height, width = frame.shape
    ys = torch.arange(height, dtype=torch.float64) + 0.5
    xs = torch.arange(width, dtype=torch.float64) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    uv = apply_homography(to_page, torch.stack([xx, yy], dim=-1).view(1, -1, 2))[0]
    u, v = uv[:, 0].view(height, width), uv[:, 1].view(height, width)

    rgb = frame[0].double()
    in_squares = torch.zeros_like(u, dtype=torch.bool)
    for box in square_boxes():
        in_squares |= _inside(u, v, box, SQUARE_PX / 4)
    in_grey = _inside(u, v, grey_box(), EDGE_PX)
    if int(in_squares.sum()) < 16 or int(in_grey.sum()) < 16:
        raise ValueError("too little of the black squares or the grey strip is in the frame")
    black = rgb[:, in_squares].median(dim=1).values.view(3, 1, 1)
    grey = rgb[:, in_grey].median(dim=1).values.view(3, 1, 1)
    page_grey = _quantise(torch.tensor(GREY)).double()
    levelled = (rgb - black) / (grey - black).clamp(min=1e-3) * page_grey
    opp = opponent(levelled[None])[0]

    out = {}
    for channel in CHANNELS:
        c = CHANNELS.index(channel)
        for cycles in CYCLES:
            box = patch_box(channel, cycles)
            keep = _inside(u, v, box, EDGE_PX)
            if int(keep.sum()) < 64:
                raise ValueError(f"the {channel} {cycles} cycle patch is not in the frame")
            pu, pv, values = u[keep], v[keep], opp[c][keep]
            phase = 2.0 * math.pi * cycles * (pu - box[0]) / TILE_PX
            cx, cy = _centre(box)
            design = torch.stack(
                [torch.ones_like(pu), (pu - cx) / TILE_PX, (pv - cy) / TILE_PX, torch.cos(phase), torch.sin(phase)], dim=1
            )
            fit = torch.linalg.lstsq(design, values[:, None]).solution[:, 0]
            out[(channel, cycles)] = math.hypot(float(fit[3]), float(fit[4])) / AMPLITUDE[channel]
    return out


def reference(cfg: dict, px_per_tile: float, draws: int = DRAWS, seed: int = SEED) -> dict[tuple[str, int], float]:
    """The simulation's amplitudes at ``px_per_tile``: ``measure_frame`` on ``draws`` draws (seeds seed, seed + 1, ...), averaged."""
    image = page()
    total: dict[tuple[str, int], float] = {}
    for k in range(draws):
        for key, value in measure_frame(camera_frame(image, cfg, px_per_tile, seed + k), px_per_tile).items():
            total[key] = total.get(key, 0.0) + value / draws
    return total


def table(phone: dict[tuple[str, int], float], sim: dict[tuple[str, int], float]) -> list[dict]:
    """One row per patch: channel, cycles, phone, simulation and their ratio (nan where the simulation keeps nothing)."""
    rows = []
    for channel in CHANNELS:
        for cycles in CYCLES:
            p, s = phone[(channel, cycles)], sim[(channel, cycles)]
            rows.append({"channel": channel, "cycles": cycles, "phone": p, "sim": s, "ratio": p / s if s > 1e-3 else math.nan})
    return rows


def verdict(rows: list[dict]) -> str:
    under = [f"{r['channel']} {r['cycles']} cycles" for r in rows if r["ratio"] < UNDER]
    if not under:
        return "colour detail matches"
    return "the phone keeps under half of the simulation's detail at " + ", ".join(under)


def compare(frame: torch.Tensor, px_per_tile: float, cfg: dict, draws: int = DRAWS, seed: int = SEED) -> list[dict]:
    return table(measure_frame(frame, px_per_tile), reference(cfg, px_per_tile, draws, seed))


# -- the commands ------------------------------------------------------------


def make(args: argparse.Namespace) -> int:
    from acceptance import PAGE_CSS
    from viz import save_image

    out = Path(args.out)
    save_image(page()[0], out / "gratings.png")
    width, height = page_size()
    mm = TILE_MM / TILE_PX
    columns = ", ".join(CHANNELS)
    rows = ", ".join(str(c) for c in CYCLES)
    text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="robots" content="noindex, nofollow">
<title>Phone gratings</title>
<style>
{PAGE_CSS}
  @page {{ margin: 10mm; }}
  body {{ margin: 0; font-size: 9pt; }}
  h1 {{ font-size: 12pt; margin: 0 0 1mm; }}
  p {{ margin: 0 0 1mm; max-width: none; }}
  .sheet {{ display: block; width: {width * mm:.2f}mm; height: {height * mm:.2f}mm; }}
</style>
</head>
<body>
<h1>Phone gratings</h1>
<p>Print at 100 percent on one page: each patch is 50 mm wide. Columns: {html.escape(columns)}. Rows: {rows} cycles
per tile. Grey strip below. Keep all four black squares in the frame; freeze and save it on the measure page.</p>
<img class="sheet" src="gratings.png" alt="grating patches">
</body>
</html>
"""
    (out / "gratings.html").write_text(text, encoding="utf-8")
    print(f"{len(CHANNELS) * len(CYCLES)} gratings and a grey strip: {out / 'gratings.html'} ({width * mm:.0f} x {height * mm:.0f} mm)")
    return 0


def measure(args: argparse.Namespace) -> int:
    from config import load_config

    if args.draws < 1:
        print("--draws must be at least 1")
        return 2
    cfg = load_config(args.config)
    try:
        rows = compare(read_frame(args.frame), args.px_per_tile, cfg, args.draws, args.seed)
    except ValueError as error:
        print(error)
        return 2
    print(f"{args.frame} at {args.px_per_tile:g} camera px per tile; simulation: {args.config}, {args.draws} draws")
    print("share of the page's grating kept")
    print(f"{'patch':<26}{'phone':>8}{'simulation':>12}{'ratio':>8}")
    for r in rows:
        ratio = "-" if math.isnan(r["ratio"]) else f"{r['ratio']:.2f}"
        print(f"{r['channel'] + ' ' + str(r['cycles']) + ' cycles':<26}{r['phone']:>8.3f}{r['sim']:>12.3f}{ratio:>8}")
    print(verdict(rows))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    making = commands.add_parser("make", help="write gratings.html")
    making.add_argument("--out", default="runs/phone_gratings")
    measuring = commands.add_parser("measure", help="compare a saved phone frame with the simulation")
    measuring.add_argument("--frame", required=True, help="PNG saved with the measure page's Save frame button")
    measuring.add_argument("--px-per-tile", type=float, required=True, help="camera px per 50 mm tile: px per cm x 5")
    measuring.add_argument("--config", default=str(DEFAULT_CONFIG), help="training config whose simulation to compare with")
    measuring.add_argument("--draws", type=int, default=DRAWS, help="simulation draws to average")
    measuring.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    return make(args) if args.command == "make" else measure(args)


if __name__ == "__main__":
    sys.exit(main())
