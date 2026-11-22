"""Acceptance: is the mark read correctly, and can it be seen?

    python acceptance.py --config configs/labels.yaml
    python acceptance.py --config configs/labels.yaml --checkpoint copies/step_00012000.pt

Takes the newest checkpoint in the config's ``checkpoint.dir`` unless one is
named. Do not point it at the folder of a run that is still training: copy
the checkpoint somewhere else first and name the copy.

Measured on 256 label images that training never saw (``data.holdout``), each
cut to one tile from the middle of the photo, with random bits that are the
same every time. Both models in eval mode, the camera simulation at full
strength, the reader given a flattened patch. One simulated shot per tile
unless --views says more. For each strength of the mark (the ``alpha``
argument of ``Embedder.forward``; the models were trained at one of them):

    read     share of tiles with at most 4 wrong bits among the first 63,
             which is what the BCH code can correct. PASS at 0.80 or more.
             Also the mean bit accuracy and the share with all 64 bits right.
    spread   the fixed-place measure of the spread check (spread.py).
             PASS below 2.5.
    seen     PSNR of the marked tile against the original, in dB: the mean
             over the tiles and the mean of the worst tenth. No pass line.

Then, unless --shots is given empty, a multi-shot table: each marked tile is
photographed 9 times by the simulation, each time with fresh random draws,
and read from the first K shots (K = 1, 3, 5, 9) by averaging the reader's
logits before taking the bits. Share within reach of the error correction
and mean bit accuracy per K and alpha. This is simulated: real frames of one
hand-held scan are less independent than these shots.

Writes into runs/acceptance/<step>/:

    results.md, results.json   the table above
    blind.html                 20 pairs of original and marked tile, side by
                               side, 50 mm wide each in print, left and right
                               shuffled: 10 pairs with busy content and 10
                               with large flat areas
    blind_answers.json         which side is the marked one. Not printed.
    pairs/                     the tiles of blind.html, as PNG

If training/data/labels/ holds images, they are measured too, reported
separately, and added to blind.html as a second section.

blind.html shows dataset photos. It carries the credit line from
``data.credit`` and is for looking at on this machine, not for publishing.
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

from checkpoint import latest_checkpoint, load_checkpoint
from config import load_config
from constants import MESSAGE_BITS, TILE_PX
from data import CenterFraction, holdout_images, list_images, load_rgb
from geometry import flatten_random_patch
from simulate import simulate, wrap_surface
from spread import busy_windows, local_energy, spread_report
from train import BCH_CODEWORD_BITS, BCH_CORRECTABLE, Trainer
from viz import save_image

HERE = Path(__file__).resolve().parent
ALPHAS = (0.30, 0.25, 0.20, 0.15)
IMAGES = 256
MIN_DECODED = 0.80      # share of tiles within reach of the error correction
MAX_FIXED_PLACE = 2.5
BLIND_PAIRS = 10        # of each kind
# A 48 px window counts as flat when the grey image differs from a 9 px blur
# of itself by less than this, as a mean square: about 2 grey levels of 255.
FLAT_LEVEL = 6e-5
SHOTS = (1, 3, 5, 9)    # multi-shot reading: logits averaged over this many simulated shots
EXTRA_DIR = HERE / "data" / "labels"
SEED = 0


class AtAlpha(torch.nn.Module):
    """The embedder at a given strength, with the call signature spread_report expects."""

    def __init__(self, embedder: torch.nn.Module, alpha: float):
        super().__init__()
        self.embedder, self.alpha = embedder, alpha

    def forward(self, tile: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        return self.embedder(tile, bits, alpha=self.alpha)


def tiles_from(files: list[Path], center_fraction: float = 1.0) -> torch.Tensor:
    """One tile per image: the middle of the photo, scaled so its short side is the tile."""
    steps = [transforms.Resize(TILE_PX, antialias=True), transforms.CenterCrop(TILE_PX), transforms.ToTensor()]
    if center_fraction < 1.0:
        steps.insert(0, CenterFraction(center_fraction))
    cut = transforms.Compose(steps)
    return torch.stack([cut(load_rgb(path)) for path in files])


def fixed_bits(count: int, seed: int) -> torch.Tensor:
    return torch.randint(0, 2, (count, MESSAGE_BITS), generator=torch.Generator().manual_seed(seed)).float()


@torch.no_grad()
def read_shots(
    trainer: Trainer, tiles: torch.Tensor, bits: torch.Tensor, alpha: float, shots: int, seed: int, chunk: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Embed at ``alpha`` once, photograph it ``shots`` times at full strength, flatten, read.

    Every shot of a tile photographs the same marked tile with fresh random
    draws of the simulation (tilt, blur, colour, resolution, noise, JPEG) and
    a fresh flattened patch. Returns (logits, psnr, marked): the reader's
    logits as (shots, N, 64), PSNR per tile as (N,), and the marked tiles on
    the CPU. The draws are seeded, so every alpha is photographed the same way.
    """
    cfg, device = trainer.cfg, trainer.device
    strength = float(cfg["schedule"]["sim_strength"]["max_strength"])
    repeat, jitter = int(cfg["surface"]["repeat"]), float(cfg["flatten"]["corner_jitter_px"])
    trainer.embedder.eval()
    trainer.reader.eval()
    logits, marked_all, psnr = [], [], []
    torch.manual_seed(seed)
    for shot_index in range(shots):
        this_shot = []
        for start in range(0, len(tiles), chunk):
            tile = tiles[start : start + chunk].to(device)
            message = bits[start : start + chunk].to(device)
            marked = (tile + trainer.embedder(tile, message, alpha=alpha)).clamp(0.0, 1.0)
            shot = simulate(wrap_surface(marked, repeat), cfg, strength)
            patch, _ = flatten_random_patch(shot.frame, shot.homography, tile_px=TILE_PX, corner_jitter_px=jitter)
            this_shot.append(trainer.reader(patch).float().cpu())
            if shot_index == 0:
                mse = (marked - tile).pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
                psnr.append((-10.0 * torch.log10(mse)).cpu())
                marked_all.append(marked.cpu())
        logits.append(torch.cat(this_shot))
    return torch.stack(logits), torch.cat(psnr), torch.cat(marked_all)


def read_tiles(
    trainer: Trainer, tiles: torch.Tensor, bits: torch.Tensor, alpha: float, views: int, seed: int, chunk: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """As read_shots, with every shot read on its own: wrong bits as (views * N, 64) of bool."""
    logits, psnr, marked = read_shots(trainer, tiles, bits, alpha, views, seed, chunk)
    return ((logits > 0).float() != bits).reshape(-1, MESSAGE_BITS), psnr, marked


def vote(logits: torch.Tensor, bits: torch.Tensor, counts: tuple[int, ...] | list[int]) -> list[dict]:
    """Read each tile from K shots: average the logits of the first K shots, then take the signs.

    ``logits`` is (shots, N, 64), ``bits`` (N, 64). One row per K in ``counts``
    with the share of tiles within reach of the error correction, the mean bit
    accuracy and the share with all 64 right. The rows for different K share
    their shots: K = 3 is the first three of the shots K = 9 uses.
    """
    rows = []
    for count in counts:
        if not 1 <= count <= len(logits):
            raise ValueError(f"{count} shots asked for, {len(logits)} taken")
        wrong = (logits[:count].mean(dim=0) > 0).float() != bits
        rows.append(
            {
                "shots": count,
                "decoded": (wrong[:, :BCH_CODEWORD_BITS].sum(dim=1) <= BCH_CORRECTABLE).float().mean().item(),
                "bit_accuracy": 1.0 - wrong.float().mean().item(),
                "all_64_right": (wrong.sum(dim=1) == 0).float().mean().item(),
            }
        )
    return rows


def measure(trainer: Trainer, tiles: torch.Tensor, bits: torch.Tensor, alpha: float, views: int, seed: int) -> dict:
    """The acceptance numbers for one strength of the mark."""
    wrong, psnr, _ = read_tiles(trainer, tiles, bits, alpha, views, seed)
    spread = spread_report(AtAlpha(trainer.embedder, alpha), tiles, bits, trainer.cfg, busy_windows(tiles, trainer.cfg))
    trainer.embedder.eval()  # spread_report leaves it in training mode
    worst = psnr.sort().values[: max(1, math.ceil(0.1 * len(psnr)))]
    decoded = (wrong[:, :BCH_CODEWORD_BITS].sum(dim=1) <= BCH_CORRECTABLE).float().mean().item()
    return {
        "alpha": alpha,
        "images": len(tiles),
        "reads": len(wrong),
        "decoded": decoded,
        "decoded_pass": decoded >= MIN_DECODED,
        "bit_accuracy": 1.0 - wrong.float().mean().item(),
        "all_64_right": (wrong.sum(dim=1) == 0).float().mean().item(),
        "fixed_place": spread["fixed_place"],
        "fixed_place_pass": spread["fixed_place"] < MAX_FIXED_PLACE,
        "strongest_spot": spread["strongest_spot"],
        "psnr_mean_db": psnr.mean().item(),
        "psnr_worst_tenth_db": worst.mean().item(),
    }


def table(rows: list[dict]) -> str:
    def verdict(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    lines = [
        "| Alpha | At most 4 wrong of the first 63 | Bit accuracy | All 64 right | Fixed place | PSNR mean, dB | PSNR worst tenth, dB |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['alpha']:.2f} | {r['decoded']:.3f} {verdict(r['decoded_pass'])} | {r['bit_accuracy']:.3f} | "
            f"{r['all_64_right']:.3f} | {r['fixed_place']:.2f} {verdict(r['fixed_place_pass'])} | "
            f"{r['psnr_mean_db']:.1f} | {r['psnr_worst_tenth_db']:.1f} |"
        )
    return "\n".join(lines)


MULTI_SHOT_NOTE = (
    "Simulated: each shot is an independent draw of the camera simulation (tilt, blur, colour, resolution, "
    "noise and JPEG all drawn afresh, and a fresh flattened patch). Real frames from one hand-held scan share "
    "the light, the angle, the distance and the camera, so they are less independent than this, and averaging "
    "them gains less than these rows show."
)


def multi_shot(trainer: Trainer, tiles: torch.Tensor, bits: torch.Tensor, alphas: list[float], counts: list[int], seed: int) -> list[dict]:
    """vote() rows for every alpha, from max(counts) shots of each tile."""
    rows = []
    for alpha in alphas:
        logits, _, _ = read_shots(trainer, tiles, bits, alpha, max(counts), seed)
        rows += [{"alpha": alpha, **row} for row in vote(logits, bits, counts)]
    return rows


def multi_table(rows: list[dict]) -> str:
    lines = [
        "| Alpha | Shots averaged | At most 4 wrong of the first 63 | Bit accuracy | All 64 right |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['alpha']:.2f} | {r['shots']} | {r['decoded']:.3f} | {r['bit_accuracy']:.3f} | {r['all_64_right']:.3f} |"
        )
    return "\n".join(lines)


def content_scores(tiles: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Per tile: (how busy it is all over, share of its area that is flat).

    Local contrast as in the spread check: the grey image minus a 9 px blur of
    itself, squared, averaged over 48 px windows. Busy is the median window;
    flat is the share of windows under FLAT_LEVEL.
    """
    grey = tiles.mean(dim=1, keepdim=True)
    blurred = F.avg_pool2d(F.pad(grey, (4, 4, 4, 4), mode="reflect"), 9, 1)
    contrast = local_energy(grey - blurred, 48, 8).flatten(1)
    return contrast.median(dim=1).values, (contrast < FLAT_LEVEL).float().mean(dim=1)


def pick_pairs(tiles: torch.Tensor, cfg: dict, each: int) -> list[tuple[int, str]]:
    """Indices of the tiles shown in blind.html: the flattest ``each`` and the busiest ``each`` of the rest."""
    busy, flat = content_scores(tiles, cfg)
    flat_picks = flat.argsort(descending=True)[:each].tolist()
    busy_picks = [i for i in busy.argsort(descending=True).tolist() if i not in flat_picks][:each]
    return [(i, "large flat areas") for i in flat_picks] + [(i, "busy content") for i in busy_picks]


def write_blind(
    directory: Path, step: int, alpha: float, sections: list[dict], credit: str | None, seed: int
) -> tuple[Path, Path]:
    """blind.html, its PNG tiles and blind_answers.json.

    ``sections``: [{"title", "originals" (K,3,H,W), "marked", "kinds", "sources"}].
    Pair order and which side holds the marked tile come from ``seed``.
    """
    rng = random.Random(seed)
    answers, body, number = [], [], 0
    for section in sections:
        order = list(range(len(section["originals"])))
        rng.shuffle(order)
        body.append(f"<h2>{html.escape(section['title'])}</h2>")
        for index in order:
            number += 1
            marked_left = rng.random() < 0.5
            left, right = (section["marked"], section["originals"]) if marked_left else (section["originals"], section["marked"])
            save_image(left[index], directory / "pairs" / f"{number:02d}_left.png")
            save_image(right[index], directory / "pairs" / f"{number:02d}_right.png")
            body.append(
                f'<div class="pair"><span class="n">{number}</span>'
                f'<img src="pairs/{number:02d}_left.png" alt="pair {number}, left">'
                f'<img src="pairs/{number:02d}_right.png" alt="pair {number}, right"></div>'
            )
            answers.append(
                {
                    "pair": number,
                    "marked": "left" if marked_left else "right",
                    "section": section["title"],
                    "kind": section["kinds"][index],
                    "source": section["sources"][index],
                }
            )
    footer = f"<footer>{html.escape(credit)}</footer>" if credit else ""
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="robots" content="noindex, nofollow">
<title>Which tile carries the mark? Step {step}</title>
<style>
  @page {{ size: A4; margin: 12mm 12mm 18mm; }}
  body {{ font: 11pt/1.4 system-ui, sans-serif; color: #111; background: #fff; margin: 12mm 12mm 20mm; }}
  h1 {{ font-size: 15pt; margin: 0 0 3mm; }}
  h2 {{ font-size: 12pt; margin: 8mm 0 4mm; }}
  p {{ max-width: 150mm; margin: 0 0 3mm; }}
  .pair {{ display: flex; align-items: center; gap: 8mm; margin: 0 0 7mm; break-inside: avoid; }}
  .pair img {{ width: 50mm; height: 50mm; display: block; }}
  .n {{ width: 9mm; font-weight: 600; text-align: right; }}
  footer {{ position: fixed; left: 0; right: 0; bottom: 0; padding: 2mm 12mm; font-size: 9pt;
            background: #fff; border-top: 0.3mm solid #999; }}
</style>
</head>
<body>
<h1>Which tile carries the mark?</h1>
<p>In every pair one tile is the original and the other carries the mark. Note, for each number, whether you
think the marked tile is the left or the right one, or that you cannot tell. The answers are in
blind_answers.json.</p>
<p>Printed at 100 percent each tile is 50 mm wide, the size of a printed tile. On a screen the size depends on
the display and its zoom.</p>
{chr(10).join(body)}
{footer}
</body>
</html>
"""
    page_path, answers_path = directory / "blind.html", directory / "blind_answers.json"
    page_path.write_text(page, encoding="utf-8")
    answers_path.write_text(
        json.dumps({"step": step, "alpha": alpha, "seed": seed, "pairs": answers}, indent=2) + "\n", encoding="utf-8"
    )
    return page_path, answers_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="checkpoint file (default: the newest in checkpoint.dir)")
    parser.add_argument("--out", default="runs/acceptance", help="results go to <out>/<step>/ (default runs/acceptance)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda when there is one)")
    parser.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS), help="strengths of the mark to measure")
    parser.add_argument("--blind-alpha", type=float, default=None, help="strength shown in blind.html (default: the trained one)")
    parser.add_argument("--views", type=int, default=1, help="simulated shots per tile, each read on its own (default 1)")
    parser.add_argument(
        "--shots", type=int, nargs="*", default=list(SHOTS),
        help="multi-shot reading: shot counts K whose logits are averaged (default 1 3 5 9; none to skip)",
    )
    parser.add_argument("overrides", nargs="*", help="dotted.key=value config overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    if cfg["train"]["reader_input"] != "flattened" or not cfg["simulation"]["enabled"]:
        print("acceptance needs train.reader_input: flattened and simulation.enabled: true")
        return 2
    held_out = holdout_images(cfg["data"]["train_dir"], int(cfg["data"].get("holdout", 0)))
    if len(held_out) < IMAGES:
        print(f"acceptance needs {IMAGES} images kept back from training; data.holdout gives {len(held_out)}")
        return 2
    path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(cfg["checkpoint"]["dir"])
    if path is None or not path.is_file():
        print(f"no checkpoint found ({args.checkpoint or cfg['checkpoint']['dir']})")
        return 2

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg["loss"]["lpips_weight"] = 0.0  # no loss is computed here, so its pretrained network is not needed
    trainer = Trainer(eval_cfg, device)
    load_checkpoint(trainer, path)
    step = trainer.step
    trained_alpha = float(cfg["model"]["embedder"]["alpha"])
    blind_alpha = trained_alpha if args.blind_alpha is None else args.blind_alpha
    directory = Path(args.out) / f"{step:08d}"
    directory.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint {path}, step {step}, image-loss weight {trainer.image_loss_scale():.3f}, device {device}")
    print(
        f"{IMAGES} held-out images from {cfg['data']['train_dir']}, simulation at strength "
        f"{float(cfg['schedule']['sim_strength']['max_strength']):g}, flattened reader input, "
        f"{args.views} simulated shot{'s' if args.views != 1 else ''} per tile; trained at alpha {trained_alpha:g}"
    )

    files = held_out[:IMAGES]
    tiles = tiles_from(files, float(cfg["data"].get("center_fraction", 1.0)))
    bits = fixed_bits(len(tiles), SEED)
    rows = [measure(trainer, tiles, bits, alpha, args.views, SEED) for alpha in args.alphas]
    report = [f"## Held-out label images ({len(tiles)})", "", table(rows)]
    results = {"step": step, "checkpoint": str(path), "views": args.views, "trained_alpha": trained_alpha, "held_out": rows}
    if args.shots:
        shot_rows = multi_shot(trainer, tiles, bits, args.alphas, sorted(set(args.shots)), SEED)
        report += [
            "",
            f"## Several shots of each tile, logits averaged ({len(tiles)} held-out tiles)",
            "",
            multi_table(shot_rows),
            "",
            MULTI_SHOT_NOTE
            + f" The rows share their shots: each K reads the first K of the same {max(args.shots)}, "
            "and the 1-shot row is the first shot, the one the table above reads.",
        ]
        results["multi_shot"] = {"note": MULTI_SHOT_NOTE, "rows": shot_rows}

    picks = pick_pairs(tiles, cfg, BLIND_PAIRS)
    chosen = torch.tensor([index for index, _ in picks])
    _, _, marked = read_tiles(trainer, tiles[chosen], bits[chosen], blind_alpha, 1, SEED)
    sections = [
        {
            "title": "Label photos",
            "originals": tiles[chosen],
            "marked": marked,
            "kinds": [kind for _, kind in picks],
            "sources": [files[index].name for index, _ in picks],
        }
    ]

    if EXTRA_DIR.is_dir() and any(EXTRA_DIR.iterdir()):
        extra_files = list_images(EXTRA_DIR)
        extra = tiles_from(extra_files)
        extra_bits = fixed_bits(len(extra), SEED + 1)
        extra_rows = [measure(trainer, extra, extra_bits, alpha, args.views, SEED) for alpha in args.alphas]
        note = (
            ""
            if len(extra) >= 64
            else f"\n\nFixed place needs many images to mean anything; {len(extra)} are too few, so read that column with care."
        )
        report += ["", f"## Images in {EXTRA_DIR.relative_to(HERE).as_posix()} ({len(extra)})", "", table(extra_rows) + note]
        results["extra"] = {"directory": str(EXTRA_DIR), "rows": extra_rows}
        shown = list(range(min(len(extra), 2 * BLIND_PAIRS)))
        _, _, extra_marked = read_tiles(trainer, extra[shown], extra_bits[shown], blind_alpha, 1, SEED)
        sections.append(
            {
                "title": "Further pairs",
                "originals": extra[shown],
                "marked": extra_marked,
                "kinds": ["own label"] * len(shown),
                "sources": [extra_files[index].name for index in shown],
            }
        )

    text = (
        f"# Acceptance at step {step}\n\n"
        f"Checkpoint `{path.as_posix()}`. Eval mode, simulation at full strength, flattened reader input, "
        f"{args.views} simulated shot{'s' if args.views != 1 else ''} per tile, fixed bits. "
        f"The models were trained at alpha {trained_alpha:g}.\n\n"
        f"PASS lines: at most 4 wrong bits of the first 63 on at least {MIN_DECODED:.2f} of the tiles; "
        f"fixed place below {MAX_FIXED_PLACE:g}. PSNR has none.\n\n" + "\n".join(report) + "\n"
    )
    (directory / "results.md").write_text(text, encoding="utf-8")
    (directory / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    page, answers = write_blind(directory, step, blind_alpha, sections, cfg["data"].get("credit"), SEED)

    print("\n" + "\n".join(report))
    print(f"\nresults:  {directory / 'results.md'}")
    print(f"blind comparison at alpha {blind_alpha:g}: {page}")
    print(f"answers (not shown here): {answers}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
