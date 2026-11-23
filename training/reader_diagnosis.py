"""Reader diagnosis on the locator's held-out set: true corners, located corners, jittered corners.

    python reader_diagnosis.py --locator <copy of a locator checkpoint> --reader A.pt --reader B.pt ... --out DIR

For every reader (a stage-1 checkpoint: its own embedder marks the tiles, its
own reader reads them) on the 256 held-out frames of ``configs/locator.yaml``
(the same frames ``HeldOut`` draws during locator training):

* true corners, the true corner order;
* the locator's corners, best of the four turns (the turn with the fewest
  wrong bits; "decodes" means at least one turn decodes, as the app tries all
  four);
* true corners with every corner moved by exactly ``d`` reader pixels in a
  random direction (``--jitter``), for the reader with the most decodes on
  true corners.

All 256 frames are marked here. ``HeldOut`` in training marks only the even
ones (odd ones stay plain to check the locator does not lean on the mark), so
its read counts are out of 128. Marking does not change the frames' geometry:
the corners are checked to be equal to the half-marked set's.

"Decodes" = at most BCH_CORRECTABLE (4) of the 63 codeword bits wrong, what
BCH(63,39) corrects (/docs/message-format.md); the spare bit 63 is not
counted. Bit accuracy and wrong bits are over all 64 bits.

No training, nothing written except ``--out``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from config import load_config
from constants import TILE_PX
from geometry import flatten_from_corners
from locator_train import BCH_CODEWORD_BITS, BCH_CORRECTABLE, HeldOut, load_stage1, turned
from models import build_locator


def wrong_bits(logits: torch.Tensor, bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(wrong of 64, wrong of the 63 codeword bits) per sample."""
    wrong = (logits > 0).float() != bits
    return wrong.sum(dim=1), wrong[:, :BCH_CODEWORD_BITS].sum(dim=1)


def stats(wrong64: torch.Tensor, decodes: torch.Tensor) -> dict:
    n = len(wrong64)
    return {
        "samples": n,
        "bit_acc": 1.0 - wrong64.float().mean().item() / 64.0,
        "mean_wrong_bits": wrong64.float().mean().item(),
        "median_wrong_bits": wrong64.float().median().item(),
        "decodes": int(decodes.sum().item()),
    }


def jitter(corners: torch.Tensor, tile_px: torch.Tensor, d: float, g: torch.Generator) -> torch.Tensor:
    """Each corner moved by exactly d reader pixels (TILE_PX per tile) in a uniformly random direction."""
    angle = torch.rand(corners.shape[:2], generator=g) * 2 * math.pi
    step = torch.stack([angle.cos(), angle.sin()], dim=-1).to(corners.device)
    return corners + step * (d * tile_px.view(-1, 1, 1) / TILE_PX)


@torch.no_grad()
def run_reader(path: Path, cfg: dict, locator, device, cache: Path, jitter_px: list[float], draws: int) -> dict:
    embedder, reader, step, _ = load_stage1(path, device)
    held = HeldOut(cfg, embedder, device, cache)
    plain_corners = torch.cat([b["corners"].cpu() for b in held.batches()])
    held.marked = torch.ones_like(held.marked)                      # read all 256
    true64, true63, loc64, loc_any, turn_best, jit = [], [], [], [], [], {d: [] for d in jitter_px}
    corners_all = []
    g = torch.Generator().manual_seed(20261123)
    for batch in held.batches():
        cam, corners, bits = batch["camera"], batch["corners"], batch["bits"]
        corners_all.append(corners.cpu())
        w64, w63 = wrong_bits(reader(flatten_from_corners(cam, corners, TILE_PX)), bits)
        true64.append(w64.cpu()); true63.append(w63.cpu())
        located = locator(batch["input"]) / batch["scale"]
        per64, per63 = [], []
        for t in range(4):
            a, b = wrong_bits(reader(flatten_from_corners(cam, turned(located, t), TILE_PX)), bits)
            per64.append(a); per63.append(b)
        per64, per63 = torch.stack(per64, 1), torch.stack(per63, 1)
        best = per64.argmin(dim=1)
        loc64.append(per64.gather(1, best.view(-1, 1)).squeeze(1).cpu())
        loc_any.append((per63 <= BCH_CORRECTABLE).any(dim=1).cpu())
        turn_best.append(best.cpu())
        for d in jitter_px:
            rows = []
            for _ in range(draws):
                moved = jitter(corners, batch["tile_px"], d, g)
                a, b = wrong_bits(reader(flatten_from_corners(cam, moved, TILE_PX)), bits)
                rows.append(torch.stack([a, b], 1).cpu())
            jit[d].append(torch.stack(rows, 0))                     # (draws, b, 2)
    if not torch.equal(torch.cat(corners_all), plain_corners):
        raise RuntimeError("marking every frame changed the frames' corners")
    true64, true63 = torch.cat(true64), torch.cat(true63)
    out = {
        "reader": str(path), "step": step,
        "true": stats(true64, true63 <= BCH_CORRECTABLE),
        "located_best_turn": stats(torch.cat(loc64), torch.cat(loc_any)),
        "located_best_turn_not_0": int((torch.cat(turn_best) != 0).sum().item()),
        "true_wrong63_histogram": torch.bincount(true63.long(), minlength=8)[:8].tolist(),
    }
    if jitter_px:
        out["jitter"] = {}
        for d in jitter_px:
            j = torch.cat(jit[d], 1)                                # (draws, 256, 2)
            per_draw = [stats(j[k, :, 0], j[k, :, 1] <= BCH_CORRECTABLE) for k in range(draws)]
            out["jitter"][str(d)] = {
                "bit_acc": sum(p["bit_acc"] for p in per_draw) / draws,
                "mean_wrong_bits": sum(p["mean_wrong_bits"] for p in per_draw) / draws,
                "decodes_mean": sum(p["decodes"] for p in per_draw) / draws,
                "decodes_per_draw": [p["decodes"] for p in per_draw],
            }
    # Cross-check with HeldOut.evaluate itself (all 256 marked): same frames, same reads.
    e = held.evaluate(locator, reader)
    out["heldout_evaluate"] = {k: e[k] for k in (
        "marked_samples", "reader_acc_true", "read_true", "reader_acc_best_turn", "read_any_turn", "mean", "median",
    )}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/locator.yaml")
    p.add_argument("--locator", required=True, help="a copy of a locator checkpoint")
    p.add_argument("--reader", action="append", required=True, help="stage-1 checkpoint; repeat for several")
    p.add_argument("--jitter", type=float, nargs="*", default=[], help="reader px; run for every reader")
    p.add_argument("--draws", type=int, default=4, help="jitter draws per sample and level")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.locator, map_location="cpu", weights_only=True)
    locator = build_locator(cfg)
    locator.load_state_dict(state["locator"])
    locator.to(device).eval()
    cache = Path(cfg["checkpoint"]["dir"]) / "heldout_tiles.pt"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for r in args.reader:
        res = run_reader(Path(r), cfg, locator, device, cache, args.jitter, args.draws)
        res["locator_step"] = int(state["step"])
        results.append(res)
        print(json.dumps(res, indent=1), flush=True)
        (out / f"reader_{res['step']:08d}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
