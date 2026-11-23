"""Stage 2: train the locator. Design: /docs/model-contract.md, "The locator: design note".

Started through train.py when ``train.stage`` is 2:

    python train.py --config configs/locator.yaml --keep-awake --max-minutes 20

The embedder and the reader are frozen. Each step takes ``train.batch_size``
generated label tiles (training seeds only), marks ``locator_train.marked_share``
of them with the frozen embedder and random messages, and draws
``locator_train.views_per_tile`` views of each with ``simulate_view``,
directly at the locator input size. The loss is the turn-free corner loss
(``corner_loss``).

Held-out evaluation (``HeldOut``): labelgen.ACCEPTANCE_SEEDS, one fixed frame
per seed (same frames at every evaluation), rendered at camera resolution
(the 1080 x 1080 central square, a tile 154 to 340 pixels) and shrunk to the
locator input as the app does. Every ``locator_train.eval_every`` steps one
line goes to the console and a row to ``<log.dir>/eval.csv``.

Corner errors are in reader pixels: pixels at TILE_PX (384) per tile, the
unit of ``flatten.corner_jitter_px``.

Checkpoints in ``checkpoint.dir`` hold the locator, the optimizer, the step,
the evaluation history and the training time; the same command resumes.
``locator_train.patience_evals`` stops the run when the held-out mean has
not improved for that many evaluations (exit code 7).
"""

from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from camera import camera_range
from checkpoint import latest_checkpoint, list_checkpoints, set_aside_checkpoints
from constants import MESSAGE_BITS, TILE_PX
from data import infinite, make_loader
from geometry import flatten_from_corners
from models import build_locator, build_models, count_parameters
from simulate import locator_crop_px, simulate_view, tile_px_in_frame

EXIT_NO_IMPROVEMENT = 7
BCH_CODEWORD_BITS = 63
BCH_CORRECTABLE = 4


# -- the loss ---------------------------------------------------------------


def turned(corners: torch.Tensor, r: int) -> torch.Tensor:
    """The same square turned by r quarter turns: place k gets corner (k + r) mod 4."""
    return corners.roll(-r, dims=1)


def corner_distances(pred: torch.Tensor, target: torch.Tensor, tile_px: torch.Tensor) -> torch.Tensor:
    """(B,4): mean corner distance in reader pixels under each turn r = 0..3 of the target."""
    to_reader = (TILE_PX / tile_px).view(-1, 1)
    return torch.stack(
        [(pred - turned(target, r)).norm(dim=-1).mean(dim=-1) for r in range(4)], dim=1
    ) * to_reader


def corner_loss(pred: torch.Tensor, target: torch.Tensor, tile_px: torch.Tensor, beta: float) -> torch.Tensor:
    """Smooth L1 on x and y in reader pixels, mean over the corners, smallest over the four turns, mean over the batch."""
    to_reader = (TILE_PX / tile_px).view(-1, 1, 1)
    per_turn = torch.stack(
        [
            F.smooth_l1_loss(pred * to_reader, turned(target, r) * to_reader, beta=beta, reduction="none").mean(dim=(1, 2))
            for r in range(4)
        ],
        dim=1,
    )
    return per_turn.min(dim=1).values.mean()


# -- the frozen stage-1 models ----------------------------------------------


def load_stage1(path: str | Path, device: torch.device):
    """Embedder and reader of a stage-1 checkpoint, frozen, in eval mode, with the step and the config they came with."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    models = build_models(state["config"])
    models.embedder.load_state_dict(state["models"]["embedder"])
    models.reader.load_state_dict(state["models"]["reader"])
    for module in (models.embedder, models.reader):
        module.to(device).eval()
        for p in module.parameters():
            p.requires_grad_(False)
    return models.embedder, models.reader, int(state["step"]), state["config"]


@torch.no_grad()
def mark(embedder, tiles: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    return (tiles + embedder(tiles, bits)).clamp(0.0, 1.0)


# -- views ------------------------------------------------------------------


def training_views(tiles: torch.Tensor, embedder, cfg: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One training batch: (locator inputs, target corners, tile sizes), all in locator-input pixels."""
    lt = cfg["locator_train"]
    b = tiles.shape[0]
    marked_n = int(round(b * float(lt["marked_share"])))
    surface = tiles.clone()
    if marked_n > 0:
        bits = torch.randint(0, 2, (marked_n, MESSAGE_BITS), device=tiles.device).float()
        surface[:marked_n] = mark(embedder, tiles[:marked_n], bits)
    views = int(lt["views_per_tile"])
    surface = surface.repeat_interleave(views, dim=0)
    # Background: a different label from the same batch.
    background = tiles.roll(1, dims=0).repeat_interleave(views, dim=0)
    side = int(lt["input_px"])
    with torch.no_grad():
        v = simulate_view(surface, cfg, 1.0, (side, side), tile_px_in_frame(cfg), background)
    return v.frame, v.corners, v.tile_px


def camera_views(surface: torch.Tensor, background: torch.Tensor, cfg: dict):
    """Camera-resolution frames (the central square of the stream) and the same shrunk to the locator input.

    Returns (camera frames, target corners in camera pixels, tile sizes in
    camera pixels, locator inputs, scale from camera to locator pixels).
    """
    crop = locator_crop_px(cfg)
    side = int(cfg["locator_train"]["input_px"])
    span = camera_range(cfg)
    if span is None:
        raise ValueError("the held-out evaluation needs camera_px_per_tile")
    v = simulate_view(surface, cfg, 1.0, (crop, crop), span, background)
    small = F.interpolate(v.frame, size=(side, side), mode="bilinear", antialias=True, align_corners=False)
    return v.frame, v.corners, v.tile_px, small.clamp(0.0, 1.0), side / crop


# -- held-out evaluation ----------------------------------------------------


class HeldOut:
    """The fixed held-out set: labelgen.ACCEPTANCE_SEEDS, even positions marked with fixed messages.

    Frames are drawn again at every evaluation from ``eval_seed``, so they
    are the same frames each time without keeping 256 camera frames in
    memory. The tiles are rendered once and cached in ``cache``.
    """

    def __init__(self, cfg: dict, embedder, device: torch.device, cache: Path, chunk: int = 8):
        from data import generated_tiles
        from labelgen import ACCEPTANCE_SEEDS, TRAIN_SEEDS

        lt = cfg["locator_train"]
        count = int(lt["eval_samples"])
        seeds = list(ACCEPTANCE_SEEDS[:count])
        if any(s in TRAIN_SEEDS for s in seeds):
            raise RuntimeError("held-out seeds overlap the training seeds")
        self.seeds = seeds
        if cache.is_file():
            stored = torch.load(cache, map_location="cpu", weights_only=True)
            if stored["seeds"] != seeds:
                raise RuntimeError(f"{cache} holds other seeds; delete it")
            tiles = stored["tiles"]
        else:
            print(f"rendering {len(seeds)} held-out tiles (seeds {seeds[0]} to {seeds[-1]}) once ...", flush=True)
            tiles = generated_tiles(seeds)
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"seeds": seeds, "tiles": tiles}, cache)
        self.tiles = tiles
        generator = torch.Generator().manual_seed(int(lt["eval_seed"]))
        self.bits = torch.randint(0, 2, (count, MESSAGE_BITS), generator=generator).float()
        self.marked = torch.arange(count) % 2 == 0
        self.cfg, self.embedder, self.device, self.chunk = cfg, embedder, device, chunk
        self.seed = int(lt["eval_seed"])

    def batches(self):
        """Yield per chunk: dict with camera frames, true corners, tile sizes, locator inputs, scale, bits, marked."""
        n = len(self.seeds)
        devices = [self.device] if self.device.type == "cuda" else []
        for start in range(0, n, self.chunk):
            idx = torch.arange(start, min(n, start + self.chunk))
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.seed + start)
                tiles = self.tiles[idx].to(self.device)
                bits = self.bits[idx].to(self.device)
                marked = self.marked[idx].to(self.device)
                surface = tiles.clone()
                if bool(marked.any()):
                    surface[marked] = mark(self.embedder, tiles[marked], bits[marked])
                # Background: the next held-out label, so it never comes from training seeds either.
                background = self.tiles[(idx + 1) % n].to(self.device)
                with torch.no_grad():
                    cam, corners, tile_px, small, scale = camera_views(surface, background, self.cfg)
            yield {
                "camera": cam, "corners": corners, "tile_px": tile_px, "input": small, "scale": scale,
                "bits": bits, "marked": marked, "index": idx,
            }

    @torch.no_grad()
    def evaluate(self, locator, reader=None, keep: int = 0) -> dict:
        """Corner error on every held-out sample; with ``reader``, bit accuracy with true and with located corners.

        ``keep``: also return the first ``keep`` locator inputs with their true and predicted corners.
        """
        was_training = locator.training
        locator.eval()
        errors, best_turn, kinds = [], [], []
        acc_true, acc_loc, read_true, read_turn = [], [], [], []
        kept = {"input": [], "true": [], "pred": []}
        for batch in self.batches():
            pred = locator(batch["input"])
            target = batch["corners"] * batch["scale"]
            tile_small = batch["tile_px"] * batch["scale"]
            d = corner_distances(pred, target, tile_small)             # reader pixels, per turn
            err, r = d.min(dim=1)
            errors.append(err.cpu())
            best_turn.append(r.cpu())
            kinds.append(batch["marked"].cpu())
            if keep and sum(len(x) for x in kept["input"]) < keep:
                kept["input"].append(batch["input"].cpu())
                kept["true"].append(target.cpu())
                kept["pred"].append(pred.cpu())
            if reader is not None and bool(batch["marked"].any()):
                m = batch["marked"]
                cam, bits = batch["camera"][m], batch["bits"][m]
                true_patch = flatten_from_corners(cam, batch["corners"][m], TILE_PX)
                logits = reader(true_patch)
                acc_true.append(((logits > 0).float() == bits).float().mean(dim=1).cpu())
                read_true.append(_within_bch(logits, bits).cpu())
                located = pred[m] / batch["scale"]                      # back to camera pixels
                per_turn_acc, per_turn_read = [], []
                for t in range(4):
                    logits = reader(flatten_from_corners(cam, turned(located, t), TILE_PX))
                    per_turn_acc.append(((logits > 0).float() == bits).float().mean(dim=1))
                    per_turn_read.append(_within_bch(logits, bits))
                per_turn_acc = torch.stack(per_turn_acc, dim=1)       # (b,4)
                per_turn_read = torch.stack(per_turn_read, dim=1)
                # The turn that puts the located corners on the true ones: the read the CRC should pick.
                rm = r[m]
                acc_loc.append(per_turn_acc.gather(1, rm.view(-1, 1)).squeeze(1).cpu())
                read_turn.append(per_turn_read.cpu())
        if was_training:
            locator.train()
        err = torch.cat(errors)
        turn = torch.cat(best_turn)
        marked = torch.cat(kinds)
        out = summarise(err)
        out.update(
            marked_mean=err[marked].mean().item() if marked.any() else float("nan"),
            plain_mean=err[~marked].mean().item() if (~marked).any() else float("nan"),
            turn_not_zero=int((turn != 0).sum().item()),
            samples=len(err),
            turns=[int((turn == t).sum().item()) for t in range(4)],
        )
        if reader is not None and acc_true:
            reads = torch.cat(read_turn)                                # (marked,4) within BCH reach per turn
            out.update(
                reader_acc_true=torch.cat(acc_true).mean().item(),
                reader_acc_located=torch.cat(acc_loc).mean().item(),
                read_true=torch.cat(read_true).float().mean().item(),
                read_any_turn=reads.any(dim=1).float().mean().item(),
                read_turn0=reads[:, 0].float().mean().item(),
                read_only_other_turn=int((~reads[:, 0] & reads[:, 1:].any(dim=1)).sum().item()),
                read_more_than_one_turn=int((reads.sum(dim=1) > 1).sum().item()),
                marked_samples=int(reads.shape[0]),
            )
        if keep:
            out["kept"] = {k: torch.cat(v)[:keep] for k, v in kept.items()}
        return out


def _within_bch(logits: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    """True where at most BCH_CORRECTABLE of the 63 codeword bits are wrong: what the code could correct."""
    wrong = ((logits > 0).float() != bits)[:, :BCH_CODEWORD_BITS].sum(dim=1)
    return wrong <= BCH_CORRECTABLE


def summarise(err: torch.Tensor) -> dict:
    """Mean, median, 95th percentile and the mean of the worst 5 percent of corner errors (reader pixels)."""
    s = err.sort().values
    worst = s[int(math.floor(len(s) * 0.95)):]
    return {
        "mean": s.mean().item(),
        "median": s.median().item(),
        "p95": torch.quantile(s, 0.95).item(),
        "worst5_mean": worst.mean().item(),
        "max": s[-1].item(),
    }


# -- sample sheet -----------------------------------------------------------


def save_sheet(kept: dict, path: Path) -> Path:
    """Locator inputs with the true square (green) and the located one (red, first corner marked)."""
    from PIL import Image, ImageDraw

    images = []
    for img, true, pred in zip(kept["input"], kept["true"], kept["pred"]):
        pil = Image.fromarray((img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8"))
        draw = ImageDraw.Draw(pil)
        for quad, colour in ((true, (0, 255, 0)), (pred, (255, 0, 0))):
            pts = [tuple(p) for p in quad.tolist()]
            draw.line(pts + [pts[0]], fill=colour, width=2)
            x, y = pts[0]
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], outline=colour, width=2)
        images.append(pil)
    cols = 4
    w, h = images[0].size
    rows = -(-len(images) // cols)
    sheet = Image.new("RGB", (cols * w + (cols - 1) * 4, rows * h + (rows - 1) * 4), (255, 255, 255))
    for i, im in enumerate(images):
        sheet.paste(im, ((i % cols) * (w + 4), (i // cols) * (h + 4)))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


# -- the run ----------------------------------------------------------------


EVAL_FIELDS = ["utc_time", "local_time", "step", "minutes", "mean", "median", "p95", "worst5_mean", "max", "marked_mean", "plain_mean", "turn_not_zero", "samples"]
PROGRESS_FIELDS = ["utc_time", "step", "minutes", "elapsed_hours", "train_corner_px", "loss", "lr", "steps_per_s"]


def _append(path: Path, fields: list[str], row: dict, fresh: bool) -> None:
    new = fresh or not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w" if new else "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def _trim(path: Path, step: int, fields: list[str]) -> None:
    """After a resume, drop rows past the checkpoint's step and write the file with ``fields``.

    Rows from an earlier version of this file are carried over: ``utc``
    becomes ``utc_time`` and ``elapsed_hours`` is filled in from ``minutes``.
    """
    if not path.is_file():
        return
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    kept = []
    for r in rows:
        if not r.get("step", "").isdigit() or int(r["step"]) > step:
            continue
        if "utc" in r and "utc_time" not in r:
            r["utc_time"] = r.pop("utc")
        if "elapsed_hours" in fields and not r.get("elapsed_hours") and r.get("minutes"):
            r["elapsed_hours"] = round(float(r["minutes"]) / 60.0, 4)
        kept.append(r)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(kept)


def eval_line(step: int, e: dict) -> str:
    return (
        f"held-out step {step}: corner error mean {e['mean']:.2f} px, median {e['median']:.2f}, "
        f"95th percentile {e['p95']:.2f}, worst 5% mean {e['worst5_mean']:.2f}, max {e['max']:.1f} "
        f"(marked {e['marked_mean']:.2f}, plain {e['plain_mean']:.2f}; reader px, {e['samples']} samples; "
        f"turn other than 0: {e['turn_not_zero']})"
    )


def run(cfg: dict, args) -> int:
    from train import StopRequest, keep_awake, pick_device, require_gpu

    lt, log_cfg, ckpt_cfg = cfg["locator_train"], cfg["log"], cfg["checkpoint"]
    log_dir, ckpt_dir = Path(log_cfg["dir"]), Path(ckpt_cfg["dir"])
    torch.manual_seed(int(cfg["train"]["seed"]))
    device = pick_device(cfg["train"]["device"])
    require_gpu(device, cfg["train"].get("expect_gpu"))
    if args.keep_awake:
        keep_awake()

    embedder, _, emb_step, _ = load_stage1(lt["embedder_checkpoint"], device)
    locator = build_locator(cfg).to(device)
    total = int(cfg["train"]["total_steps"])
    optimizer = torch.optim.Adam(locator.parameters(), lr=float(cfg["train"]["lr"]))
    lr0, lr_end = float(cfg["train"]["lr"]), float(lt["lr_end"])

    def lr_at(step: int) -> float:
        return lr_end + 0.5 * (lr0 - lr_end) * (1.0 + math.cos(math.pi * min(step, total) / total))

    span = tile_px_in_frame(cfg)
    print(
        f"stage 2, locator: {count_parameters(locator):,} parameters, input {locator.input_px} px, "
        f"tile {span[0]:.1f} to {span[1]:.1f} px in training frames, reference square {locator.reference_px:.1f} px"
    )
    print(f"frozen embedder: {lt['embedder_checkpoint']} (step {emb_step}); {float(lt['marked_share']):.0%} of tiles marked")
    print(
        f"rotation 0 to {cfg['simulation']['perspective']['rotation_deg']:g} deg, tilt up to "
        f"{cfg['simulation']['perspective']['max_tilt_deg']:g} deg, print edges "
        f"{cfg['simulation'].get('surface_extent_tiles')} tiles from the centre"
    )

    step, elapsed_before, history, best, since_best = 0, 0.0, [], None, 0
    if args.fresh:
        moved = set_aside_checkpoints(ckpt_dir, time.strftime("%Y%m%d-%H%M%S"))
        print("starting fresh" + (f"; earlier checkpoints moved to {moved}" if moved else ""))
    else:
        newest = latest_checkpoint(ckpt_dir)
        if newest is not None:
            state = torch.load(newest, map_location=device, weights_only=True)
            locator.load_state_dict(state["locator"])
            optimizer.load_state_dict(state["optimizer"])
            step, elapsed_before = int(state["step"]), float(state["elapsed_seconds"])
            history, best, since_best = list(state["history"]), state["best"], int(state["since_best"])
            torch.manual_seed(int(cfg["train"]["seed"]) + step)
            print(f"resumed from {newest} at step {step}, {elapsed_before / 60:.1f} minutes of training before now")
        else:
            print(f"starting from step 0 (no checkpoint in {ckpt_dir})")
    fresh_logs = step == 0
    if not fresh_logs:
        _trim(log_dir / "eval.csv", step, EVAL_FIELDS)
        _trim(log_dir / "progress.csv", step, PROGRESS_FIELDS)

    held = HeldOut(cfg, embedder, device, ckpt_dir / "heldout_tiles.pt")
    patience = lt.get("patience_evals")
    min_gain = float(lt.get("min_improvement", 0.01))
    if patience:
        print(
            f"early stop: the run stops when the held-out mean corner error has not improved by {min_gain:.0%} "
            f"on its best for {int(patience)} evaluations in a row ({int(patience) * int(lt['eval_every'])} steps)"
        )
    t0 = time.time()

    def minutes() -> float:
        return (elapsed_before + time.time() - t0) / 60.0

    def evaluate_now(sheet: bool) -> dict:
        nonlocal best, since_best
        e = held.evaluate(locator, keep=8 if sheet else 0)
        if "kept" in e:
            p = save_sheet(e.pop("kept"), log_dir / "samples" / f"step_{step:08d}.png")
            print(f"sample sheet (green true, red located, circle on its first corner): {p}", flush=True)
        row = {
            "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "step": step, "minutes": round(minutes(), 2), **e,
        }
        _append(log_dir / "eval.csv", EVAL_FIELDS, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, fresh=False)
        history.append({"step": step, "mean": e["mean"], "p95": e["p95"], "worst5_mean": e["worst5_mean"]})
        if best is None or e["mean"] < best["mean"] * (1.0 - min_gain):
            best, since_best = {"step": step, "mean": e["mean"]}, 0
        else:
            since_best += 1
        print(eval_line(step, e) + f"  best {best['mean']:.2f} at step {best['step']}, {since_best} evals since", flush=True)
        return e

    def checkpoint_now() -> Path:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"step_{step:08d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(
            {
                "step": step, "locator": locator.state_dict(), "optimizer": optimizer.state_dict(),
                "elapsed_seconds": elapsed_before + time.time() - t0, "history": history, "best": best,
                "since_best": since_best, "embedder_checkpoint": str(lt["embedder_checkpoint"]), "config": cfg,
            },
            tmp,
        )
        os.replace(tmp, path)
        keep = int(ckpt_cfg["keep"])
        if keep > 0:
            for old in list_checkpoints(ckpt_dir)[:-keep]:
                old.unlink()
        memory = f"  (peak GPU memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB)" if device.type == "cuda" else ""
        print(f"saved {path}{memory}", flush=True)
        return path

    if fresh_logs:
        for name in ("eval.csv", "progress.csv"):
            if (log_dir / name).is_file():
                (log_dir / name).unlink()
        evaluate_now(sheet=True)

    batches = infinite(make_loader(cfg, device))
    stop = StopRequest()
    max_seconds = None if args.max_minutes is None else args.max_minutes * 60.0
    if max_seconds is not None:
        print(f"time limit: the run stops after {args.max_minutes:g} minutes of training in this process")
    beta = float(lt["smooth_l1_beta"])
    recent_err, recent_loss, window_t, window_step = [], [], time.time(), step
    reason = None
    locator.train()
    try:
        while step < total and stop.signal_name is None:
            tiles = next(batches).to(device, non_blocking=True)
            inputs, target, tile_px = training_views(tiles, embedder, cfg)
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            pred = locator(inputs)
            loss = corner_loss(pred, target, tile_px, beta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(locator.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
            optimizer.step()
            step += 1
            with torch.no_grad():
                recent_err.append(corner_distances(pred, target, tile_px).min(dim=1).values.mean().item())
            recent_loss.append(loss.item())
            if log_cfg["print_every"] and step % int(log_cfg["print_every"]) == 0:
                n = int(log_cfg["print_every"])
                rate = (step - window_step) / max(1e-9, time.time() - window_t)
                print(
                    f"step {step:6d}  train corner error {sum(recent_err[-n:]) / len(recent_err[-n:]):.2f} px  "
                    f"loss {sum(recent_loss[-n:]) / len(recent_loss[-n:]):.3f}  lr {lr_at(step):.2e}  {rate:.2f} it/s",
                    flush=True,
                )
            if step % int(log_cfg["report_every"]) == 0:
                rate = (step - window_step) / max(1e-9, time.time() - window_t)
                _append(
                    log_dir / "progress.csv", PROGRESS_FIELDS,
                    {
                        "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "step": step,
                        "minutes": round(minutes(), 2), "elapsed_hours": round(minutes() / 60.0, 4),
                        "train_corner_px": round(sum(recent_err) / len(recent_err), 4),
                        "loss": round(sum(recent_loss) / len(recent_loss), 5), "lr": lr_at(step),
                        "steps_per_s": round(rate, 3),
                    },
                    fresh=False,
                )
                recent_err, recent_loss, window_t, window_step = [], [], time.time(), step
            if step % int(lt["eval_every"]) == 0:
                evaluate_now(sheet=bool(log_cfg.get("sheet_every")) and step % int(log_cfg["sheet_every"]) == 0)
                if patience and since_best >= int(patience):
                    reason = "no_improvement"
            if step % int(ckpt_cfg["every"]) == 0 or reason:
                checkpoint_now()
            if reason:
                break
            if max_seconds is not None and time.time() - t0 >= max_seconds:
                reason = "time"
                break
    except Exception:
        if stop.signal_name is None:
            raise

    if reason != "no_improvement" and (not history or history[-1]["step"] != step):
        evaluate_now(sheet=True)
    if not list_checkpoints(ckpt_dir) or latest_checkpoint(ckpt_dir).name != f"step_{step:08d}.pt":
        checkpoint_now()
    rate = step / max(1e-9, minutes() * 60.0)
    if reason == "no_improvement":
        print(
            f"Stopped: the held-out mean corner error has not improved by {min_gain:.0%} for {since_best} evaluations; "
            f"best {best['mean']:.2f} px at step {best['step']}. Stopped at step {step} after {minutes():.1f} minutes."
        )
        return EXIT_NO_IMPROVEMENT
    if stop.signal_name is not None:
        print(f"stopped by {stop.signal_name} at step {step}; the same command resumes from here")
        return 130
    if reason == "time":
        print(f"time limit reached: stopped at step {step} after {minutes():.1f} minutes of training in total; the same command resumes")
        return 0
    print(f"finished {step} steps in {minutes():.1f} minutes")
    return 0
