"""Stage 1 training: embedder and reader.

    python train.py --config configs/default.yaml data.train_dir=/path/to/images

Resumes automatically from the newest checkpoint in ``checkpoint.dir``.
Pass --fresh to start over; existing checkpoints are moved aside, not deleted.

Every ``log.report_every`` steps one line goes to the console and to
``<log.dir>/progress.csv``: step, bit accuracy, PSNR and, with
``--price-per-hour``, the cost so far.

    python train.py --config configs/runpod_direct.yaml --price-per-hour 0.69

Ctrl-C or SIGTERM saves a checkpoint and exits. Exit codes: 0 finished,
3 stopped by the learning check (``early_stop``), 130 interrupted.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from collections import deque

import torch
import torch.nn.functional as F

from checkpoint import load_latest_checkpoint, save_checkpoint, set_aside_checkpoints
from config import load_config
from constants import MESSAGE_BITS, TILE_PX
from data import infinite, make_loader
from geometry import center_crop, flatten_random_patch
from models import build_models, count_parameters
from runlog import ProgressLog, RunClock
from simulate import simulate, wrap_surface
from viz import residual_image, sample_sheet

READER_INPUTS = ("flattened", "direct")
# BCH(63,39) in /docs/message-format.md corrects up to 4 errors in bits 0-62.
BCH_CODEWORD_BITS = 63
BCH_CORRECTABLE = 4


def pick_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-10))


def bit_metrics(logits: torch.Tensor, bits: torch.Tensor) -> dict[str, float]:
    """Bit accuracy and the two full-message rates for a batch."""
    wrong = (logits > 0).float() != bits
    codeword_errors = wrong[:, :BCH_CODEWORD_BITS].sum(dim=1)
    return {
        "bit_acc": 1.0 - wrong.float().mean().item(),
        # full-message decode rate without error correction: all 64 bits right
        "message_exact": (wrong.sum(dim=1) == 0).float().mean().item(),
        # full-message decode rate with it: at most 4 errors in the 63-bit codeword
        "message_correctable": (codeword_errors <= BCH_CORRECTABLE).float().mean().item(),
    }


class Trainer:
    """Owns the models, the optimizer and the two schedules.

    Schedules (config section ``schedule``):

    * ``image_loss``: the image-loss weight is 0 until the smoothed bit
      accuracy first passes ``acc_threshold``, then ramps linearly to 1 over
      ``ramp_steps`` steps.
    * ``sim_strength``: the simulation strength is 0 until ``start_step``,
      then ramps linearly to ``max_strength`` over ``ramp_steps`` steps
      (0 steps: full strength from ``start_step``).
    """

    def __init__(self, cfg: dict, device: torch.device):
        if cfg["train"]["stage"] != 1:
            raise NotImplementedError("only stage 1 (embedder + reader) is written")
        if cfg["train"]["reader_input"] not in READER_INPUTS:
            raise ValueError(f"train.reader_input must be one of {READER_INPUTS}")
        self.cfg = cfg
        self.device = device
        self.models = build_models(cfg)
        for module in self.models.named().values():
            module.to(device)
        self.embedder, self.reader = self.models.embedder, self.models.reader
        self.params = list(self.embedder.parameters()) + list(self.reader.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=cfg["train"]["lr"])
        # Built up front so a missing package or weight file fails at start, not mid-run.
        self.perceptual = None
        if cfg["loss"]["lpips_weight"] > 0:
            from losses import PerceptualLoss

            self.perceptual = PerceptualLoss(cfg["loss"]["lpips_net"]).to(device)
        self.step = 0
        self.acc_ema = 0.5
        self.image_ramp_start: int | None = None

    # -- schedules ---------------------------------------------------------

    def image_loss_scale(self) -> float:
        if self.image_ramp_start is None:
            return 0.0
        steps = int(self.cfg["schedule"]["image_loss"]["ramp_steps"])
        if steps <= 0:
            return 1.0
        return min(1.0, (self.step - self.image_ramp_start) / steps)

    def sim_strength(self) -> float:
        if not self.cfg["simulation"]["enabled"]:
            return 0.0
        sched = self.cfg["schedule"]["sim_strength"]
        since_start = self.step - int(sched["start_step"])
        if since_start < 0:
            return 0.0
        steps = int(sched["ramp_steps"])
        progress = 1.0 if steps <= 0 else min(1.0, since_start / steps)
        return float(sched["max_strength"]) * progress

    # -- forward pass shared by training and evaluation --------------------

    def reader_view(self, frame: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
        """What the reader is given. The only place the two reader_input modes differ."""
        if self.cfg["train"]["reader_input"] == "flattened":
            patch, _ = flatten_random_patch(
                frame,
                homography,
                tile_px=TILE_PX,
                corner_jitter_px=float(self.cfg["flatten"]["corner_jitter_px"]),
            )
            return patch
        return center_crop(frame, TILE_PX)

    def forward(self, tiles: torch.Tensor, bits: torch.Tensor) -> dict[str, torch.Tensor]:
        residual = self.embedder(tiles, bits)
        marked = (tiles + residual).clamp(0.0, 1.0)
        surface = wrap_surface(marked, int(self.cfg["surface"]["repeat"]))
        direct = self.cfg["train"]["reader_input"] == "direct"
        sim = simulate(surface, self.cfg, self.sim_strength(), direct=direct)
        view = self.reader_view(sim.frame, sim.homography)
        logits = self.reader(view)
        return {"residual": residual, "marked": marked, "frame": sim.frame, "view": view, "logits": logits}

    def random_bits(self, batch: int) -> torch.Tensor:
        return torch.randint(0, 2, (batch, MESSAGE_BITS), device=self.device).float()

    # -- one optimisation step ---------------------------------------------

    def train_step(self, tiles: torch.Tensor, keep_outputs: bool = False) -> dict:
        weights = self.cfg["loss"]
        self.embedder.train()
        self.reader.train()
        tiles = tiles.to(self.device, non_blocking=True)
        bits = self.random_bits(tiles.shape[0])

        out = self.forward(tiles, bits)
        bit_loss = F.binary_cross_entropy_with_logits(out["logits"], bits)
        # L2 on the residual before the [0,1] clamp, so pixels pushed past
        # the clamp are still pulled back.
        l2_loss = out["residual"].pow(2).mean()
        image_scale = self.image_loss_scale()
        loss = weights["bit_weight"] * bit_loss
        lpips_loss = None
        if image_scale > 0:
            loss = loss + image_scale * weights["l2_weight"] * l2_loss
            if self.perceptual is not None:
                lpips_loss = self.perceptual(out["marked"], tiles)
                loss = loss + image_scale * weights["lpips_weight"] * lpips_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip = self.cfg["train"]["grad_clip"]
        if clip:
            torch.nn.utils.clip_grad_norm_(self.params, clip)
        strength = self.sim_strength()
        self.optimizer.step()
        self.step += 1

        with torch.no_grad():
            metrics: dict = bit_metrics(out["logits"], bits)
            sched = self.cfg["schedule"]["image_loss"]
            self.acc_ema = sched["acc_ema"] * self.acc_ema + (1.0 - sched["acc_ema"]) * metrics["bit_acc"]
            if self.image_ramp_start is None and self.acc_ema > sched["acc_threshold"]:
                self.image_ramp_start = self.step
            metrics.update(
                loss=loss.item(),
                bit_loss=bit_loss.item(),
                l2_loss=l2_loss.item(),
                bit_acc_ema=self.acc_ema,
                psnr=psnr(out["marked"], tiles),
                image_scale=image_scale,
                sim_strength=strength,
            )
            if lpips_loss is not None:
                metrics["lpips_loss"] = lpips_loss.item()
            if keep_outputs:
                metrics["outputs"] = {
                    "original": tiles.detach(),
                    "marked": out["marked"].detach(),
                    "view": out["view"].detach(),
                }
            return metrics

    @torch.no_grad()
    def evaluate(self, tiles: torch.Tensor, keep_outputs: bool = False) -> dict:
        """Metrics in eval mode on fresh random bits and a fresh view."""
        self.embedder.eval()
        self.reader.eval()
        tiles = tiles.to(self.device)
        bits = self.random_bits(tiles.shape[0])
        out = self.forward(tiles, bits)
        metrics: dict = bit_metrics(out["logits"], bits)
        metrics["psnr"] = psnr(out["marked"], tiles)
        if keep_outputs:
            metrics["outputs"] = {"original": tiles, "marked": out["marked"], "view": out["view"]}
        return metrics


# Scalars written to TensorBoard, as tag -> key in the metrics of train_step.
SCALARS = {
    "read/bit_accuracy": "bit_acc",
    "read/message_decode_rate_exact": "message_exact",
    "read/message_decode_rate_bch": "message_correctable",
    "image/psnr_db": "psnr",
    "loss/total": "loss",
    "loss/bit": "bit_loss",
    "loss/l2": "l2_loss",
    "loss/lpips": "lpips_loss",
    "schedule/image_loss_scale": "image_scale",
    "schedule/sim_strength": "sim_strength",
}


def log_scalars(writer, metrics: dict, step: int) -> None:
    for tag, key in SCALARS.items():
        if key in metrics:
            writer.add_scalar(tag, metrics[key], step)


def log_images(writer, outputs: dict, step: int, count: int) -> None:
    original, marked, view = (outputs[k][:count].float().cpu() for k in ("original", "marked", "view"))
    writer.add_images("samples/1_original", original, step)
    writer.add_images("samples/2_marked", marked.clamp(0, 1), step)
    writer.add_images("samples/3_residual_x10", residual_image(marked - original), step)
    writer.add_images("samples/4_reader_input", view.clamp(0, 1), step)
    writer.add_image("samples/sheet_original_marked_residual_x10", sample_sheet(original, marked), step)


class StopRequest:
    """Turns Ctrl-C and SIGTERM into a stop at the next step boundary, with a checkpoint.

    A second signal stops at once, without one.
    """

    def __init__(self) -> None:
        self.signal_name: str | None = None
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame) -> None:
        if self.signal_name is not None:
            raise KeyboardInterrupt
        self.signal_name = signal.Signals(signum).name
        print(f"\n{self.signal_name} received: saving a checkpoint after this step (send it again to stop now)", flush=True)


DIRECT_NOTICE = (
    "NOTE: reader_input=direct (no locator, no flatten) has not yet been shown to learn. "
    "This run is an experiment, not a known-good configuration."
)

EXIT_NOT_LEARNING = 3
EXIT_INTERRUPTED = 130


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fresh", action="store_true", help="do not resume; existing checkpoints are moved aside")
    parser.add_argument(
        "--price-per-hour",
        type=float,
        default=None,
        help="what the machine costs per hour, in dollars; adds the cost so far to every report line",
    )
    parser.add_argument("overrides", nargs="*", help="dotted.key=value config overrides")
    args = parser.parse_args()
    if args.price_per_hour is not None and args.price_per_hour < 0:
        parser.error("--price-per-hour cannot be negative")

    cfg = load_config(args.config, args.overrides)
    ckpt_cfg, log_cfg = cfg["checkpoint"], cfg["log"]
    early = cfg.get("early_stop") or {}
    check_enabled = bool(early.get("enabled", False))
    check_after_s = float(early.get("after_minutes", 30)) * 60.0
    check_min_acc = float(early.get("min_bit_acc", 0.55))
    report_every = int(log_cfg.get("report_every", 1000))

    clock = RunClock(log_cfg["dir"], args.price_per_hour, fresh=args.fresh)
    if check_enabled and clock.learning_check and clock.learning_check["result"] == "failed":
        c = clock.learning_check
        print(
            f"This run was stopped by the learning check at step {c['step']}: bit accuracy {c['bit_acc']:.3f} "
            f"after {c['elapsed_minutes']:.0f} minutes, below {c['min_bit_acc']}.\n"
            "It will not resume while the check is on. To continue anyway add early_stop.enabled=false; "
            "to start over add --fresh."
        )
        return EXIT_NOT_LEARNING

    torch.manual_seed(cfg["train"]["seed"])
    device = pick_device(cfg["train"]["device"])
    trainer = Trainer(cfg, device)
    print(
        f"device={device}  embedder={count_parameters(trainer.embedder) / 1e6:.2f}M params  "
        f"reader={count_parameters(trainer.reader) / 1e6:.2f}M params  "
        f"reader_input={cfg['train']['reader_input']}"
    )
    print(trainer.models.band.describe())
    if cfg["train"]["reader_input"] == "direct":
        print(DIRECT_NOTICE)

    newest = None
    if args.fresh:
        moved = set_aside_checkpoints(ckpt_cfg["dir"], time.strftime("%Y%m%d-%H%M%S"))
        print("starting fresh" + (f"; earlier checkpoints moved to {moved}" if moved else ""))
    else:
        newest = load_latest_checkpoint(trainer, ckpt_cfg["dir"])
        if newest is not None:
            # Different random draws after a resume than in the steps already done.
            torch.manual_seed(cfg["train"]["seed"] + trainer.step)
            print(f"resumed from {newest} at step {trainer.step}")
        else:
            print(f"starting from step 0 (no checkpoint in {ckpt_cfg['dir']})")
    print(f"checkpoints: {ckpt_cfg['dir']}   logs: {log_cfg['dir']}")
    if clock.elapsed_seconds > 0:
        print(f"time on this run before now: {clock.elapsed_seconds / 3600.0:.2f} h, {clock.cost_text()}")
    if args.price_per_hour is not None and clock.unpriced_seconds > 60:
        print(
            f"note: {clock.unpriced_seconds / 3600.0:.2f} h of this run had no --price-per-hour "
            "and is not in the cost"
        )
    check_pending = check_enabled and clock.learning_check is None
    if check_pending:
        print(
            f"learning check: the run stops if mean bit accuracy is below {check_min_acc} "
            f"once it has trained for {check_after_s / 60:.0f} minutes"
        )

    from torch.utils.tensorboard import SummaryWriter

    # After a resume, drop only what was logged past the checkpoint.
    writer = SummaryWriter(log_cfg["dir"], purge_step=trainer.step + 1 if newest is not None else 0)
    progress = ProgressLog(log_cfg["dir"], fresh=args.fresh)
    batches = infinite(make_loader(cfg, device))
    total = int(cfg["train"]["total_steps"])
    t0, start_step = time.time(), trainer.step
    # (bit accuracy, PSNR) of the most recent steps, for the report line and the learning check
    recent: deque[tuple[float, float]] = deque(maxlen=max(1, report_every))
    stop = StopRequest()
    not_learning = False

    def checkpoint_now() -> None:
        path = save_checkpoint(trainer, ckpt_cfg["dir"], int(ckpt_cfg["keep"]))
        clock.save()
        memory = ""
        if device.type == "cuda":
            memory = f"  (peak GPU memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB)"
        print(f"saved {path}{memory}", flush=True)

    try:
        while trainer.step < total and stop.signal_name is None:
            want_images = (trainer.step + 1) % log_cfg["image_every"] == 0
            m = trainer.train_step(next(batches), keep_outputs=want_images)
            step = trainer.step
            clock.tick()
            recent.append((m["bit_acc"], m["psnr"]))
            if step % log_cfg["scalar_every"] == 0:
                log_scalars(writer, m, step)
            if want_images:
                log_images(writer, m["outputs"], step, int(log_cfg["num_images"]))
            if log_cfg["print_every"] and step % log_cfg["print_every"] == 0:
                rate = (step - start_step) / max(1e-9, time.time() - t0)
                print(
                    f"step {step:6d}  bit_acc {m['bit_acc']:.3f}  decoded {m['message_correctable']:.2f}  "
                    f"psnr {m['psnr']:.1f} dB  loss {m['loss']:.4f}  image_scale {m['image_scale']:.2f}  "
                    f"sim {m['sim_strength']:.2f}  {rate:.2f} it/s",
                    flush=True,
                )
            if report_every and step % report_every == 0:
                mean_acc = sum(a for a, _ in recent) / len(recent)
                mean_psnr = sum(p for _, p in recent) / len(recent)
                print(progress.write(step, mean_acc, mean_psnr, len(recent), clock), flush=True)
                if clock.price_per_hour is not None:
                    writer.add_scalar("run/cost_usd", clock.cost_usd, step)
                clock.save()
            if step % ckpt_cfg["every"] == 0 or step == total:
                checkpoint_now()
            # Wait for 100 steps so that a restart just before the mark is not judged on a handful.
            if check_pending and clock.elapsed_seconds >= check_after_s and len(recent) >= min(100, recent.maxlen):
                mean_acc = sum(a for a, _ in recent) / len(recent)
                not_learning = mean_acc < check_min_acc
                clock.learning_check = {
                    "result": "failed" if not_learning else "passed",
                    "step": step,
                    "bit_acc": mean_acc,
                    "min_bit_acc": check_min_acc,
                    "elapsed_minutes": clock.elapsed_seconds / 60.0,
                }
                check_pending = False
                clock.save()
                print(
                    f"learning check at step {step}, {clock.elapsed_seconds / 60:.0f} minutes in: "
                    f"mean bit accuracy over the last {len(recent)} steps is {mean_acc:.3f} "
                    f"(chance is 0.500, required {check_min_acc}) -> "
                    + ("NOT LEARNING, stopping the run" if not_learning else "passed, the run continues"),
                    flush=True,
                )
                if not_learning:
                    break
    except Exception:
        # Ctrl-C also reaches the data-loader workers; if their exit is what
        # failed here, the stop was asked for and the checkpoint below still happens.
        if stop.signal_name is None:
            raise
    finally:
        writer.close()
        clock.save()

    if not_learning:
        checkpoint_now()
        print(
            f"Stopped: bit accuracy is still near 50 percent. {clock.cost_text()}.\n"
            "The machine is still running and still billed; see RUNPOD.md for how to stop it."
        )
        return EXIT_NOT_LEARNING
    if stop.signal_name is not None:
        checkpoint_now()
        print(f"stopped by {stop.signal_name} at step {trainer.step}; the same command resumes from here")
        return EXIT_INTERRUPTED
    print(f"finished {trainer.step} steps in {clock.elapsed_seconds / 3600.0:.2f} h, {clock.cost_text()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
