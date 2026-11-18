"""Training loop for the tileable watermark encoder/decoder."""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from config import load_config
from data import infinite, make_loader
from models import build_models, count_parameters
from simulate import simulate


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


class Trainer:
    """Owns the models, optimizer and the two schedules.

    Schedules:
    * Image-loss weight is 0 until the running bit accuracy first passes
      ``loss.image_ramp.acc_threshold``, then ramps linearly to 1 over
      ``loss.image_ramp.steps`` steps.
    * Distortion strength ramps linearly from 0 to 1 over
      ``distortion.ramp_steps`` steps.
    """

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.encoder, self.decoder = build_models(cfg)
        self.encoder.to(device)
        self.decoder.to(device)
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=cfg["train"]["lr"])
        self.step = 0
        self.acc_ema = 0.5
        self.image_ramp_start: int | None = None

    # -- schedules ---------------------------------------------------------

    def image_loss_scale(self) -> float:
        if self.image_ramp_start is None:
            return 0.0
        steps = max(1, int(self.cfg["loss"]["image_ramp"]["steps"]))
        return min(1.0, (self.step - self.image_ramp_start) / steps)

    def distortion_strength(self) -> float:
        dist = self.cfg["distortion"]
        if not dist.get("enabled", True):
            return 0.0
        return min(1.0, self.step / max(1, int(dist["ramp_steps"])))

    # -- one optimisation step ---------------------------------------------

    def forward(self, images: torch.Tensor, messages: torch.Tensor) -> dict[str, torch.Tensor]:
        residual = self.encoder(images, messages)
        encoded = (images + residual).clamp(0.0, 1.0)
        view = simulate(encoded, self.cfg, self.distortion_strength())
        logits = self.decoder(view)
        return {"residual": residual, "encoded": encoded, "view": view, "logits": logits}

    def train_step(self, images: torch.Tensor) -> dict[str, float]:
        cfg_loss = self.cfg["loss"]
        self.encoder.train()
        self.decoder.train()
        images = images.to(self.device, non_blocking=True)
        bits = self.cfg["model"]["message_bits"]
        messages = torch.randint(0, 2, (images.shape[0], bits), device=self.device).float()

        out = self.forward(images, messages)
        bit_loss = F.binary_cross_entropy_with_logits(out["logits"], messages)
        # L2 on the raw residual, so pixels pushed past the [0,1] clamp still
        # get pulled back.
        l2_loss = out["residual"].pow(2).mean()
        image_scale = self.image_loss_scale()
        loss = cfg_loss["bit_weight"] * bit_loss + image_scale * cfg_loss["l2_weight"] * l2_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip = self.cfg["train"].get("grad_clip")
        if clip:
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()), clip
            )
        self.optimizer.step()
        self.step += 1

        with torch.no_grad():
            bit_acc = ((out["logits"] > 0).float() == messages).float().mean().item()
            decay = cfg_loss["image_ramp"]["acc_ema"]
            self.acc_ema = decay * self.acc_ema + (1.0 - decay) * bit_acc
            if self.image_ramp_start is None and self.acc_ema > cfg_loss["image_ramp"]["acc_threshold"]:
                self.image_ramp_start = self.step
            return {
                "loss": loss.item(),
                "bit_loss": bit_loss.item(),
                "l2_loss": l2_loss.item(),
                "bit_acc": bit_acc,
                "bit_acc_ema": self.acc_ema,
                "psnr": psnr(out["encoded"], images),
                "image_scale": image_scale,
                "distortion_strength": self.distortion_strength(),
            }

    @torch.no_grad()
    def evaluate(self, images: torch.Tensor) -> float:
        """Bit accuracy in eval mode on fresh random messages and crop offsets."""
        self.encoder.eval()
        self.decoder.eval()
        images = images.to(self.device)
        bits = self.cfg["model"]["message_bits"]
        messages = torch.randint(0, 2, (images.shape[0], bits), device=self.device).float()
        out = self.forward(images, messages)
        return ((out["logits"] > 0).float() == messages).float().mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("overrides", nargs="*", help="dotted.key=value config overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    torch.manual_seed(cfg["train"]["seed"])
    device = pick_device(cfg["train"]["device"])
    trainer = Trainer(cfg, device)
    print(
        f"device={device}  encoder={count_parameters(trainer.encoder) / 1e6:.2f}M params  "
        f"decoder={count_parameters(trainer.decoder) / 1e6:.2f}M params"
    )

    batches = infinite(make_loader(cfg, device))
    t0 = time.time()
    while trainer.step < cfg["train"]["total_steps"]:
        m = trainer.train_step(next(batches))
        if trainer.step % cfg["log"]["print_every"] == 0:
            print(
                f"step {trainer.step:6d}  bit_acc {m['bit_acc']:.3f}  psnr {m['psnr']:.1f} dB  "
                f"loss {m['loss']:.4f}  image_scale {m['image_scale']:.2f}  "
                f"distortion {m['distortion_strength']:.2f}  {time.time() - t0:.0f}s"
            )


if __name__ == "__main__":
    main()
