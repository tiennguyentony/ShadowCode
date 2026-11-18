"""Encoder and decoder for the tileable neural watermark.

Adapted from StegaStamp (Tancik et al., CVPR 2020). What differs, and why:

* Every encoder conv uses circular padding, so the residual is a periodic
  function of position and tiles without a seam.
* The decoder has no spatial transformer and no flatten. Everything it reads
  is globally average-pooled, so the output does not depend on where the
  tile boundaries fall in the view.
* A pooled decoder cannot see where a pattern is, only how much of each
  texture is present. Two things follow, both needed for training to start:

  - The message map is laid out at pixel resolution and tiled, and starts
    with each bit switching one fine grating on or off. A StegaStamp-style
    map (coarse, position-coded, sign-keyed) looks identical to a pooled
    decoder whatever the message is.
  - The decoder has a second-order branch: a linear filter bank on the
    high-passed input, with the products of all filter pairs pooled. That is
    a local power spectrum, in which an on/off texture code is linear. Plain
    pooled ReLU features did not learn the same code at all in 400 steps.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircConv(nn.Sequential):
    """3x3 conv with circular padding, GroupNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, padding_mode="circular"),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True),
        )


class Encoder(nn.Module):
    """U-Net that turns (image, message) into a seamless-tiling residual.

    Layer layout follows the StegaStamp encoder. The message goes through a
    dense layer, is reshaped to ``message_channels x grid x grid``, upsampled
    by ``message_upsample`` and tiled to the image size, then concatenated
    with the image. The residual is bounded to +-max_residual with a tanh so
    it cannot saturate the image while the image loss is still switched off.

    The image side must be a multiple of 16 so circular padding stays
    consistent at every scale.
    """

    def __init__(
        self,
        message_bits: int = 64,
        image_size: int = 256,
        base_channels: int = 32,
        message_grid: int = 16,
        message_upsample: int = 1,
        message_channels: int = 16,
        message_init: str = "gratings",
        max_residual: float = 0.3,
    ):
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError("image_size must be a multiple of 16")
        if image_size % (message_grid * message_upsample) != 0:
            raise ValueError("image_size must be a multiple of message_grid * message_upsample")
        self.message_bits = message_bits
        self.image_size = image_size
        self.message_grid = message_grid
        self.message_upsample = message_upsample
        self.message_channels = message_channels
        self.max_residual = max_residual

        self.message_fc = nn.Linear(message_bits, message_channels * message_grid * message_grid)
        if message_init == "gratings":
            self._init_message_gratings()
        elif message_init != "dense":
            raise ValueError(f"unknown message_init {message_init!r}")

        c = base_channels
        in_ch = 3 + message_channels
        self.conv1 = CircConv(in_ch, c)
        self.conv2 = CircConv(c, c, stride=2)
        self.conv3 = CircConv(c, 2 * c, stride=2)
        self.conv4 = CircConv(2 * c, 4 * c, stride=2)
        self.conv5 = CircConv(4 * c, 8 * c, stride=2)
        self.up6 = CircConv(8 * c, 4 * c)
        self.conv6 = CircConv(8 * c, 4 * c)
        self.up7 = CircConv(4 * c, 2 * c)
        self.conv7 = CircConv(4 * c, 2 * c)
        self.up8 = CircConv(2 * c, c)
        self.conv8 = CircConv(2 * c, c)
        self.up9 = CircConv(c, c)
        # The last stage also sees the raw input, as in StegaStamp.
        self.conv9 = CircConv(2 * c + in_ch, c)
        self.out = nn.Conv2d(c, 3, 3, padding=1, padding_mode="circular")

    def _init_message_gratings(self) -> None:
        """Start with each bit switching one grating of the message grid on or off.

        Bit k owns one spatial frequency of the grid and a random direction
        in channel space. The dense layer sees the message as +-1, and the
        bias is set to the sum of the weights, so a 0 bit removes its grating
        and a 1 bit adds it. It is still an ordinary dense layer and is free
        to learn any other code from here.
        """
        c, g, bits = self.message_channels, self.message_grid, self.message_bits
        h = g // 2
        # One frequency from each +-pair, skipping DC, the lowest frequencies
        # and the bins that are their own mirror image.
        freqs = [
            (u, v)
            for u in range(0, h + 1)
            for v in range(-h + 1, h + 1)
            if (u > 0 or v > 0)
            and not (u == 0 and v == h)
            and not (u == h and v < 0)
            and math.hypot(u, v) >= 1.5
        ]
        if len(freqs) < bits:
            raise ValueError(f"message_grid {g} has only {len(freqs)} usable frequencies for {bits} bits")
        pick = torch.randperm(len(freqs))[:bits].tolist()
        yy, xx = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
        weight = torch.empty(c, g, g, bits)
        for k, j in enumerate(pick):
            u, v = freqs[j]
            phase = torch.rand(1) * 2 * math.pi
            direction = torch.randn(c)
            direction = direction / direction.norm() * (c / (bits / 2)) ** 0.5
            weight[..., k] = direction[:, None, None] * torch.cos(2 * math.pi * (u * xx + v * yy) / g + phase)
        with torch.no_grad():
            self.message_fc.weight.copy_(weight.reshape(c * g * g, bits) * 0.5)
            self.message_fc.bias.copy_(weight.sum(-1).reshape(-1) * 0.5)

    def message_map(self, message: torch.Tensor, height: int, width: int) -> torch.Tensor:
        m = self.message_fc(message * 2.0 - 1.0)
        m = m.view(-1, self.message_channels, self.message_grid, self.message_grid)
        if self.message_upsample > 1:
            m = F.interpolate(m, scale_factor=self.message_upsample, mode="nearest")
        return m.repeat(1, 1, height // m.shape[-2], width // m.shape[-1])

    def forward(self, image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """image: (B,3,H,W) in [0,1]. message: (B,bits) of 0/1. Returns the residual."""
        h, w = image.shape[-2:]
        x0 = torch.cat([image - 0.5, self.message_map(message, h, w)], dim=1)

        c1 = self.conv1(x0)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        x = self.conv5(c4)
        x = self.conv6(torch.cat([self.up6(_up(x)), c4], dim=1))
        x = self.conv7(torch.cat([self.up7(_up(x)), c3], dim=1))
        x = self.conv8(torch.cat([self.up8(_up(x)), c2], dim=1))
        x = self.conv9(torch.cat([self.up9(_up(x)), c1, x0], dim=1))
        return self.max_residual * torch.tanh(self.out(x))


def _up(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, scale_factor=2, mode="nearest")


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x + y)


class Decoder(nn.Module):
    """Small CNN whose features are all globally average-pooled, then 64 logits.

    Fully convolutional up to the pooling, so the input size is not fixed and
    the output does not depend on where the tile boundaries fall.

    Two sets of pooled features feed the head:

    * the mean of every stage of a four-stage stride-2 CNN;
    * the second-order branch: ``stem_filters`` linear filters on the input
      with its local mean removed, and the mean over the image of the product
      of every pair of filter responses.

    The pooled vector is batch-normalised before the head. Without that the
    message is a small ripple on large, image-dependent offsets and the head
    learns slowly. Normalisation inside the CNN is BatchNorm as well:
    GroupNorm divides out the per-image statistics that carry the message.

    ``head_dim`` 0 gives a single linear layer from pooled features to logits.
    """

    def __init__(
        self,
        message_bits: int = 64,
        widths=(32, 64, 128, 192),
        stem_filters: int = 48,
        stem_kernel: int = 8,
        stem_stride: int = 2,
        head_dim: int = 0,
    ):
        super().__init__()
        w0, w1, w2, w3 = widths

        def down(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        self.stages = nn.ModuleList(
            [
                down(3, w0),
                nn.Sequential(down(w0, w1), _ResBlock(w1)),
                nn.Sequential(down(w1, w2), _ResBlock(w2)),
                nn.Sequential(down(w2, w3), _ResBlock(w3), _ResBlock(w3)),
            ]
        )
        self.stem = nn.Conv2d(3, stem_filters, stem_kernel, stride=stem_stride, bias=False)
        pairs = torch.triu_indices(stem_filters, stem_filters)
        self.register_buffer("pair_index", pairs[0] * stem_filters + pairs[1], persistent=False)

        pooled = sum(widths) + pairs.shape[1]
        self.pool_norm = nn.BatchNorm1d(pooled)
        if head_dim:
            self.head = nn.Sequential(
                nn.Linear(pooled, head_dim),
                nn.ReLU(inplace=True),
                nn.Linear(head_dim, message_bits),
            )
        else:
            self.head = nn.Linear(pooled, message_bits)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image: (B,3,H,W) in [0,1]. Returns (B,bits) logits."""
        x = image * 2.0 - 1.0
        feats = []

        # Second-order branch. Removing the local mean first matters: image
        # content is mostly low-frequency and otherwise swamps the products.
        detail = x - F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
        u = self.stem(detail).flatten(2)                         # (B,F,N)
        products = torch.bmm(u, u.transpose(1, 2)) / u.shape[-1]  # (B,F,F), mean over positions
        feats.append(products.flatten(1)[:, self.pair_index])

        for stage in self.stages:
            x = stage(x)
            feats.append(x.mean(dim=(2, 3)))
        return self.head(self.pool_norm(torch.cat(feats, dim=1)))


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_models(cfg: dict) -> tuple[Encoder, Decoder]:
    m = cfg["model"]
    enc_cfg, dec_cfg = m["encoder"], m["decoder"]
    encoder = Encoder(
        message_bits=m["message_bits"],
        image_size=cfg["data"]["image_size"],
        base_channels=enc_cfg["base_channels"],
        message_grid=enc_cfg["message_grid"],
        message_upsample=enc_cfg["message_upsample"],
        message_channels=enc_cfg["message_channels"],
        message_init=enc_cfg["message_init"],
        max_residual=enc_cfg["max_residual"],
    )
    decoder = Decoder(
        message_bits=m["message_bits"],
        widths=tuple(dec_cfg["widths"]),
        stem_filters=dec_cfg["stem_filters"],
        stem_kernel=dec_cfg["stem_kernel"],
        stem_stride=dec_cfg["stem_stride"],
        head_dim=dec_cfg["head_dim"],
    )
    return encoder, decoder
