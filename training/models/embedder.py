"""Embedder: U-Net that writes MESSAGE_BITS bits into one tile as a seamless residual.

Runs offline only, so its size is not a concern.
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


def _up(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, scale_factor=2, mode="nearest")


class Embedder(nn.Module):
    """(tile, bits) -> residual = alpha * tanh(U-Net(tile, message map)).

    The bits go through a dense layer, are reshaped to a small
    ``message_channels x message_grid x message_grid`` map, upsampled (nearest)
    to the tile size and concatenated with the tile.

    Seamless repetition: every conv uses circular padding, the down path is
    stride-2 convs on even sizes, the up path and the message map use nearest
    upsampling. The network therefore treats the tile as a torus and adds no
    edge of its own. ``tile_px`` must be a multiple of 16 and of
    ``message_grid``.

    ``message_init``:

    * ``gratings``: the dense layer starts with each bit switching one grating
      of the message map on or off (see ``_init_message_gratings``). It is
      still an ordinary dense layer and is free to learn any other code.
    * ``dense``: PyTorch's default init.
    """

    def __init__(
        self,
        message_bits: int,
        tile_px: int,
        base_channels: int = 32,
        message_grid: int = 128,
        message_channels: int = 16,
        message_init: str = "gratings",
        grating_step: int = 8,
        alpha: float = 0.3,
    ):
        super().__init__()
        if tile_px % 16 != 0:
            raise ValueError("tile_px must be a multiple of 16")
        if tile_px % message_grid != 0:
            raise ValueError("tile_px must be a multiple of message_grid")
        self.message_bits = message_bits
        self.tile_px = tile_px
        self.message_grid = message_grid
        self.message_channels = message_channels
        self.alpha = alpha

        self.message_fc = nn.Linear(message_bits, message_channels * message_grid * message_grid)
        if message_init == "gratings":
            self._init_message_gratings(grating_step)
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
        # The last stage also sees the raw input (tile and message map).
        self.conv9 = CircConv(2 * c + in_ch, c)
        self.out = nn.Conv2d(c, 3, 3, padding=1, padding_mode="circular")

    def _init_message_gratings(self, step: int) -> None:
        """Start with each bit switching one grating of the message map on or off.

        The reader pools over position, so at the start it can only tell
        messages apart by how much of each texture is present, not by where
        anything is. This init gives it that from step 0.

        Bit k owns one spatial frequency (u, v) = step * (a, b) cycles per
        tile and a random direction in channel space. The lowest-frequency
        candidates are used, so the starting code sits in the band most
        likely to survive loss of resolution. ``step`` spaces the gratings
        apart so a reader with a window of tile_px / step pixels can tell
        neighbours apart. The dense layer sees the message as +-1 and the
        bias is the sum of the weights, so a 0 bit removes its grating and a
        1 bit adds it.
        """
        c, g, bits = self.message_channels, self.message_grid, self.message_bits
        limit = (g // 2 - 1) // step  # stay strictly below the grid's Nyquist frequency
        # One of each +-frequency pair, without DC and the lowest few.
        candidates = [
            (a, b)
            for a in range(0, limit + 1)
            for b in range(-limit, limit + 1)
            if (a > 0 or b > 0) and math.hypot(a, b) >= 1.5
        ]
        if len(candidates) < bits:
            raise ValueError(
                f"message_grid {g} with grating_step {step} has only {len(candidates)} "
                f"usable frequencies for {bits} bits"
            )
        candidates.sort(key=lambda ab: (math.hypot(*ab), ab))
        chosen = [candidates[i] for i in torch.randperm(bits).tolist()]
        yy, xx = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
        weight = torch.empty(c, g, g, bits)
        for k, (a, b) in enumerate(chosen):
            phase = torch.rand(1) * 2 * math.pi
            direction = torch.randn(c)
            direction = direction / direction.norm() * (c / (bits / 2)) ** 0.5
            wave = torch.cos(2 * math.pi * step * (a * xx + b * yy) / g + phase)
            weight[..., k] = direction[:, None, None] * wave
        with torch.no_grad():
            self.message_fc.weight.copy_(weight.reshape(c * g * g, bits) * 0.5)
            self.message_fc.bias.copy_(weight.sum(-1).reshape(-1) * 0.5)

    def message_map(self, bits: torch.Tensor) -> torch.Tensor:
        m = self.message_fc(bits * 2.0 - 1.0)
        m = m.view(-1, self.message_channels, self.message_grid, self.message_grid)
        return F.interpolate(m, scale_factor=self.tile_px // self.message_grid, mode="nearest")

    def forward(self, tile: torch.Tensor, bits: torch.Tensor, alpha: float | None = None) -> torch.Tensor:
        """tile: (B,3,TILE_PX,TILE_PX) in [0,1]. bits: (B,MESSAGE_BITS) of 0/1.

        Returns the residual, already scaled: alpha * tanh(.). The marked
        tile is ``tile + residual``.
        """
        if tile.shape[-2:] != (self.tile_px, self.tile_px):
            raise ValueError(f"expected a {self.tile_px}x{self.tile_px} tile, got {tuple(tile.shape[-2:])}")
        alpha = self.alpha if alpha is None else alpha
        x0 = torch.cat([tile - 0.5, self.message_map(bits)], dim=1)

        c1 = self.conv1(x0)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        x = self.conv5(c4)
        x = self.conv6(torch.cat([self.up6(_up(x)), c4], dim=1))
        x = self.conv7(torch.cat([self.up7(_up(x)), c3], dim=1))
        x = self.conv8(torch.cat([self.up8(_up(x)), c2], dim=1))
        x = self.conv9(torch.cat([self.up9(_up(x)), c1, x0], dim=1))
        return alpha * torch.tanh(self.out(x))
