"""Locator: the third model, stage 2. Design: /docs/model-contract.md, "The locator: design note".

* Input: the locator input, (B,3,input_px,input_px) in [0,1]: the central
  square of the camera stream shrunk to input_px (384) by the app.
* Output: four corners, (B,4,2), in locator-input pixels (x right, y down,
  pixel centres at +0.5), of one tile-sized square of the surface centred on
  the point seen at the middle of the input. Order top-left, top-right,
  bottom-right, bottom-left up to a turn by a multiple of 90 degrees: the
  locator does not know which side is the top, and the app tries all four
  (roll the list by 0 to 3).
* ``geometry.flatten_from_corners`` takes the corners, scaled back to stream
  pixels, to the reader's TILE_PX x TILE_PX input.

The network: six stride-2 stages with one residual block each, a 1x1
convolution, the map flattened (not pooled, so the output keeps where things
are), two fully connected layers. The 8 outputs are offsets from a reference
square centred on the input middle, in units of its side; forward() adds the
reference back. The last layer starts at zero, so an untrained locator returns
the reference square.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Stage(nn.Module):
    """3x3 stride-2 convolution, then one residual block of two 3x3 convolutions."""

    def __init__(self, i: int, o: int):
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(o, o, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(o)
        self.conv2 = nn.Conv2d(o, o, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x + y)


class Locator(nn.Module):
    def __init__(
        self,
        input_px: int = 384,
        widths: tuple[int, ...] | list[int] = (32, 64, 96, 128, 160, 192),
        head_channels: int = 64,
        hidden: int = 128,
        reference_px: float = 82.0,
    ):
        super().__init__()
        self.input_px = int(input_px)
        stages, i = [], 3
        for o in widths:
            stages.append(_Stage(i, o))
            i = o
        self.stages = nn.Sequential(*stages)
        side = self.input_px // (2 ** len(widths))
        if side * 2 ** len(widths) != self.input_px:
            raise ValueError(f"input_px {input_px} is not divisible by 2^{len(widths)}")
        self.head = nn.Sequential(nn.Conv2d(i, head_channels, 1), nn.ReLU(inplace=True))
        self.fc1 = nn.Linear(head_channels * side * side, hidden)
        self.fc2 = nn.Linear(hidden, 8)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        m, r = self.input_px / 2.0, float(reference_px) / 2.0
        reference = torch.tensor([[m - r, m - r], [m + r, m - r], [m + r, m + r], [m - r, m + r]])
        self.register_buffer("reference", reference, persistent=True)
        self.reference_px = float(reference_px)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stages(image - 0.5)
        x = self.head(x).flatten(1)
        offsets = self.fc2(F.relu(self.fc1(x))).view(-1, 4, 2)
        return self.reference.unsqueeze(0) + offsets * self.reference_px
