"""Locator: the third model. Not built yet; this file is its slot.

Stage 1 trains the embedder and the reader and flattens the reader's patch
with the TRUE homography returned by the simulation. The locator will replace
that at run time.

Contract the rest of the code already assumes:

* Input: a camera frame, (B,3,H,W) in [0,1].
* Output: the four corners, in frame pixels (x, y), of one tile-sized patch
  of the surface, shape (B,4,2), ordered top-left, top-right, bottom-right,
  bottom-left. Any offset within the tile pattern is acceptable; the reader
  is trained on random offsets.
* ``geometry.flatten_from_corners`` turns those corners into the reader's
  TILE_PX x TILE_PX input. ``flatten.corner_jitter_px`` in the config is the
  corner error the reader is trained to tolerate.

Where it plugs in: ``build_models`` returns it in ``Models.locator`` (None
while ``model.locator`` is null in the config), checkpoints carry a
``locator`` entry, and ``train.stage`` selects the training step.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Locator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        raise NotImplementedError("the locator is not built yet (stage 1 trains embedder and reader only)")

    def forward(self, frame: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError
