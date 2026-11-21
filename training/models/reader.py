"""Reader: small CNN that turns one view of the surface into MESSAGE_BITS logits.

Target: under 3M parameters (it runs on the phone).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TextureBranch(nn.Module):
    """Globally pooled products of a strided linear filter bank.

    The input is average-pooled by ``downsample``, filtered by ``filters``
    zero-mean linear filters of ``kernel`` pixels at ``stride``, and the mean
    over position of the product of every pair of filter responses is
    returned. That is a local power spectrum: it says how much of each
    texture is present and ignores where, so a code that switches textures
    on and off is linear in it. The window is ``downsample * kernel`` input
    pixels.
    """

    def __init__(self, filters: int = 48, kernel: int = 24, stride: int = 4, downsample: int = 2):
        super().__init__()
        self.stride = stride
        self.downsample = downsample
        self.weight = nn.Parameter(torch.randn(filters, 3, kernel, kernel) * (3 * kernel * kernel) ** -0.5)
        pairs = torch.triu_indices(filters, filters)
        self.register_buffer("pair_index", pairs[0] * filters + pairs[1], persistent=False)
        self.out_features = pairs.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample > 1:
            x = F.avg_pool2d(x, self.downsample)
        # Zero-mean filters: flat areas and slow gradients give no response.
        weight = self.weight - self.weight.mean(dim=(2, 3), keepdim=True)
        u = F.conv2d(x, weight, stride=self.stride).flatten(2)   # (B,F,N)
        products = torch.bmm(u, u.transpose(1, 2)) / u.shape[-1]  # (B,F,F), mean over position
        # index_select, not products.flatten(1)[:, pair_index]: the same values,
        # but it exports to ONNX as a plain Gather instead of GatherND, which
        # ONNX Runtime Web supports less widely (see /docs/model-contract.md).
        return products.flatten(1).index_select(1, self.pair_index)


class RunningNorm(nn.Module):
    """Standardise features with running statistics, in training as well as in eval.

    BatchNorm would use the current batch's own mean and variance while
    training. On a pooled feature vector that is one value per image, so with
    small batches each feature would be measured against a handful of other
    images and most of the signal would be lost. Here the batch only updates
    the running mean and variance (no gradient flows through them), and the
    running values do the normalising.
    """

    def __init__(self, features: int, momentum: float = 0.05, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.register_buffer("running_mean", torch.zeros(features))
        self.register_buffer("running_var", torch.ones(features))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                mean = x.mean(dim=0)
                var = (x - mean).pow(2).mean(dim=0) if x.shape[0] > 1 else self.running_var
                # Plain average over the first updates, then an exponential one.
                rate = max(self.momentum, 1.0 / (self.updates.item() + 1))
                self.running_var.lerp_(var + (mean - self.running_mean).pow(2) * (1.0 - rate), rate)
                self.running_mean.lerp_(mean, rate)
                self.updates += 1
        x = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
        return x * self.weight + self.bias


class Reader(nn.Module):
    """Stride-2 conv stages, global average pooling, MESSAGE_BITS logits.

    Fully convolutional up to the pooling, so the same weights take the
    flattened TILE_PX patch (reader_input: flattened) or a centre crop of the
    camera frame (reader_input: direct).

    The pooled output of every stage is concatenated, normalised and fed to a
    linear head. ``pool_norm`` is ``running`` (RunningNorm) or ``batch``
    (plain BatchNorm1d). ``widths`` gives one stride-2 stage per entry; every
    stage except the first and the last is followed by a residual
    block. With ``texture`` set, a TextureBranch adds its pooled features to
    the same vector.
    """

    def __init__(
        self,
        message_bits: int,
        widths: tuple[int, ...] = (32, 64, 128, 192, 256),
        texture: dict | None = None,
        pool_norm: str = "running",
    ):
        super().__init__()

        def down(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        stages = []
        prev = 3
        for n, width in enumerate(widths):
            layers = [down(prev, width)]
            if 0 < n < len(widths) - 1:
                layers.append(_ResBlock(width))
            stages.append(nn.Sequential(*layers))
            prev = width
        self.stages = nn.ModuleList(stages)

        pooled = sum(widths)
        self.texture = TextureBranch(**texture) if texture else None
        if self.texture is not None:
            pooled += self.texture.out_features
        if pool_norm == "running":
            self.pool_norm = RunningNorm(pooled)
        elif pool_norm == "batch":
            self.pool_norm = nn.BatchNorm1d(pooled)
        else:
            raise ValueError(f"unknown pool_norm {pool_norm!r}")
        self.head = nn.Linear(pooled, message_bits)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image: (B,3,H,W) in [0,1]. Returns (B,MESSAGE_BITS) logits."""
        x = image * 2.0 - 1.0
        feats = [] if self.texture is None else [self.texture(x)]
        for stage in self.stages:
            x = stage(x)
            feats.append(x.mean(dim=(2, 3)))
        return self.head(self.pool_norm(torch.cat(feats, dim=1)))
