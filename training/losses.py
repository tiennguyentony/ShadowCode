"""Image losses between the marked tile and the original."""

from __future__ import annotations

import torch
import torch.nn as nn


class PerceptualLoss(nn.Module):
    """LPIPS distance (Zhang et al., 2018), frozen. Inputs are (B,3,H,W) in [0,1].

    Uses the ``lpips`` package. Its backbone weights are pretrained and are
    downloaded by torchvision on first use. See THIRD_PARTY.md.
    """

    def __init__(self, net: str = "alex"):
        super().__init__()
        try:
            import lpips
        except ImportError as error:  # pragma: no cover
            raise ImportError("loss.lpips_weight > 0 needs the lpips package: pip install lpips") from error
        self.model = lpips.LPIPS(net=net, verbose=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True) -> "PerceptualLoss":
        return super().train(False)  # always in eval mode

    def forward(self, marked: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        return self.model(marked * 2.0 - 1.0, original * 2.0 - 1.0).mean()
