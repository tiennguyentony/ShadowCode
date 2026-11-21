"""Print-and-camera simulation between the embedder and the reader.

``simulate`` takes the wrapped surface (the marked tile repeated) and returns
the camera frame together with the exact homography it applied, mapping
surface pixels to frame pixels.

Only the pass-through exists so far: with the simulation disabled, or at
strength 0, the frame is the surface and the homography is the identity.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class SimResult(NamedTuple):
    frame: torch.Tensor        # (B,3,H,W) in [0,1]: what the camera sees
    homography: torch.Tensor   # (B,3,3): surface pixels -> frame pixels, exact


def wrap_surface(marked: torch.Tensor, repeat: int = 3) -> torch.Tensor:
    """Repeat the marked tile repeat x repeat times: (B,C,H,W) -> (B,C,repeat*H,repeat*W)."""
    return marked.repeat(1, 1, repeat, repeat)


def simulate(surface: torch.Tensor, cfg: dict, strength: float, direct: bool = False) -> SimResult:
    """Wrapped surface -> (camera frame, surface-to-frame homography).

    ``strength`` is the single knob, 0 to 1. ``direct`` asks for the extra
    random rotation and scale used when the reader gets the frame unflattened.
    """
    identity = torch.eye(3, device=surface.device, dtype=surface.dtype).repeat(surface.shape[0], 1, 1)
    if not cfg["simulation"]["enabled"] or strength <= 0.0:
        return SimResult(surface, identity)
    raise NotImplementedError("the print-and-camera distortions are not written yet")
