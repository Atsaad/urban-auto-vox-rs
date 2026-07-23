"""Exponential moving average of model parameters (Polyak averaging).

Standard in diffusion training (claude.md §11 STEP 10) -- the EMA copy is the
one used for sampling and evaluation, because it tracks a smoother trajectory
than the raw SGD parameters.
"""

from __future__ import annotations

import copy
from typing import Iterator

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        # Deep-copy the model so EMA params are independent + don't backprop.
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(p.detach(), alpha=1.0 - d)
        # Buffers (e.g. GroupNorm running stats are unused here but keep parity).
        for sb, pb in zip(self.shadow.buffers(), model.buffers()):
            sb.copy_(pb)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = float(sd["decay"])
        self.shadow.load_state_dict(sd["shadow"])

    def parameters(self) -> Iterator[torch.Tensor]:
        return self.shadow.parameters()
