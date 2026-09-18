"""Swin-T with a configurable stage-3 block schedule (RK-scaled)."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import Swin_T_Weights, swin_t

from .rk_integrate import residual_rk_step


class InterpoledSwinT(nn.Module):
    """Pretrained Swin-T with a configurable stage-3 block schedule.

    torchvision Swin-T ``features`` layout:
      0 stem, 1 stage1 (2), 2 PatchMerging,
      3 stage2 (2), 4 PatchMerging,
      5 stage3 (6), 6 PatchMerging,
      7 stage4 (2)

    ``blocks`` is the sequence of stage-3 indices (0..5). PatchMerging + stage4
    always run once after that sequence. Each scheduled block is integrated with
    ``euler_step`` via ``method`` (RK1/Euler, RK2, RK4).

    Module weights stay identical to torchvision ``swin_t``.
    """

    STAGE3_INDEX = 5

    def __init__(
        self,
        blocks: list[int] | None = None,
        euler_step: float = 1.0,
        method: str | None = "RK1",
        weights: Swin_T_Weights | str | None = Swin_T_Weights.DEFAULT,
    ):
        super().__init__()
        if isinstance(weights, str):
            weights = Swin_T_Weights[weights]
        self.backbone = swin_t(weights=weights)
        self.n_blocks = len(self.backbone.features[self.STAGE3_INDEX])
        self.euler_step = euler_step
        self.method = method
        self.blocks = list(range(self.n_blocks)) if blocks is None else list(blocks)

    def set_schedule(
        self,
        blocks: list[int],
        euler_step: float,
        method: str | None = "RK1",
    ) -> None:
        bad = [i for i in blocks if not 0 <= i < self.n_blocks]
        if bad:
            raise ValueError(f"block indices {bad} out of range [0, {self.n_blocks})")
        self.blocks = list(blocks)
        self.euler_step = euler_step
        self.method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        stage3 = b.features[self.STAGE3_INDEX]
        for i, module in enumerate(b.features):
            if i == self.STAGE3_INDEX:
                for j in self.blocks:
                    x = residual_rk_step(stage3[j], x, self.euler_step, self.method)
            else:
                x = module(x)
        x = b.norm(x)
        x = b.permute(x)
        x = b.avgpool(x)
        x = b.flatten(x)
        x = b.head(x)
        return x
