"""Swin-T with a configurable stage-3 block schedule (RK-scaled)."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import Swin_T_Weights, swin_t

from .rk_integrate import bilinear_time_steps, residual_rk_step, residual_rk_step_lerped


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

    ``weight_interpolation``:
      - ``"plain"``: θ(t) = θ_⌊t⌋
      - ``"bilinear"``: θ(t) = (1-α)θ_k + α θ_{k+1}
    """

    STAGE3_INDEX = 5

    def __init__(
        self,
        blocks: list[int] | None = None,
        euler_step: float = 1.0,
        method: str | None = "RK1",
        weight_interpolation: str = "plain",
        weights: Swin_T_Weights | str | None = Swin_T_Weights.DEFAULT,
    ):
        super().__init__()
        if isinstance(weights, str):
            weights = Swin_T_Weights[weights]
        self.backbone = swin_t(weights=weights)
        self.n_blocks = len(self.backbone.features[self.STAGE3_INDEX])
        self.euler_step = euler_step
        self.method = method
        self.weight_interpolation = weight_interpolation
        self.blocks = list(range(self.n_blocks)) if blocks is None else list(blocks)

    def set_schedule(
        self,
        blocks: list[int],
        euler_step: float,
        method: str | None = "RK1",
        weight_interpolation: str = "plain",
    ) -> None:
        bad = [i for i in blocks if not 0 <= i < self.n_blocks]
        if bad:
            raise ValueError(f"block indices {bad} out of range [0, {self.n_blocks})")
        if weight_interpolation not in ("plain", "bilinear"):
            raise ValueError(
                f"weight_interpolation must be 'plain' or 'bilinear', got {weight_interpolation!r}"
            )
        self.blocks = list(blocks)
        self.euler_step = euler_step
        self.method = method
        self.weight_interpolation = weight_interpolation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        stage3 = b.features[self.STAGE3_INDEX]
        for i, module in enumerate(b.features):
            if i == self.STAGE3_INDEX:
                if self.weight_interpolation == "plain":
                    for j in self.blocks:
                        x = residual_rk_step(stage3[j], x, self.euler_step, self.method)
                else:
                    for k, alpha in bilinear_time_steps(
                        self.n_blocks, len(self.blocks), self.euler_step
                    ):
                        if alpha == 0.0 or k >= self.n_blocks - 1:
                            x = residual_rk_step(
                                stage3[k], x, self.euler_step, self.method
                            )
                        else:
                            x = residual_rk_step_lerped(
                                stage3[k],
                                stage3[k + 1],
                                alpha,
                                x,
                                self.euler_step,
                                self.method,
                            )
            else:
                x = module(x)
        x = b.norm(x)
        x = b.permute(x)
        x = b.avgpool(x)
        x = b.flatten(x)
        x = b.head(x)
        return x