"""ResNet-50 with a configurable stage-3 residual schedule (Euler-scaled)."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class InterpoledResNet50(nn.Module):
    """Pretrained ResNet-50 with a configurable stage-3 residual schedule.

    ``layer3`` has 6 Bottlenecks: index 0 downsamples (stride-2 + channel expand)
    and always runs once. Indices 1..5 are identity residuals and are the ones
    scheduled via ``blocks`` / ``euler_step``:

        x <- x + euler_step * (layer3[i](x) - x)

    ``blocks`` indexes those identity residuals as 0..4 (mapping to layer3[1..5]).
    Module weights stay identical to torchvision ``resnet50``.
    """

    def __init__(
        self,
        blocks: list[int] | None = None,
        euler_step: float = 1.0,
        weights: ResNet50_Weights | str | None = ResNet50_Weights.DEFAULT,
    ):
        super().__init__()
        if isinstance(weights, str):
            weights = ResNet50_Weights[weights]
        self.backbone = resnet50(weights=weights)
        # Identity residuals only (skip the downsample head of layer3).
        self.n_blocks = len(self.backbone.layer3) - 1
        self.euler_step = euler_step
        self.blocks = list(range(self.n_blocks)) if blocks is None else list(blocks)

    def set_schedule(self, blocks: list[int], euler_step: float) -> None:
        bad = [i for i in blocks if not 0 <= i < self.n_blocks]
        if bad:
            raise ValueError(f"block indices {bad} out of range [0, {self.n_blocks})")
        self.blocks = list(blocks)
        self.euler_step = euler_step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        # Stage-3 downsample head (always once).
        x = b.layer3[0](x)
        for i in self.blocks:
            blk = b.layer3[i + 1]
            x = x + self.euler_step * (blk(x) - x)
        x = b.layer4(x)
        x = b.avgpool(x)
        x = torch.flatten(x, 1)
        x = b.fc(x)
        return x
