"""ResNet-50 / ResNet-101 with a configurable stage-3 residual schedule (RK-scaled)."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    ResNet50_Weights,
    ResNet101_Weights,
    resnet50,
    resnet101,
)

from .rk_integrate import residual_rk_step


class InterpoledResNet(nn.Module):
    """Pretrained ResNet with a configurable stage-3 residual schedule.

    ``layer3[0]`` downsamples (stride-2 + channel expand) and always runs once.
    The remaining identity Bottlenecks are scheduled via ``blocks`` / ``euler_step``
    and integrated with ``method`` (RK1/Euler, RK2, RK4):

        x <- RK(f, x, euler_step)  with  f(y) = layer3[i](y) - y

    ``blocks`` indexes those identity residuals as ``0 .. n_blocks-1``
    (mapping to ``layer3[1 ..]``). Weights stay identical to torchvision.
    """

    def __init__(
        self,
        backbone: nn.Module,
        blocks: list[int] | None = None,
        euler_step: float = 1.0,
        method: str | None = "RK1",
    ):
        super().__init__()
        self.backbone = backbone
        self.n_blocks = len(self.backbone.layer3) - 1
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
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3[0](x)
        for i in self.blocks:
            x = residual_rk_step(b.layer3[i + 1], x, self.euler_step, self.method)
        x = b.layer4(x)
        x = b.avgpool(x)
        x = torch.flatten(x, 1)
        x = b.fc(x)
        return x


class InterpoledResNet50(InterpoledResNet):
    """ResNet-50: 5 schedulable identity residuals in stage 3 (layer3[1..5])."""

    def __init__(
        self,
        blocks: list[int] | None = None,
        euler_step: float = 1.0,
        method: str | None = "RK1",
        weights: ResNet50_Weights | str | None = ResNet50_Weights.DEFAULT,
    ):
        if isinstance(weights, str):
            weights = ResNet50_Weights[weights]
        super().__init__(
            resnet50(weights=weights),
            blocks=blocks,
            euler_step=euler_step,
            method=method,
        )


class InterpoledResNet101(InterpoledResNet):
    """ResNet-101: 22 schedulable identity residuals in stage 3 (layer3[1..22])."""

    def __init__(
        self,
        blocks: list[int] | None = None,
        euler_step: float = 1.0,
        method: str | None = "RK1",
        weights: ResNet101_Weights | str | None = ResNet101_Weights.DEFAULT,
    ):
        if isinstance(weights, str):
            weights = ResNet101_Weights[weights]
        super().__init__(
            resnet101(weights=weights),
            blocks=blocks,
            euler_step=euler_step,
            method=method,
        )
