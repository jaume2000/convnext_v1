import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for NCHW feature maps (ConvNeXt-style)."""

    def __init__(self, num_channels: int, eps: float = 1e-6, weight_init: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(weight_init * torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.normalized_shape = (num_channels,)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W]. Viewing it as NHWC puts the normalized axis last, which lets
        # F.layer_norm run as one fused kernel instead of ~8 elementwise passes; both
        # permutes are stride-only no-ops when x is channels_last. F.layer_norm also
        # keeps the statistics in fp32 under autocast, unlike a manual mean/var.
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)
