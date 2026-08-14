import torch
import torch.nn as nn


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        # gamma: [C] -> [1, C, 1, 1, ...] for NCHW (or ND) broadcast.
        return x * self.gamma.view(1, -1, *([1] * (x.ndim - 2)))
