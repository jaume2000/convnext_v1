import torch
from models.blocks.convenxt_v1 import ConvnextBlock
from torch import nn

from models.blocks.layerNorm2d import LayerNorm2d
from models.blocks.layerScale import LayerScale
from models.blocks.stochasticDepth import StochasticDepth


class DeltaConvnextBlock(ConvnextBlock):
    def __init__(self, inCh, schocasticDepth=0.1):
        super().__init__(inCh, schocasticDepth)
        self.inCh = inCh

    def rewire(self, sharedBlock):
        self.block = sharedBlock
        if len(self.block) != 6:
            raise ValueError("Shared block must have 6 layers")
        # Initialize with 1e-6
        self.blockDeltas = [
            nn.Conv2d(self.inCh, self.inCh, kernel_size=7, padding=3, groups=self.inCh),
            LayerNorm2d(self.inCh, weight_init=1e-6),
            nn.Conv2d(self.inCh, self.inCh * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.inCh * 4, self.inCh, kernel_size=1),
            LayerScale(self.inCh, init_value=1e-6),
        ]
        for i in [0,2,4]:
            self.blockDeltas[i].weight = nn.Parameter(1e-6 * torch.ones(self.blockDeltas[i].weight.shape))
            self.blockDeltas[i].bias = nn.Parameter(torch.zeros(self.blockDeltas[i].bias.shape))
        self.blockDeltas = nn.Sequential(*self.blockDeltas)
        for param in self.blockDeltas.parameters():
            param.isDelta = True

    # This can be optimized. We are builing it sequentially, but could be done in parallel.
    def forward(self, x):
        h = x
        for i in range(len(self.block)):
            h = self.block[i](h) + self.blockDeltas[i](h)
        return x + self.stochasticDepth(h)
        
