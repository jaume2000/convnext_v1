from torch import nn

from models.blocks.layerScale import LayerScale
from models.blocks.stochasticDepth import StochasticDepth


class ConvnextBlock(nn.Module):
    def __init__(self, inCh, schocasticDepth=0.1):
        super().__init__()
        self.schocasticDepth = schocasticDepth
        self.block = nn.Sequential(
            nn.Conv2d(inCh, inCh, kernel_size=7, padding=3, groups=inCh),
            nn.LayerNorm(inCh),
            nn.Conv2d(inCh, inCh*4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(inCh*4, inCh, kernel_size=1),
            LayerScale(inCh)
        )
        self.stochasticDepth = StochasticDepth(self.schocasticDepth)

    def forward(self, x):
        return x + self.stochasticDepth(self.block(x))

