import torch
import torch.nn.functional as F
from torch import nn

from models.blocks.stochasticDepth import StochasticDepth


class DeltaConvnextBlock(nn.Module):
    def __init__(self, sharedBlock, stochasticDepth=0.1):
        super().__init__()
        if len(sharedBlock) != 6:
            raise ValueError("Shared block must have 6 layers")
        # Referencia sin registrar: no duplica claves en el state_dict.
        object.__setattr__(self, "_shared", sharedBlock)

        dw, ln, pw1, _, pw2, ls = sharedBlock
        z = lambda t: nn.Parameter(torch.zeros_like(t))
        self.dwWDelta,  self.dwBDelta  = z(dw.weight),  z(dw.bias)
        self.lnWDelta,  self.lnBDelta  = z(ln.weight),  z(ln.bias)
        self.pw1WDelta, self.pw1BDelta = z(pw1.weight), z(pw1.bias)
        self.pw2WDelta, self.pw2BDelta = z(pw2.weight), z(pw2.bias)
        self.lsDelta                   = z(ls.gamma)

        self.stochasticDepth = StochasticDepth(stochasticDepth)

    def deltas(self):
        """(name, delta) for each part of the block, named as in sharedWeights().

        Biases are left out: they are one order of magnitude smaller than the weights
        they sit next to, so a ratio against them says more about the bias than about
        how far this block has drifted from the shared block.
        """
        return [
            ("dw",  self.dwWDelta),
            ("ln",  self.lnWDelta),
            ("pw1", self.pw1WDelta),
            ("pw2", self.pw2WDelta),
            ("ls",  self.lsDelta),
        ]

    @staticmethod
    def sharedWeights(sharedBlock):
        """(name, weight) that the deltas of every block are added to.

        Takes the shared block rather than an instance because these tensors are the
        same objects for all the delta blocks: anything reducing over them (a norm, say)
        only has to do it once for the whole stage.
        """
        dw, ln, pw1, _, pw2, ls = sharedBlock
        return [
            ("dw",  dw.weight),
            ("ln",  ln.weight),
            ("pw1", pw1.weight),
            ("pw2", pw2.weight),
            ("ls",  ls.gamma),
        ]

    def forward(self, x):
        dw, ln, pw1, _, pw2, ls = self._shared

        h = F.conv2d(x, dw.weight + self.dwWDelta, dw.bias + self.dwBDelta,
                     padding=dw.padding, groups=dw.groups)

        h = h.permute(0, 2, 3, 1)
        h = F.layer_norm(h, ln.normalized_shape,
                         ln.weight + self.lnWDelta,
                         ln.bias + self.lnBDelta, ln.eps)
        h = h.permute(0, 3, 1, 2)

        h = F.conv2d(h, pw1.weight + self.pw1WDelta, pw1.bias + self.pw1BDelta)
        h = F.gelu(h)                                    # una sola vez
        h = F.conv2d(h, pw2.weight + self.pw2WDelta, pw2.bias + self.pw2BDelta)

        h = h * (ls.gamma + self.lsDelta).view(1, -1, 1, 1)
        return x + self.stochasticDepth(h)