import torch
import torch.nn.functional as F
from torch import nn

from models.blocks.stochasticDepth import StochasticDepth


class DeltaConvnextBlock(nn.Module):
    def __init__(self, sharedBlock, stochasticDepth=0.1, useDeltas=True):
        super().__init__()
        if len(sharedBlock) != 6:
            raise ValueError("Shared block must have 6 layers")
        # Referencia sin registrar: no duplica claves en el state_dict.
        object.__setattr__(self, "_shared", sharedBlock)
        self.useDeltas = useDeltas

        dw, ln, pw1, _, pw2, ls = sharedBlock
        z = lambda t: nn.Parameter(torch.zeros_like(t))
        self.dwWDelta,  self.dwBDelta  = z(dw.weight),  z(dw.bias)
        self.lnWDelta,  self.lnBDelta  = z(ln.weight),  z(ln.bias)
        self.pw1WDelta, self.pw1BDelta = z(pw1.weight), z(pw1.bias)
        self.pw2WDelta, self.pw2BDelta = z(pw2.weight), z(pw2.bias)
        self.lsDelta                   = z(ls.gamma)

        self.stochasticDepth = StochasticDepth(stochasticDepth)

    # Public delta name -> attribute holding it. The two only differ for the layer scale,
    # whose parameter is named after the module and not after sharedWeights().
    DELTA_ATTRS = {
        "dw.weight":  "dwWDelta",
        "ln.weight":  "lnWDelta",
        "pw1.weight": "pw1WDelta",
        "pw2.weight": "pw2WDelta",
        "ls.gamma":  "lsDelta",

        "dw.bias":  "dwBDelta",
        "ln.bias":  "lnBDelta",
        "pw1.bias": "pw1BDelta",
        "pw2.bias": "pw2BDelta",
    }

    def deltas(self):
        """(name, delta) for each part of the block, named as in sharedWeights()."""
        return [(name, getattr(self, attr)) for name, attr in self.DELTA_ATTRS.items()]

    @torch.no_grad()
    def setDeltas(self, deltas: dict[str, torch.Tensor]):
        """Overwrite the given deltas in place, keyed as in deltas()."""
        for key, delta in deltas.items():
            attr = self.DELTA_ATTRS.get(key)
            if attr is None:
                raise KeyError(f"Unknown delta {key!r}, expected one of {list(self.DELTA_ATTRS)}")
            # copy_ instead of setattr: it keeps the Parameter (and its identity in the
            # optimizer) and stops every block from ending up with the same tensor object.
            getattr(self, attr).copy_(delta)

    @staticmethod
    @torch.no_grad()
    def setSharedBlockWeights(sharedBlock, sharedWeights: dict[str, torch.Tensor]):
        """Overwrite shared weights in place, keyed as in sharedWeights()."""
        weight_map = dict(DeltaConvnextBlock.sharedWeights(sharedBlock))
        for name, weight in sharedWeights.items():
            param = weight_map.get(name)
            if param is None:
                raise KeyError(f"Unknown shared weight {name!r}, expected one of {list(weight_map)}")
            param.copy_(weight)

    @torch.no_grad()
    def setSharedWeights(self, sharedWeights: dict[str, torch.Tensor]):
        """Overwrite the given shared weights in place, keyed as in sharedWeights()."""
        self.setSharedBlockWeights(self._shared, sharedWeights)

    def setUseDeltas(self, useDeltas: bool):
        self.useDeltas = useDeltas

    @staticmethod
    def sharedWeights(sharedBlock):
        """(name, weight) that the deltas of every block are added to.

        Takes the shared block rather than an instance because these tensors are the
        same objects for all the delta blocks: anything reducing over them (a norm, say)
        only has to do it once for the whole stage.
        """
        dw, ln, pw1, _, pw2, ls = sharedBlock
        return [
            ("dw.weight",  dw.weight),
            ("ln.weight",  ln.weight),
            ("pw1.weight", pw1.weight),
            ("pw2.weight", pw2.weight),
            ("ls.gamma",   ls.gamma),
            ("dw.bias",    dw.bias),
            ("ln.bias",    ln.bias),
            ("pw1.bias",   pw1.bias),
            ("pw2.bias",   pw2.bias),
        ]

    @torch.no_grad()
    def reparametrize_delta(self, mean_deltas: dict[str, torch.Tensor]):
        """
        Reparametrize substracting the mean_delta from the original weight.
        """
        for name, delta in self.deltas():
            n_meanDelta = mean_deltas[name]
            self.setDeltas({name: delta - n_meanDelta})

    def forward(self, x):
        dw, ln, pw1, _, pw2, ls = self._shared
        weights = dw.weight
        biases = dw.bias
        if self.useDeltas:
            weights = weights + self.dwWDelta
            biases = biases + self.dwBDelta
        h = F.conv2d(x, weights, biases, padding=dw.padding, groups=dw.groups)

        h = h.permute(0, 2, 3, 1)

        weights = ln.weight
        biases = ln.bias
        if self.useDeltas:
            weights = weights + self.lnWDelta
            biases = biases + self.lnBDelta
        h = F.layer_norm(h, ln.normalized_shape, weights, biases, ln.eps)
        h = h.permute(0, 3, 1, 2)

        weights = pw1.weight
        biases = pw1.bias
        if self.useDeltas:
            weights = weights + self.pw1WDelta
            biases = biases + self.pw1BDelta
        h = F.conv2d(h, weights, biases)

        h = F.gelu(h)                                    # una sola vez

        weights = pw2.weight
        biases = pw2.bias
        if self.useDeltas:
            weights = weights + self.pw2WDelta
            biases = biases + self.pw2BDelta
        h = F.conv2d(h, weights, biases)

        gamma = ls.gamma
        if self.useDeltas:
            gamma = gamma + self.lsDelta
        h = h * gamma.view(1, -1, 1, 1)
        return x + self.stochasticDepth(h)