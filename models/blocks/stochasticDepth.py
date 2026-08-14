from torch import nn
import torch

class StochasticDepth(nn.Module):
    
    def __init__(self, p=1.0) -> None:
        super().__init__()
        self.keepProb = 1-p

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (len(x.shape)-1)
        maskTensor = (self.keepProb + torch.rand(shape, dtype=x.dtype, device=x.device)).floor()
        maskTensor /= self.keepProb  # Estimation now is scaled up to 1.
        return maskTensor * x

if __name__ == "__main__":
    with torch.no_grad():
        StDepth = StochasticDepth(0.25)
        x = torch.rand((8,2,2,1))   # Simulate Bz = 8
        x = StDepth(x)
        x = x.reshape((8,2,2))
        print(x)