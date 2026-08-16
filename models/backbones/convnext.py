from torch import nn
from torch.nn.init import trunc_normal_

from ..blocks.convenxt_v1 import ConvnextBlock
from ..blocks.layerNorm2d import LayerNorm2d


class ConvNextV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=4, stride=4, padding=0),
            LayerNorm2d(96),
        )
        self.stage1 = nn.Sequential(
            *[ConvnextBlock(96, 0.1) for _ in range(3)],
            LayerNorm2d(96 * 2),
            nn.Conv2d(96, 96 * 2, kernel_size=2, stride=2, padding=0),
        )
        self.stage2 = nn.Sequential(
            *[ConvnextBlock(96 * 2, 0.1) for _ in range(3)],
            LayerNorm2d(96 * 4),
            nn.Conv2d(96 * 2, 96 * 4, kernel_size=2, stride=2, padding=0),
        )
        self.stage3 = nn.Sequential(
            *[ConvnextBlock(96 * 4, 0.1) for _ in range(9)],
            LayerNorm2d(96 * 8),
            nn.Conv2d(96 * 4, 96 * 8, kernel_size=2, stride=2, padding=0),
        )
        self.stage4 = nn.Sequential(*[ConvnextBlock(96 * 8, 0.1) for _ in range(3)])
        self.globalPool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            LayerNorm2d(96 * 8),
            nn.Flatten(),
        )
        self.fc = nn.Linear(96 * 8, 1000)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        """trunc_normal_(std=0.02), as in the ConvNeXt/DeiT recipes.

        PyTorch's default Kaiming-uniform scales with 1/sqrt(fan_in), which for the 7x7
        depthwise conv (fan_in 49) means std 0.083 and for the 1x1 expansion std 0.059,
        i.e. 3-4x wider than intended. LayerScale's 1e-6 hides that at step 0, but the
        residual branches then carry far more gain than the recipe's LR was tuned for
        once gamma grows. LayerNorm2d and LayerScale keep their own inits.
        """
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.globalPool(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = ConvNextV1()
    print(f"The model has {sum(p.numel() for p in model.parameters())} parameters")
