from torch import nn

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
            nn.Conv2d(96, 96 * 2, kernel_size=2, stride=2, padding=0),
            LayerNorm2d(96 * 2),
        )
        self.stage2 = nn.Sequential(
            *[ConvnextBlock(96 * 2, 0.4) for _ in range(3)],
            nn.Conv2d(96 * 2, 96 * 4, kernel_size=2, stride=2, padding=0),
            LayerNorm2d(96 * 4),
        )
        self.stage3 = nn.Sequential(
            *[ConvnextBlock(96 * 4, 0.5) for _ in range(9)],
            nn.Conv2d(96 * 4, 96 * 8, kernel_size=2, stride=2, padding=0),
            LayerNorm2d(96 * 8),
        )
        self.stage4 = nn.Sequential(*[ConvnextBlock(96 * 8, 0.5) for _ in range(3)])
        self.globalPool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            LayerNorm2d(96 * 8),
            nn.Flatten(),
        )
        self.fc = nn.Linear(96 * 8, 1000)

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
