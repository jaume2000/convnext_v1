from .convnext import ConvNextV1
from .rk_integrate import residual_rk_step


class InterpoledConvNextV1(ConvNextV1):
    """Pretrained ConvNeXt-T with a configurable stage-3 residual schedule.

    ``blocks`` is the sequence of stage-3 residual indices (0..8). The downsample
    tail (9, 10) always runs once after that sequence. Each residual is integrated
    with ``euler_step`` via ``method`` (RK1/Euler, RK2, RK4). Module names stay
    identical to ``ConvNextV1``.
    """

    def __init__(
        self,
        blocks: list[int] | None = None,
        euler_step: float = 1.0,
        method: str | None = "RK1",
    ):
        super().__init__()
        self.euler_step = euler_step
        self.method = method
        self.blocks = list(range(self.depths[2])) if blocks is None else list(blocks)

    def set_schedule(
        self,
        blocks: list[int],
        euler_step: float,
        method: str | None = "RK1",
    ) -> None:
        n_blocks = self.depths[2]
        bad = [i for i in blocks if not 0 <= i < n_blocks]
        if bad:
            raise ValueError(f"block indices {bad} out of range [0, {n_blocks})")
        self.blocks = list(blocks)
        self.euler_step = euler_step
        self.method = method

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        n_blocks = self.depths[2]
        for i in self.blocks:
            x = residual_rk_step(self.stage3[i], x, self.euler_step, self.method)
        x = self.stage3[n_blocks](x)
        x = self.stage3[n_blocks + 1](x)
        x = self.stage4(x)
        x = self.globalPool(x)
        x = self.fc(x)
        return x
