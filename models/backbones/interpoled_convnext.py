from .convnext import ConvNextV1
from .rk_integrate import bilinear_time_steps, residual_rk_step, residual_rk_step_lerped


class InterpoledConvNextV1(ConvNextV1):
    """Pretrained ConvNeXt-T with a configurable stage-3 residual schedule.

    ``blocks`` is the sequence of stage-3 residual indices (0..8). The downsample
    tail (9, 10) always runs once after that sequence. Each residual is integrated
    with ``euler_step`` via ``method`` (RK1/Euler, RK2, RK4). Module names stay
    identical to ``ConvNextV1``.

    ``weight_interpolation``:
      - ``"plain"``: θ(t) = θ_⌊t⌋ (use scheduled block indices as-is)
      - ``"bilinear"``: θ(t) = (1-α)θ_k + α θ_{k+1} on a uniform Euler grid
        over [0, n_blocks) with the same number of steps as ``blocks``
    """

    def __init__(
        self,
        blocks: list[int] | None = None,
        euler_step: float = 1.0,
        method: str | None = "RK1",
        weight_interpolation: str = "plain",
    ):
        super().__init__()
        self.euler_step = euler_step
        self.method = method
        self.weight_interpolation = weight_interpolation
        self.blocks = list(range(self.depths[2])) if blocks is None else list(blocks)

    def set_schedule(
        self,
        blocks: list[int],
        euler_step: float,
        method: str | None = "RK1",
        weight_interpolation: str = "plain",
    ) -> None:
        n_blocks = self.depths[2]
        bad = [i for i in blocks if not 0 <= i < n_blocks]
        if bad:
            raise ValueError(f"block indices {bad} out of range [0, {n_blocks})")
        if weight_interpolation not in ("plain", "bilinear"):
            raise ValueError(
                f"weight_interpolation must be 'plain' or 'bilinear', got {weight_interpolation!r}"
            )
        self.blocks = list(blocks)
        self.euler_step = euler_step
        self.method = method
        self.weight_interpolation = weight_interpolation

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        n_blocks = self.depths[2]
        if self.weight_interpolation == "plain":
            for i in self.blocks:
                x = residual_rk_step(self.stage3[i], x, self.euler_step, self.method)
        else:
            for k, alpha in bilinear_time_steps(n_blocks, len(self.blocks), self.euler_step):
                if alpha == 0.0 or k >= n_blocks - 1:
                    x = residual_rk_step(self.stage3[k], x, self.euler_step, self.method)
                else:
                    x = residual_rk_step_lerped(
                        self.stage3[k],
                        self.stage3[k + 1],
                        alpha,
                        x,
                        self.euler_step,
                        self.method,
                    )
        x = self.stage3[n_blocks](x)
        x = self.stage3[n_blocks + 1](x)
        x = self.stage4(x)
        x = self.globalPool(x)
        x = self.fc(x)
        return x
