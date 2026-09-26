from typing import NotRequired, TypedDict

import torch
from torch import nn

from models.blocks.deltaBlock import DeltaConvnextBlock
from .convnext import ConvNextV1


class CustomForwardConfig(TypedDict):
    block_indices: list[int]
    euler_step: NotRequired[float]
    method: NotRequired[str | None]  # None / "RK1" -> Euler; "RK2", "RK3", "RK4"

class DeltaConvNext(ConvNextV1):
    def __init__(self, stage3_length: int=9, useDeltas: bool=True):
        super().__init__()
        self.stage3_length = stage3_length
        self.useDeltas = useDeltas
        self.eulerStep = 1.0

    def _stage3_drop_paths(self) -> list[float]:
        """Drop-path rates for the (possibly unrolled) stage3 steps."""
        stage3_offset = self.accum_depths[2]
        n_base = self.depths[2]
        if self.stage3_length == n_base:
            return [self.drop_paths[i + stage3_offset] for i in range(self.stage3_length)]
        # Expand/contract the original stage3 segment so D != 9 still gets a smooth schedule.
        return torch.linspace(
            self.drop_paths[stage3_offset],
            self.drop_paths[stage3_offset + n_base - 1],
            self.stage3_length,
        ).tolist()

    def rewire(self, sharedBlock: int = 5):
        self.sharedBlock: nn.Sequential = self.stage3[sharedBlock].block  # único registro del bloque

        deltablocks = [
            DeltaConvnextBlock(
                self.sharedBlock,
                stochasticDepth=dp,
                useDeltas=self.useDeltas,
                eulerStep=self.eulerStep,
            )
            for dp in self._stage3_drop_paths()
        ]
        # stage3 no son solo los 9 bloques: acaba en LayerNorm2d + conv stride 2 que lleva
        # 384 -> 768 canales para stage4. Si ese tramo no se copia, stage4 recibe 384.
        self.tail = self.stage3[self.depths[2]:]
        self.deltifiedStage3 = nn.Sequential(*deltablocks, *self.tail)
        del self.stage3
        mode = "shared-only (no delta params)" if not self.useDeltas else f"{self.stage3_length} delta blocks"
        print(f"Rewired: stage3 -> 1 shared block + {mode} + {len(self.tail)} tail layers")

    def rewire_all_stages(self, shared_indices: list[int] | None = None):
        """One shared block per stage, looped depths[i] times (channels differ per stage).

        Default shared index is the middle block of each stage (stage3 -> 4 for depth 9).
        Keeps each stage's LN+downsample tail. Stages 1-4 become deltifiedStage{1..4}.
        """
        if shared_indices is None:
            shared_indices = [d // 2 for d in self.depths]
        if len(shared_indices) != 4:
            raise ValueError(f"Expected 4 shared indices, got {shared_indices}")

        stages = [self.stage1, self.stage2, self.stage3, self.stage4]
        shared = {}
        for s, (stage, idx, n) in enumerate(zip(stages, shared_indices, self.depths), start=1):
            if not 0 <= idx < n:
                raise ValueError(f"shared index {idx} out of range for stage{s} depth {n}")
            shared_block = stage[idx].block
            shared[str(s)] = shared_block
            offset = self.accum_depths[s - 1]
            deltablocks = [
                DeltaConvnextBlock(
                    shared_block,
                    stochasticDepth=self.drop_paths[i + offset],
                    useDeltas=self.useDeltas,
                    eulerStep=self.eulerStep,
                )
                for i in range(n)
            ]
            # Stages 1-3 end with LayerNorm2d + stride-2 conv; stage4 is blocks only.
            tail = list(stage[n:]) if s < 4 else []
            setattr(self, f"deltifiedStage{s}", nn.Sequential(*deltablocks, *tail))
            if s == 3:
                self.tail = stage[n:]
                self.stage3_length = n

        self.sharedBlocks = nn.ModuleDict(shared)
        # Alias for stage3-only helpers (getSharedWeights, delta_ratios, …).
        # Not registered again: sharedBlocks already owns the module.
        object.__setattr__(self, "sharedBlock", shared["3"])
        del self.stage1, self.stage2, self.stage3, self.stage4
        mode = "shared-only (no delta params)" if not self.useDeltas else "with deltas"
        loops = ", ".join(f"S{s}x{n}" for s, n in enumerate(self.depths, start=1))
        print(f"Rewired all stages ({loops}): 4 shared blocks, {mode}")


    def setUseDeltas(self, useDeltas: bool):
        """Toggle whether blocks add deltas in forward. Does not delete them."""
        self.useDeltas = useDeltas
        for i in range(self.stage3_length):
            block = self.deltifiedStage3[i]
            if isinstance(block, DeltaConvnextBlock):
                block.setUseDeltas(useDeltas)

    def init_deltas(self):
        """Allocate zero delta Parameters on every block that lacks them."""
        for block in self.deltifiedStage3:
            if isinstance(block, DeltaConvnextBlock):
                block.init_deltas()
        if self.useDeltas:
            self.unfreeze_deltas()

    def delete_deltas(self):
        """Remove delta Parameters from every block; shared-only mode."""
        for block in self.deltifiedStage3:
            if isinstance(block, DeltaConvnextBlock):
                block.delete_deltas()
        self.useDeltas = False

    def freeze_deltas(self):
        """Keep delta params but stop training them."""
        for block in self.deltifiedStage3:
            if isinstance(block, DeltaConvnextBlock):
                block.freeze_deltas()

    def unfreeze_deltas(self):
        for block in self.deltifiedStage3:
            if isinstance(block, DeltaConvnextBlock):
                for _, delta in block.deltas():
                    delta.requires_grad_(True)

    def getSharedWeights(self):
        return DeltaConvnextBlock.sharedWeights(self.sharedBlock)

    def setSharedWeights(self, sharedWeights: dict[str, torch.Tensor]):
        DeltaConvnextBlock.setSharedBlockWeights(self.sharedBlock, sharedWeights)

    @torch.no_grad()
    def delta_ratios(self) -> dict[str, float]:
        """||delta|| / ||shared weight|| for every part of every delta block.

        The denominators are five norms, not 45: every block divides by the same shared
        tensors, and those hold ~1.2M of the parameters the call reduces over.

        The ratios are stacked and read back once, since 45 separate .item() calls would
        mean 45 device syncs inside the training step.
        """
        # The tail (LayerNorm2d + downsampling conv) shares the Sequential but has no deltas.
        blocks = [
            b for b in self.deltifiedStage3
            if isinstance(b, DeltaConvnextBlock) and b.has_deltas()
        ]
        if not blocks:
            return {}
        weight_norms = {
            name: weight.norm().clamp_min(1e-12)
            for name, weight in DeltaConvnextBlock.sharedWeights(self.sharedBlock)
        }
        names, ratios = [], []
        for i, block in enumerate(blocks):
            for name, delta in block.deltas():
                names.append(f"b{i}/{name}")
                ratios.append(delta.norm() / weight_norms[name])
        if not ratios:
            return {}
        return dict(zip(names, torch.stack(ratios).float().tolist()))

    @torch.no_grad()
    def _reparametrize_shared(self, mean_deltas: dict[str, torch.Tensor]):
        for name, weight in self.getSharedWeights():
            n_meanDelta = mean_deltas[name]
            self.setSharedWeights({name: weight + n_meanDelta})

    @torch.no_grad()
    def _reparametrize_deltas(self, mean_deltas: dict[str, torch.Tensor]):
        for i in range(self.stage3_length):
            self.deltifiedStage3[i].reparametrize_delta(mean_deltas)

    @torch.no_grad()
    def _getMeanDeltas(self) -> dict[str, torch.Tensor]:
        mean_deltas = {}
        for block in self.deltifiedStage3:
            if isinstance(block, DeltaConvnextBlock):
                for name, delta in block.deltas():
                    mean_deltas[name] = mean_deltas.get(name, torch.zeros_like(delta)) + delta / self.stage3_length
        return mean_deltas

    @torch.no_grad()
    def reparametrize(self):
        blocks = [b for b in self.deltifiedStage3 if isinstance(b, DeltaConvnextBlock) and b.has_deltas()]
        if not blocks:
            return
        mean_deltas = self._getMeanDeltas()
        self._reparametrize_shared(mean_deltas)
        self._reparametrize_deltas(mean_deltas)

    def setStage3Length(self, stage3Length: int):
        """Unroll stage3 to ``stage3Length`` steps with EU = base_length / stage3Length.

        Keeps the already-loaded ``sharedBlock`` and ``tail``. When ``useDeltas`` is
        True, each new step gets its own zero-init delta Parameters.
        """
        self.extendedStage3Length = stage3Length
        self.eulerStep = self.stage3_length / stage3Length
        deltablocks = [
            DeltaConvnextBlock(
                self.sharedBlock,
                stochasticDepth=0.0,
                useDeltas=self.useDeltas,
                eulerStep=self.eulerStep,
            )
            for _ in range(stage3Length)
        ]
        self.deltifiedStage3 = nn.Sequential(*deltablocks, *self.tail)
        self.stage3_length = stage3Length
        mode = "shared-only" if not self.useDeltas else f"{stage3Length} delta blocks"
        print(f"Stage3 length -> {stage3Length} ({mode}, eulerStep={self.eulerStep:g})")

    def freezeStages(self, stages: list[int]):
        if 0 in stages:
            self.stem.requires_grad_(False)
        if 1 in stages:
            if hasattr(self, "deltifiedStage1"):
                self.deltifiedStage1.requires_grad_(False)
                self.sharedBlocks["1"].requires_grad_(False)
            else:
                self.stage1.requires_grad_(False)
        if 2 in stages:
            if hasattr(self, "deltifiedStage2"):
                self.deltifiedStage2.requires_grad_(False)
                self.sharedBlocks["2"].requires_grad_(False)
            else:
                self.stage2.requires_grad_(False)
        if 3 in stages:
            self.deltifiedStage3.requires_grad_(False)
            self.sharedBlock.requires_grad_(False)
        if 4 in stages:
            if hasattr(self, "deltifiedStage4"):
                self.deltifiedStage4.requires_grad_(False)
                self.sharedBlocks["4"].requires_grad_(False)
            else:
                self.stage4.requires_grad_(False)
        if 5 in stages:
            self.globalPool.requires_grad_(False)
            self.fc.requires_grad_(False)


    def _integrate_block(self, block: nn.Module, x: torch.Tensor, euler_step: float, method: str | None):
        # f(y) = block(y) - y; must subtract the evaluation point, not x.
        def f(y: torch.Tensor) -> torch.Tensor:
            return block(y) - y

        h = euler_step
        if method in (None, "RK1", "EULER"):
            return x + f(x) * h
        if method == "RK2":
            k1 = f(x)
            k2 = f(x + h * k1)
            return x + h / 2 * (k1 + k2)
        if method == "RK3":
            # Kutta's third order rule.
            k1 = f(x)
            k2 = f(x + h / 2 * k1)
            k3 = f(x - h * k1 + 2 * h * k2)
            return x + h / 6 * (k1 + 4 * k2 + k3)
        if method == "RK4":
            k1 = f(x)
            k2 = f(x + h / 2 * k1)
            k3 = f(x + h / 2 * k2)
            k4 = f(x + h * k3)
            return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        raise ValueError(f"Unknown integration method: {method!r}")

    def _stage(self, idx: int) -> nn.Module:
        """Stage ``idx`` (1-4): deltified if rewired, else the original Sequential."""
        deltified = getattr(self, f"deltifiedStage{idx}", None)
        if deltified is not None:
            return deltified
        return getattr(self, f"stage{idx}")

    def custom_forward(self, x, cnf: CustomForwardConfig):
        block_indices = cnf["block_indices"]
        euler_step = cnf.get("euler_step", 1.0)
        method = cnf.get("method")
        if method is not None:
            method = method.upper()
        x = self.stem(x)
        x = self._stage(1)(x)
        x = self._stage(2)(x)
        tail = {self.stage3_length, self.stage3_length + 1}
        for i in block_indices:
            if i not in tail:
                x = self._integrate_block(self.deltifiedStage3[i], x, euler_step, method)
            else:
                x = self.deltifiedStage3[i](x)
        x = self._stage(4)(x)
        x = self.globalPool(x)
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self.stem(x)
        x = self._stage(1)(x)
        x = self._stage(2)(x)
        x = self._stage(3)(x)
        x = self._stage(4)(x)
        x = self.globalPool(x)
        x = self.fc(x)
        return x