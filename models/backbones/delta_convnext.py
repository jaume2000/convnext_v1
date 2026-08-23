import torch
from torch import nn

from models.blocks.deltaBlock import DeltaConvnextBlock
from .convnext import ConvNextV1

class DeltaConvNext(ConvNextV1):
    def __init__(self, stage3_length: int=9):
        super().__init__()
        self.stage3_length = stage3_length

    def rewire(self, sharedBlock: int = 5):
        stage3_offset = self.accum_depths[2]
        self.sharedBlock = self.stage3[sharedBlock].block   # único registro del bloque

        deltablocks = [
            DeltaConvnextBlock(
                self.sharedBlock,
                stochasticDepth=self.drop_paths[i + stage3_offset],
            )
            for i in range(self.stage3_length)
        ]
        # stage3 no son solo los 9 bloques: acaba en LayerNorm2d + conv stride 2 que lleva
        # 384 -> 768 canales para stage4. Si ese tramo no se copia, stage4 recibe 384.
        tail = self.stage3[self.depths[2]:]
        self.deltifiedStage3 = nn.Sequential(*deltablocks, *tail)
        del self.stage3
        print(f"Rewired: stage3 -> 1 shared block + {self.stage3_length} deltas + {len(tail)} tail layers")

    @torch.no_grad()
    def delta_ratios(self) -> dict[str, float]:
        """||delta|| / ||shared weight|| for every part of every delta block.

        The denominators are five norms, not 45: every block divides by the same shared
        tensors, and those hold ~1.2M of the parameters the call reduces over.

        The ratios are stacked and read back once, since 45 separate .item() calls would
        mean 45 device syncs inside the training step.
        """
        # The tail (LayerNorm2d + downsampling conv) shares the Sequential but has no deltas.
        blocks = [b for b in self.deltifiedStage3 if isinstance(b, DeltaConvnextBlock)]
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
        return dict(zip(names, torch.stack(ratios).float().tolist()))

    def forward(self,x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.deltifiedStage3(x)
        x = self.stage4(x)
        x = self.globalPool(x)
        x = self.fc(x)
        return x
