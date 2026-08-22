import torch
from torch import nn

from models.blocks.deltaBlock import DeltaConvnextBlock
from .convnext import ConvNextV1

class DeltaConvNext(ConvNextV1):
    def __init__(self, stage3_length: int=9):
        super().__init__()
        self.stage3_length = stage3_length

    def rewire(self, sharedBlock: int=5):
        #This will be the shared block
        self.sharedBlock = self.stage3[sharedBlock].block
        stage3_offset = self.accum_depths[2]
        deltablocks = [
            DeltaConvnextBlock(
                inCh=self.stage_dims[2],
                schocasticDepth=self.drop_paths[i + stage3_offset],
            )
            for i in range(self.stage3_length)
        ]
        for deltablock in deltablocks:
            deltablock.rewire(self.sharedBlock)
        self.deltifiedStage3 = nn.Sequential(*deltablocks, self.stage3[self.stage3_length:])
        
        # Delete the non-shared parameters
        del self.stage3
        print("Model rewired from old ConvNextV1!")

    def forward(self,x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.deltifiedStage3(x)
        x = self.stage4(x)
        x = self.globalPool(x)
        x = self.fc(x)
        return x
