"""Reparametrization centers deltas around zero and folds the mean into shared."""

import torch

from models.backbones.delta_convnext import DeltaConvNext
from models.blocks.deltaBlock import DeltaConvnextBlock


def test_reparametrize_centers_deltas_and_updates_shared():
    """Shared=1, delta_i = i  →  shared=5, delta_i = i - 4.

    Mean of {0..8} is 4, so after reparametrize:
      shared <- 1 + 4 = 5
      block i <- i - 4   (block 0: -4, block 4: 0, block 8: +4)
    """
    stage3_length = 9
    model = DeltaConvNext(stage3_length=stage3_length)
    model.rewire()

    shared = {name: torch.ones_like(w) for name, w in model.getSharedWeights()}
    model.setSharedWeights(shared)

    for i in range(stage3_length):
        block = model.deltifiedStage3[i]
        assert isinstance(block, DeltaConvnextBlock)
        block.setDeltas({name: torch.full_like(delta, float(i)) for name, delta in block.deltas()})

    model.reparametrize()

    for name, weight in model.getSharedWeights():
        assert torch.allclose(weight, torch.full_like(weight, 5.0)), f"shared {name}"

    for i in range(stage3_length):
        expected = float(i - 4)
        for name, delta in model.deltifiedStage3[i].deltas():
            assert torch.allclose(delta, torch.full_like(delta, expected)), (
                f"block {i} {name}: got {delta.flatten()[0].item()}, expected {expected}"
            )


def test_reparametrize_preserves_effective_weights():
    """shared + delta must be unchanged by reparametrization."""
    stage3_length = 9
    model = DeltaConvNext(stage3_length=stage3_length)
    model.rewire()

    model.setSharedWeights({name: torch.ones_like(w) for name, w in model.getSharedWeights()})
    for i in range(stage3_length):
        block = model.deltifiedStage3[i]
        block.setDeltas({name: torch.full_like(d, float(i)) for name, d in block.deltas()})

    before = []
    shared_before = dict(model.getSharedWeights())
    for i in range(stage3_length):
        before.append(
            {
                name: shared_before[name] + delta
                for name, delta in model.deltifiedStage3[i].deltas()
            }
        )

    model.reparametrize()

    shared_after = dict(model.getSharedWeights())
    for i in range(stage3_length):
        for name, delta in model.deltifiedStage3[i].deltas():
            effective = shared_after[name] + delta
            assert torch.allclose(effective, before[i][name]), f"block {i} {name}"
