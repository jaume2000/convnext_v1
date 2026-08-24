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


def test_shared_only_has_no_delta_params():
    model = DeltaConvNext(useDeltas=False)
    model.rewire()
    delta_params = [n for n, _ in model.named_parameters() if "Delta" in n]
    assert delta_params == []

    model = DeltaConvNext(useDeltas=True)
    model.rewire()
    assert sum(1 for n, _ in model.named_parameters() if "Delta" in n) == 81
    model.delete_deltas()
    assert sum(1 for n, _ in model.named_parameters() if "Delta" in n) == 0


def test_init_deltas_recreates_zero_params():
    model = DeltaConvNext(useDeltas=False)
    model.rewire()
    assert sum(1 for n, _ in model.named_parameters() if "Delta" in n) == 0

    model.init_deltas()
    assert sum(1 for n, _ in model.named_parameters() if "Delta" in n) == 81
    for block in model.deltifiedStage3:
        if isinstance(block, DeltaConvnextBlock):
            for _, delta in block.deltas():
                assert torch.all(delta == 0)


def test_set_use_deltas_does_not_delete():
    """useDeltas=False toggles forward only; params stay allocated."""
    model = DeltaConvNext(useDeltas=True)
    model.rewire()
    block = model.deltifiedStage3[0]
    assert isinstance(block, DeltaConvnextBlock)
    block.setDeltas({name: torch.full_like(d, 1.0) for name, d in block.deltas()})

    model.setUseDeltas(False)
    assert sum(1 for n, _ in model.named_parameters() if "Delta" in n) == 81
    assert all(not p.requires_grad for n, p in model.named_parameters() if "Delta" in n)
    assert torch.all(block.dwWDelta == 1.0)

    model.setUseDeltas(True)
    assert all(p.requires_grad for n, p in model.named_parameters() if "Delta" in n)


def test_set_use_deltas_generates_missing_deltas():
    model = DeltaConvNext(useDeltas=False)
    model.rewire()
    model.setUseDeltas(True)
    assert sum(1 for n, _ in model.named_parameters() if "Delta" in n) == 81


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
