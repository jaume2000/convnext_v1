"""ImageNet val of shared ConvNeXt with refined depth / RK schedules.

Shared residual applied D=R times with ES=9/R (train horizon T=9), integrated
with RK1/RK2/RK4 for R in {1,2,4,8,16,32,64,128}. Weight interpolation is
``plain`` only (a single shared θ; bilinear between identical weights is a
no-op). CSV columns match the interpoled ``results2.csv`` layout.

  python scripts/shared_convnext_interpolation.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.backbones.delta_convnext import CustomForwardConfig, DeltaConvNext
from scripts.validate import validate
from utils.env import load_dotenv

load_dotenv()

CHECKPOINT = Path(
    os.environ.get(
        "SHARED_CONVNEXT_CHECKPOINT",
        str(_REPO_ROOT / "outputs" / "shared_convnextv1_imagenet" / "weights" / "last.pth"),
    )
)
OUTPUT_PATH = Path(
    os.environ.get(
        "SHARED_CONVNEXT_INTERP_OUT",
        str(_REPO_ROOT / "outputs" / "shared_convnext_interpolation"),
    )
)
BATCH_SIZE = 128
N_BLOCKS = 9  # native shared stage-3 depth (train horizon units)
TAIL = [N_BLOCKS, N_BLOCKS + 1]

REPEATS = [1, 2, 4, 8, 16, 32, 64, 128]
METHODS = ["RK1", "RK2", "RK4"]
# Shared has one θ — only plain is meaningful.
WEIGHT_INTERPOLATIONS = ["plain"]

EXPERIMENTS = [
    {
        "name": f"R{r}_ES{N_BLOCKS / r:g}_{m}_{wi}",
        "repeats": r,
        "euler_step": N_BLOCKS / r,
        "method": m,
        "weight_interpolation": wi,
    }
    for r in REPEATS
    for m in METHODS
    for wi in WEIGHT_INTERPOLATIONS
]
EXPERIMENTS += [
    {
        "name": f"R10_ES{N_BLOCKS / 10:g}_{m}_{wi}",
        "repeats": 10,
        "euler_step": N_BLOCKS / 10,
        "method": m,
        "weight_interpolation": wi,
    }
    for m in METHODS
    for wi in WEIGHT_INTERPOLATIONS
]
EXPERIMENTS += [
    {
        "name": f"R10_ES1_RK1_{wi}",
        "repeats": 10,
        "euler_step": 1.0,
        "method": "RK1",
        "weight_interpolation": wi,
    }
    for wi in WEIGHT_INTERPOLATIONS
]
EXPERIMENTS += [
    {
        "name": f"R100_ES1_RK1_{wi}",
        "repeats": 100,
        "euler_step": 1.0,
        "method": "RK1",
        "weight_interpolation": wi,
    }
    for wi in WEIGHT_INTERPOLATIONS
]


class _CustomForwardModel(nn.Module):
    """Expose ``custom_forward`` as ``forward`` for ``scripts.validate``."""

    def __init__(self, model: DeltaConvNext, cnf: CustomForwardConfig):
        super().__init__()
        self.model = model
        self.cnf = cnf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.custom_forward(x, self.cnf)


def load_shared_convnext(checkpoint: Path) -> DeltaConvNext:
    model = DeltaConvNext(useDeltas=False)
    model.rewire()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    print(f"Loaded {checkpoint}" + (f" (epoch {epoch})" if epoch is not None else ""))
    return model


def make_config(spec: dict) -> CustomForwardConfig:
    r = int(spec["repeats"])
    cnf: CustomForwardConfig = {
        "block_indices": [0] * r + TAIL,
        "euler_step": float(spec["euler_step"]),
        "method": spec.get("method", "RK1"),
    }
    return cnf


def last_metric(history, name: str) -> float:
    return float(history.get(name).compute_last())


if __name__ == "__main__":
    results2 = OUTPUT_PATH / "results2.csv"
    if results2.exists():
        print(f"Skip: {results2} already exists")
        sys.exit(0)

    model = load_shared_convnext(CHECKPOINT)
    results = []
    for spec in EXPERIMENTS:
        cnf = make_config(spec)
        name = spec["name"]
        blocks = cnf["block_indices"][:-2]
        print(
            f"\n=== {name} ===\n"
            f"shared schedule: {','.join(map(str, blocks))}+tail  "
            f"(euler_step={cnf['euler_step']:g}, method={cnf.get('method')}, "
            f"weight_interpolation={spec['weight_interpolation']})"
        )
        wrapped = _CustomForwardModel(model, cnf)
        history = validate(
            model=wrapped,
            batch_size=BATCH_SIZE,
            output_path=OUTPUT_PATH / name,
        )
        results.append({
            "name": name,
            "repeats": spec["repeats"],
            "blocks": ",".join(map(str, blocks)),
            "euler_step": spec["euler_step"],
            "method": spec.get("method", "RK1"),
            "weight_interpolation": spec["weight_interpolation"],
            "top1acc": last_metric(history, "top1acc"),
            "loss": last_metric(history, "loss"),
        })

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(results2, index=False)

    print("\n=== summary ===")
    print(f"{'name':<36} {'wi':<10} {'method':<6} {'euler':>8} {'top1':>10} {'loss':>10}")
    for row in results:
        print(
            f"{row['name']:<36} {row['weight_interpolation']:<10} {row['method']:<6} "
            f"{row['euler_step']:>8g} {row['top1acc']:>10.5f} {row['loss']:>10.6f}"
        )
    print(f"\nSaved {len(results)} rows to {results2}")
