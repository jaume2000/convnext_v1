"""ImageNet val of pretrained ConvNeXt-T with interpolated stage-3 blocks.

Each experiment repeats every residual 0..8, R times, with ES=1/R, integrated
with RK1/RK2/RK4 for R in {1,2,4,8,16,32,64,128}. The downsample tail (9, 10)
always runs once. Edit EXPERIMENTS and run:

  python scripts/convnext_interpolation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.backbones.interpoled_convnext import InterpoledConvNextV1
from scripts.validate import validate
from utils.env import load_dotenv

load_dotenv()

CHECKPOINT = _REPO_ROOT / "outputs" / "convnextv1_imagenet_droppath0" / "weights" / "last.pth"
OUTPUT_PATH = _REPO_ROOT / "outputs" / "convnext_interpolation_droppath0"
BATCH_SIZE = 128

REPEATS = [1, 2, 4, 8, 16, 32, 64, 128]
METHODS = ["RK1", "RK2", "RK4"]
EXPERIMENTS = [
    {
        "name": f"R{r}_ES{1 / r:g}_{m}",
        "repeats": r,
        "euler_step": 1 / r,
        "method": m,
    }
    for r in REPEATS
    for m in METHODS
]


def resolve_blocks(spec: dict, n_blocks: int) -> list[int]:
    if "blocks" in spec and "repeats" in spec:
        raise ValueError("Specify either blocks or repeats, not both")
    if "blocks" in spec:
        return list(spec["blocks"])
    repeats = spec.get("repeats", 1)
    return [i for i in range(n_blocks) for _ in range(repeats)]


def experiment_name(spec: dict, blocks: list[int]) -> str:
    if spec.get("name"):
        return spec["name"]
    es = spec["euler_step"]
    method = spec.get("method", "RK1")
    if "repeats" in spec:
        return f"R{spec['repeats']}_ES{es:g}_{method}"
    return f"B{'-'.join(map(str, blocks))}_ES{es:g}_{method}".replace("/", "div")


def load_interpoled_convnext(checkpoint: Path) -> InterpoledConvNextV1:
    model = InterpoledConvNextV1()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    print(f"Loaded {checkpoint}" + (f" (epoch {epoch})" if epoch is not None else ""))
    return model


def apply_experiment(model: InterpoledConvNextV1, spec: dict) -> tuple[list[int], str]:
    n_blocks = model.depths[2]
    blocks = resolve_blocks(spec, n_blocks)
    euler_step = spec["euler_step"]
    method = spec.get("method", "RK1")
    model.set_schedule(blocks, euler_step, method=method)
    name = experiment_name(spec, blocks)
    print(
        f"\n=== {name} ===\n"
        f"stage3 schedule: {','.join(map(str, blocks))},{n_blocks},{n_blocks + 1}  "
        f"(euler_step={euler_step:g}, method={method})"
    )
    return blocks, name


def last_metric(history, name: str) -> float:
    return float(history.get(name).compute_last())


def next_results_csv(output_path: Path) -> Path:
    candidate = output_path / "results.csv"
    if not candidate.exists():
        return candidate
    n = 2
    while True:
        candidate = output_path / f"results{n}.csv"
        if not candidate.exists():
            return candidate
        n += 1


if __name__ == "__main__":
    model = load_interpoled_convnext(CHECKPOINT)
    results = []
    for spec in EXPERIMENTS:
        blocks, name = apply_experiment(model, spec)
        history = validate(
            model=model,
            batch_size=BATCH_SIZE,
            output_path=OUTPUT_PATH / name,
        )
        results.append({
            "name": name,
            "repeats": spec.get("repeats"),
            "blocks": ",".join(map(str, blocks)),
            "euler_step": spec["euler_step"],
            "method": spec.get("method", "RK1"),
            "top1acc": last_metric(history, "top1acc"),
            "loss": last_metric(history, "loss"),
        })

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    csv_path = next_results_csv(OUTPUT_PATH)
    pd.DataFrame(results).to_csv(csv_path, index=False)

    print("\n=== summary ===")
    print(f"{'name':<28} {'method':<6} {'euler':>8} {'top1':>10} {'loss':>10}")
    for row in results:
        print(
            f"{row['name']:<28} {row['method']:<6} {row['euler_step']:>8g} "
            f"{row['top1acc']:>10.5f} {row['loss']:>10.6f}"
        )
    print(f"\nSaved {len(results)} rows to {csv_path}")
