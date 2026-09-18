"""ImageNet val of pretrained ResNet-50 with interpolated stage-3 blocks.

Stage-3 ``layer3`` has 6 Bottlenecks: index 0 downsamples and always runs once.
Schedulable identity residuals are indices 0..4 (mapping to layer3[1..5]).

Grid: every residual repeated R times with ES=1/R, integrated with RK1/RK2/RK4
for R in {1,2,4,8,16,32,64,128}. Edit EXPERIMENTS and run:

  python scripts/resnet_interpolation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.backbones.interpoled_resnet import InterpoledResNet50
from scripts.validate import validate
from utils.env import load_dotenv

load_dotenv()

OUTPUT_PATH = _REPO_ROOT / "outputs" / "resnet_interpolation"
BATCH_SIZE = 128
# Identity residuals in stage 3 (layer3[1..5]).
N_BLOCKS = 5

# Previous Euler / block-schedule sweeps (kept for reference; not run).
OLD_EXPERIMENTS = [
    # Single residual, one giant Euler step (integrate the whole stage-3 interval).
    {"name": "BASELINE", "blocks": [0,1,2,3,4], "euler_step": 1},
    {"name": "B0_ES5", "blocks": [0], "euler_step": 5},
    {"name": "B1_ES5", "blocks": [1], "euler_step": 5},
    {"name": "B2_ES5", "blocks": [2], "euler_step": 5},
    {"name": "B3_ES5", "blocks": [3], "euler_step": 5},
    {"name": "B4_ES5", "blocks": [4], "euler_step": 5},
    # Single residual, unit step.
    {"name": "B0_ES1", "blocks": [0], "euler_step": 1},
    {"name": "B1_ES1", "blocks": [1], "euler_step": 1},
    {"name": "B2_ES1", "blocks": [2], "euler_step": 1},
    {"name": "B3_ES1", "blocks": [3], "euler_step": 1},
    {"name": "B4_ES1", "blocks": [4], "euler_step": 1},
    # Two residuals, half the original depth each.
    {"name": "B0-4_ES2.5", "blocks": [0, 4], "euler_step": 5 / 2},
    {"name": "B1-3_ES2.5", "blocks": [1, 3], "euler_step": 5 / 2},
    {"name": "B2-2_ES2.5", "blocks": [2, 2], "euler_step": 5 / 2},
    {"name": "B3-1_ES2.5", "blocks": [3, 1], "euler_step": 5 / 2},
    {"name": "B4-0_ES2.5", "blocks": [4, 0], "euler_step": 5 / 2},
    # Two residuals, ES=1.
    {"name": "B0-4_ES1", "blocks": [0, 4], "euler_step": 1},
    {"name": "B1-3_ES1", "blocks": [1, 3], "euler_step": 1},
    {"name": "B2-2_ES1", "blocks": [2, 2], "euler_step": 1},
    {"name": "B3-1_ES1", "blocks": [3, 1], "euler_step": 1},
    {"name": "B4-0_ES1", "blocks": [4, 0], "euler_step": 1},
    # Ends + middle, varying step.
    {"name": "B0-2-4_ES2.5", "blocks": [0, 2, 4], "euler_step": 5 / 2},
    {"name": "B0-2-4_ES5div3", "blocks": [0, 2, 4], "euler_step": 5 / 3},
    {"name": "B0-2-4_ES1.25", "blocks": [0, 2, 4], "euler_step": 5 / 4},
    # Length-5 schedules at unit step.
    {"name": "B0-2-4x_ES1", "blocks": [0, 2, 4, 0, 2], "euler_step": 1},
    {"name": "B0x2-2x2-4_ES1", "blocks": [0, 0, 2, 2, 4], "euler_step": 1},
    {"name": "B0x3-4x2_ES1", "blocks": [0, 0, 0, 4, 4], "euler_step": 1},
    # All 5 residuals, refined Euler grid.
    {"name": "R2_ES0.5", "repeats": 2, "euler_step": 0.5},
    {"name": "R4_ES0.25", "repeats": 4, "euler_step": 0.25},
    {"name": "R10_ES0.1", "repeats": 10, "euler_step": 0.1},
    {"name": "R100_ES0.01", "repeats": 100, "euler_step": 0.01},
    # All 5 residuals, unit step.
    {"name": "R2_ES1", "repeats": 2, "euler_step": 1},
    {"name": "R4_ES1", "repeats": 4, "euler_step": 1},
    {"name": "R10_ES1", "repeats": 10, "euler_step": 1},
    {"name": "R100_ES1", "repeats": 100, "euler_step": 1},
    # Baseline (native depth, unit step).
    {"name": "baseline_ES1", "repeats": 1, "euler_step": 1},
]

# Active RK grid (what __main__ runs).
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
# Extra: R10 ES=0.1 with all RKs; R10/R100 ES=1 Euler-only (no RK sweep).
EXPERIMENTS += [
    {"name": f"R10_ES0.1_{m}", "repeats": 10, "euler_step": 0.1, "method": m}
    for m in METHODS
]
EXPERIMENTS += [
    {"name": "R10_ES1_RK1", "repeats": 10, "euler_step": 1, "method": "RK1"},
    {"name": "R100_ES1_RK1", "repeats": 100, "euler_step": 1, "method": "RK1"},
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


def load_interpoled_resnet() -> InterpoledResNet50:
    model = InterpoledResNet50(weights="DEFAULT")
    print(f"Loaded torchvision ResNet-50 (DEFAULT ImageNet weights), n_blocks={model.n_blocks}")
    return model


def apply_experiment(model: InterpoledResNet50, spec: dict) -> tuple[list[int], str]:
    n_blocks = model.n_blocks
    blocks = resolve_blocks(spec, n_blocks)
    euler_step = spec["euler_step"]
    method = spec.get("method", "RK1")
    model.set_schedule(blocks, euler_step, method=method)
    name = experiment_name(spec, blocks)
    print(
        f"\n=== {name} ===\n"
        f"stage3 schedule: layer3[0] + "
        f"{','.join(f'layer3[{i + 1}]' for i in blocks)}  "
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
    model = load_interpoled_resnet()
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
