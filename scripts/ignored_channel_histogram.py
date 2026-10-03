"""Which stage-3 channel would ``ignore_top_k_channels=1`` drop, image by image?

For each model/schedule and each of N images (one per randomly drawn class), integrate
stage 3 (Euler, plain weights) and rank channels with the explorer criterion
``score_c = || (||h[t, c]||_F)_t ||_2`` (``channel_spatial_norms``). Also report the channel
picked on the mean trajectory over all N images (what ``feature_map_explorer`` does).

Local:
  python scripts/ignored_channel_histogram.py            # N=200
  python scripts/ignored_channel_histogram.py --n 50 --runs convnext_R1 shared_D9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.feature_map_explorer as fme
from data.imagenet import ImageNetDataset
from data.transforms.transforms import build_val_transforms

OUT_DIR = _REPO_ROOT / "outputs" / "ignored_channel_histogram"

# name -> (model key, repeats or shared depth D, euler step)
RUNS: dict[str, tuple[str, int, float]] = {
    "shared_D9": (fme._SHARED_MODEL_KEY, 9, 1.0),
    "shared_D100": (fme._SHARED_MODEL_KEY, 100, 0.09),
    "convnext_R1": ("convnext", 1, 1.0),
    "convnext_R100": ("convnext", 100, 0.01),
    "convnext_droppath0_R1": ("convnext_droppath0", 1, 1.0),
    "swin_R1": ("swin", 1, 1.0),
    "resnet50_R1": ("resnet50", 1, 1.0),
    "resnet101_R1": ("resnet101", 1, 1.0),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=200, help="Images (one per distinct random class)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
    return p.parse_args()


def sample_indices(labels: list[int], n: int, seed: int) -> list[int]:
    """First validation image of each of ``n`` distinct classes drawn at random."""
    rng = np.random.default_rng(seed)
    first: dict[int, int] = {}
    for i, y in enumerate(labels):
        first.setdefault(int(y), i)
    classes = rng.choice(sorted(first), size=n, replace=False)
    return sorted(first[int(c)] for c in classes)


def build_run(model_key: str, depth: int, es: float, model):
    """(enter_fn, field_blocks) for one schedule, matching ``run_one``."""
    if model_key == fme._SHARED_MODEL_KEY:
        enter = lambda b: model.stage2(model.stage1(model.stem(b)))
        return enter, [model.deltifiedStage3[0]] * depth
    blocks = fme.resolve_blocks({"repeats": depth}, fme.stage3_n_blocks(model_key, model))
    fields = fme.stage3_field_blocks(model_key, model, blocks, euler_step=es, weight_interpolation="plain")
    return (lambda b: fme.enter_stage3(model_key, model, b)), fields


@torch.no_grad()
def channel_scores(enter, fields, es: float, dataset, indices: list[int], batch_size: int):
    """Per-image channel scores [N, C] and the summed h trajectory [D+1, C, H, W]."""
    D = len(fields)
    scores, sum_h = [], None
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        batch = torch.stack([dataset[i][0] for i in idx]).to(fme.device)
        x = enter(batch)
        sq = torch.zeros(x.shape[:2], device=x.device, dtype=torch.float64)
        if sum_h is None:
            sum_h = torch.zeros((D + 1, *x.shape[1:]), device=x.device)
        for d in range(D + 1):
            block = fields[min(d, D - 1)]
            h = block(x) - x
            sq += h.flatten(2).norm(dim=2).double() ** 2
            sum_h[d] += h.sum(0)
            if d < D:
                x = x + es * h
        scores.append(sq.sqrt().cpu())
        print(f"    {min(start + batch_size, len(indices))}/{len(indices)}", flush=True)
    return torch.cat(scores), sum_h.cpu()


def plot_histograms(results: dict[str, dict], path: Path, n: int) -> None:
    k = len(results)
    cols = 2
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.0 * rows), squeeze=False, layout="constrained")
    for ax, (name, r) in zip(axes.flat, results.items()):
        counts = r["counts"]
        ch = np.array(sorted(counts))
        ax.bar(ch, [counts[c] for c in ch], width=max(1.0, r["n_channels"] / 150), color="#4a6fa5")
        ax.axvline(r["mean_traj_channel"], color="#c0392b", lw=1, ls="--", label=f"mean-traj pick: {r['mean_traj_channel']}")
        top = max(counts, key=counts.get)
        ax.set_title(
            f"{name}: {len(counts)} distinct; ch {top} in {counts[top]}/{n} "
            f"(median share {r['median_top_share']:.0%})",
            fontsize=9,
        )
        ax.set_xlim(-0.5, r["n_channels"] - 0.5)
        ax.set_xlabel("channel id")
        ax.set_ylabel("count")
        ax.legend(frameon=False, fontsize=8)
    for ax in list(axes.flat)[k:]:
        ax.axis("off")
    fig.suptitle(f"Top-1 channel by ||h|| (WxH, L2 over t), per image (n={n})")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = ImageNetDataset(split="validation", transforms=build_val_transforms())
    labels = dataset.ds.data.column("label").to_pylist()
    indices = sample_indices(labels, args.n, args.seed)
    print(f"device={fme.device}  n={len(indices)}  seed={args.seed}")

    results: dict[str, dict] = {}
    models: dict[str, torch.nn.Module] = {}
    for name in args.runs:
        model_key, depth, es = RUNS[name]
        if model_key not in models:
            models.clear()
            torch.cuda.empty_cache()
            models[model_key] = fme.load_interpoled_model(model_key).to(fme.device).eval()
        print(f"\n=== {name} ({model_key}, depth={depth}, ES={es:g}) ===")
        enter, fields = build_run(model_key, depth, es, models[model_key])
        scores, sum_h = channel_scores(enter, fields, es, dataset, indices, args.batch_size)

        top = scores.argmax(1)
        share = scores.max(1).values ** 2 / (scores**2).sum(1)
        second = scores.topk(2, dim=1).values
        ratio = second[:, 0] / second[:, 1]
        mean_pick = int(fme.top_norm_channels(sum_h / len(indices), 1)[0])
        counts = pd.Series(top.numpy()).value_counts().to_dict()

        pd.DataFrame(
            {
                "image_index": indices,
                "label": [labels[i] for i in indices],
                "top_channel": top.numpy(),
                "top_share": share.numpy(),
                "top_over_second": ratio.numpy(),
            }
        ).to_csv(OUT_DIR / f"{name}_per_image.csv", index=False)
        results[name] = {
            "counts": {int(c): int(v) for c, v in counts.items()},
            "n_channels": int(scores.shape[1]),
            "mean_traj_channel": mean_pick,
            "median_top_share": float(share.median()),
            "median_top_over_second": float(ratio.median()),
        }
        best = max(counts, key=counts.get)
        print(
            f"  per-image picks: {len(counts)} distinct, most common ch {best} "
            f"({counts[best]}/{len(indices)}); mean-traj pick ch {mean_pick}; "
            f"median share {share.median():.1%}, median top/2nd {ratio.median():.1f}x"
        )

    plot_histograms(results, OUT_DIR / f"histogram_n{args.n}.png", args.n)
    (OUT_DIR / f"summary_n{args.n}.json").write_text(
        json.dumps({"n": args.n, "seed": args.seed, "image_indices": indices, "runs": results}, indent=2)
    )
    print(f"\nwrote {OUT_DIR / f'histogram_n{args.n}.png'}")


if __name__ == "__main__":
    main()
