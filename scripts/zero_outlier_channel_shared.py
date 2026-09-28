"""Zero outlier / random stage-3 channels on shared ConvNeXt (R1/R10).

Naming (feature-map convention for shared):
  R = multiplier of the native stage-3 depth (N_BLOCKS=9)
  D = N_BLOCKS * R   →  R1 ⇒ D=9,  R10 ⇒ D=90
  ES = N_BLOCKS / D  →  R1 ⇒ ES=1, R10 ⇒ ES=0.1   (train horizon T = D·ES = 9)

Flow:
  1. Run R1 (D=9, ES=1), rank channels by ||x|| at end of stage-3 residuals → top-1.
  2. Re-integrate with that channel forced to 0 after every residual step.
  3. Save feature-map metrics + ImageNet val loss/top1 for R1 and R10 (outlier + baseline).
  4. Zero each of 20 random channels independently; val loss → histogram of Δloss vs channel.

Leonardo:
  source .env && sbatch --account="$SLURM_ACCOUNT" jobs/zero_outlier_channel_shared.sh

Local:
  python scripts/zero_outlier_channel_shared.py
  python scripts/zero_outlier_channel_shared.py --skip-val
  python scripts/zero_outlier_channel_shared.py --skip-metrics
  python scripts/zero_outlier_channel_shared.py --R 1 --n-random 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.imagenet import ImageNetDataset
from data.transforms.transforms import build_val_transforms
from models.backbones.delta_convnext import CustomForwardConfig, DeltaConvNext
from scripts.feature_map_explorer import (
    SHARED_CHECKPOINT,
    load_shared_convnext,
    save_metric_plots,
    save_tables_and_config,
    trajectory_stats,
)
from scripts.shared_convnext_interpolation import N_BLOCKS, last_metric, make_config
from scripts.validate import validate
from utils.env import load_dotenv

load_dotenv(_REPO_ROOT / ".env")

CLASS_ID = 289
IMAGE_INDEX_FALLBACK = 644  # snow leopard sample used in feature-map runs
BATCH_SIZE_VAL = 128
N_RANDOM_CHANNELS = 20
RANDOM_SEED = 0

# R multiplier → (D, ES) with fixed train horizon T = N_BLOCKS.
R_SCHEDULE = {
    1: {"D": N_BLOCKS * 1, "euler_step": 1.0},
    10: {"D": N_BLOCKS * 10, "euler_step": N_BLOCKS / (N_BLOCKS * 10)},
}


def _out_root() -> Path:
    """Bulky outputs under $WORK on Leonardo, else repo outputs/."""
    explicit = os.environ.get("ZERO_OUTLIER_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    work = os.environ.get("WORK")
    if work:
        return Path(work).expanduser() / "zero_outlier_channel_shared"
    return _REPO_ROOT / "outputs" / "zero_outlier_channel_shared"


class ZeroChannelCustomForward(nn.Module):
    """``custom_forward`` with channel(s) zeroed after every residual integrate."""

    def __init__(self, model: DeltaConvNext, cnf: CustomForwardConfig, channels: list[int]):
        super().__init__()
        self.model = model
        self.cnf = cnf
        self.channels = [int(c) for c in channels]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model = self.model
        cnf = self.cnf
        block_indices = cnf["block_indices"]
        euler_step = cnf.get("euler_step", 1.0)
        method = cnf.get("method")
        if method is not None:
            method = method.upper()

        x = model.stem(x)
        x = model._stage(1)(x)
        x = model._stage(2)(x)
        # Native shared stage3_length stays 9; block_indices repeat index 0 then LN+downsample.
        tail = {model.stage3_length, model.stage3_length + 1}
        for i in block_indices:
            if i not in tail:
                x = model._integrate_block(model.deltifiedStage3[i], x, euler_step, method)
                x = x.clone()
                for c in self.channels:
                    x[:, c].zero_()
            else:
                x = model.deltifiedStage3[i](x)
        x = model._stage(4)(x)
        x = model.globalPool(x)
        x = model.fc(x)
        return x


class _CustomForward(nn.Module):
    def __init__(self, model: DeltaConvNext, cnf: CustomForwardConfig):
        super().__init__()
        self.model = model
        self.cnf = cnf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.custom_forward(x, self.cnf)


@torch.no_grad()
def stage3_exit_channel_norms(
    model: DeltaConvNext,
    image: torch.Tensor,
    *,
    D: int,
    euler_step: float,
    method: str | None = "RK1",
) -> torch.Tensor:
    """||x_c||_F at the end of stage-3 residuals (before LN+downsample tail)."""
    model.eval()
    x = model.stage2(model.stage1(model.stem(image.unsqueeze(0))))
    block = model.deltifiedStage3[0]
    for _ in range(D):
        x = model._integrate_block(block, x, euler_step, method)
    return x[0].flatten(1).norm(dim=1).cpu()


def resolve_image_index(dataset: ImageNetDataset, class_id: int) -> int:
    for i in range(len(dataset)):
        if int(dataset[i][1]) == class_id:
            return i
    return IMAGE_INDEX_FALLBACK


def save_detection_plot(norms: torch.Tensor, outlier: int, out_path: Path) -> None:
    """Bar of all channel norms + highlight the detected outlier."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vals = norms.numpy()
    fig, ax = plt.subplots(figsize=(10, 3.6), layout="constrained")
    ax.bar(np.arange(len(vals)), vals, width=1.0, color="#7a8a99", linewidth=0)
    ax.bar([outlier], [vals[outlier]], width=1.0, color="#c0392b", linewidth=0, label=f"outlier ch={outlier}")
    ax.set_xlabel("channel")
    ax.set_ylabel(r"$\|x_c\|_F$ at stage-3 exit (R1)")
    ax.set_title("Stage-3 exit channel norms (shared ConvNeXt, R1/D=9)")
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run_feature_maps(
    model: DeltaConvNext,
    dataset: ImageNetDataset,
    class_names: list[str],
    *,
    R: int,
    channels: list[int],
    tag: str,
    image_index: int,
    out_dir: Path,
) -> dict:
    import scripts.feature_map_explorer as fme

    fme.FORCE_RECOMPUTE = True
    fme.METRICS_ONLY = True

    sched = R_SCHEDULE[R]
    D = int(sched["D"])
    euler_step = float(sched["euler_step"])
    if channels:
        ch_label = "ch" + "-".join(str(c) for c in channels[:3])
        if len(channels) > 3:
            ch_label += f"_n{len(channels)}"
    else:
        ch_label = "noch"
    name = f"R{R}_D{D}_ES{euler_step:g}_{tag}_{ch_label}_c{CLASS_ID}_n1"
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    field_blocks = [model.deltifiedStage3[0]] * D
    enter_fn = lambda batch, m=model: m.stage2(m.stage1(m.stem(batch)))
    print(
        f"\n=== feature maps {name} ===\n"
        f"R={R} D={D} ES={euler_step:g} zero_channels={channels} image={image_index}"
    )
    res = trajectory_stats(
        enter_fn,
        field_blocks,
        [image_index],
        euler_step=euler_step,
        method="RK1",
        batch_size=1,
        dataset=dataset,
        block_schedule=[0] * D,
        zero_channels_after_step=channels,
    )
    res["name"] = name
    res["spec"] = {
        "name": name,
        "model": "convnext_shared",
        "R": R,
        "D": D,
        "euler_step": euler_step,
        "method": "RK1",
        "class_id": CLASS_ID,
        "max_images": 1,
        "batch_size": 1,
        "ignore_top_k_channels": 0,
        "zero_channels": channels,
        "tag": tag,
        "image_indices": [image_index],
        "fps": 1 if D <= 9 else 10,
    }
    res["blocks"] = None
    res["model"] = "convnext_shared"
    res["weight_interpolation"] = "plain"
    res["overlay_label"] = f"R{R}"
    res["ignored_channels"] = list(channels)
    res["resolved_channels"] = list(channels)

    mean_x = res["mean_x"]
    for ch in channels[:5]:
        ch_norms = mean_x[:, ch].flatten(1).norm(dim=1)
        print(f"  ||mean_x[ch={ch}]|| along traj: {[float(v) for v in ch_norms[:8]]}...")

    save_tables_and_config(res, class_names, run_dir)
    save_metric_plots(res, run_dir)
    print(f"  metrics → {run_dir / 'metrics' / 'metrics.png'}")
    return {
        "name": name,
        "tag": tag,
        "R": R,
        "D": D,
        "euler_step": euler_step,
        "channels": channels,
        "run_dir": str(run_dir),
        "rectitude_mean": float(res.get("rectitude_mean", float("nan"))),
        "L_mean": float(res.get("L_mean", float("nan"))),
        "N_mean": float(res.get("N_mean", float("nan"))),
        "metrics_png": str(run_dir / "metrics" / "metrics.png"),
    }


def run_val(
    model: DeltaConvNext,
    *,
    R: int,
    channels: list[int] | None,
    tag: str,
    out_dir: Path,
) -> dict:
    sched = R_SCHEDULE[R]
    D = int(sched["D"])
    euler_step = float(sched["euler_step"])
    # make_config: repeats = # residual applications (= D for shared).
    ch_suffix = ""
    if channels is not None:
        ch_suffix = "_ch" + "-".join(str(c) for c in channels[:3])
        if len(channels) > 3:
            ch_suffix += f"_n{len(channels)}"
    name = f"R{R}_D{D}_ES{euler_step:g}_RK1_plain_{tag}{ch_suffix}"
    spec = {
        "name": name,
        "repeats": D,
        "euler_step": euler_step,
        "method": "RK1",
        "weight_interpolation": "plain",
    }
    cnf = make_config(spec)
    print(
        f"\n=== val {name} ===\n"
        f"blocks={cnf['block_indices'][:3]}...+tail  n_res={D} ES={cnf['euler_step']:g} "
        f"zero_channels={channels}"
    )
    if channels is None:
        wrapped: nn.Module = _CustomForward(model, cnf)
    else:
        wrapped = ZeroChannelCustomForward(model, cnf, channels)
    history = validate(
        model=wrapped,
        batch_size=BATCH_SIZE_VAL,
        output_path=out_dir / name,
        num_workers=2,
    )
    row = {
        "name": name,
        "tag": tag,
        "R": R,
        "D": D,
        "euler_step": float(cnf["euler_step"]),
        "zero_channels": channels,
        "channel": None if channels is None else (int(channels[0]) if len(channels) == 1 else None),
        "top1acc": last_metric(history, "top1acc"),
        "loss": last_metric(history, "loss"),
    }
    print(f"  top1={row['top1acc']:.5f}  loss={row['loss']:.6f}")
    return row


def save_loss_histograms(
    val_rows: list[dict],
    *,
    outlier: int,
    random_channels: list[int],
    out_dir: Path,
) -> list[str]:
    """Per-R bar + histogram of Δloss when dropping one channel."""
    out_dir.mkdir(parents=True, exist_ok=True)
    baselines = {
        int(r["R"]): float(r["loss"])
        for r in val_rows
        if r.get("tag") == "baseline" and r.get("zero_channels") is None
    }
    written: list[str] = []
    Rs = sorted({int(r["R"]) for r in val_rows})

    for R in Rs:
        base = baselines.get(R)
        if base is None:
            continue
        single = [
            r
            for r in val_rows
            if int(r["R"]) == R
            and r.get("channel") is not None
            and r.get("tag") in {"outlier", "random1"}
        ]
        if not single:
            continue

        rows = []
        for r in single:
            ch = int(r["channel"])
            loss = float(r["loss"])
            rows.append(
                {
                    "R": R,
                    "D": int(r["D"]),
                    "channel": ch,
                    "tag": r["tag"],
                    "loss": loss,
                    "delta_loss": loss - base,
                    "top1acc": float(r["top1acc"]),
                    "is_outlier": ch == outlier,
                }
            )
        df = pd.DataFrame(rows).sort_values("channel")
        csv_path = out_dir / f"R{R}_per_channel_loss.csv"
        df.to_csv(csv_path, index=False)
        written.append(str(csv_path))

        # Bar: channel → Δloss (outlier highlighted).
        fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(df) + 2), 4.0), layout="constrained")
        colors = ["#c0392b" if bool(o) else "#4a6fa5" for o in df["is_outlier"]]
        ax.bar(
            [str(c) for c in df["channel"]],
            df["delta_loss"],
            color=colors,
            width=0.8,
            linewidth=0,
        )
        ax.axhline(0.0, color="0.35", lw=0.8)
        ax.set_xlabel("zeroed channel")
        ax.set_ylabel(r"$\Delta$loss vs baseline")
        ax.set_title(
            f"R{R}/D={int(df['D'].iloc[0])}  "
            f"baseline loss={base:.4f}  "
            f"(red=outlier ch={outlier}, blue=random)"
        )
        fig.savefig(out_dir / f"R{R}_delta_loss_by_channel.png", dpi=140)
        plt.close(fig)
        written.append(str(out_dir / f"R{R}_delta_loss_by_channel.png"))

        # Histogram of Δloss for random channels; outlier as vertical line.
        rand_df = df[df["tag"] == "random1"]
        out_df = df[df["tag"] == "outlier"]
        fig, ax = plt.subplots(figsize=(7.5, 4.0), layout="constrained")
        if len(rand_df):
            ax.hist(
                rand_df["delta_loss"].to_numpy(),
                bins=min(12, max(5, len(rand_df))),
                color="#4a6fa5",
                edgecolor="white",
                alpha=0.9,
                label=f"random n={len(rand_df)}",
            )
        if len(out_df):
            d_out = float(out_df["delta_loss"].iloc[0])
            ax.axvline(
                d_out,
                color="#c0392b",
                lw=2.0,
                label=f"outlier ch={outlier}  Δloss={d_out:.4f}",
            )
        ax.axvline(0.0, color="0.35", lw=0.8, ls="--", label="baseline")
        ax.set_xlabel(r"$\Delta$loss vs baseline (drop one channel)")
        ax.set_ylabel("count")
        ax.set_title(f"R{R}/D={int(df['D'].iloc[0])}: loss rise when zeroing a channel")
        ax.legend(frameon=False)
        fig.savefig(out_dir / f"R{R}_delta_loss_hist.png", dpi=140)
        plt.close(fig)
        written.append(str(out_dir / f"R{R}_delta_loss_hist.png"))

    # Combined overlay histogram for all R if ≥2.
    if len(Rs) >= 2 and all(r in baselines for r in Rs):
        fig, ax = plt.subplots(figsize=(8.0, 4.2), layout="constrained")
        palette = {1: "#4a6fa5", 10: "#2a9d8f"}
        for R in Rs:
            single = [
                r
                for r in val_rows
                if int(r["R"]) == R and r.get("tag") == "random1" and r.get("channel") is not None
            ]
            if not single:
                continue
            deltas = [float(r["loss"]) - baselines[R] for r in single]
            ax.hist(
                deltas,
                bins=min(12, max(5, len(deltas))),
                color=palette.get(R, None),
                edgecolor="white",
                alpha=0.55,
                label=f"R{R} random n={len(deltas)}",
            )
            out_row = next(
                (
                    r
                    for r in val_rows
                    if int(r["R"]) == R and r.get("tag") == "outlier" and r.get("channel") == outlier
                ),
                None,
            )
            if out_row is not None:
                ax.axvline(
                    float(out_row["loss"]) - baselines[R],
                    color=palette.get(R, "0.2"),
                    lw=2.0,
                    ls="-",
                    label=f"R{R} outlier Δloss={float(out_row['loss']) - baselines[R]:.4f}",
                )
        ax.axvline(0.0, color="0.35", lw=0.8, ls="--")
        ax.set_xlabel(r"$\Delta$loss vs baseline")
        ax.set_ylabel("count")
        ax.set_title("Δloss when zeroing one channel (random + outlier)")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(out_dir / "delta_loss_hist_all_R.png", dpi=140)
        plt.close(fig)
        written.append(str(out_dir / "delta_loss_hist_all_R.png"))

    # Keep random channel list for the plot caption / reproducibility.
    (out_dir / "random_channels.json").write_text(
        json.dumps({"outlier": outlier, "random_channels": random_channels}, indent=2)
    )
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-val", action="store_true")
    p.add_argument("--skip-metrics", action="store_true")
    p.add_argument(
        "--with-random-metrics",
        action="store_true",
        help="Also save metrics plots when zeroing each random channel (default: off).",
    )
    p.add_argument("--R", nargs="+", type=int, default=[1, 10], choices=sorted(R_SCHEDULE))
    p.add_argument("--n-random", type=int, default=N_RANDOM_CHANNELS)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument(
        "--outlier-channel",
        type=int,
        default=None,
        help="Skip detection and force this channel as the outlier.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    skip_random_metrics = not args.with_random_metrics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA required")
    out_root = _out_root()
    print(f"device={device}  ckpt={SHARED_CHECKPOINT}")
    print(f"out_root={out_root}")
    print(f"schedule: { {r: R_SCHEDULE[r] for r in args.R} }")

    out_root.mkdir(parents=True, exist_ok=True)
    if not SHARED_CHECKPOINT.is_file():
        raise SystemExit(f"Shared checkpoint missing: {SHARED_CHECKPOINT}")
    model = load_shared_convnext(SHARED_CHECKPOINT).to(device).eval()

    dataset = ImageNetDataset(split="validation", transforms=build_val_transforms())
    class_names = list(getattr(dataset, "classes", [str(i) for i in range(1000)]))
    image_index = resolve_image_index(dataset, CLASS_ID)
    image, label = dataset[image_index]
    print(
        f"probe image index={image_index} label={label} "
        f"({class_names[label] if label < len(class_names) else label})"
    )

    # --- 1) detect outlier channel on R1 (D=9, ES=1) at end of stage-3 residuals ---
    r1 = R_SCHEDULE[1]
    norms = stage3_exit_channel_norms(
        model,
        image.to(device),
        D=int(r1["D"]),
        euler_step=float(r1["euler_step"]),
        method="RK1",
    )
    n_channels = int(norms.numel())
    topk = norms.topk(min(10, n_channels))
    print(f"\n=== R1 (D={r1['D']}, ES={r1['euler_step']:g}) stage-3 exit channel norms (top-10) ===")
    for rank, (ch, val) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), start=1):
        print(
            f"  #{rank:2d}  ch={ch:3d}  ||x||={val:10.3f}  "
            f"×median={val / norms.median().item():6.1f}"
        )
    outlier = int(args.outlier_channel) if args.outlier_channel is not None else int(topk.indices[0].item())
    print(
        f"\n→ outlier channel = {outlier}  "
        f"(||x||={norms[outlier].item():.3f}, median={norms.median().item():.3f}, C={n_channels})"
    )
    save_detection_plot(norms, outlier, out_root / "detected_channel_norms.png")

    detect = {
        "channel": outlier,
        "R": 1,
        "D": int(r1["D"]),
        "euler_step": float(r1["euler_step"]),
        "image_index": image_index,
        "n_channels": n_channels,
        "forced": args.outlier_channel is not None,
        "top10": [
            {"channel": int(c), "norm": float(v)}
            for c, v in zip(topk.indices.tolist(), topk.values.tolist())
        ],
    }
    (out_root / "detected_channel.json").write_text(json.dumps(detect, indent=2))

    # --- random control channels (exclude the outlier); ablated one-by-one ---
    g = torch.Generator().manual_seed(args.seed)
    pool = [c for c in range(n_channels) if c != outlier]
    perm = torch.randperm(len(pool), generator=g).tolist()
    random_channels = sorted(pool[i] for i in perm[: args.n_random])
    print(f"\n→ random control channels (n={len(random_channels)}, seed={args.seed}): {random_channels}")

    metrics_rows: list[dict] = []
    val_rows: list[dict] = []

    # baselines without zeroing (once per R): metrics + val
    for R in args.R:
        if not args.skip_metrics:
            metrics_rows.append(
                run_feature_maps(
                    model,
                    dataset,
                    class_names,
                    R=R,
                    channels=[],
                    tag="baseline",
                    image_index=image_index,
                    out_dir=out_root / "featureMaps",
                )
            )
        if not args.skip_val:
            val_rows.append(
                run_val(
                    model,
                    R=R,
                    channels=None,
                    tag="baseline",
                    out_dir=out_root / "val",
                )
            )

    # outlier: metrics + val for each R
    for R in args.R:
        if not args.skip_metrics:
            metrics_rows.append(
                run_feature_maps(
                    model,
                    dataset,
                    class_names,
                    R=R,
                    channels=[outlier],
                    tag="outlier",
                    image_index=image_index,
                    out_dir=out_root / "featureMaps",
                )
            )
        if not args.skip_val:
            val_rows.append(
                run_val(
                    model,
                    R=R,
                    channels=[outlier],
                    tag="outlier",
                    out_dir=out_root / "val",
                )
            )

    # 20 random channels, one at a time
    for ch in random_channels:
        for R in args.R:
            if not skip_random_metrics and not args.skip_metrics:
                metrics_rows.append(
                    run_feature_maps(
                        model,
                        dataset,
                        class_names,
                        R=R,
                        channels=[ch],
                        tag="random1",
                        image_index=image_index,
                        out_dir=out_root / "featureMaps",
                    )
                )
            if not args.skip_val:
                val_rows.append(
                    run_val(
                        model,
                        R=R,
                        channels=[ch],
                        tag="random1",
                        out_dir=out_root / "val",
                    )
                )

    hist_paths: list[str] = []
    if val_rows:
        hist_paths = save_loss_histograms(
            val_rows,
            outlier=outlier,
            random_channels=random_channels,
            out_dir=out_root / "histograms",
        )

    # JSON-serializable val rows
    val_rows_json = []
    for row in val_rows:
        val_rows_json.append({**row, "zero_channels": row["zero_channels"]})

    summary = {
        "detected": detect,
        "random_channels": random_channels,
        "random_seed": args.seed,
        "schedule": {str(r): R_SCHEDULE[r] for r in args.R},
        "metrics": metrics_rows,
        "val": val_rows_json,
        "histograms": hist_paths,
        "out_root": str(out_root),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    if val_rows:
        pd.DataFrame(val_rows_json).to_csv(out_root / "val_summary.csv", index=False)

    print("\n=== summary ===")
    print(f"outlier channel: {outlier}")
    print(f"random channels ({len(random_channels)}): {random_channels}")
    if val_rows:
        print(f"{'name':<64} {'top1':>10} {'loss':>12}")
        for row in val_rows:
            print(f"{row['name']:<64} {row['top1acc']:>10.5f} {row['loss']:>12.6f}")
    for row in metrics_rows:
        print(f"metrics: {row['metrics_png']}")
    for p in hist_paths:
        print(f"histogram: {p}")
    print(f"wrote {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
