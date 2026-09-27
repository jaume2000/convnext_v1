#!/usr/bin/env python3
"""Plot interpolation Top-1 accuracy vs R (log2) from results2.csv.

By default plots every category that has a results2.csv (swin, convnext
droppath0, resnet, resnet101, …). Writes ``results_plain.png`` and
``results_bilinear.png`` next to each CSV.

Usage (from repo root):
    python scripts/plot_swin_interpolation.py
    python scripts/plot_swin_interpolation.py --categories swin_interpolation resnet_interpolation
    python scripts/plot_swin_interpolation.py --csv outputs/swin_interpolation/results2.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

METHODS = ("RK1", "RK2", "RK4")
POWER_OF_TWO_R = (1, 2, 4, 8, 16, 32, 64, 128)

# Default categories to plot when --csv is not given.
DEFAULT_CATEGORIES = (
    "swin_interpolation",
    "convnext_interpolation",
    "convnext_interpolation_droppath0",
    "resnet_interpolation",
    "resnet101_interpolation",
    "shared_convnext_interpolation",
)

CATEGORY_TITLES = {
    "swin_interpolation": "Swin",
    "convnext_interpolation": "ConvNeXt",
    "convnext_interpolation_droppath0": "ConvNeXt (droppath0)",
    "resnet_interpolation": "ResNet-50",
    "resnet101_interpolation": "ResNet-101",
    "shared_convnext_interpolation": "Shared ConvNeXt",
}


def plot_interpolation(
    df: pd.DataFrame,
    weight_interpolation: str,
    out_path: Path,
    title: str,
) -> bool:
    if "weight_interpolation" not in df.columns:
        print(f"  skip {weight_interpolation}: no weight_interpolation column")
        return False

    subset = df[
        (df["weight_interpolation"] == weight_interpolation)
        & (df["method"].isin(METHODS))
        & (df["repeats"].isin(POWER_OF_TWO_R))
    ].copy()
    if subset.empty:
        print(f"  skip {weight_interpolation}: no rows")
        return False
    subset = subset.sort_values("repeats")

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for method in METHODS:
        m = subset[subset["method"] == method]
        if m.empty:
            continue
        ax.plot(
            m["repeats"],
            m["top1acc"],
            marker="o",
            linewidth=2,
            markersize=6,
            label=method,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        print(f"  skip {weight_interpolation}: no method series")
        return False

    ax.set_xscale("log", base=2)
    ax.set_xticks(list(POWER_OF_TWO_R))
    ax.set_xticklabels([str(r) for r in POWER_OF_TWO_R])
    ax.set_xlabel("R (repeats)")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")
    return True


def plot_category(csv_path: Path, out_dir: Path | None = None, title_prefix: str | None = None) -> None:
    if not csv_path.exists():
        print(f"[miss] {csv_path}")
        return
    out_dir = out_dir or csv_path.parent
    if title_prefix is None:
        title_prefix = CATEGORY_TITLES.get(csv_path.parent.name, csv_path.parent.name)
    print(f"[plot] {csv_path}")
    df = pd.read_csv(csv_path)
    plot_interpolation(
        df,
        "plain",
        out_dir / "results_plain.png",
        f"{title_prefix} interpolation — plain (RK1 / RK2 / RK4)",
    )
    plot_interpolation(
        df,
        "bilinear",
        out_dir / "results_bilinear.png",
        f"{title_prefix} interpolation — bilinear (RK1 / RK2 / RK4)",
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Single results2.csv (overrides --categories)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir for --csv mode (default: CSV parent)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        help="Category dirs under outputs/ to plot (default: all known)",
    )
    args = parser.parse_args()

    if args.csv is not None:
        plot_category(args.csv, args.out_dir)
        return

    for name in args.categories:
        csv_path = repo_root / "outputs" / name / "results2.csv"
        plot_category(csv_path)


if __name__ == "__main__":
    main()
