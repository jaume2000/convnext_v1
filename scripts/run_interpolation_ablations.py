#!/usr/bin/env python3
"""Run plain/bilinear R×RK interpolation ablations for every backbone category.

Skips a category when ``outputs/<category>/results2.csv`` already exists.

  python scripts/run_interpolation_ablations.py
  python scripts/run_interpolation_ablations.py --only convnext_interpolation shared_convnext_interpolation
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# (category_dir_name under outputs/, python script, optional env overrides)
CATEGORIES: list[tuple[str, str, dict[str, str]]] = [
    ("swin_interpolation", "scripts/swin_interpolation.py", {}),
    (
        "convnext_interpolation",
        "scripts/convnext_interpolation.py",
        {
            "CONVNEXT_CHECKPOINT": str(
                _REPO_ROOT / "outputs" / "convnextv1_imagenet" / "weights" / "last.pth"
            ),
            "CONVNEXT_INTERP_OUT": str(_REPO_ROOT / "outputs" / "convnext_interpolation"),
        },
    ),
    (
        "convnext_interpolation_droppath0",
        "scripts/convnext_interpolation.py",
        {
            "CONVNEXT_CHECKPOINT": str(
                _REPO_ROOT / "outputs" / "convnextv1_imagenet_droppath0" / "weights" / "last.pth"
            ),
            "CONVNEXT_INTERP_OUT": str(
                _REPO_ROOT / "outputs" / "convnext_interpolation_droppath0"
            ),
        },
    ),
    ("resnet_interpolation", "scripts/resnet_interpolation.py", {}),
    ("resnet101_interpolation", "scripts/resnet101_interpolation.py", {}),
    (
        "shared_convnext_interpolation",
        "scripts/shared_convnext_interpolation.py",
        {
            "SHARED_CONVNEXT_CHECKPOINT": str(
                _REPO_ROOT / "outputs" / "shared_convnextv1_imagenet" / "weights" / "last.pth"
            ),
            "SHARED_CONVNEXT_INTERP_OUT": str(
                _REPO_ROOT / "outputs" / "shared_convnext_interpolation"
            ),
        },
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Optional subset of category names to consider",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print skip/run decisions without executing",
    )
    args = parser.parse_args()

    only = set(args.only) if args.only else None
    for category, script, env_extra in CATEGORIES:
        if only is not None and category not in only:
            continue
        out_dir = _REPO_ROOT / "outputs" / category
        results2 = out_dir / "results2.csv"
        if results2.exists():
            print(f"[skip] {category}: {results2} exists")
            continue
        print(f"[run]  {category}: {script}")
        if args.dry_run:
            continue
        env = os.environ.copy()
        env.update(env_extra)
        subprocess.run(
            [sys.executable, str(_REPO_ROOT / script)],
            cwd=_REPO_ROOT,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
