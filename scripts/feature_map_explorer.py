"""Stage-3 feature-map trajectories for shared ConvNeXt or interpoled backbones.

Default (``BACKBONE = "interpoled"``): all probes in one job —
  1. random-init six (LayerScale=1): SHARED | NON-shared plain | NON-shared bilinear
     at R1/D=9 and R100/D=900
  2. pretrained ``convnext_shared`` residual × D sweep
  3. interpoled stage-3 on swin / resnet* / convnext* (plain + bilinear)

``BACKBONE = "random_init"`` keeps only the six random-weight probes.
``BACKBONE = "shared"`` keeps the pretrained shared-only sweep.

Submit:
  source .env && sbatch --account="$SLURM_ACCOUNT" jobs/feature_map_explorer.sh

Or locally:
  python scripts/feature_map_explorer.py
  python scripts/feature_map_explorer.py --list-only
  python scripts/feature_map_explorer.py --only resnet50_baseline_R1_ES1_c289_n1
  python scripts/feature_map_explorer.py --only convnext_shared_baseline_D9_ES1_c289_n1
  python scripts/feature_map_explorer.py --force   # overwrite existing figures / recompute
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from tqdm import tqdm

from data.imagenet import ImageNetDataset
from data.transforms.transforms import IMAGENET_MEAN, IMAGENET_STD, build_val_transforms
from models.backbones.delta_convnext import DeltaConvNext
from models.backbones.interpoled_convnext import InterpoledConvNextV1
from models.backbones.interpoled_resnet import InterpoledResNet50, InterpoledResNet101
from models.backbones.interpoled_swin import InterpoledSwinT
from models.backbones.rk_integrate import bilinear_time_steps, call_lerped
from models.blocks.layerScale import LayerScale
from utils.env import load_dotenv

# --------------------------------------------------------------------------- config
# "interpoled"   = random-init six + pretrained shared + interpoled sweeps (default)
# "random_init"  = only the six random-weight probes
# "shared"       = only pretrained DeltaConvNext shared residual × D
BACKBONE = "interpoled"  # "interpoled" | "random_init" | "shared"
RANDOM_INIT_SEED = 0
# Override LayerScale gamma after random init (recipe default is 1e-6).
RANDOM_INIT_LAYERSCALE = 1.0

# Models included when BACKBONE == "interpoled" (docs / startup log).
INTERPOLED_MODELS = [
    "convnext_shared_rand",  # random-init DeltaConvNext (LayerScale override)
    "convnext_rand",         # random-init InterpoledConvNext (LayerScale override)
    "convnext_shared",       # outputs/shared_convnextv1_imagenet (shared residual × D)
    "convnext",              # outputs/convnextv1_imagenet (with drop-path)
    "convnext_droppath0",    # outputs/convnextv1_imagenet_droppath0
    "resnet50",
    "resnet101",
    "swin",
]

SHARED_CHECKPOINT = _REPO_ROOT / "outputs" / "shared_convnextv1_imagenet" / "weights" / "last.pth"
INTERPOLED_CONVNEXT_CHECKPOINT = (
    _REPO_ROOT / "outputs" / "convnextv1_imagenet" / "weights" / "last.pth"
)
INTERPOLED_CONVNEXT_DROPPATH0_CHECKPOINT = (
    _REPO_ROOT / "outputs" / "convnextv1_imagenet_droppath0" / "weights" / "last.pth"
)
OUT_DIR_SHARED = _REPO_ROOT / "outputs" / "featureMaps"
OUT_DIR_INTERPOLED = _REPO_ROOT / "outputs" / "featureMaps_interpoled"

SPLIT = "validation"
IMAGE_INDICES = [0, 17, 4242]
CLASS_ID: int | None = 207
MAX_IMAGES_PER_CLASS: int | None = 100
BATCH_SIZE = 8
FPS = 10.0

# Shared ConvNeXt feature maps (same last.pth as shared_convnext_ablation).
_FM_COMMON = {"class_id": 289, "max_images": 1, "batch_size": 1}
_SHARED = {"model": "convnext_shared", **_FM_COMMON}
RUNS_SHARED: list[dict] = [
    # Baseline D=9 ES=1 ± ignore (Euler).
    {"name": "convnext_shared_baseline_D9_ES1_c289_n1", "D": 9, "euler_step": 1.0, "fps": 1, "ignore_top_k_channels": 0, "method": None, **_SHARED},
    {"name": "convnext_shared_baseline_D9_ES1_c289_n1_ignore1", "D": 9, "euler_step": 1.0, "fps": 1, "ignore_top_k_channels": 1, "method": None, **_SHARED},
    # R100-style refined / large-step Euler.
    {"name": "convnext_shared_D100_ES0.01_c289_n1", "D": 100, "euler_step": 0.01, "fps": 80, "ignore_top_k_channels": 0, "method": None, **_SHARED},
    {"name": "convnext_shared_D100_ES0.01_c289_n1_ignore1", "D": 100, "euler_step": 0.01, "fps": 80, "ignore_top_k_channels": 1, "method": None, **_SHARED},
    {"name": "convnext_shared_D100_ES0.1_c289_n1", "D": 100, "euler_step": 0.1, "fps": 80, "ignore_top_k_channels": 0, "method": None, **_SHARED},
    {"name": "convnext_shared_D100_ES0.1_c289_n1_ignore1", "D": 100, "euler_step": 0.1, "fps": 80, "ignore_top_k_channels": 1, "method": None, **_SHARED},
    {"name": "convnext_shared_D100_ES1_c289_n1", "D": 100, "euler_step": 1.0, "fps": 80, "ignore_top_k_channels": 0, "method": None, **_SHARED},
    {"name": "convnext_shared_D100_ES1_c289_n1_ignore1", "D": 100, "euler_step": 1.0, "fps": 80, "ignore_top_k_channels": 1, "method": None, **_SHARED},
    # RK4 probe at refined step (vs Euler D100 ES=0.01).
    {"name": "convnext_shared_D100_ES0.01_RK4_c289_n1", "D": 100, "euler_step": 0.01, "fps": 80, "ignore_top_k_channels": 0, "method": "RK4", **_SHARED},
    {"name": "convnext_shared_D100_ES0.01_RK4_c289_n1_ignore1", "D": 100, "euler_step": 0.01, "fps": 80, "ignore_top_k_channels": 1, "method": "RK4", **_SHARED},
]

# Interpoled: bilinear θ first (swin → … → convnext), then RK4 probes,
# then plain θ last (swin → … → convnext).
_FM_INTERP_SPECS: list[tuple[str, dict]] = [
    ("baseline_R1_ES1_c289_n1", {"repeats": 1, "euler_step": 1.0, "fps": 1, "ignore_top_k_channels": 0}),
    ("baseline_R1_ES1_c289_n1_ignore1", {"repeats": 1, "euler_step": 1.0, "fps": 1, "ignore_top_k_channels": 1}),
    ("R100_ES0.01_c289_n1", {"repeats": 100, "euler_step": 0.01, "fps": 80, "ignore_top_k_channels": 0}),
    ("R100_ES0.01_c289_n1_ignore1", {"repeats": 100, "euler_step": 0.01, "fps": 80, "ignore_top_k_channels": 1}),
    ("R100_ES0.1_c289_n1", {"repeats": 100, "euler_step": 0.1, "fps": 80, "ignore_top_k_channels": 0}),
    ("R100_ES0.1_c289_n1_ignore1", {"repeats": 100, "euler_step": 0.1, "fps": 80, "ignore_top_k_channels": 1}),
    ("R100_ES1_c289_n1", {"repeats": 100, "euler_step": 1.0, "fps": 80, "ignore_top_k_channels": 0}),
    ("R100_ES1_c289_n1_ignore1", {"repeats": 100, "euler_step": 1.0, "fps": 80, "ignore_top_k_channels": 1}),
]
# Bilinear weight interpolation (θ=(1-α)θ_k+αθ_{k+1}), same grids as the
# interpolation scripts. Sweep order: swin first → convnext last.
_FM_BILINEAR_MODELS = ("swin", "resnet101", "resnet50", "convnext_droppath0", "convnext")
_FM_PLAIN_MODELS = ("swin", "resnet101", "resnet50", "convnext_droppath0", "convnext")
_FM_BILINEAR_SPECS: list[tuple[str, dict]] = [
    ("R100_ES0.01_bilinear_c289_n1", {"repeats": 100, "euler_step": 0.01, "fps": 80, "ignore_top_k_channels": 0}),
    ("R100_ES0.01_bilinear_c289_n1_ignore1", {"repeats": 100, "euler_step": 0.01, "fps": 80, "ignore_top_k_channels": 1}),
    ("R100_ES0.1_bilinear_c289_n1", {"repeats": 100, "euler_step": 0.1, "fps": 80, "ignore_top_k_channels": 0}),
    ("R100_ES0.1_bilinear_c289_n1_ignore1", {"repeats": 100, "euler_step": 0.1, "fps": 80, "ignore_top_k_channels": 1}),
    ("R100_ES1_bilinear_c289_n1", {"repeats": 100, "euler_step": 1.0, "fps": 80, "ignore_top_k_channels": 0}),
    ("R100_ES1_bilinear_c289_n1_ignore1", {"repeats": 100, "euler_step": 1.0, "fps": 80, "ignore_top_k_channels": 1}),
]
INTERPOLED_EXPERIMENTS: list[dict] = [
    {
        "model": model,
        "name": f"{model}_{suffix}",
        "class_id": 289,
        "max_images": 1,
        "batch_size": 1,
        "weight_interpolation": "bilinear",
        **kw,
    }
    for model in _FM_BILINEAR_MODELS
    for suffix, kw in _FM_BILINEAR_SPECS
]
# RK4 probe: R100 ES=0.01 on both ConvNeXt checkpoints (± ignore), bilinear then plain.
INTERPOLED_EXPERIMENTS += [
    {
        "model": model,
        "name": f"{model}_R100_ES0.01_RK4{wi_sfx}_c289_n1{ign_sfx}",
        "repeats": 100,
        "euler_step": 0.01,
        "method": "RK4",
        "class_id": 289,
        "max_images": 1,
        "batch_size": 1,
        "fps": 80,
        "ignore_top_k_channels": ign,
        **({"weight_interpolation": "bilinear"} if wi == "bilinear" else {}),
    }
    for model in ("convnext_droppath0", "convnext")
    for wi, wi_sfx in (("bilinear", "_bilinear"), ("plain", ""))
    for ign_sfx, ign in (("", 0), ("_ignore1", 1))
]
# Plain θ last (swin → … → convnext).
INTERPOLED_EXPERIMENTS += [
    {
        "model": model,
        "name": f"{model}_{suffix}",
        "class_id": 289,
        "max_images": 1,
        "batch_size": 1,
        **kw,
    }
    for model in _FM_PLAIN_MODELS
    for suffix, kw in _FM_INTERP_SPECS
]

INTERPOLED_MODEL_KEYS = (
    "convnext_shared",
    "convnext_shared_rand",
    "convnext",
    "convnext_rand",
    "convnext_droppath0",
    "resnet50",
    "resnet101",
    "swin",
)
_CONVNEXT_KEYS = ("convnext", "convnext_rand", "convnext_droppath0")
_SHARED_MODEL_KEY = "convnext_shared"
_SHARED_RAND_MODEL_KEY = "convnext_shared_rand"
_SHARED_MODEL_KEYS = (_SHARED_MODEL_KEY, _SHARED_RAND_MODEL_KEY)


def is_shared_run(spec: dict) -> bool:
    return spec.get("model") in _SHARED_MODEL_KEYS or BACKBONE == "shared"


def out_dir_for(spec: dict) -> Path:
    """Shared residual runs stay under featureMaps/; interpoled under featureMaps_interpoled/."""
    if is_shared_run(spec):
        return OUT_DIR_SHARED
    return OUT_DIR_INTERPOLED


def build_shared_runs(experiments: list[dict]) -> list[dict]:
    """Tag shared-residual runs with ``model=convnext_shared``."""
    runs: list[dict] = []
    for exp in experiments:
        run = dict(exp)
        run["model"] = _SHARED_MODEL_KEY
        base = run.get("name") or "run"
        if not base.startswith(f"{_SHARED_MODEL_KEY}_"):
            if base.startswith("shared_"):
                base = base[len("shared_") :]
            run["name"] = f"{_SHARED_MODEL_KEY}_{base}"
        runs.append(run)
    return runs


def build_interpoled_runs(experiments: list[dict]) -> list[dict]:
    """Normalize interpoled runs that already carry a ``model`` field."""
    runs: list[dict] = []
    for exp in experiments:
        model = exp.get("model")
        if model in _SHARED_MODEL_KEYS:
            raise ValueError("Use RUNS_SHARED / build_shared_runs / RUNS_RANDOM_INIT for shared")
        if model not in INTERPOLED_MODEL_KEYS:
            raise ValueError(f"Unknown interpoled model {model!r}; expected one of {INTERPOLED_MODEL_KEYS}")
        run = dict(exp)
        base = run.get("name") or "run"
        if not base.startswith(f"{model}_"):
            run["name"] = f"{model}_{base}"
        runs.append(run)
    return runs


# Random-init probes:
#   1. SHARED  — DeltaConvNext, one residual × D (D=9 ↔ R1, D=900 ↔ R100)
#   2. NON-shared plain
#   3. NON-shared bilinear
# Dedicated model keys so they coexist with pretrained checkpoints in one job.
# Bilinear only applies to non-shared (distinct θ_k); never to shared.
_FM_RAND = {
    "class_id": 289,
    "max_images": 1,
    "batch_size": 1,
    "random_init": True,
    "fps": 1,
    "ignore_top_k_channels": 0,
    "method": None,
}
_RAND_NONSHARED = {"model": "convnext_rand", **_FM_RAND}
_RAND_SHARED = {"model": _SHARED_RAND_MODEL_KEY, **_FM_RAND}
RUNS_RANDOM_INIT: list[dict] = [
    # --- R1 / D=9 ---
    {
        **_RAND_SHARED,
        "name": "convnext_shared_rand_D9_ES1_ls1_c289_n1",
        "D": 9,
        "euler_step": 1.0,
    },
    {
        **_RAND_NONSHARED,
        "name": "convnext_rand_nonshared_R1_ES1_ls1_c289_n1",
        "repeats": 1,
        "euler_step": 1.0,
        "weight_interpolation": "plain",
    },
    {
        **_RAND_NONSHARED,
        "name": "convnext_rand_nonshared_R1_ES1_bilinear_ls1_c289_n1",
        "repeats": 1,
        "euler_step": 1.0,
        "weight_interpolation": "bilinear",
    },
    # --- R100 / D=900 (9×100 steps) ---
    {
        **_RAND_SHARED,
        "name": "convnext_shared_rand_D900_ES1_ls1_c289_n1",
        "D": 900,
        "euler_step": 1.0,
        "fps": 80,
    },
    {
        **_RAND_NONSHARED,
        "name": "convnext_rand_nonshared_R100_ES1_ls1_c289_n1",
        "repeats": 100,
        "euler_step": 1.0,
        "fps": 80,
        "weight_interpolation": "plain",
    },
    {
        **_RAND_NONSHARED,
        "name": "convnext_rand_nonshared_R100_ES1_bilinear_ls1_c289_n1",
        "repeats": 100,
        "euler_step": 1.0,
        "fps": 80,
        "weight_interpolation": "bilinear",
    },
]

if BACKBONE == "random_init":
    OUT_DIR = OUT_DIR_INTERPOLED
    RUNS = list(RUNS_RANDOM_INIT)
elif BACKBONE == "shared":
    OUT_DIR = OUT_DIR_SHARED
    RUNS = build_shared_runs(RUNS_SHARED)
elif BACKBONE == "interpoled":
    OUT_DIR = OUT_DIR_INTERPOLED
    # Random-init probes first, then pretrained shared, then interpoled sweeps.
    RUNS = (
        list(RUNS_RANDOM_INIT)
        + build_shared_runs(RUNS_SHARED)
        + build_interpoled_runs(INTERPOLED_EXPERIMENTS)
    )
else:
    raise ValueError(
        f"Unknown BACKBONE={BACKBONE!r}; use 'interpoled', 'random_init', or 'shared'"
    )

CHANNELS: list[int] | None = None
N_AUTO_CHANNELS = 3
# Default when a RUNS entry omits ignore_top_k_channels. 0 disables.
IGNORE_TOP_K_CHANNELS = 1
VIDEO_MAPS = [
    "h",
    "x",
    "cos_h",
    "norm_h",
    "l2_h",
    "h_CH",
    "x_CH",
    "scatter_ch_x",
    "scatter_ch_h",
    "scatter_h",
    "scatter_ch_x_means",
    "scatter_ch_h_means",
    "scatter_h_means",
]

CMAP_X = "viridis"
CMAP_H = "RdBu_r"
CMAP_COS = "magma"
CMAP_NORM = "magma"
SHARED_SCALE = True
# Persist mean maps so later runs can add figures without re-integrating.
SAVE_TENSORS = True
# Skip writing an artifact when a non-empty file already exists (use --force to overwrite).
SKIP_EXISTING_FIGURES = True
FORCE_RECOMPUTE = False
GRID_MAX_FRAMES = 36
SCATTER_MAX_POINTS = 8000
# None = all channels in static scatter; int = that channel only.
SCATTER_CHANNEL: int | None = None
# After all RUNS, write a combined residual scatter coloured/legended by D.
SCATTER_OVERLAY_BY_D = True
# Animation: sample N features once, track the same indices across depth.
SCATTER_ANIM_MAX_POINTS = 8000
SCATTER_ANIM_SEED = 0
# Cap how many depths a static overlay draws (D=900 × all features OOMs hard).
SCATTER_STATIC_MAX_DEPTHS = 48
# turbo: high local contrast (nearby channels look distinct); better than viridis here.
SCATTER_ANIM_CMAP = "turbo"
# Spaghetti: trajectories d ↦ value for a fixed feature sample (colour = channel).
SPAGHETTI_MAX_LINES = 200
SPAGHETTI_SEED = 0
# Delete PNG frame dirs after the video is written (saves a lot of disk).
KEEP_FRAMES = True
MEAN_MAPS_NAME = "mean_maps.pt"
FIGSIZE, DPI = (4.4, 4.2), 100
# C×H frames: one square pixel block per (H, channel) cell — zoom to see vertical channels.
CH_PX_PER_CELL = 8
CH_DPI = 100
CH_TITLE_PX = 56  # 3-line title: run name / formula / depth
CH_LEFT_PX = 44  # room for H tick labels / ylabel
CH_XLABEL_PX = 52  # channel ticks + "channel" under the map
CH_CBAR_GAP_PX = 40  # gap for colorbar tick labels (drawn above the strip)
CH_CBAR_PX = 18  # colorbar strip height
CH_BOTTOM_PAD_PX = 10
CH_GRID_PX_PER_CELL = 3  # smaller cells for multi-panel grids

KIND_TITLES = {
    "h": "h_d = block(x_d) - x_d",
    "x": "x_d",
    "cos_h": "ω = arccos(cos(h_d[i,j], h_{d+1}[i,j])) / ES  [rad/t]",
    "cos_x": "1 - cos(x_d[i,j], x_{d+1}[i,j])",
    "norm_h": "||h_d[:,i,j]||",
    "l2_h": "||h_d[:,i,j] - h_{d+1}[:,i,j]||",
    "h_CH": "h_d C×H slice at W//2",
    "x_CH": "x_d C×H slice at W//2",
    "scatter_ch_x": "state vs channel: C → x_d",
    "scatter_ch_h": "field vs channel: C → h_d (= Δx_d / ES)",
    "scatter_h": "residual field: x_d → h_d (= Δx_d / ES)",
    "scatter_ch_x_means": "channel means: C → mean(x_d)",
    "scatter_ch_h_means": "channel means: C → mean(h_d)",
    "scatter_h_means": "channel means: mean(x_d) → mean(h_d)",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Truncated ffmpeg kills leave ~48-byte ftyp stubs; treat those as missing.
_MIN_VIDEO_BYTES = 1024


def _exists_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _exists_video(path: Path) -> bool:
    return path.is_file() and path.stat().st_size >= _MIN_VIDEO_BYTES


def should_write(path: Path) -> bool:
    """False when SKIP_EXISTING_FIGURES and a non-empty file already exists (unless FORCE)."""
    if FORCE_RECOMPUTE or not SKIP_EXISTING_FIGURES:
        return True
    if path.suffix.lower() == ".mp4":
        return not _exists_video(path)
    return not _exists_nonempty(path)


def video_products_ready(out_stem: Path, n_frames: int | None = None) -> bool:
    """True if the mp4 ``write_video`` would produce already exists."""
    del n_frames  # kept for call-site compatibility
    if FORCE_RECOMPUTE or not SKIP_EXISTING_FIGURES:
        return False
    return _exists_video(out_stem.with_suffix(".mp4"))


def write_static(path: Path, fn) -> None:
    """Run ``fn()`` only if ``path`` should be written; otherwise print a skip line."""
    if not should_write(path):
        print(f"  skip existing {path.name}")
        return
    print(f"  writing {path.name} …", flush=True)
    fn()
    plt.close("all")
    gc.collect()


# --------------------------------------------------------------------------- helpers
def run_folder_name(spec: dict) -> str:
    if spec.get("name"):
        return spec["name"]
    method = (spec.get("method") or "RK1").lower()
    wi = spec.get("weight_interpolation", "plain")
    if "D" in spec:
        parts = [f"D{spec['D']}_{method}"]
    elif "repeats" in spec:
        parts = [f"R{spec['repeats']}_ES{spec['euler_step']:g}"]
    elif "blocks" in spec:
        parts = [f"B{'-'.join(map(str, spec['blocks']))}_ES{spec['euler_step']:g}"]
    else:
        raise ValueError("run needs name, or D / blocks / repeats")
    if wi != "plain":
        parts.append(wi)
    class_id = spec.get("class_id", CLASS_ID)
    max_images = spec.get("max_images", MAX_IMAGES_PER_CLASS)
    batch_size = spec.get("batch_size", BATCH_SIZE)
    if spec.get("image_indices") is not None:
        parts.append(f"n{len(spec['image_indices'])}")
    elif class_id is not None:
        parts.append(f"c{class_id}")
        if max_images is not None:
            parts.append(f"n{max_images}")
    if batch_size is not None:
        parts.append(f"bs{batch_size}")
    return "_".join(parts).replace("/", "div")


def resolve_blocks(spec: dict, n_blocks: int) -> list[int]:
    """``blocks`` XOR ``repeats`` → stage-3 residual index schedule (like interpolation)."""
    if "blocks" in spec and "repeats" in spec:
        raise ValueError("Specify either blocks or repeats, not both")
    if "blocks" in spec:
        blocks = list(spec["blocks"])
    else:
        repeats = spec.get("repeats")
        if repeats is None:
            raise ValueError("Interpoled run needs blocks or repeats")
        blocks = [i for i in range(n_blocks) for _ in range(int(repeats))]
    bad = [i for i in blocks if not 0 <= i < n_blocks]
    if bad:
        raise ValueError(f"block indices {bad} out of range [0, {n_blocks})")
    if not blocks:
        raise ValueError("empty block schedule")
    return blocks


@torch.no_grad()
def set_layerscale_value(model: nn.Module, value: float) -> int:
    """Fill every LayerScale γ with ``value``. Returns how many modules were updated."""
    n = 0
    for m in model.modules():
        if isinstance(m, LayerScale):
            m.gamma.fill_(value)
            n += 1
    return n


def load_shared_convnext(checkpoint: Path | None = None) -> DeltaConvNext:
    model = DeltaConvNext(useDeltas=False)
    model.rewire()
    if checkpoint is None:
        n_ls = set_layerscale_value(model, RANDOM_INIT_LAYERSCALE)
        print(
            f"Random-init shared ConvNeXt (seed={RANDOM_INIT_SEED}, "
            f"LayerScale={RANDOM_INIT_LAYERSCALE:g} on {n_ls} modules)"
        )
        return model
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    stale = [k for k in missing if not k.startswith("tail.")]
    if stale or unexpected:
        raise RuntimeError(f"missing={stale} unexpected={unexpected}")
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    print(f"Loaded shared {checkpoint}" + (f" (epoch {epoch})" if epoch is not None else ""))
    return model


def load_interpoled_convnext(
    checkpoint: Path | None = None,
    *,
    share_stage3: bool = False,
    share_src: int = 5,
) -> InterpoledConvNextV1:
    model = InterpoledConvNextV1()
    if checkpoint is None:
        if share_stage3:
            share_stage3_weights(model, src=share_src)
        n_ls = set_layerscale_value(model, RANDOM_INIT_LAYERSCALE)
        tied = (
            f", stage-3 tied to block {share_src}" if share_stage3 else ""
        )
        print(
            f"Random-init interpoled ConvNeXt (seed={RANDOM_INIT_SEED}{tied}, "
            f"LayerScale={RANDOM_INIT_LAYERSCALE:g} on {n_ls} modules)"
        )
        return model
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    print(f"Loaded interpoled convnext {checkpoint}" + (f" (epoch {epoch})" if epoch is not None else ""))
    if share_stage3:
        share_stage3_weights(model, src=share_src)
        print(f"  tied stage-3 residual weights to block {share_src}")
    return model


@torch.no_grad()
def share_stage3_weights(model: InterpoledConvNextV1, src: int = 5) -> None:
    """Copy stage-3 residual ``src`` weights into all 9 residual slots (1×9 shared)."""
    n = int(model.depths[2])
    if not 0 <= src < n:
        raise ValueError(f"share_src={src} out of range [0, {n})")
    src_state = model.stage3[src].state_dict()
    for i in range(n):
        if i != src:
            model.stage3[i].load_state_dict(src_state)


def _seed_random_init() -> None:
    torch.manual_seed(RANDOM_INIT_SEED)
    np.random.seed(RANDOM_INIT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_INIT_SEED)


def load_interpoled_model(model_key: str, *, random_init: bool = False) -> nn.Module:
    if random_init or model_key in ("convnext_rand", _SHARED_RAND_MODEL_KEY):
        _seed_random_init()
    if model_key == _SHARED_MODEL_KEY:
        if random_init:
            return load_shared_convnext(None)
        if not SHARED_CHECKPOINT.is_file():
            raise FileNotFoundError(f"Shared ConvNeXt checkpoint not found: {SHARED_CHECKPOINT}")
        return load_shared_convnext(SHARED_CHECKPOINT)
    if model_key == _SHARED_RAND_MODEL_KEY:
        return load_shared_convnext(None)
    if model_key == "convnext":
        if random_init:
            return load_interpoled_convnext(None)
        if not INTERPOLED_CONVNEXT_CHECKPOINT.is_file():
            raise FileNotFoundError(f"ConvNeXt checkpoint not found: {INTERPOLED_CONVNEXT_CHECKPOINT}")
        return load_interpoled_convnext(INTERPOLED_CONVNEXT_CHECKPOINT)
    if model_key == "convnext_rand":
        return load_interpoled_convnext(None)
    if model_key == "convnext_droppath0":
        if random_init:
            _seed_random_init()
            return load_interpoled_convnext(None)
        if not INTERPOLED_CONVNEXT_DROPPATH0_CHECKPOINT.is_file():
            raise FileNotFoundError(
                f"ConvNeXt droppath0 checkpoint not found: {INTERPOLED_CONVNEXT_DROPPATH0_CHECKPOINT}"
            )
        return load_interpoled_convnext(INTERPOLED_CONVNEXT_DROPPATH0_CHECKPOINT)
    if model_key == "resnet50":
        model = InterpoledResNet50(weights=None if random_init else "DEFAULT")
        print(f"Loaded torchvision ResNet-50 {'random' if random_init else 'DEFAULT'}, n_blocks={model.n_blocks}")
        return model
    if model_key == "resnet101":
        model = InterpoledResNet101(weights=None if random_init else "DEFAULT")
        print(f"Loaded torchvision ResNet-101 {'random' if random_init else 'DEFAULT'}, n_blocks={model.n_blocks}")
        return model
    if model_key == "swin":
        model = InterpoledSwinT(weights=None if random_init else "DEFAULT")
        print(f"Loaded torchvision Swin-T {'random' if random_init else 'DEFAULT'}, n_blocks={model.n_blocks}")
        return model
    raise ValueError(f"Unknown model_key={model_key!r}")


class _NHWCBlockAsNCHW(nn.Module):
    """Run an NHWC residual block with NCHW tensors (permute in/out)."""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x.permute(0, 2, 3, 1).contiguous())
        return y.permute(0, 3, 1, 2).contiguous()


class _LerpedBlock(nn.Module):
    """Residual module with bilinear weights: θ = (1-α)θ_a + α θ_b."""

    def __init__(self, block_a: nn.Module, block_b: nn.Module, alpha: float):
        super().__init__()
        self.block_a = block_a
        self.block_b = block_b
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return call_lerped(self.block_a, self.block_b, self.alpha, x)


class _LerpedNHWCAsNCHW(nn.Module):
    """Lerped NHWC residual with NCHW tensors (permute in/out)."""

    def __init__(self, block_a: nn.Module, block_b: nn.Module, alpha: float):
        super().__init__()
        self.block_a = block_a
        self.block_b = block_b
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = call_lerped(
            self.block_a,
            self.block_b,
            self.alpha,
            x.permute(0, 2, 3, 1).contiguous(),
        )
        return y.permute(0, 3, 1, 2).contiguous()


def stage3_n_blocks(model_key: str, model: nn.Module) -> int:
    if model_key in _CONVNEXT_KEYS:
        return int(model.depths[2])
    if model_key in ("resnet50", "resnet101"):
        return int(model.n_blocks)
    if model_key == "swin":
        return int(model.n_blocks)
    raise ValueError(model_key)


def stage3_raw_blocks(model_key: str, model: nn.Module) -> list[nn.Module]:
    """Native stage-3 residual modules (indices 0..n_blocks-1), NHWC for Swin."""
    n = stage3_n_blocks(model_key, model)
    if model_key in _CONVNEXT_KEYS:
        return [model.stage3[i] for i in range(n)]
    if model_key in ("resnet50", "resnet101"):
        return [model.backbone.layer3[i + 1] for i in range(n)]
    if model_key == "swin":
        stage3 = model.backbone.features[InterpoledSwinT.STAGE3_INDEX]
        return [stage3[i] for i in range(n)]
    raise ValueError(model_key)


def enter_stage3(model_key: str, model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Map images → stage-3 feature state in NCHW."""
    if model_key in _CONVNEXT_KEYS:
        return model.stage2(model.stage1(model.stem(batch)))
    if model_key in ("resnet50", "resnet101"):
        b = model.backbone
        x = b.conv1(batch)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        return b.layer3[0](x)
    if model_key == "swin":
        x = batch
        for i in range(InterpoledSwinT.STAGE3_INDEX):
            x = model.backbone.features[i](x)
        # features output is NHWC → NCHW for shared analysis code.
        return x.permute(0, 3, 1, 2).contiguous()
    raise ValueError(model_key)


def _wrap_stage3_block(model_key: str, block: nn.Module) -> nn.Module:
    if model_key == "swin":
        return _NHWCBlockAsNCHW(block)
    return block


def _wrap_lerped_stage3(
    model_key: str, block_a: nn.Module, block_b: nn.Module, alpha: float
) -> nn.Module:
    if model_key == "swin":
        return _LerpedNHWCAsNCHW(block_a, block_b, alpha)
    return _LerpedBlock(block_a, block_b, alpha)


def stage3_field_blocks(
    model_key: str,
    model: nn.Module,
    blocks: list[int],
    *,
    euler_step: float = 1.0,
    weight_interpolation: str = "plain",
) -> list[nn.Module]:
    """Modules whose forward is residual ``x + f(x)``; used as ``h = block(x) - x``.

    ``plain``: one field per scheduled stage-3 index.
    ``bilinear``: uniform Euler grid on [0, n_blocks) with ``len(blocks)`` steps,
    θ(t)=(1-α)θ_k+αθ_{k+1} (same as the interpolation scripts; block *indices*
    are ignored except for the step count).
    """
    if weight_interpolation not in ("plain", "bilinear"):
        raise ValueError(
            f"weight_interpolation must be 'plain' or 'bilinear', got {weight_interpolation!r}"
        )
    raw = stage3_raw_blocks(model_key, model)
    n_blocks = len(raw)
    if weight_interpolation == "plain":
        return [_wrap_stage3_block(model_key, raw[i]) for i in blocks]

    fields: list[nn.Module] = []
    for k, alpha in bilinear_time_steps(n_blocks, len(blocks), euler_step):
        if alpha == 0.0 or k >= n_blocks - 1:
            fields.append(_wrap_stage3_block(model_key, raw[k]))
        else:
            fields.append(_wrap_lerped_stage3(model_key, raw[k], raw[k + 1], alpha))
    return fields


def denormalize(img: torch.Tensor) -> torch.Tensor:
    mean = img.new_tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = img.new_tensor(IMAGENET_STD).view(-1, 1, 1)
    return (img * std + mean).clamp(0, 1)


def rk_step(f, x, h, method, k1=None):
    if k1 is None:
        k1 = f(x)
    if method in (None, "RK1", "EULER"):
        return x + h * k1
    if method == "RK2":
        k2 = f(x + h * k1)
        return x + h / 2 * (k1 + k2)
    if method == "RK3":
        k2 = f(x + h / 2 * k1)
        k3 = f(x - h * k1 + 2 * h * k2)
        return x + h / 6 * (k1 + 4 * k2 + k3)
    if method == "RK4":
        k2 = f(x + h / 2 * k1)
        k3 = f(x + h / 2 * k2)
        k4 = f(x + h * k3)
        return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    raise ValueError(f"Unknown integration method: {method!r}")


def participation_ratio(singular_values: torch.Tensor) -> float:
    """PR = (sum s_i^2)^2 / sum s_i^4 over singular values."""
    s2 = singular_values.double().clamp_min(0.0).square()
    num = s2.sum().square()
    den = s2.square().sum().clamp_min(1e-30)
    return (num / den).item()


def svd_participation_ratio(matrix: torch.Tensor) -> float:
    """Thin SVD participation ratio of a 2D matrix."""
    if matrix.numel() == 0 or min(matrix.shape) == 0:
        return float("nan")
    # float32 SVD is enough for a scalar PR; keep matrix on CPU.
    m = matrix.float().cpu()
    # Divergent ODE steps (large ES) can produce Inf/NaN; SVD refuses those.
    if not torch.isfinite(m).all():
        return float("nan")
    try:
        _, s, _ = torch.linalg.svd(m, full_matrices=False)
    except RuntimeError:
        return float("nan")
    return participation_ratio(s)


def _r1_group_boundaries(block_schedule: list[int], D: int) -> tuple[set[int], set[int]]:
    """Run-length groups of equal schedule ids → step indices that start / end a group.

    Within each group the multi-step path applies the same residual R times while a
    parallel R1 path applies it once; after the group, R1 continues from its own state.
    """
    if len(block_schedule) != D:
        raise ValueError(f"block_schedule length {len(block_schedule)} != D={D}")
    starts: set[int] = set()
    ends: set[int] = set()
    d = 0
    while d < D:
        starts.add(d)
        j = d + 1
        while j < D and block_schedule[j] == block_schedule[d]:
            j += 1
        ends.add(j - 1)  # last micro-step index of this group
        d = j
    return starts, ends


@torch.no_grad()
def trajectory_stats(
    enter_fn,
    field_blocks,
    indices,
    *,
    euler_step,
    method,
    batch_size,
    dataset,
    block_schedule: list[int] | None = None,
):
    """Integrate a sequence of residual fields and collect per-image dynamics + mean maps.

    ``enter_fn(batch)`` returns the stage-3 state in NCHW. ``field_blocks[d]`` is the
    residual module at step d (``forward`` returns ``x + f(x)``). Shared mode passes
    the same block D times; interpoled mode passes the scheduled stage-3 residuals.

    Naming: h := block(x) - x = Δx / ES (ODE field). Discrete step: Δx = ES · h,
    i.e. x ← x + ES · h. Stored mean_h is this h (not Δx).

    ``block_schedule`` (length D) groups consecutive equal ids into residual "blocks".
    A parallel R1 trajectory takes one step per group; after each multi-step micro-step
    we record ||x^{(n)} - x_{R1}|| (and the relative distance).
    """
    method = method.upper() if isinstance(method, str) else method
    D = len(field_blocks)
    if D < 1:
        raise ValueError("field_blocks must be non-empty")
    n = len(indices)
    w = 1.0 / n
    es = float(euler_step)
    mean_x = mean_h = cos_map_h = cos_map_x = norm_map_h = l2_map_h = None
    schedule = list(block_schedule) if block_schedule is not None else [0] * D
    group_starts, group_ends = _r1_group_boundaries(schedule, D)

    # Per-image series (filled in index order).
    norm_h_i = torch.zeros(n, D + 1, dtype=torch.float64)
    norm_x_i = torch.zeros(n, D + 1, dtype=torch.float64)
    cos_flat_i = torch.zeros(n, D, dtype=torch.float64)  # 1 - cos(h_d, h_{d+1}) flat
    omega_spatial_i = torch.zeros(n, D, dtype=torch.float64)  # mean_{i,j} arccos/ES
    omega_channel_i = torch.zeros(n, D, dtype=torch.float64)  # mean_c arccos/ES on HW
    cos_x_flat_i = torch.zeros(n, D, dtype=torch.float64)
    cos_x_spatial_i = torch.zeros(n, D, dtype=torch.float64)
    omega_i = torch.zeros(n, D, dtype=torch.float64)  # arccos(cos)/ES flat
    align_h0_i = torch.zeros(n, D + 1, dtype=torch.float64)  # cos(h_0, h_d)
    dist_x_r1_i = torch.zeros(n, D, dtype=torch.float64)  # ||x_n - x_R1|| after step d
    dist_x_r1_rel_i = torch.zeros(n, D, dtype=torch.float64)  # / ||x_R1||
    rect_L_i = torch.zeros(n, dtype=torch.float64)
    rect_N_i = torch.zeros(n, dtype=torch.float64)
    rect_R_i = torch.zeros(n, dtype=torch.float64)
    pr_depth_i = torch.zeros(n, dtype=torch.float64)
    pr_spatial_i = torch.zeros(n, dtype=torch.float64)

    def field_at(block):
        def f(y):
            return block(y) - y

        return f

    def load_batch(idxs: list[int]) -> torch.Tensor:
        return torch.stack([dataset[i][0] for i in idxs])

    starts = range(0, n, batch_size)
    desc = f"D={D} h={es:.4g} m={method or 'RK1'}"
    for start in tqdm(starts, desc=desc, leave=True):
        batch_idx = indices[start : start + batch_size]
        batch = load_batch(batch_idx).to(device, non_blocking=True)
        bsz = batch.shape[0]
        x = enter_fn(batch)
        if mean_x is None:
            mean_x = torch.zeros((D + 1, *x.shape[1:]), device=device)
            mean_h = torch.zeros_like(mean_x)
            cos_map_h = torch.zeros((D, *x.shape[2:]), device=device)
            cos_map_x = torch.zeros_like(cos_map_h)
            norm_map_h = torch.zeros((D + 1, *x.shape[2:]), device=device)
            l2_map_h = torch.zeros((D, *x.shape[2:]), device=device)

        # Per-image residual trajectory for SVD / rectitude.
        f0 = field_at(field_blocks[0])
        h = f0(x)
        x0 = x.detach()
        x_r1 = x  # parallel R1 path (one step per residual group)
        x_r1_ref = None  # R1 state after one step of the current group
        h0_flat = h.flatten(1).detach()
        H_rows: list[torch.Tensor] = []
        path_len = torch.zeros(bsz, dtype=torch.float64, device=device)

        for d in range(D + 1):
            # At state d, evaluate the field of the block that acts at this index
            # (last block again after the final step, matching shared-mode storage).
            block = field_blocks[min(d, D - 1)]
            f = field_at(block)
            if d > 0:
                h = f(x)

            mean_x[d] += w * x.sum(0)
            mean_h[d] += w * h.sum(0)
            norm_map_h[d] += w * h.norm(dim=1).sum(0)

            h_flat = h.flatten(1)
            x_flat = x.flatten(1)
            n_h = h_flat.norm(dim=1)
            n_x = x_flat.norm(dim=1)
            norm_h_i[start : start + bsz, d] = n_h.double().cpu()
            norm_x_i[start : start + bsz, d] = n_x.double().cpu()
            align_h0_i[start : start + bsz, d] = F.cosine_similarity(h0_flat, h_flat, dim=1).double().cpu()
            H_rows.append(h_flat.detach().cpu())
            # Path length L = sum ||Δx_d|| = ES * sum ||h_d||.
            if d < D:
                path_len = path_len + es * n_h.double()

            if d == D:
                break

            # Start of a residual group: R1 takes one step from its own input.
            if d in group_starts:
                f_r1 = field_at(field_blocks[d])
                h_r1 = f_r1(x_r1)
                x_r1_ref = rk_step(f_r1, x_r1, euler_step, method, k1=h_r1)

            # Transition d → d+1 uses field_blocks[d] (h already matches for Euler).
            f_step = field_at(field_blocks[d])
            x_next = rk_step(f_step, x, euler_step, method, k1=h)
            h_next = field_at(field_blocks[min(d + 1, D - 1)])(x_next)
            h_next_flat = h_next.flatten(1)
            x_next_flat = x_next.flatten(1)

            # Distance of multi-step state after n steps of this block vs R1's one-step x.
            assert x_r1_ref is not None
            dist = (x_next - x_r1_ref).flatten(1).norm(dim=1)
            rel = dist / x_r1_ref.flatten(1).norm(dim=1).clamp_min(1e-30)
            dist_x_r1_i[start : start + bsz, d] = dist.double().cpu()
            dist_x_r1_rel_i[start : start + bsz, d] = rel.double().cpu()
            if d in group_ends:
                x_r1 = x_r1_ref

            cos_hh = F.cosine_similarity(h_flat, h_next_flat, dim=1).clamp(-1.0, 1.0)
            cos_flat_i[start : start + bsz, d] = (1.0 - cos_hh).double().cpu()
            omega_i[start : start + bsz, d] = torch.arccos(cos_hh).double().cpu() / max(es, 1e-12)
            cos_x_flat_i[start : start + bsz, d] = (
                1.0 - F.cosine_similarity(x_flat, x_next_flat, dim=1)
            ).double().cpu()

            loc_cos_h = F.cosine_similarity(h, h_next, dim=1).clamp(-1.0, 1.0)
            loc_omega_h = torch.arccos(loc_cos_h) / max(es, 1e-12)
            # Per-channel ω: each channel is an HW vector.
            cos_ch = F.cosine_similarity(h.flatten(2), h_next.flatten(2), dim=2).clamp(-1.0, 1.0)
            omega_ch = torch.arccos(cos_ch) / max(es, 1e-12)
            loc_x = 1.0 - F.cosine_similarity(x, x_next, dim=1)
            cos_map_h[d] += w * loc_omega_h.sum(0)
            cos_map_x[d] += w * loc_x.sum(0)
            omega_spatial_i[start : start + bsz, d] = loc_omega_h.mean(dim=(1, 2)).double().cpu()
            omega_channel_i[start : start + bsz, d] = omega_ch.mean(dim=1).double().cpu()
            cos_x_spatial_i[start : start + bsz, d] = loc_x.mean(dim=(1, 2)).double().cpu()
            l2_map_h[d] += w * (h - h_next).norm(dim=1).sum(0)
            x, h = x_next, h_next

        # Rectitude: L = ES * sum ||h_d||, N = ||x_D - x_0||, R = N/L.
        disp = (x - x0).flatten(1).norm(dim=1).double()
        rect_L_i[start : start + bsz] = path_len.cpu()
        rect_N_i[start : start + bsz] = disp.cpu()
        rect_R_i[start : start + bsz] = (disp / path_len.clamp_min(1e-30)).cpu()

        # SVD PR on H_i shaped (D, dim) using h_0..h_{D-1}, and mid-step (H*W, C).
        H_stack = torch.stack(H_rows[:D], dim=1)  # [B, D, dim]
        mid = D // 2
        if not torch.isfinite(H_stack).all():
            print(
                f"  WARNING: non-finite residuals at batch start={start} "
                f"(ES={es:g}, D={D}); PR will be NaN for those images"
            )
        for bi in range(bsz):
            pr_depth_i[start + bi] = svd_participation_ratio(H_stack[bi])
            h_mid = H_rows[mid][bi].reshape(x.shape[1], -1).T  # (H*W, C)
            pr_spatial_i[start + bi] = svd_participation_ratio(h_mid)

        del batch, x, h, x0, x_r1, x_r1_ref, H_rows, H_stack
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def mean_std(arr: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        return arr.mean(0).numpy(), arr.std(0, unbiased=False).numpy()

    depths = torch.arange(D + 1)
    norm_h_mean, norm_h_std = mean_std(norm_h_i)
    norm_x_mean, norm_x_std = mean_std(norm_x_i)
    # h = Δx/ES = stored residual; plot ||h|| and ||h||/||x|| (no extra /ES).
    norm_h_over_x_i = norm_h_i / norm_x_i.clamp_min(1e-30)
    norm_h_over_x_mean, norm_h_over_x_std = mean_std(norm_h_over_x_i)

    cos_flat_mean, cos_flat_std = mean_std(cos_flat_i)
    omega_spatial_mean, omega_spatial_std = mean_std(omega_spatial_i)
    omega_channel_mean, omega_channel_std = mean_std(omega_channel_i)
    omega_mean, omega_std = mean_std(omega_i)
    align_mean, align_std = mean_std(align_h0_i)
    dist_r1_mean, dist_r1_std = mean_std(dist_x_r1_i)
    dist_r1_rel_mean, dist_r1_rel_std = mean_std(dist_x_r1_rel_i)

    # Correlation between flat and spatial ω series (on the mean curves).
    if D >= 2 and omega_mean.std() > 0 and omega_spatial_mean.std() > 0:
        cos_flat_spatial_corr = float(np.corrcoef(omega_mean, omega_spatial_mean)[0, 1])
    else:
        cos_flat_spatial_corr = float("nan")

    norms = pd.DataFrame(
        {
            "d": depths.numpy(),
            "t": (depths * es).numpy(),
            "norm_h": norm_h_mean,
            "norm_h_std": norm_h_std,
            "norm_x": norm_x_mean,
            "norm_x_std": norm_x_std,
            "norm_h_over_norm_x": norm_h_over_x_mean,
            "norm_h_over_norm_x_std": norm_h_over_x_std,
            "cos_h0_hd": align_mean,
            "cos_h0_hd_std": align_std,
        }
    )

    pairs = pd.DataFrame(
        {
            "d": torch.arange(D).numpy(),
            "pair": [f"{d}->{d + 1}" for d in range(D)],
            "t": (torch.arange(D).numpy() * es),
            "cos_dist_h": cos_flat_mean,
            "cos_dist_h_std": cos_flat_std,
            "omega_h_spatial": omega_spatial_mean,
            "omega_h_spatial_std": omega_spatial_std,
            "omega_h_channel": omega_channel_mean,
            "omega_h_channel_std": omega_channel_std,
            "omega_h": omega_mean,
            "omega_h_std": omega_std,
            "cos_dist_x": mean_std(cos_x_flat_i)[0],
            "cos_dist_x_spatial": mean_std(cos_x_spatial_i)[0],
            "dist_x_r1": dist_r1_mean,
            "dist_x_r1_std": dist_r1_std,
            "dist_x_r1_rel": dist_r1_rel_mean,
            "dist_x_r1_rel_std": dist_r1_rel_std,
        }
    )

    scalars = pd.DataFrame(
        {
            "image_index": list(indices),
            "L": rect_L_i.numpy(),
            "N": rect_N_i.numpy(),
            "R": rect_R_i.numpy(),
            "PR_depth": pr_depth_i.numpy(),
            "PR_spatial_mid": pr_spatial_i.numpy(),
        }
    )

    # PR on the mean residual trajectory (D, dim) and mid-step (HW, C).
    H_mean = mean_h[:D].reshape(D, -1)
    pr_depth_mean_traj = svd_participation_ratio(H_mean)
    c, hh, ww = mean_h.shape[1:]
    pr_spatial_mean_traj = svd_participation_ratio(mean_h[D // 2].reshape(c, -1).T)

    return {
        "D": D,
        "euler_step": es,
        "method": method,
        "n_images": n,
        "mean_x": mean_x.cpu(),
        "mean_h": mean_h.cpu(),
        "cos_h": cos_map_h.cpu(),
        "cos_x": cos_map_x.cpu(),
        "norm_h": norm_map_h.cpu(),
        "l2_h": l2_map_h.cpu(),
        "norms": norms,
        "pairs": pairs,
        "scalars": scalars,
        "cos_flat_spatial_corr": cos_flat_spatial_corr,
        "rectitude_mean": float(rect_R_i.mean()),
        "rectitude_std": float(rect_R_i.std(unbiased=False)),
        "L_mean": float(rect_L_i.mean()),
        "N_mean": float(rect_N_i.mean()),
        "PR_depth_mean": float(pr_depth_i.mean()),
        "PR_depth_std": float(pr_depth_i.std(unbiased=False)),
        "PR_spatial_mean": float(pr_spatial_i.mean()),
        "PR_spatial_std": float(pr_spatial_i.std(unbiased=False)),
        "PR_depth_on_mean_traj": pr_depth_mean_traj,
        "PR_spatial_on_mean_traj": pr_spatial_mean_traj,
    }


def most_active_channels(mean_map: torch.Tensor, k: int) -> list[int]:
    total_variation = (mean_map[1:] - mean_map[:-1]).flatten(2).norm(dim=2).sum(0)
    return total_variation.argsort(descending=True)[:k].tolist()


def channel_spatial_norms(maps: torch.Tensor) -> torch.Tensor:
    """Per-channel norm: each channel is a WxH vector; aggregate over depth with L2.

    maps: [T, C, H, W] → [C], where score_c = || (||maps[t,c]||_F)_t ||_2
    """
    # [T, C] Frobenius norms over H×W, then L2 over T
    return maps.flatten(2).norm(dim=-1).norm(dim=0)


def top_norm_channels(maps: torch.Tensor, k: int) -> list[int]:
    if k <= 0:
        return []
    k = min(k, maps.shape[1])
    return channel_spatial_norms(maps).topk(k).indices.tolist()


def zero_channels(maps: torch.Tensor, channels: list[int]) -> torch.Tensor:
    if not channels:
        return maps
    out = maps.clone()
    out[:, channels] = 0
    return out


def spatial_maps_from_means(
    mean_x: torch.Tensor,
    mean_h: torch.Tensor,
    *,
    euler_step: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Rebuild H×W video tensors from (possibly channel-masked) mean maps."""
    D = mean_x.shape[0] - 1
    es = max(float(euler_step), 1e-12)
    cos_h = torch.stack(
        [
            torch.arccos(F.cosine_similarity(mean_h[d], mean_h[d + 1], dim=0).clamp(-1.0, 1.0)) / es
            for d in range(D)
        ]
    )
    cos_x = torch.stack(
        [1 - F.cosine_similarity(mean_x[d], mean_x[d + 1], dim=0) for d in range(D)]
    )
    l2_h = torch.stack([(mean_h[d] - mean_h[d + 1]).norm(dim=0) for d in range(D)])
    return {
        "cos_h": cos_h,
        "cos_x": cos_x,
        "norm_h": mean_h.norm(dim=1),
        "l2_h": l2_h,
    }


def tables_from_means(mean_x: torch.Tensor, mean_h: torch.Tensor, *, euler_step: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Norm / pair tables from mean maps (used after channel masking)."""
    D = mean_x.shape[0] - 1
    es = float(euler_step)
    depths = torch.arange(D + 1)
    norm_h = mean_h.flatten(1).norm(dim=1)
    norm_x = mean_x.flatten(1).norm(dim=1)
    align = torch.stack(
        [F.cosine_similarity(mean_h[0].flatten(), mean_h[d].flatten(), dim=0) for d in range(D + 1)]
    )
    norms = pd.DataFrame(
        {
            "d": depths.numpy(),
            "t": (depths * es).numpy(),
            "norm_h": norm_h.numpy(),
            "norm_h_std": 0.0,
            "norm_x": norm_x.numpy(),
            "norm_x_std": 0.0,
            "norm_h_over_norm_x": (norm_h / norm_x.clamp_min(1e-30)).numpy(),
            "norm_h_over_norm_x_std": 0.0,
            "cos_h0_hd": align.numpy(),
            "cos_h0_hd_std": 0.0,
        }
    )

    cos_h = []
    omega_spatial = []
    omega_channel = []
    omega = []
    cos_x = []
    cos_x_spatial = []
    for d in range(D):
        c = F.cosine_similarity(mean_h[d].flatten(), mean_h[d + 1].flatten(), dim=0).clamp(-1, 1)
        cos_h.append((1 - c).item())
        omega.append((torch.arccos(c) / max(es, 1e-12)).item())
        cos_x.append(
            (1 - F.cosine_similarity(mean_x[d].flatten(), mean_x[d + 1].flatten(), dim=0)).item()
        )
        loc_omega = torch.arccos(
            F.cosine_similarity(mean_h[d], mean_h[d + 1], dim=0).clamp(-1.0, 1.0)
        ) / max(es, 1e-12)
        ch_cos = F.cosine_similarity(
            mean_h[d].flatten(1), mean_h[d + 1].flatten(1), dim=1
        ).clamp(-1.0, 1.0)
        ch_omega = torch.arccos(ch_cos) / max(es, 1e-12)
        loc_x = 1 - F.cosine_similarity(mean_x[d], mean_x[d + 1], dim=0)
        omega_spatial.append(loc_omega.mean().item())
        omega_channel.append(ch_omega.mean().item())
        cos_x_spatial.append(loc_x.mean().item())
    pairs = pd.DataFrame(
        {
            "d": list(range(D)),
            "pair": [f"{d}->{d + 1}" for d in range(D)],
            "t": [d * es for d in range(D)],
            "cos_dist_h": cos_h,
            "cos_dist_h_std": 0.0,
            "omega_h_spatial": omega_spatial,
            "omega_h_spatial_std": 0.0,
            "omega_h_channel": omega_channel,
            "omega_h_channel_std": 0.0,
            "omega_h": omega,
            "omega_h_std": 0.0,
            "cos_dist_x": cos_x,
            "cos_dist_x_spatial": cos_x_spatial,
        }
    )
    return norms, pairs


def apply_ignore_top_k_channels(res: dict, k: int) -> list[int]:
    """Zero top-K channels (ranked on mean_h WxH norms) in mean maps; refresh video tensors.

    Per-image curve tables (norms/pairs/scalars) are left unchanged — they were computed
    on the full tensor before masking.
    """
    if k <= 0:
        res["ignored_channels"] = []
        return []
    ignored = top_norm_channels(res["mean_h"], k)
    res["mean_h"] = zero_channels(res["mean_h"], ignored)
    res["mean_x"] = zero_channels(res["mean_x"], ignored)
    res.update(spatial_maps_from_means(res["mean_x"], res["mean_h"], euler_step=res["euler_step"]))
    res["ignored_channels"] = ignored
    return ignored


def is_pair_kind(kind: str) -> bool:
    return kind in ("cos_h", "cos_x", "l2_h")


def depth_line(kind: str, d: int, last: int) -> str:
    """Third title line: dX → dX+1 (d/D) for pair maps, else dX (d/D)."""
    if is_pair_kind(kind):
        return f"d{d} → d{d + 1} ({d}/{last})"
    return f"d{d} ({d}/{last})"


def frame_heading(run_name: str, formula: str, depth: str) -> str:
    """Three-line frame title: experiment / formula / depth."""
    return f"{run_name}\n{formula}\n{depth}"


def _padded_finite_limits(
    *tensors: torch.Tensor,
    pad_frac: float = 0.02,
    default: float = 1.0,
) -> tuple[float, float]:
    """Axis limits from finite values only (large ES can yield Inf/NaN)."""
    chunks: list[torch.Tensor] = []
    for t in tensors:
        f = t[torch.isfinite(t)]
        if f.numel():
            chunks.append(f.reshape(-1))
    if not chunks:
        return -default, default
    finite = torch.cat(chunks)
    lo = float(finite.min().item())
    hi = float(finite.max().item())
    pad = pad_frac * (hi - lo) if hi > lo else default
    return lo - pad, hi + pad


def draw_style(maps: torch.Tensor, kind: str) -> dict:
    finite = maps[torch.isfinite(maps)]

    def _finite_max(default: float = 1.0) -> float:
        return float(finite.max().item()) if finite.numel() else default

    def _finite_min(default: float = 0.0) -> float:
        return float(finite.min().item()) if finite.numel() else default

    if kind.startswith("cos"):
        cmap = CMAP_COS
        vmax = _finite_max() if SHARED_SCALE else None
        return {"cmap": cmap, "vmin": 0.0, **({"vmax": vmax} if vmax is not None else {})}
    if kind in ("norm_h", "l2_h"):
        cmap = CMAP_NORM
        vmax = _finite_max() if SHARED_SCALE else None
        return {"cmap": cmap, "vmin": 0.0, **({"vmax": vmax} if vmax is not None else {})}
    signed = kind in ("h", "h_CH")
    cmap = CMAP_H if signed else CMAP_X
    if not SHARED_SCALE:
        return {"cmap": cmap}
    if signed:
        limit = float(finite.abs().max().item()) if finite.numel() else 1.0
        return {"cmap": cmap, "vmin": -limit, "vmax": limit}
    return {"cmap": cmap, "vmin": _finite_min(), "vmax": _finite_max()}


def ch_cut(maps: torch.Tensor) -> torch.Tensor:
    """Slice at W//2 → [T, H, C]: x=channel (vertical lines), y=spatial H."""
    w = maps.shape[-1] // 2
    return maps[..., w].permute(0, 2, 1)


def ch_canvas_px(n_h: int, n_c: int, *, px_per_cell: int) -> tuple[int, int]:
    """Figure size in pixels: data n_c×n_h cells + title, left pad, labels, colorbar."""
    data_w = n_c * px_per_cell
    data_h = n_h * px_per_cell
    fig_w = data_w + CH_LEFT_PX
    fig_h = (
        data_h
        + CH_TITLE_PX
        + CH_XLABEL_PX
        + CH_CBAR_GAP_PX
        + CH_CBAR_PX
        + CH_BOTTOM_PAD_PX
    )
    return fig_w, fig_h


def render_ch_frame(
    frame: torch.Tensor,
    path: Path,
    *,
    heading: str,
    style: dict,
    px_per_cell: int = CH_PX_PER_CELL,
) -> None:
    """Save a C×H slice with native map aspect (channels = vertical columns)."""
    n_h, n_c = frame.shape
    fig_w_px, fig_h_px = ch_canvas_px(n_h, n_c, px_per_cell=px_per_cell)
    data_w_px = n_c * px_per_cell
    data_h_px = n_h * px_per_cell

    # Bottom → top: pad | cbar | cbar tick labels | channel labels | data | title
    y_cbar = CH_BOTTOM_PAD_PX
    y_data = y_cbar + CH_CBAR_PX + CH_CBAR_GAP_PX + CH_XLABEL_PX

    def _rect(x_px: float, y_px: float, w_px: float, h_px: float) -> list[float]:
        return [x_px / fig_w_px, y_px / fig_h_px, w_px / fig_w_px, h_px / fig_h_px]

    fig = plt.figure(figsize=(fig_w_px / CH_DPI, fig_h_px / CH_DPI), dpi=CH_DPI)
    fig.text(
        (CH_LEFT_PX + data_w_px / 2) / fig_w_px,
        1 - CH_TITLE_PX / (2 * fig_h_px),
        heading,
        ha="center",
        va="center",
        fontsize=8,
        linespacing=1.25,
    )

    ax = fig.add_axes(_rect(CH_LEFT_PX, y_data, data_w_px, data_h_px))
    im = ax.imshow(
        frame.numpy(),
        aspect="equal",
        origin="lower",
        interpolation="nearest",
        **style,
    )
    ax.set_xlim(-0.5, n_c - 0.5)
    ax.set_ylim(-0.5, n_h - 0.5)
    ax.set_xlabel("channel", fontsize=8, labelpad=6)
    ax.set_ylabel("H", fontsize=8, labelpad=4)
    step = max(1, n_c // 8)
    xticks = list(range(0, n_c, step))
    if (n_c - 1) % step:
        xticks.append(n_c - 1)
    ax.set_xticks(xticks)
    ax.set_yticks(range(n_h))
    ax.tick_params(axis="x", labelsize=7, pad=3)
    ax.tick_params(axis="y", labelsize=7, pad=2)

    cax = fig.add_axes(_rect(CH_LEFT_PX, y_cbar, data_w_px, CH_CBAR_PX))
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_ticks([])
    # Color scale numbers in figure coords (above the strip) so they cannot be clipped.
    vmin, vmax = im.get_clim()
    n_ticks = 7
    tick_vals = [vmin + (vmax - vmin) * i / (n_ticks - 1) for i in range(n_ticks)]
    y_tick = (y_cbar + CH_CBAR_PX + 6) / fig_h_px
    for val in tick_vals:
        x = (CH_LEFT_PX + data_w_px * ((val - vmin) / (vmax - vmin) if vmax > vmin else 0.0)) / fig_w_px
        fig.text(x, y_tick, f"{val:.3g}", ha="center", va="bottom", fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def render_frames(
    maps: torch.Tensor,
    out_dir: Path,
    *,
    run_name: str,
    formula: str,
    kind: str,
) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    style = draw_style(maps, kind)
    is_ch = kind.endswith("_CH")
    last = len(maps) - 1
    for d, frame in enumerate(maps):
        heading = frame_heading(run_name, formula, depth_line(kind, d, last))
        if is_ch:
            render_ch_frame(
                frame,
                out_dir / f"frame_{d:04d}.png",
                heading=heading,
                style=style,
            )
            continue
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout="constrained")
        im = ax.imshow(frame.numpy(), interpolation="nearest", aspect="auto", origin="lower", **style)
        ax.set_title(heading, fontsize=9, linespacing=1.2)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.savefig(out_dir / f"frame_{d:04d}.png")
        plt.close(fig)


def save_grid(maps: torch.Tensor, path: Path, *, title: str, kind: str, ncols: int = 5) -> None:
    if not should_write(path):
        print(f"  skip existing {path.name}")
        return
    n = len(maps)
    ncols = min(ncols, n)
    nrows = -(-n // ncols)
    style = draw_style(maps, kind)
    is_ch = kind.endswith("_CH")
    if is_ch:
        n_h, n_c = maps.shape[1], maps.shape[2]
        cell = CH_GRID_PX_PER_CELL
        panel_w = n_c * cell / CH_DPI + 0.6
        panel_h = n_h * cell / CH_DPI + 0.9
        figsize = (panel_w * ncols, panel_h * nrows)
    else:
        figsize = (2.1 * ncols, 2.3 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    im = None
    for d, (ax, frame) in enumerate(zip(axes.flat, maps)):
        aspect = "equal" if is_ch else "auto"
        im = ax.imshow(
            frame.numpy(), interpolation="nearest", aspect=aspect, origin="lower", **style
        )
        if is_ch:
            ax.set_xlabel("ch", fontsize=7)
            ax.set_ylabel("H", fontsize=7)
        ax.set_title(f"d={d}→{d + 1}" if is_pair_kind(kind) else f"d={d}", fontsize=9)
    if im is not None:
        fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.03, pad=0.04)
    fig.suptitle(title)
    fig.savefig(path, dpi=CH_DPI if is_ch else 140)
    plt.close(fig)


# MPEG-4 Part 2 rejects frames with any side > 8191 (ResNet C×H at 8px/cell is 8236 wide).
_MPEG4_MAX_SIDE = 8190


def _even_fit_size(w: int, h: int, max_side: int = _MPEG4_MAX_SIDE) -> tuple[int, int]:
    """Shrink (w, h) to fit in max_side×max_side, then force even sides for yuv420p/mpeg4."""
    scale = min(1.0, max_side / max(w, 1), max_side / max(h, 1))
    out_w = max(2, int(w * scale))
    out_h = max(2, int(h * scale))
    return out_w - (out_w % 2), out_h - (out_h % 2)


def _write_mp4_ffmpeg(frame_dir: Path, mp4: Path, fps: float) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    # Prefer libx264; fall back to mpeg4 when the build has no x264 (common on HPC images).
    # mpeg4 needs the side cap; libx264 only needs even dimensions.
    even = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    fit = (
        f"scale='min(iw,{_MPEG4_MAX_SIDE})':'min(ih,{_MPEG4_MAX_SIDE})'"
        f":force_original_aspect_ratio=decrease,{even}"
    )
    for codec_args, vf in (
        (["-c:v", "libx264", "-pix_fmt", "yuv420p"], even),
        (["-c:v", "mpeg4", "-q:v", "5"], fit),
    ):
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(fps),
                "-i", str(frame_dir / "frame_%04d.png"),
                *codec_args,
                "-vf", vf,
                str(mp4),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and _exists_video(mp4):
            return True
        if mp4.is_file() and not _exists_video(mp4):
            mp4.unlink(missing_ok=True)
        print(f"  ffmpeg {' '.join(codec_args)} failed: {proc.stderr.strip()[:200]}")
    return False


def _write_mp4_opencv(frame_dir: Path, mp4: Path, fps: float) -> bool:
    try:
        import cv2
    except ImportError:
        return False
    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        return False
    first = cv2.imread(str(frames[0]))
    if first is None:
        return False
    h0, w0 = first.shape[:2]
    w, h = _even_fit_size(w0, h0)
    # mp4v is widely available; avc1/H264 often is not in OpenCV builds.
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1e-3), (w, h))
    if not writer.isOpened():
        return False
    for path in frames:
        img = cv2.imread(str(path))
        if img is None:
            continue
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h))
        writer.write(img)
    writer.release()
    if not _exists_video(mp4):
        if mp4.is_file():
            mp4.unlink(missing_ok=True)
        return False
    return True


def write_video(frame_dir: Path, out_stem: Path, *, fps: float, n_frames: int) -> dict[str, Path]:
    """Write mp4 (ffmpeg or OpenCV). Never delete frames if nothing was written."""
    del n_frames  # retained for call-site compatibility
    written: dict[str, Path] = {}
    mp4 = out_stem.with_suffix(".mp4")

    if should_write(mp4):
        if _write_mp4_ffmpeg(frame_dir, mp4, fps) or _write_mp4_opencv(frame_dir, mp4, fps):
            written["mp4"] = mp4
    elif _exists_video(mp4):
        written["mp4"] = mp4
        print(f"  skip existing {mp4.name}")

    if not written:
        print(f"  WARNING: no video written for {out_stem.name}; keeping frames in {frame_dir}")
    elif not KEEP_FRAMES:
        shutil.rmtree(frame_dir, ignore_errors=True)
    return written


def map_tensor(res: dict, kind: str) -> torch.Tensor:
    if kind in ("h_CH", "x_CH"):
        return ch_cut(res[f"mean_{kind[0]}"])
    return res[kind]


def save_inputs(spec: dict, dataset, class_names: list[str], run_dir: Path) -> None:
    out = run_dir / "inputs.png"
    if not should_write(out):
        print(f"  skip existing {out.name}")
        return
    indices = spec["image_indices"]
    cid = spec["class_id"]
    subject = f"class {cid} — {class_names[cid]}" if cid is not None else "hand-picked images"
    preview = indices[:8]
    fig, axes = plt.subplots(1, len(preview), figsize=(2.2 * len(preview), 2.8), squeeze=False)
    for ax, idx in zip(axes[0], preview):
        img, label = dataset[idx]
        ax.imshow(denormalize(img).permute(1, 2, 0).numpy())
        ax.set_title(f"#{idx}\n{class_names[label].split(',')[0]}", fontsize=8)
        ax.axis("off")
    fig.suptitle(
        f"{spec['name']} — averaging over {subject}"
        + (f" — showing {len(preview)}" if len(preview) < len(indices) else "")
    )
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_tables_and_config(res: dict, class_names: list[str], run_dir: Path) -> None:
    spec = res["spec"]
    res["norms"].to_csv(run_dir / "table_norms.csv", index=False)
    res["pairs"].to_csv(run_dir / "table_pairs.csv", index=False)
    if "scalars" in res:
        res["scalars"].to_csv(run_dir / "table_scalars.csv", index=False)
    config = {
        "name": res["name"],
        "backbone": "shared" if is_shared_run(spec) else "interpoled",
        "model": res.get("model"),
        "D": res["D"],
        "blocks": list(res.get("blocks") or []),
        "euler_step": res["euler_step"],
        "method": res["method"] or "RK1",
        "weight_interpolation": res.get("weight_interpolation", "plain"),
        "split": SPLIT,
        "class_id": spec["class_id"],
        "class_name": class_names[spec["class_id"]] if spec["class_id"] is not None else None,
        "max_images": spec["max_images"],
        "n_images": len(spec["image_indices"]),
        "image_indices": list(spec["image_indices"]),
        "batch_size": spec["batch_size"],
        "fps": spec["fps"],
        "video_maps": list(VIDEO_MAPS),
        "channels": CHANNELS,
        "n_auto_channels": N_AUTO_CHANNELS,
        "ignore_top_k_channels": spec["ignore_top_k_channels"],
        "ignored_channels": list(res.get("ignored_channels", [])),
        "cos_flat_spatial_corr": res.get("cos_flat_spatial_corr"),
        "rectitude_mean": res.get("rectitude_mean"),
        "rectitude_std": res.get("rectitude_std"),
        "L_mean": res.get("L_mean"),
        "N_mean": res.get("N_mean"),
        "PR_depth_mean": res.get("PR_depth_mean"),
        "PR_depth_std": res.get("PR_depth_std"),
        "PR_spatial_mean": res.get("PR_spatial_mean"),
        "PR_spatial_std": res.get("PR_spatial_std"),
        "PR_depth_on_mean_traj": res.get("PR_depth_on_mean_traj"),
        "PR_spatial_on_mean_traj": res.get("PR_spatial_on_mean_traj"),
        "scatter_channel": SCATTER_CHANNEL,
        "resolved_channels": list(res.get("resolved_channels") or []),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if SAVE_TENSORS:
        torch.save(
            {
                "mean_x": res["mean_x"],
                "mean_h": res["mean_h"],
                "cos_h": res["cos_h"],
                "cos_x": res["cos_x"],
                "norm_h": res["norm_h"],
                "l2_h": res["l2_h"],
            },
            run_dir / MEAN_MAPS_NAME,
        )


def load_cached_run(run_dir: Path, spec: dict) -> dict | None:
    """Reload maps + tables from a previous run (skip trajectory integration)."""
    if FORCE_RECOMPUTE:
        return None
    pt = run_dir / MEAN_MAPS_NAME
    cfg_path = run_dir / "config.json"
    norms_path = run_dir / "table_norms.csv"
    pairs_path = run_dir / "table_pairs.csv"
    if not (pt.is_file() and cfg_path.is_file() and norms_path.is_file() and pairs_path.is_file()):
        return None
    cfg = json.loads(cfg_path.read_text())
    try:
        maps = torch.load(pt, map_location="cpu", weights_only=True)
    except TypeError:
        maps = torch.load(pt, map_location="cpu")
    required = ("mean_x", "mean_h", "cos_h", "cos_x", "norm_h", "l2_h")
    if any(k not in maps for k in required):
        return None
    res = {
        "name": cfg.get("name", spec["name"]),
        "spec": spec,
        "D": int(cfg["D"]),
        "euler_step": float(cfg["euler_step"]),
        "method": cfg.get("method") or "RK1",
        "blocks": list(cfg.get("blocks") or []),
        "model": cfg.get("model"),
        "weight_interpolation": cfg.get("weight_interpolation", "plain"),
        "overlay_label": cfg.get("name", spec["name"]),
        "n_images": int(cfg.get("n_images", len(spec["image_indices"]))),
        "ignored_channels": list(cfg.get("ignored_channels") or []),
        "resolved_channels": list(cfg.get("resolved_channels") or []),
        "mean_x": maps["mean_x"],
        "mean_h": maps["mean_h"],
        "cos_h": maps["cos_h"],
        "cos_x": maps["cos_x"],
        "norm_h": maps["norm_h"],
        "l2_h": maps["l2_h"],
        "norms": pd.read_csv(norms_path),
        "pairs": pd.read_csv(pairs_path),
        "cos_flat_spatial_corr": cfg.get("cos_flat_spatial_corr"),
        "rectitude_mean": cfg.get("rectitude_mean"),
        "rectitude_std": cfg.get("rectitude_std"),
        "L_mean": cfg.get("L_mean"),
        "N_mean": cfg.get("N_mean"),
        "PR_depth_mean": cfg.get("PR_depth_mean"),
        "PR_depth_std": cfg.get("PR_depth_std"),
        "PR_spatial_mean": cfg.get("PR_spatial_mean"),
        "PR_spatial_std": cfg.get("PR_spatial_std"),
        "PR_depth_on_mean_traj": cfg.get("PR_depth_on_mean_traj"),
        "PR_spatial_on_mean_traj": cfg.get("PR_spatial_on_mean_traj"),
    }
    # New R1-distance columns require a trajectory pass; stale caches must recompute.
    if "dist_x_r1" not in res["pairs"].columns:
        return None
    scalars_path = run_dir / "table_scalars.csv"
    if scalars_path.is_file():
        res["scalars"] = pd.read_csv(scalars_path)
    return res


def _plot_mean_std(ax, x, mean, std, *, label: str, **plot_kw):
    mean = np.asarray(mean)
    std = np.asarray(std)
    ax.plot(x, mean, marker="o", ms=3, label=label, **plot_kw)
    ax.fill_between(x, mean - std, mean + std, alpha=0.25)


def save_metric_plots(res: dict, run_dir: Path) -> None:
    out = run_dir / "metrics.png"
    if not should_write(out):
        print(f"  skip existing {out.name}")
        return
    norms, pairs = res["norms"], res["pairs"]
    es = res["euler_step"]
    label = f"{res['name']} (D={res['D']}, n={res['n_images']})"
    t_state = norms["t"]
    t_pair = pairs["t"] if "t" in pairs.columns else pairs["d"] * es
    has_r1 = "dist_x_r1" in pairs.columns

    nrows = 3 if has_r1 else 2
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows), layout="constrained")

    # (0,0) ||h|| and ||h||/||x||  (h = Δx/ES)
    ax = axes[0, 0]
    _plot_mean_std(
        ax, t_state, norms["norm_h"], norms.get("norm_h_std", 0.0), label=r"$\|h_d\|$"
    )
    _plot_mean_std(
        ax,
        t_state,
        norms["norm_h_over_norm_x"],
        norms.get("norm_h_over_norm_x_std", 0.0),
        label=r"$\|h_d\| / \|x_d\|$",
    )
    ax.set(xlabel="t = d · ES", ylabel="norm", title=r"field norm ($h=\Delta x/\mathrm{ES}$)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (0,1) angular velocity from flattened cos
    ax = axes[0, 1]
    _plot_mean_std(
        ax, t_pair, pairs["omega_h"], pairs.get("omega_h_std", 0.0), label=label
    )
    ax.set(
        xlabel="t",
        ylabel=r"$\omega$ [rad / t]",
        title=r"angular speed  $\arccos(\cos(h_d,h_{d+1}))/\mathrm{ES}$",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (1,0) alignment with h_0
    ax = axes[1, 0]
    _plot_mean_std(
        ax, t_state, norms["cos_h0_hd"], norms.get("cos_h0_hd_std", 0.0), label=label
    )
    ax.axhline(0.0, color="0.7", lw=1, zorder=0)
    ax.set_ylim(-1.05, 1.05)
    ax.set(xlabel="t", ylabel=r"$\cos(h_0, h_d)$", title="alignment with initial residual")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (1,1) mean angular speed: per-location (C-vec) and per-channel (HW-vec)
    ax = axes[1, 1]
    _plot_mean_std(
        ax,
        t_pair,
        pairs["omega_h_spatial"],
        pairs.get("omega_h_spatial_std", 0.0),
        label=r"per-location $\mathrm{mean}_{i,j}\,\omega$",
    )
    _plot_mean_std(
        ax,
        t_pair,
        pairs["omega_h_channel"],
        pairs.get("omega_h_channel_std", 0.0),
        label=r"per-channel $\mathrm{mean}_c\,\omega$",
    )
    ax.set(
        xlabel="t",
        ylabel=r"$\omega$ [rad / t]",
        title=r"mean angular speed (spatial / channel)",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (2,*) distance of multi-step x to parallel R1 trajectory (per residual group)
    if has_r1:
        ax = axes[2, 0]
        _plot_mean_std(
            ax,
            t_pair,
            pairs["dist_x_r1"],
            pairs.get("dist_x_r1_std", 0.0),
            label=r"$\|x^{(n)} - x_{\mathrm{R1}}\|$",
        )
        ax.set(
            xlabel="t",
            ylabel=r"$\|x^{(n)} - x_{\mathrm{R1}}\|$",
            title="distance to parallel R1 (same block, one step)",
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[2, 1]
        _plot_mean_std(
            ax,
            t_pair,
            pairs["dist_x_r1_rel"],
            pairs.get("dist_x_r1_rel_std", 0.0),
            label=r"$\|x^{(n)} - x_{\mathrm{R1}}\| / \|x_{\mathrm{R1}}\|$",
        )
        ax.set(
            xlabel="t",
            ylabel="relative distance",
            title="relative distance to parallel R1",
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    else:
        print("  note: dist_x_r1 missing from cache — re-run with --force to compute R1 distance")

    fig.savefig(out, dpi=140)
    plt.close(fig)

    corr = res.get("cos_flat_spatial_corr", float("nan"))
    print(f"  corr(flat ω, spatial ω) = {corr:.6f}")
    print(
        f"  rectitude R = N/L : mean={res.get('rectitude_mean', float('nan')):.6f} "
        f"± {res.get('rectitude_std', float('nan')):.6f} "
        f"(L̄={res.get('L_mean', float('nan')):.4g}, N̄={res.get('N_mean', float('nan')):.4g})"
    )
    print(
        f"  PR_depth (H:[D,dim]) : mean={res.get('PR_depth_mean', float('nan')):.4f} "
        f"± {res.get('PR_depth_std', float('nan')):.4f} "
        f"(on mean traj={res.get('PR_depth_on_mean_traj', float('nan')):.4f})"
    )
    print(
        f"  PR_spatial (mid h:[HW,C]) : mean={res.get('PR_spatial_mean', float('nan')):.4f} "
        f"± {res.get('PR_spatial_std', float('nan')):.4f} "
        f"(on mean traj={res.get('PR_spatial_on_mean_traj', float('nan')):.4f})"
    )


def _channel_spatial_means(maps: torch.Tensor) -> torch.Tensor:
    """Per-depth, per-channel mean over H×W. ``maps`` is (T, C, H, W) → (T, C)."""
    return maps.flatten(2).mean(-1)


def scatter_io(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    *,
    title: str,
    path: Path,
    xlabel: str,
    ylabel: str,
    max_points: int = SCATTER_MAX_POINTS,
    channel: int | None = SCATTER_CHANNEL,
    draw_y_equals_x: bool = False,
    label: str | None = None,
    ax: plt.Axes | None = None,
    color=None,
) -> plt.Axes:
    """Scatter (input[d], output[d]) for features, one colour per depth (or solid if ax shared)."""
    assert inputs.shape == outputs.shape and inputs.ndim == 4
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6.5, 6), layout="constrained")
    else:
        fig = ax.figure

    n, n_c, n_h, n_w = inputs.shape
    if channel is not None:
        if not 0 <= channel < n_c:
            raise ValueError(f"scatter channel {channel} outside 0..{n_c - 1}")
        inputs = inputs[:, channel : channel + 1]
        outputs = outputs[:, channel : channel + 1]
        n_c = 1

    cmap = plt.cm.viridis
    flat_in = inputs.reshape(n, -1)
    flat_out = outputs.reshape(n, -1)
    feat_idx = _sample_feature_indices(flat_in.shape[1], max_points, SCATTER_ANIM_SEED)
    lo, hi = _padded_finite_limits(flat_in[:, feat_idx], flat_out[:, feat_idx])
    if draw_y_equals_x:
        ax.plot([lo, hi], [lo, hi], color="0.7", lw=1, zorder=0, label="y = x")
    ax.axhline(0.0, color="0.85", lw=1, zorder=0)
    # Fixed feature sample across depths; subsample depths when D is huge.
    depth_idx = _depth_indices(n)
    for d in depth_idx:
        a = flat_in[d, feat_idx]
        b = flat_out[d, feat_idx]
        c_kw = {}
        if color is not None:
            c_kw["color"] = color
        else:
            c_kw["c"] = [cmap(d / max(n - 1, 1))]
        ax.scatter(
            a.numpy(),
            b.numpy(),
            s=4,
            alpha=0.25,
            linewidths=0,
            label=(label if d == depth_idx[0] else None)
            if label is not None
            else (f"d={d}" if len(depth_idx) <= 12 or d in (0, n // 2, n - 1) else None),
            **c_kw,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if own_fig:
        ax.set_title(title)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    if own_fig:
        ax.legend(fontsize=7, markerscale=2)
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return ax


def scatter_io_means(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    *,
    title: str,
    path: Path,
    xlabel: str,
    ylabel: str,
    channel: int | None = SCATTER_CHANNEL,
    label: str | None = None,
    ax: plt.Axes | None = None,
    color=None,
) -> plt.Axes:
    """Scatter of per-channel spatial means only (colour = depth, or solid if ax shared)."""
    assert inputs.shape == outputs.shape and inputs.ndim == 4
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6.5, 6), layout="constrained")
    else:
        fig = ax.figure

    n, n_c = inputs.shape[:2]
    if channel is not None:
        if not 0 <= channel < n_c:
            raise ValueError(f"scatter channel {channel} outside 0..{n_c - 1}")
        inputs = inputs[:, channel : channel + 1]
        outputs = outputs[:, channel : channel + 1]
        n_c = 1

    mean_in = _channel_spatial_means(inputs)
    mean_out = _channel_spatial_means(outputs)
    lo, hi = _padded_finite_limits(mean_in, mean_out)
    cmap = plt.cm.viridis
    ch_colors = np.arange(n_c, dtype=np.float64)
    ax.axhline(0.0, color="0.85", lw=1, zorder=0)
    depth_idx = _depth_indices(n)
    for d in depth_idx:
        c_kw = {}
        if color is not None:
            c_kw["color"] = color
        else:
            c_kw["c"] = ch_colors
            c_kw["cmap"] = SCATTER_ANIM_CMAP
            c_kw["vmin"] = 0
            c_kw["vmax"] = max(n_c - 1, 1)
        ax.scatter(
            mean_in[d].numpy(),
            mean_out[d].numpy(),
            s=28,
            alpha=0.85,
            linewidths=0,
            zorder=5,
            label=(label if d == depth_idx[0] else None)
            if label is not None
            else (f"d={d}" if len(depth_idx) <= 12 or d in (0, n // 2, n - 1) else None),
            **c_kw,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if own_fig:
        ax.set_title(title)
        if color is None and n_c > 1:
            sm = plt.cm.ScalarMappable(
                cmap=plt.get_cmap(SCATTER_ANIM_CMAP),
                norm=plt.Normalize(vmin=0, vmax=max(n_c - 1, 1)),
            )
            sm.set_array([])
            fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="channel")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    if own_fig:
        ax.legend(fontsize=7, markerscale=1.5)
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return ax


def scatter_residual_field(
    res: dict,
    *,
    path: Path,
    channel: int | None = SCATTER_CHANNEL,
    max_points: int = SCATTER_MAX_POINTS,
) -> None:
    """Static scatter of (x_d, h_d) with h = Δx/ES = block(x)-x."""
    scatter_io(
        res["mean_x"],
        res["mean_h"],
        title=f"{res['name']} | {KIND_TITLES['scatter_h']}",
        path=path,
        xlabel="x_d",
        ylabel=r"$h_d\ (=\Delta x_d/\mathrm{ES})$",
        max_points=max_points,
        channel=channel,
        draw_y_equals_x=False,
    )


def scatter_residual_field_means(
    res: dict,
    *,
    path: Path,
    channel: int | None = SCATTER_CHANNEL,
) -> None:
    """Static scatter of per-channel means (mean x_d, mean h_d)."""
    scatter_io_means(
        res["mean_x"],
        res["mean_h"],
        title=f"{res['name']} | {KIND_TITLES['scatter_h_means']}",
        path=path,
        xlabel=r"$\mathrm{mean}_{H,W}(x_d)$",
        ylabel=r"$\mathrm{mean}_{H,W}(h_d)$",
        channel=channel,
    )


def scatter_overlay_by_d(
    results: list[dict],
    *,
    path: Path,
    channel: int | None = SCATTER_CHANNEL,
    max_points: int = SCATTER_MAX_POINTS,
) -> None:
    """Overlay residual-field scatters for several D on the same axes."""
    if not results:
        return
    fig, ax = plt.subplots(figsize=(6.5, 6), layout="constrained")
    cmap = plt.cm.turbo
    for i, res in enumerate(results):
        color = cmap(i / max(len(results) - 1, 1))
        scatter_io(
            res["mean_x"],
            res["mean_h"],
            title="",
            path=path,
            xlabel="x_d",
            ylabel=r"$h_d\ (=\Delta x_d/\mathrm{ES})$",
            max_points=max_points,
            channel=channel,
            draw_y_equals_x=False,
            label=res.get("overlay_label", f"D={res['D']}"),
            ax=ax,
            color=color,
        )
    ax.set_title("residual field overlay by D")
    ax.legend(fontsize=8, markerscale=2)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def scatter_overlay_means_by_d(
    results: list[dict],
    *,
    path: Path,
    channel: int | None = SCATTER_CHANNEL,
) -> None:
    """Overlay residual-field channel-mean scatters for several D."""
    if not results:
        return
    fig, ax = plt.subplots(figsize=(6.5, 6), layout="constrained")
    cmap = plt.cm.turbo
    for i, res in enumerate(results):
        color = cmap(i / max(len(results) - 1, 1))
        scatter_io_means(
            res["mean_x"],
            res["mean_h"],
            title="",
            path=path,
            xlabel=r"$\mathrm{mean}_{H,W}(x_d)$",
            ylabel=r"$\mathrm{mean}_{H,W}(h_d)$",
            channel=channel,
            label=res.get("overlay_label", f"D={res['D']}"),
            ax=ax,
            color=color,
        )
    ax.set_title("channel-mean residual field overlay by D")
    ax.legend(fontsize=8, markerscale=1.5)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _sample_feature_indices(n_feat: int, max_points: int, seed: int) -> torch.Tensor:
    if n_feat > max_points:
        g = torch.Generator().manual_seed(seed)
        return torch.randperm(n_feat, generator=g)[:max_points]
    return torch.arange(n_feat)


def _depth_indices(n: int, max_depths: int = SCATTER_STATIC_MAX_DEPTHS) -> np.ndarray:
    """Subsample depth indices for static overlays (keeps endpoints)."""
    if n <= max_depths:
        return np.arange(n, dtype=int)
    return np.unique(np.round(np.linspace(0, n - 1, max_depths)).astype(int))


def spaghetti_trajectories(
    maps: torch.Tensor,
    *,
    title: str,
    path: Path,
    ylabel: str,
    euler_step: float,
    max_lines: int = SPAGHETTI_MAX_LINES,
    seed: int = SPAGHETTI_SEED,
    channel: int | None = SCATTER_CHANNEL,
) -> None:
    """Plot d ↦ value trajectories for a fixed random feature sample (colour = channel)."""
    assert maps.ndim == 4
    n, n_c, n_h, n_w = maps.shape
    if channel is not None:
        if not 0 <= channel < n_c:
            raise ValueError(f"spaghetti channel {channel} outside 0..{n_c - 1}")
        maps = maps[:, channel : channel + 1]
        n_c = 1
    flat = maps.reshape(n, -1)
    spatial = n_h * n_w
    idx = _sample_feature_indices(flat.shape[1], max_lines, seed)
    ch = (idx // spatial).numpy()
    ys = flat[:, idx].numpy()  # [T, N]
    t = np.arange(n, dtype=np.float64) * float(euler_step)
    cmap = plt.get_cmap(SCATTER_ANIM_CMAP)
    colors = cmap(ch / max(n_c - 1, 1))

    fig, ax = plt.subplots(figsize=(8.0, 5.0), layout="constrained")
    ax.axhline(0.0, color="0.85", lw=1, zorder=0)
    for i in range(ys.shape[1]):
        ax.plot(t, ys[:, i], color=colors[i], alpha=0.35, lw=0.9, solid_capstyle="round")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(n_c - 1, 1)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="channel")
    ax.set_xlabel("t = d · ES")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n({ys.shape[1]} trajectories)")
    ax.set_ylim(*_padded_finite_limits(flat[:, idx]))
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def spaghetti_trajectories_means(
    maps: torch.Tensor,
    *,
    title: str,
    path: Path,
    ylabel: str,
    euler_step: float,
    channel: int | None = SCATTER_CHANNEL,
) -> None:
    """Plot d ↦ spatial-mean trajectories, one line per channel (colour = channel)."""
    assert maps.ndim == 4
    n, n_c = maps.shape[:2]
    if channel is not None:
        if not 0 <= channel < n_c:
            raise ValueError(f"spaghetti channel {channel} outside 0..{n_c - 1}")
        maps = maps[:, channel : channel + 1]
        n_c = 1
    ys = _channel_spatial_means(maps)
    t = np.arange(n, dtype=np.float64) * float(euler_step)
    cmap = plt.get_cmap(SCATTER_ANIM_CMAP)
    colors = cmap(np.arange(n_c) / max(n_c - 1, 1))

    fig, ax = plt.subplots(figsize=(8.0, 5.0), layout="constrained")
    ax.axhline(0.0, color="0.85", lw=1, zorder=0)
    for c in range(n_c):
        ax.plot(t, ys[:, c].numpy(), color=colors[c], alpha=0.55, lw=1.0, solid_capstyle="round")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(n_c - 1, 1)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="channel")
    ax.set_xlabel("t = d · ES")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n({n_c} channel means)")
    ax.set_ylim(*_padded_finite_limits(ys))
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def scatter_vs_channel(
    maps: torch.Tensor,
    *,
    title: str,
    path: Path,
    ylabel: str,
    max_points: int = SCATTER_MAX_POINTS,
    seed: int = SCATTER_ANIM_SEED,
    channel: int | None = SCATTER_CHANNEL,
) -> None:
    """Static overlay of (channel, value_d) for all depths (colour = depth)."""
    assert maps.ndim == 4
    n, n_c, n_h, n_w = maps.shape
    if channel is not None:
        if not 0 <= channel < n_c:
            raise ValueError(f"scatter channel {channel} outside 0..{n_c - 1}")
        maps = maps[:, channel : channel + 1]
        n_c = 1
    flat = maps.reshape(n, -1)
    spatial = n_h * n_w
    idx = _sample_feature_indices(flat.shape[1], max_points, seed)
    ch = (idx // spatial).numpy().astype(np.float64)
    rng = np.random.default_rng(seed)
    ch_plot = ch + rng.uniform(-0.35, 0.35, size=ch.shape)
    vals = flat[:, idx]
    lo, hi = _padded_finite_limits(vals)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), layout="constrained")
    cmap = plt.cm.viridis
    ax.axhline(0.0, color="0.85", lw=1, zorder=0)
    depth_idx = _depth_indices(n)
    for d in depth_idx:
        ax.scatter(
            ch_plot,
            vals[d].numpy(),
            s=4,
            alpha=0.25,
            c=[cmap(d / max(n - 1, 1))],
            linewidths=0,
            label=f"d={d}" if len(depth_idx) <= 12 or d in (0, n // 2, n - 1) else None,
        )
    ax.set_xlabel("channel")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(-0.5, n_c - 0.5)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, markerscale=2)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def scatter_vs_channel_means(
    maps: torch.Tensor,
    *,
    title: str,
    path: Path,
    ylabel: str,
    channel: int | None = SCATTER_CHANNEL,
) -> None:
    """Static scatter of per-channel spatial means vs channel index (colour = channel)."""
    assert maps.ndim == 4
    n, n_c = maps.shape[:2]
    if channel is not None:
        if not 0 <= channel < n_c:
            raise ValueError(f"scatter channel {channel} outside 0..{n_c - 1}")
        maps = maps[:, channel : channel + 1]
        n_c = 1
    ch_means = _channel_spatial_means(maps)
    ch_axis = np.arange(n_c, dtype=np.float64)
    lo, hi = _padded_finite_limits(ch_means)
    cmap = plt.get_cmap(SCATTER_ANIM_CMAP)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), layout="constrained")
    ax.axhline(0.0, color="0.85", lw=1, zorder=0)
    depth_idx = _depth_indices(n)
    for d in depth_idx:
        ax.scatter(
            ch_axis,
            ch_means[d].numpy(),
            s=28,
            alpha=0.85,
            c=ch_axis,
            cmap=SCATTER_ANIM_CMAP,
            vmin=0,
            vmax=max(n_c - 1, 1),
            linewidths=0,
            zorder=5,
            label=f"d={d}" if len(depth_idx) <= 12 or d in (0, n // 2, n - 1) else None,
        )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(n_c - 1, 1)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="channel")
    ax.set_xlabel("channel")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(-0.5, n_c - 0.5)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, markerscale=1.5)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def scatter_vs_channel_animation(
    maps: torch.Tensor,
    *,
    title: str,
    frame_dir: Path,
    out_stem: Path,
    ylabel: str,
    fps: float,
    max_points: int = SCATTER_ANIM_MAX_POINTS,
    seed: int = SCATTER_ANIM_SEED,
) -> dict[str, Path]:
    """Animate (channel, value_d) for a fixed feature sample across depth."""
    assert maps.ndim == 4
    n, n_c, n_h, n_w = maps.shape
    flat = maps.reshape(n, -1)
    spatial = n_h * n_w
    idx = _sample_feature_indices(flat.shape[1], max_points, seed)
    ch = (idx // spatial).numpy().astype(np.float64)
    rng = np.random.default_rng(seed)
    ch_plot = ch + rng.uniform(-0.35, 0.35, size=ch.shape)
    vals = flat[:, idx]
    lo, hi = _padded_finite_limits(vals)

    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)

    for d in range(n):
        fig, ax = plt.subplots(figsize=(7.5, 5.5), layout="constrained")
        ax.axhline(0.0, color="0.85", lw=1, zorder=0)
        sc = ax.scatter(
            ch_plot,
            vals[d].numpy(),
            s=8,
            alpha=0.55,
            c=ch,
            cmap=SCATTER_ANIM_CMAP,
            vmin=0,
            vmax=max(n_c - 1, 1),
            linewidths=0,
        )
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="channel")
        ax.set_xlim(-0.5, n_c - 0.5)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("channel")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nd={d}  ({idx.numel()} features)")
        ax.grid(alpha=0.3)
        fig.savefig(frame_dir / f"frame_{d:04d}.png", dpi=140)
        plt.close(fig)

    return write_video(frame_dir, out_stem, fps=fps, n_frames=n)


def scatter_vs_channel_means_animation(
    maps: torch.Tensor,
    *,
    title: str,
    frame_dir: Path,
    out_stem: Path,
    ylabel: str,
    fps: float,
) -> dict[str, Path]:
    """Animate per-channel spatial means vs channel index across depth."""
    assert maps.ndim == 4
    n, n_c = maps.shape[:2]
    ch_means = _channel_spatial_means(maps)
    ch_axis = np.arange(n_c, dtype=np.float64)
    lo, hi = _padded_finite_limits(ch_means)

    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)

    for d in range(n):
        fig, ax = plt.subplots(figsize=(7.5, 5.5), layout="constrained")
        ax.axhline(0.0, color="0.85", lw=1, zorder=0)
        sc = ax.scatter(
            ch_axis,
            ch_means[d].numpy(),
            s=36,
            alpha=0.9,
            c=ch_axis,
            cmap=SCATTER_ANIM_CMAP,
            vmin=0,
            vmax=max(n_c - 1, 1),
            linewidths=0,
        )
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="channel")
        ax.set_xlim(-0.5, n_c - 0.5)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("channel")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nd={d}  ({n_c} channel means)")
        ax.grid(alpha=0.3)
        fig.savefig(frame_dir / f"frame_{d:04d}.png", dpi=140)
        plt.close(fig)

    return write_video(frame_dir, out_stem, fps=fps, n_frames=n)


def scatter_io_animation(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    *,
    title: str,
    frame_dir: Path,
    out_stem: Path,
    xlabel: str,
    ylabel: str,
    fps: float,
    max_points: int = SCATTER_ANIM_MAX_POINTS,
    seed: int = SCATTER_ANIM_SEED,
) -> dict[str, Path]:
    """Animate (input[d], output[d]) for a fixed random feature subset across depth.

    Colour encodes channel index (stable over depth).
    """
    assert inputs.shape == outputs.shape and inputs.ndim == 4
    n, n_c, n_h, n_w = inputs.shape
    flat_in = inputs.reshape(n, -1)
    flat_out = outputs.reshape(n, -1)
    spatial = n_h * n_w
    idx = _sample_feature_indices(flat_in.shape[1], max_points, seed)
    tracked_in = flat_in[:, idx]
    tracked_out = flat_out[:, idx]
    channels = (idx // spatial).numpy()
    lo, hi = _padded_finite_limits(tracked_in, tracked_out)

    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)

    for d in range(n):
        fig, ax = plt.subplots(figsize=(6.5, 6), layout="constrained")
        ax.axhline(0.0, color="0.85", lw=1, zorder=0)
        sc = ax.scatter(
            tracked_in[d].numpy(),
            tracked_out[d].numpy(),
            s=8,
            alpha=0.55,
            c=channels,
            cmap=SCATTER_ANIM_CMAP,
            vmin=0,
            vmax=max(n_c - 1, 1),
            linewidths=0,
        )
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="channel")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nd={d}  ({idx.numel()} features, colour=channel)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        fig.savefig(frame_dir / f"frame_{d:04d}.png", dpi=140)
        plt.close(fig)

    return write_video(frame_dir, out_stem, fps=fps, n_frames=n)


def scatter_io_means_animation(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    *,
    title: str,
    frame_dir: Path,
    out_stem: Path,
    xlabel: str,
    ylabel: str,
    fps: float,
) -> dict[str, Path]:
    """Animate per-channel spatial means (input mean, output mean) across depth."""
    assert inputs.shape == outputs.shape and inputs.ndim == 4
    n, n_c = inputs.shape[:2]
    mean_in = _channel_spatial_means(inputs)
    mean_out = _channel_spatial_means(outputs)
    ch = np.arange(n_c, dtype=np.float64)
    lo, hi = _padded_finite_limits(mean_in, mean_out)

    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)

    for d in range(n):
        fig, ax = plt.subplots(figsize=(6.5, 6), layout="constrained")
        ax.axhline(0.0, color="0.85", lw=1, zorder=0)
        sc = ax.scatter(
            mean_in[d].numpy(),
            mean_out[d].numpy(),
            s=36,
            alpha=0.9,
            c=ch,
            cmap=SCATTER_ANIM_CMAP,
            vmin=0,
            vmax=max(n_c - 1, 1),
            linewidths=0,
        )
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="channel")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nd={d}  ({n_c} channel means)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        fig.savefig(frame_dir / f"frame_{d:04d}.png", dpi=140)
        plt.close(fig)

    return write_video(frame_dir, out_stem, fps=fps, n_frames=n)


def write_videos(res: dict, channels: list[int], run_dir: Path) -> None:
    fps = float(res["spec"]["fps"])
    run_name = res["name"]
    channel_kinds = [k for k in VIDEO_MAPS if k in ("h", "x")]
    map_kinds = [k for k in VIDEO_MAPS if k in ("cos_h", "cos_x", "norm_h", "l2_h", "h_CH", "x_CH")]
    scatter_kinds = [
        k
        for k in VIDEO_MAPS
        if k
        in (
            "scatter_ch_x",
            "scatter_ch_h",
            "scatter_h",
            "scatter_ch_x_means",
            "scatter_ch_h_means",
            "scatter_h_means",
        )
    ]

    for kind in channel_kinds:
        mean_map = res[f"mean_{kind}"]
        n_frames = mean_map.shape[0]
        for channel in channels:
            stem = f"{kind}_ch{channel:03d}"
            out_stem = run_dir / stem
            formula = f"{KIND_TITLES[kind]}  |  channel {channel}"
            grid_path = run_dir / f"{stem}_grid.png"
            need_grid = n_frames <= GRID_MAX_FRAMES and should_write(grid_path)
            if video_products_ready(out_stem, n_frames):
                print(f"  skip existing {stem}")
            else:
                frame_dir = run_dir / "frames" / stem
                render_frames(
                    mean_map[:, channel],
                    frame_dir,
                    run_name=run_name,
                    formula=formula,
                    kind=kind,
                )
                written = write_video(frame_dir, out_stem, fps=fps, n_frames=n_frames)
                print(f"  {stem}: {[p.name for p in written.values()]}")
            if need_grid:
                save_grid(
                    mean_map[:, channel],
                    grid_path,
                    title=f"{run_name}\n{formula}",
                    kind=kind,
                )

    for kind in map_kinds:
        maps = map_tensor(res, kind)
        n_frames = maps.shape[0]
        out_stem = run_dir / kind
        formula = KIND_TITLES[kind]
        grid_path = run_dir / f"{kind}_grid.png"
        need_grid = n_frames <= GRID_MAX_FRAMES and should_write(grid_path)
        if video_products_ready(out_stem, n_frames):
            print(f"  skip existing {kind}")
        else:
            frame_dir = run_dir / "frames" / kind
            render_frames(maps, frame_dir, run_name=run_name, formula=formula, kind=kind)
            written = write_video(frame_dir, out_stem, fps=fps, n_frames=n_frames)
            print(f"  {kind}: {[p.name for p in written.values()]}")
        if need_grid:
            save_grid(maps, grid_path, title=f"{run_name}\n{formula}", kind=kind)

    for kind in scatter_kinds:
        out_stem = run_dir / kind
        n_frames = res["mean_x"].shape[0]
        if video_products_ready(out_stem, n_frames):
            print(f"  skip existing {kind}")
            continue
        if kind == "scatter_ch_x":
            written = scatter_vs_channel_animation(
                res["mean_x"],
                title=f"{run_name} | {KIND_TITLES[kind]}",
                frame_dir=run_dir / "frames" / kind,
                out_stem=out_stem,
                ylabel="x_d",
                fps=fps,
            )
        elif kind == "scatter_ch_h":
            written = scatter_vs_channel_animation(
                res["mean_h"],
                title=f"{run_name} | {KIND_TITLES[kind]}",
                frame_dir=run_dir / "frames" / kind,
                out_stem=out_stem,
                ylabel=r"$h_d\ (=\Delta x_d/\mathrm{ES})$",
                fps=fps,
            )
        elif kind == "scatter_h":
            written = scatter_io_animation(
                res["mean_x"],
                res["mean_h"],
                title=f"{run_name} | {KIND_TITLES[kind]}",
                frame_dir=run_dir / "frames" / kind,
                out_stem=out_stem,
                xlabel="x_d",
                ylabel=r"$h_d\ (=\Delta x_d/\mathrm{ES})$",
                fps=fps,
            )
        elif kind == "scatter_ch_x_means":
            written = scatter_vs_channel_means_animation(
                res["mean_x"],
                title=f"{run_name} | {KIND_TITLES[kind]}",
                frame_dir=run_dir / "frames" / kind,
                out_stem=out_stem,
                ylabel=r"$\mathrm{mean}_{H,W}(x_d)$",
                fps=fps,
            )
        elif kind == "scatter_ch_h_means":
            written = scatter_vs_channel_means_animation(
                res["mean_h"],
                title=f"{run_name} | {KIND_TITLES[kind]}",
                frame_dir=run_dir / "frames" / kind,
                out_stem=out_stem,
                ylabel=r"$\mathrm{mean}_{H,W}(h_d)$",
                fps=fps,
            )
        else:
            written = scatter_io_means_animation(
                res["mean_x"],
                res["mean_h"],
                title=f"{run_name} | {KIND_TITLES[kind]}",
                frame_dir=run_dir / "frames" / kind,
                out_stem=out_stem,
                xlabel=r"$\mathrm{mean}_{H,W}(x_d)$",
                ylabel=r"$\mathrm{mean}_{H,W}(h_d)$",
                fps=fps,
            )
        print(f"  {kind}: {[p.name for p in written.values()]}")
        plt.close("all")
        gc.collect()


def resolve_run(spec: dict, *, labels: list[int], class_names: list[str]) -> dict:
    out = dict(spec)
    out["class_id"] = out.get("class_id", CLASS_ID)
    out["max_images"] = out.get("max_images", MAX_IMAGES_PER_CLASS)
    out["batch_size"] = out.get("batch_size", BATCH_SIZE)
    out["fps"] = out.get("fps", FPS)
    out["ignore_top_k_channels"] = out.get("ignore_top_k_channels", IGNORE_TOP_K_CHANNELS)
    out["weight_interpolation"] = out.get("weight_interpolation", "plain")
    if out.get("image_indices") is None:
        if out["class_id"] is not None:
            cid = out["class_id"]
            if not 0 <= cid < len(class_names):
                raise ValueError(f"class_id {cid} outside 0..{len(class_names) - 1}")
            found = [i for i, label in enumerate(labels) if label == cid]
            limit = out["max_images"]
            out["image_indices"] = found if limit is None else found[:limit]
        else:
            out["image_indices"] = list(IMAGE_INDICES)
    out["name"] = run_folder_name(out)
    return out


def run_one(model, dataset, class_names: list[str], spec: dict) -> dict:
    name = spec["name"]
    run_dir = out_dir_for(spec) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ===")
    save_inputs(spec, dataset, class_names, run_dir)

    method = spec.get("method")
    cached = load_cached_run(run_dir, spec)
    if cached is not None:
        res = cached
        model_key = res.get("model") or (
            _SHARED_MODEL_KEY if is_shared_run(spec) else spec.get("model")
        )
        blocks = res.get("blocks")
        overlay_label = res.get("overlay_label") or name
        res["name"] = name
        res["spec"] = spec
        res["model"] = model_key
        res["overlay_label"] = overlay_label
        print(
            f"  loaded {MEAN_MAPS_NAME} (skip trajectory) "
            f"D={res['D']} ES={res['euler_step']:g} method={res.get('method') or 'RK1'}"
        )
    else:
        if is_shared_run(spec):
            D = int(spec["D"])
            step = spec.get("euler_step")
            if step is None:
                step = model.stage3_length / D
            field_blocks = [model.deltifiedStage3[0]] * D
            blocks = None
            wi = "plain"
            model_key = spec.get("model") or _SHARED_MODEL_KEY
            overlay_label = f"D={D}"
            enter_fn = lambda batch, m=model: m.stage2(m.stage1(m.stem(batch)))
            print(
                f"shared D={D} euler_step={step:.4g} method={method or 'RK1'} "
                f"batch={spec['batch_size']} n={len(spec['image_indices'])}"
            )
        else:
            model_key = spec["model"]
            n_blocks = stage3_n_blocks(model_key, model)
            blocks = resolve_blocks(spec, n_blocks)
            step = float(spec["euler_step"])
            wi = spec.get("weight_interpolation", "plain")
            field_blocks = stage3_field_blocks(
                model_key,
                model,
                blocks,
                euler_step=step,
                weight_interpolation=wi,
            )
            D = len(field_blocks)
            overlay_label = name
            enter_fn = lambda batch, m=model, k=model_key: enter_stage3(k, m, batch)
            print(
                f"{model_key} schedule={blocks} (D={D}) euler_step={step:.4g} "
                f"method={method or 'RK1'} weight_interpolation={wi} "
                f"batch={spec['batch_size']} n={len(spec['image_indices'])}"
            )

        res = trajectory_stats(
            enter_fn,
            field_blocks,
            spec["image_indices"],
            euler_step=step,
            method=method,
            batch_size=spec["batch_size"],
            dataset=dataset,
            block_schedule=blocks if blocks is not None else [0] * D,
        )
        res["name"] = name
        res["spec"] = spec
        res["blocks"] = blocks
        res["model"] = model_key
        res["weight_interpolation"] = wi
        res["overlay_label"] = overlay_label

        ignored = apply_ignore_top_k_channels(res, spec["ignore_top_k_channels"])
        if ignored:
            print(f"ignored top-{spec['ignore_top_k_channels']} channels by ||h|| (WxH): {ignored}")

    channel_kinds = [k for k in VIDEO_MAPS if k in ("h", "x")]
    rank_kind = channel_kinds[0] if channel_kinds else "h"
    if CHANNELS is not None:
        channels = list(CHANNELS)
    elif res.get("resolved_channels"):
        channels = [int(c) for c in res["resolved_channels"]]
    else:
        channels = most_active_channels(res[f"mean_{rank_kind}"], N_AUTO_CHANNELS)
    res["resolved_channels"] = channels
    print(f"channels={channels}")

    save_tables_and_config(res, class_names, run_dir)
    save_metric_plots(res, run_dir)
    write_videos(res, channels, run_dir)

    # Static overlays (always). Animations are gated by VIDEO_MAPS via write_videos.
    write_static(
        run_dir / "scatter_ch_x.png",
        lambda: scatter_vs_channel(
            res["mean_x"],
            title=f"{name} | {KIND_TITLES['scatter_ch_x']}",
            path=run_dir / "scatter_ch_x.png",
            ylabel="x_d",
            channel=SCATTER_CHANNEL,
        ),
    )
    write_static(
        run_dir / "scatter_ch_h.png",
        lambda: scatter_vs_channel(
            res["mean_h"],
            title=f"{name} | {KIND_TITLES['scatter_ch_h']}",
            path=run_dir / "scatter_ch_h.png",
            ylabel=r"$h_d\ (=\Delta x_d/\mathrm{ES})$",
            channel=SCATTER_CHANNEL,
        ),
    )
    write_static(
        run_dir / "scatter_h.png",
        lambda: scatter_residual_field(res, path=run_dir / "scatter_h.png", channel=SCATTER_CHANNEL),
    )
    write_static(
        run_dir / "scatter_ch_x_means.png",
        lambda: scatter_vs_channel_means(
            res["mean_x"],
            title=f"{name} | {KIND_TITLES['scatter_ch_x_means']}",
            path=run_dir / "scatter_ch_x_means.png",
            ylabel=r"$\mathrm{mean}_{H,W}(x_d)$",
            channel=SCATTER_CHANNEL,
        ),
    )
    write_static(
        run_dir / "scatter_ch_h_means.png",
        lambda: scatter_vs_channel_means(
            res["mean_h"],
            title=f"{name} | {KIND_TITLES['scatter_ch_h_means']}",
            path=run_dir / "scatter_ch_h_means.png",
            ylabel=r"$\mathrm{mean}_{H,W}(h_d)$",
            channel=SCATTER_CHANNEL,
        ),
    )
    write_static(
        run_dir / "scatter_h_means.png",
        lambda: scatter_residual_field_means(
            res, path=run_dir / "scatter_h_means.png", channel=SCATTER_CHANNEL
        ),
    )
    write_static(
        run_dir / "spaghetti_x.png",
        lambda: spaghetti_trajectories(
            res["mean_x"],
            title=f"{name} | spaghetti x_d(t)",
            path=run_dir / "spaghetti_x.png",
            ylabel="x_d",
            euler_step=res["euler_step"],
            channel=SCATTER_CHANNEL,
        ),
    )
    write_static(
        run_dir / "spaghetti_h.png",
        lambda: spaghetti_trajectories(
            res["mean_h"],
            title=f"{name} | spaghetti h_d(t)",
            path=run_dir / "spaghetti_h.png",
            ylabel=r"$h_d\ (=\Delta x_d/\mathrm{ES})$",
            euler_step=res["euler_step"],
            channel=SCATTER_CHANNEL,
        ),
    )
    write_static(
        run_dir / "spaghetti_x_means.png",
        lambda: spaghetti_trajectories_means(
            res["mean_x"],
            title=f"{name} | spaghetti mean(x_d)(t)",
            path=run_dir / "spaghetti_x_means.png",
            ylabel=r"$\mathrm{mean}_{H,W}(x_d)$",
            euler_step=res["euler_step"],
            channel=SCATTER_CHANNEL,
        ),
    )
    write_static(
        run_dir / "spaghetti_h_means.png",
        lambda: spaghetti_trajectories_means(
            res["mean_h"],
            title=f"{name} | spaghetti mean(h_d)(t)",
            path=run_dir / "spaghetti_h_means.png",
            ylabel=r"$\mathrm{mean}_{H,W}(h_d)$",
            euler_step=res["euler_step"],
            channel=SCATTER_CHANNEL,
        ),
    )
    print(f"done -> {run_dir}")

    # Do NOT keep mean_x/mean_h in memory across runs (~800MB each for ResNet R100).
    # Overlay reloads from mean_maps.pt on disk.
    light = {
        "name": res["name"],
        "model": model_key,
        "D": res["D"],
        "blocks": blocks,
        "overlay_label": overlay_label,
        "euler_step": res["euler_step"],
        "n_images": res["n_images"],
        "run_dir": str(run_dir),
        "map_shape": tuple(res["mean_x"].shape[1:]),
    }
    del res
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return light


def load_overlay_maps(meta: dict) -> dict:
    """Reload mean_x/mean_h from disk for SCATTER_OVERLAY_BY_D."""
    pt = Path(meta["run_dir"]) / MEAN_MAPS_NAME
    try:
        maps = torch.load(pt, map_location="cpu", weights_only=True)
    except TypeError:
        maps = torch.load(pt, map_location="cpu")
    out = dict(meta)
    out["mean_x"] = maps["mean_x"]
    out["mean_h"] = maps["mean_h"]
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list-only", action="store_true", help="Print resolved RUNS and exit")
    p.add_argument("--only", nargs="+", default=None, help="Only these run names")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already have config.json (coarse; per-figure skip is on by default)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing figures and recompute trajectories (ignore mean_maps.pt cache)",
    )
    p.add_argument("--keep-frames", action="store_true", help="Keep PNG frame directories")
    return p.parse_args()


def main() -> None:
    global KEEP_FRAMES, FORCE_RECOMPUTE
    load_dotenv(_REPO_ROOT / ".env")
    args = parse_args()
    if args.keep_frames:
        KEEP_FRAMES = True
    if args.force:
        FORCE_RECOMPUTE = True
        print("FORCE_RECOMPUTE: overwriting existing figures / ignoring trajectory cache")

    OUT_DIR_SHARED.mkdir(parents=True, exist_ok=True)
    OUT_DIR_INTERPOLED.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  backbone={BACKBONE}  out_shared={OUT_DIR_SHARED}")
    print(f"  out_interpoled={OUT_DIR_INTERPOLED}")
    if BACKBONE == "random_init":
        print(
            f"random_init seed={RANDOM_INIT_SEED}  "
            f"LayerScale={RANDOM_INIT_LAYERSCALE:g}  runs={len(RUNS)}"
        )
    elif BACKBONE == "interpoled":
        print(f"models={INTERPOLED_MODELS}")
        print(
            f"random_init probes={len(RUNS_RANDOM_INIT)}  "
            f"seed={RANDOM_INIT_SEED}  LayerScale={RANDOM_INIT_LAYERSCALE:g}"
        )
        print(f"shared ckpt={SHARED_CHECKPOINT}")
        print(f"convnext ckpt={INTERPOLED_CONVNEXT_CHECKPOINT}")
        print(f"convnext_droppath0 ckpt={INTERPOLED_CONVNEXT_DROPPATH0_CHECKPOINT}")
        print(f"total runs={len(RUNS)}")
    elif BACKBONE == "shared":
        print(f"shared ckpt={SHARED_CHECKPOINT}")

    dataset = ImageNetDataset(split=SPLIT, transforms=build_val_transforms())
    labels: list[int] = dataset.ds.data.column("label").to_pylist()
    class_names: list[str] = dataset.ds.features["label"].names
    print(f"{SPLIT}: {len(dataset)} images, {len(class_names)} classes")

    runs = [resolve_run(spec, labels=labels, class_names=class_names) for spec in RUNS]
    if args.only:
        wanted = set(args.only)
        runs = [r for r in runs if r["name"] in wanted]
        missing = wanted - {r["name"] for r in runs}
        if missing:
            raise SystemExit(f"unknown --only names: {sorted(missing)}")

    print("runs:")
    for spec in runs:
        cid = spec["class_id"]
        who = f"class {cid} — {class_names[cid]}" if cid is not None else "hand-picked"
        if is_shared_run(spec):
            sched = f"D={spec['D']}"
            model_s = spec.get("model") or _SHARED_MODEL_KEY
        else:
            model_s = spec.get("model", "?")
            if "repeats" in spec:
                sched = f"repeats={spec['repeats']} ES={spec['euler_step']:g}"
            else:
                sched = f"blocks={spec['blocks']} ES={spec['euler_step']:g}"
        print(
            f"  {spec['name']}: model={model_s}, {who}, {sched}, "
            f"wi={spec.get('weight_interpolation', 'plain')}, "
            f"n={len(spec['image_indices'])}, bs={spec['batch_size']}, "
            f"fps={spec['fps']}, ignore_top_k={spec['ignore_top_k_channels']}"
        )

    if args.list_only:
        return

    needed_rand: dict[str, bool] = {}
    for r in runs:
        key = r.get("model") or _SHARED_MODEL_KEY
        needed_rand[key] = needed_rand.get(key, False) or bool(r.get("random_init"))
    models: dict[str, nn.Module] = {}
    for key, rand in sorted(needed_rand.items()):
        models[key] = load_interpoled_model(key, random_init=rand).to(device).eval()
        if key in _SHARED_MODEL_KEYS:
            assert models[key].deltifiedStage3[0].eulerStep == 1.0

    completed: list[dict] = []
    for spec in runs:
        marker = out_dir_for(spec) / spec["name"] / "config.json"
        if args.skip_existing and marker.is_file():
            print(f"skip existing {spec['name']}")
            continue
        model_key = spec.get("model") or _SHARED_MODEL_KEY
        completed.append(run_one(models[model_key], dataset, class_names, spec))

    if SCATTER_OVERLAY_BY_D and completed:
        by_key: dict[tuple, list[dict]] = {}
        for res in completed:
            key = (res.get("model"), tuple(res.get("map_shape") or ()), res["n_images"])
            by_key.setdefault(key, []).append(res)
        for group_meta in by_key.values():
            if len(group_meta) < 2:
                continue
            group_meta = sorted(group_meta, key=lambda r: (r["D"], r["name"]))
            tag = "-".join(r.get("overlay_label", str(r["D"])) for r in group_meta)
            tag = tag.replace(" ", "_").replace("/", "div")[:120]
            overlay_root = Path(group_meta[0]["run_dir"]).parent
            group = [load_overlay_maps(m) for m in group_meta]
            try:
                out = overlay_root / f"scatter_h_overlay_{tag}.png"
                write_static(
                    out,
                    lambda g=group, p=out: scatter_overlay_by_d(g, path=p, channel=SCATTER_CHANNEL),
                )
                if _exists_nonempty(out):
                    print(f"overlay scatter -> {out}")
                out_means = overlay_root / f"scatter_h_means_overlay_{tag}.png"
                write_static(
                    out_means,
                    lambda g=group, p=out_means: scatter_overlay_means_by_d(
                        g, path=p, channel=SCATTER_CHANNEL
                    ),
                )
                if _exists_nonempty(out_means):
                    print(f"overlay scatter means -> {out_means}")
            finally:
                del group
                gc.collect()


if __name__ == "__main__":
    main()
