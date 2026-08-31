"""Stage-3 feature-map trajectories on the shared D=9 ConvNeXt checkpoint.

Headless port of notebooks/featureMapExplorer.ipynb for Leonardo. Processes one
RUN at a time, writes under outputs/featureMaps/<run name>/, and prefers mp4
over giant in-memory GIFs (the notebook crash mode on a laptop).

Submit:
  source .env && sbatch --account="$SLURM_ACCOUNT" jobs/feature_map_explorer.sh

Or locally:
  python scripts/feature_map_explorer.py
  python scripts/feature_map_explorer.py --list-only
  python scripts/feature_map_explorer.py --only D9_rk1_c289_n1_bs1
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
from tqdm import tqdm

from data.imagenet import ImageNetDataset
from data.transforms.transforms import IMAGENET_MEAN, IMAGENET_STD, build_val_transforms
from models.backbones.delta_convnext import DeltaConvNext
from utils.env import load_dotenv

# --------------------------------------------------------------------------- config
CHECKPOINT = _REPO_ROOT / "outputs" / "shared_convnextv1_imagenet" / "weights" / "last.pth"
OUT_DIR = _REPO_ROOT / "outputs" / "featureMaps"

SPLIT = "validation"
IMAGE_INDICES = [0, 17, 4242]
CLASS_ID: int | None = 207
# Default N images to average curves over (mean ± std).
MAX_IMAGES_PER_CLASS: int | None = 100
BATCH_SIZE = 8
FPS = 10.0

# Edit this list for the experiments to run on Leonardo.
# Optional per-run field: ignore_top_k_channels (default = IGNORE_TOP_K_CHANNELS below).
RUNS: list[dict] = [
    {"name": "D100_rk1_c7_n50_bs16_no_ignore", "D": 100, "method": None, "class_id": 7, "max_images": 50, "batch_size": 4, "fps": 10, "ignore_top_k_channels": 0},
    {"name": "D100_rk1_c7_n50_bs16", "D": 100, "method": None, "class_id": 7, "max_images": 50, "batch_size": 4, "fps": 10, "ignore_top_k_channels": 1},
    {"name": "D100_rk1_c7_n1_bs1", "D": 100, "method": None, "class_id": 7, "max_images": 1, "batch_size": 1, "fps": 10, "ignore_top_k_channels": 1},
    {"name": "D100_rk1_c207_n50_bs16", "D": 100, "method": None, "class_id": 207, "max_images": 50, "batch_size": 4, "fps": 10, "ignore_top_k_channels": 1},
    {"name": "D100_rk1_c207_n1_bs1", "D": 100, "method": None, "class_id": 207, "max_images": 1, "batch_size": 1, "fps": 10, "ignore_top_k_channels": 1},
    {"name": "D100_rk1_c282_n50_bs16", "D": 100, "method": None, "class_id": 282, "max_images": 50, "batch_size": 4, "fps": 10, "ignore_top_k_channels": 1},
    {"name": "D100_rk1_c282_n1_bs1", "D": 100, "method": None, "class_id": 282, "max_images": 1, "batch_size": 1, "fps": 10, "ignore_top_k_channels": 1},
    {"name": "D100_rk1_c289_n50_bs16", "D": 100, "method": None, "class_id": 289, "max_images": 50, "batch_size": 4, "fps": 10, "ignore_top_k_channels": 1},
    {"name": "D100_rk1_c289_n1_bs1", "D": 100, "method": None, "class_id": 289, "max_images": 1, "batch_size": 1, "fps": 10, "ignore_top_k_channels": 1},
    {"name": "D50_rk1_c289_n1_bs1", "D": 50, "method": None, "class_id": 289, "max_images": 1, "batch_size": 1, "fps": 5, "ignore_top_k_channels": 1},
    {"name": "D25_rk1_c289_n1_bs1", "D": 25, "method": None, "class_id": 289, "max_images": 1, "batch_size": 1, "fps": 3, "ignore_top_k_channels": 1},
    {"name": "D9_rk1_c289_n1_bs1", "D": 9, "method": None, "class_id": 289, "max_images": 1, "batch_size": 1, "fps": 1, "ignore_top_k_channels": 1},
    {"name": "D8_rk1_c289_n1_bs1", "D": 8, "method": None, "class_id": 289, "max_images": 1, "batch_size": 1, "fps": 1, "ignore_top_k_channels": 1},
    {"name": "D5_rk1_c289_n1_bs1", "D": 5, "method": None, "class_id": 289, "max_images": 1, "batch_size": 1, "fps": 0.5, "ignore_top_k_channels": 1},
    {"name": "D3_rk1_c289_n1_bs1", "D": 3, "method": None, "class_id": 289, "max_images": 1, "batch_size": 1, "fps": 0.5, "ignore_top_k_channels": 1},
]

CHANNELS: list[int] | None = None
N_AUTO_CHANNELS = 3
# Default when a RUNS entry omits ignore_top_k_channels. 0 disables.
IGNORE_TOP_K_CHANNELS = 1
VIDEO_MAPS = ["h", "x", "cos_h", "norm_h", "l2_h", "h_CH", "x_CH", "scatter_x", "scatter_h"]

CMAP_X = "viridis"
CMAP_H = "RdBu_r"
CMAP_COS = "magma"
CMAP_NORM = "magma"
SHARED_SCALE = True
SAVE_TENSORS = False
GRID_MAX_FRAMES = 36
SCATTER_MAX_POINTS = 99999999
# None = all channels in static scatter; int = that channel only.
SCATTER_CHANNEL: int | None = None
# After all RUNS, write a combined residual scatter coloured/legended by D.
SCATTER_OVERLAY_BY_D = True
# Animation: sample N features once, track the same indices across depth.
SCATTER_ANIM_MAX_POINTS = 8000
SCATTER_ANIM_SEED = 0
# turbo: high local contrast (nearby channels look distinct); better than viridis here.
SCATTER_ANIM_CMAP = "turbo"
# Building a GIF loads every frame into RAM — skip when D is large.
GIF_MAX_FRAMES = 9999
# Delete PNG frame dirs after the video is written (saves a lot of disk).
KEEP_FRAMES = True
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
    "cos_h": "1 - cos(h_d[i,j], h_{d+1}[i,j])",
    "cos_x": "1 - cos(x_d[i,j], x_{d+1}[i,j])",
    "norm_h": "||h_d[:,i,j]||",
    "l2_h": "||h_d[:,i,j] - h_{d+1}[:,i,j]||",
    "h_CH": "h_d C×H slice at W//2",
    "x_CH": "x_d C×H slice at W//2",
    "scatter_x": "X cumulative: x_d → x_{d+1}",
    "scatter_h": "residual field: x_d → f(x_d) = h_d / ES",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- helpers
def run_folder_name(spec: dict) -> str:
    if spec.get("name"):
        return spec["name"]
    method = (spec.get("method") or "RK1").lower()
    parts = [f"D{spec['D']}_{method}"]
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
    return "_".join(parts)


def load_shared_convnext(checkpoint: Path) -> DeltaConvNext:
    model = DeltaConvNext(useDeltas=False)
    model.rewire()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    stale = [k for k in missing if not k.startswith("tail.")]
    if stale or unexpected:
        raise RuntimeError(f"missing={stale} unexpected={unexpected}")
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    print(f"Loaded {checkpoint}" + (f" (epoch {epoch})" if epoch is not None else ""))
    return model


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
    _, s, _ = torch.linalg.svd(matrix.float().cpu(), full_matrices=False)
    return participation_ratio(s)


@torch.no_grad()
def trajectory_stats(model, shared_block, indices, *, D, euler_step, method, batch_size, dataset):
    """Integrate the shared block and collect per-image dynamics + mean maps.

    Stored residual maps keep the existing convention h = block(x) - x (no ES).
    Derived field used in plots/metrics: f = h / ES.
    """
    method = method.upper() if isinstance(method, str) else method
    n = len(indices)
    w = 1.0 / n
    es = float(euler_step)
    mean_x = mean_h = cos_map_h = cos_map_x = norm_map_h = l2_map_h = None

    # Per-image series (filled in index order).
    norm_h_i = torch.zeros(n, D + 1, dtype=torch.float64)
    norm_x_i = torch.zeros(n, D + 1, dtype=torch.float64)
    cos_flat_i = torch.zeros(n, D, dtype=torch.float64)  # 1 - cos(h_d, h_{d+1}) flat
    cos_spatial_i = torch.zeros(n, D, dtype=torch.float64)
    cos_x_flat_i = torch.zeros(n, D, dtype=torch.float64)
    cos_x_spatial_i = torch.zeros(n, D, dtype=torch.float64)
    omega_i = torch.zeros(n, D, dtype=torch.float64)  # arccos(cos)/ES
    align_h0_i = torch.zeros(n, D + 1, dtype=torch.float64)  # cos(h_0, h_d)
    rect_L_i = torch.zeros(n, dtype=torch.float64)
    rect_N_i = torch.zeros(n, dtype=torch.float64)
    rect_R_i = torch.zeros(n, dtype=torch.float64)
    pr_depth_i = torch.zeros(n, dtype=torch.float64)
    pr_spatial_i = torch.zeros(n, dtype=torch.float64)

    def f(y):
        return shared_block(y) - y

    def load_batch(idxs: list[int]) -> torch.Tensor:
        return torch.stack([dataset[i][0] for i in idxs])

    starts = range(0, n, batch_size)
    desc = f"D={D} h={es:.4g} m={method or 'RK1'}"
    for start in tqdm(starts, desc=desc, leave=True):
        batch_idx = indices[start : start + batch_size]
        batch = load_batch(batch_idx).to(device, non_blocking=True)
        bsz = batch.shape[0]
        x = model.stage2(model.stage1(model.stem(batch)))
        if mean_x is None:
            mean_x = torch.zeros((D + 1, *x.shape[1:]), device=device)
            mean_h = torch.zeros_like(mean_x)
            cos_map_h = torch.zeros((D, *x.shape[2:]), device=device)
            cos_map_x = torch.zeros_like(cos_map_h)
            norm_map_h = torch.zeros((D + 1, *x.shape[2:]), device=device)
            l2_map_h = torch.zeros((D, *x.shape[2:]), device=device)

        # Per-image residual trajectory for SVD / rectitude: [B, D+1, C, H, W] is heavy;
        # keep flattened [B, D+1, dim] on CPU.
        h = f(x)
        x0 = x.detach()
        h0_flat = h.flatten(1).detach()
        H_rows: list[torch.Tensor] = []
        path_len = torch.zeros(bsz, dtype=torch.float64, device=device)

        for d in range(D + 1):
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
            # L = ES * sum ||f_d|| with f = h/ES → sum ||h_d|| over applied steps.
            if d < D:
                path_len = path_len + n_h.double()

            if d == D:
                break

            x_next = rk_step(f, x, euler_step, method, k1=h)
            h_next = f(x_next)
            h_next_flat = h_next.flatten(1)
            x_next_flat = x_next.flatten(1)

            cos_hh = F.cosine_similarity(h_flat, h_next_flat, dim=1).clamp(-1.0, 1.0)
            cos_flat_i[start : start + bsz, d] = (1.0 - cos_hh).double().cpu()
            omega_i[start : start + bsz, d] = torch.arccos(cos_hh).double().cpu() / max(es, 1e-12)
            cos_x_flat_i[start : start + bsz, d] = (
                1.0 - F.cosine_similarity(x_flat, x_next_flat, dim=1)
            ).double().cpu()

            loc_h = 1.0 - F.cosine_similarity(h, h_next, dim=1)
            loc_x = 1.0 - F.cosine_similarity(x, x_next, dim=1)
            cos_map_h[d] += w * loc_h.sum(0)
            cos_map_x[d] += w * loc_x.sum(0)
            cos_spatial_i[start : start + bsz, d] = loc_h.mean(dim=(1, 2)).double().cpu()
            cos_x_spatial_i[start : start + bsz, d] = loc_x.mean(dim=(1, 2)).double().cpu()
            l2_map_h[d] += w * (h - h_next).norm(dim=1).sum(0)
            x, h = x_next, h_next

        # Rectitude: L = sum_d ||h_d|| (= ES * sum ||f_d||), N = ||x_D - x_0||, R = N/L.
        disp = (x - x0).flatten(1).norm(dim=1).double()
        rect_L_i[start : start + bsz] = path_len.cpu()
        rect_N_i[start : start + bsz] = disp.cpu()
        rect_R_i[start : start + bsz] = (disp / path_len.clamp_min(1e-30)).cpu()

        # SVD PR on H_i shaped (D, dim) using h_0..h_{D-1}, and mid-step (H*W, C).
        H_stack = torch.stack(H_rows[:D], dim=1)  # [B, D, dim]
        mid = D // 2
        for bi in range(bsz):
            pr_depth_i[start + bi] = svd_participation_ratio(H_stack[bi])
            h_mid = H_rows[mid][bi].reshape(x.shape[1], -1).T  # (H*W, C)
            pr_spatial_i[start + bi] = svd_participation_ratio(h_mid)

        del batch, x, h, x0, H_rows, H_stack
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def mean_std(arr: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        return arr.mean(0).numpy(), arr.std(0, unbiased=False).numpy()

    depths = torch.arange(D + 1)
    norm_h_mean, norm_h_std = mean_std(norm_h_i)
    norm_x_mean, norm_x_std = mean_std(norm_x_i)
    # f = h / ES
    norm_f_mean, norm_f_std = norm_h_mean / max(es, 1e-12), norm_h_std / max(es, 1e-12)
    norm_f_over_x_mean = norm_f_mean / np.clip(norm_x_mean, 1e-30, None)
    # delta method-ish std for ratio: skip exact; use std of per-image ratios.
    norm_f_over_x_i = (norm_h_i / max(es, 1e-12)) / norm_x_i.clamp_min(1e-30)
    norm_f_over_x_std = norm_f_over_x_i.std(0, unbiased=False).numpy()

    cos_flat_mean, cos_flat_std = mean_std(cos_flat_i)
    cos_spatial_mean, cos_spatial_std = mean_std(cos_spatial_i)
    omega_mean, omega_std = mean_std(omega_i)
    align_mean, align_std = mean_std(align_h0_i)

    # Correlation between flat and spatial 1-cos series (on the mean curves).
    if D >= 2 and cos_flat_mean.std() > 0 and cos_spatial_mean.std() > 0:
        cos_flat_spatial_corr = float(np.corrcoef(cos_flat_mean, cos_spatial_mean)[0, 1])
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
            "norm_f": norm_f_mean,
            "norm_f_std": norm_f_std,
            "norm_f_over_norm_x": norm_f_over_x_mean,
            "norm_f_over_norm_x_std": norm_f_over_x_std,
            "cos_h0_hd": align_mean,
            "cos_h0_hd_std": align_std,
        }
    )
    # Keep legacy column name used elsewhere.
    norms["norm_h_over_norm_x"] = norms["norm_h"] / norms["norm_x"].clip(lower=1e-30)

    pairs = pd.DataFrame(
        {
            "d": torch.arange(D).numpy(),
            "pair": [f"{d}->{d + 1}" for d in range(D)],
            "t": (torch.arange(D).numpy() * es),
            "cos_dist_h": cos_flat_mean,
            "cos_dist_h_std": cos_flat_std,
            "cos_dist_h_spatial": cos_spatial_mean,
            "cos_dist_h_spatial_std": cos_spatial_std,
            "omega_h": omega_mean,
            "omega_h_std": omega_std,
            "cos_dist_x": mean_std(cos_x_flat_i)[0],
            "cos_dist_x_spatial": mean_std(cos_x_spatial_i)[0],
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


def spatial_maps_from_means(mean_x: torch.Tensor, mean_h: torch.Tensor) -> dict[str, torch.Tensor]:
    """Rebuild H×W video tensors from (possibly channel-masked) mean maps."""
    D = mean_x.shape[0] - 1
    cos_h = torch.stack(
        [1 - F.cosine_similarity(mean_h[d], mean_h[d + 1], dim=0) for d in range(D)]
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
    norm_f = norm_h / max(es, 1e-12)
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
            "norm_f": norm_f.numpy(),
            "norm_f_std": 0.0,
            "norm_f_over_norm_x": (norm_f / norm_x.clamp_min(1e-30)).numpy(),
            "norm_f_over_norm_x_std": 0.0,
            "cos_h0_hd": align.numpy(),
            "cos_h0_hd_std": 0.0,
        }
    )
    norms["norm_h_over_norm_x"] = norms["norm_h"] / norms["norm_x"].clip(lower=1e-30)

    cos_h = []
    cos_spatial = []
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
        loc_h = 1 - F.cosine_similarity(mean_h[d], mean_h[d + 1], dim=0)
        loc_x = 1 - F.cosine_similarity(mean_x[d], mean_x[d + 1], dim=0)
        cos_spatial.append(loc_h.mean().item())
        cos_x_spatial.append(loc_x.mean().item())
    pairs = pd.DataFrame(
        {
            "d": list(range(D)),
            "pair": [f"{d}->{d + 1}" for d in range(D)],
            "t": [d * es for d in range(D)],
            "cos_dist_h": cos_h,
            "cos_dist_h_std": 0.0,
            "cos_dist_h_spatial": cos_spatial,
            "cos_dist_h_spatial_std": 0.0,
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
    res.update(spatial_maps_from_means(res["mean_x"], res["mean_h"]))
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


def draw_style(maps: torch.Tensor, kind: str) -> dict:
    if kind.startswith("cos"):
        cmap = CMAP_COS
        vmax = maps.max().item() if SHARED_SCALE else None
        return {"cmap": cmap, "vmin": 0.0, **({"vmax": vmax} if vmax is not None else {})}
    if kind in ("norm_h", "l2_h"):
        cmap = CMAP_NORM
        vmax = maps.max().item() if SHARED_SCALE else None
        return {"cmap": cmap, "vmin": 0.0, **({"vmax": vmax} if vmax is not None else {})}
    signed = kind in ("h", "h_CH")
    cmap = CMAP_H if signed else CMAP_X
    if not SHARED_SCALE:
        return {"cmap": cmap}
    if signed:
        limit = maps.abs().max().item()
        return {"cmap": cmap, "vmin": -limit, "vmax": limit}
    return {"cmap": cmap, "vmin": maps.min().item(), "vmax": maps.max().item()}


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


def _load_rgb_frames(frame_dir: Path) -> list[Image.Image]:
    frames = sorted(frame_dir.glob("frame_*.png"))
    images = [Image.open(p).convert("RGB") for p in frames]
    if len({im.size for im in images}) <= 1:
        return images
    width = max(im.width for im in images)
    height = max(im.height for im in images)
    padded = []
    for im in images:
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(im, ((width - im.width) // 2, (height - im.height) // 2))
        im.close()
        padded.append(canvas)
    return padded


def _write_mp4_ffmpeg(frame_dir: Path, mp4: Path, fps: float) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    # Prefer libx264; fall back to mpeg4 when the build has no x264 (common on HPC images).
    for codec_args in (
        ["-c:v", "libx264", "-pix_fmt", "yuv420p"],
        ["-c:v", "mpeg4", "-q:v", "5"],
    ):
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(fps),
                "-i", str(frame_dir / "frame_%04d.png"),
                *codec_args,
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                str(mp4),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and mp4.is_file():
            return True
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
    h, w = first.shape[:2]
    w -= w % 2
    h -= h % 2
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
    return mp4.is_file() and mp4.stat().st_size > 0


def write_video(frame_dir: Path, out_stem: Path, *, fps: float, n_frames: int) -> dict[str, Path]:
    """Write mp4 (ffmpeg or OpenCV) and/or gif. Never delete frames if nothing was written."""
    written: dict[str, Path] = {}
    mp4 = out_stem.with_suffix(".mp4")
    if _write_mp4_ffmpeg(frame_dir, mp4, fps) or _write_mp4_opencv(frame_dir, mp4, fps):
        written["mp4"] = mp4

    # GIF loads every frame in RAM. Prefer mp4 when D is large; if mp4 failed, always
    # write a gif anyway so the run still produces a viewable animation.
    want_gif = n_frames <= GIF_MAX_FRAMES or "mp4" not in written
    if want_gif:
        images = _load_rgb_frames(frame_dir)
        gif = out_stem.with_suffix(".gif")
        images[0].save(
            gif,
            save_all=True,
            append_images=images[1:],
            duration=max(1, int(round(1000 / max(fps, 1e-3)))),
            loop=0,
        )
        written["gif"] = gif
        for im in images:
            im.close()
        if n_frames > GIF_MAX_FRAMES and "mp4" not in written:
            print(f"  no ffmpeg/opencv mp4 — wrote gif with {n_frames} frames instead")

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
    fig.savefig(run_dir / "inputs.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_tables_and_config(res: dict, class_names: list[str], run_dir: Path) -> None:
    spec = res["spec"]
    res["norms"].to_csv(run_dir / "table_norms.csv", index=False)
    res["pairs"].to_csv(run_dir / "table_pairs.csv", index=False)
    if "scalars" in res:
        res["scalars"].to_csv(run_dir / "table_scalars.csv", index=False)
    config = {
        "name": res["name"],
        "D": res["D"],
        "euler_step": res["euler_step"],
        "method": res["method"] or "RK1",
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
            run_dir / "mean_maps.pt",
        )


def _plot_mean_std(ax, x, mean, std, *, label: str, **plot_kw):
    mean = np.asarray(mean)
    std = np.asarray(std)
    ax.plot(x, mean, marker="o", ms=3, label=label, **plot_kw)
    ax.fill_between(x, mean - std, mean + std, alpha=0.25)


def save_metric_plots(res: dict, run_dir: Path) -> None:
    norms, pairs = res["norms"], res["pairs"]
    es = res["euler_step"]
    label = f"{res['name']} (D={res['D']}, n={res['n_images']})"
    t_state = norms["t"]
    t_pair = pairs["t"] if "t" in pairs.columns else pairs["d"] * es

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")

    # (0,0) ||f|| and ||f||/||x||
    ax = axes[0, 0]
    _plot_mean_std(
        ax, t_state, norms["norm_f"], norms.get("norm_f_std", 0.0), label=r"$\|f_d\|$"
    )
    _plot_mean_std(
        ax,
        t_state,
        norms["norm_f_over_norm_x"],
        norms.get("norm_f_over_norm_x_std", 0.0),
        label=r"$\|f_d\| / \|x_d\|$",
    )
    ax.set(xlabel="t = d · ES", ylabel="norm", title=r"field norm ($\mathrm{f}=h/\mathrm{ES}$)")
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

    # (1,1) spatial mean 1-cos (kept; flat series only used for correlation)
    ax = axes[1, 1]
    _plot_mean_std(
        ax,
        t_pair,
        pairs["cos_dist_h_spatial"],
        pairs.get("cos_dist_h_spatial_std", 0.0),
        label=label,
    )
    ax.set(
        xlabel="t",
        ylabel=r"mean$_{i,j}\,[1-\cos(h_d[i,j], h_{d+1}[i,j])]$",
        title="mean per-location cosine distance (h)",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.savefig(run_dir / "metrics.png", dpi=140)
    plt.close(fig)

    corr = res.get("cos_flat_spatial_corr", float("nan"))
    print(f"  corr(flat 1-cos, spatial 1-cos) = {corr:.6f}")
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
    lo = min(flat_in.min().item(), flat_out.min().item())
    hi = max(flat_in.max().item(), flat_out.max().item())
    if draw_y_equals_x:
        ax.plot([lo, hi], [lo, hi], color="0.7", lw=1, zorder=0, label="y = x")
    ax.axhline(0.0, color="0.85", lw=1, zorder=0)
    for d in range(n):
        a = flat_in[d]
        b = flat_out[d]
        if a.numel() > max_points:
            idx = torch.randperm(a.numel())[:max_points]
            a, b = a[idx], b[idx]
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
            label=(label if d == 0 else None)
            if label is not None
            else (f"d={d}" if n <= 12 or d in (0, n // 2, n - 1) else None),
            **c_kw,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if own_fig:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    if own_fig:
        ax.legend(fontsize=7, markerscale=2)
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
    """Static scatter of (x_d, f(x_d)) with f = h/ES."""
    es = float(res["euler_step"])
    scatter_io(
        res["mean_x"],
        res["mean_h"] / max(es, 1e-12),
        title=f"{res['name']} | {KIND_TITLES['scatter_h']}",
        path=path,
        xlabel="x_d",
        ylabel=r"$f(x_d) = h_d / \mathrm{ES}$",
        max_points=max_points,
        channel=channel,
        draw_y_equals_x=False,
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
        es = float(res["euler_step"])
        color = cmap(i / max(len(results) - 1, 1))
        scatter_io(
            res["mean_x"],
            res["mean_h"] / max(es, 1e-12),
            title="",
            path=path,
            xlabel="x_d",
            ylabel=r"$f(x_d) = h_d / \mathrm{ES}$",
            max_points=max_points,
            channel=channel,
            draw_y_equals_x=False,
            label=f"D={res['D']}",
            ax=ax,
            color=color,
        )
    ax.set_title("residual field overlay by D")
    ax.legend(fontsize=8, markerscale=2)
    fig.savefig(path, dpi=140)
    plt.close(fig)


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

    Samples feature indices once, then draws the same points at every depth so the
    cloud can be watched moving. Colour encodes channel index (stable over depth).
    """
    assert inputs.shape == outputs.shape and inputs.ndim == 4
    n, n_c, n_h, n_w = inputs.shape
    flat_in = inputs.reshape(n, -1)
    flat_out = outputs.reshape(n, -1)
    n_feat = flat_in.shape[1]
    spatial = n_h * n_w
    if n_feat > max_points:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n_feat, generator=g)[:max_points]
    else:
        idx = torch.arange(n_feat)
    tracked_in = flat_in[:, idx]
    tracked_out = flat_out[:, idx]
    channels = (idx // spatial).numpy()
    lo = min(tracked_in.min().item(), tracked_out.min().item())
    hi = max(tracked_in.max().item(), tracked_out.max().item())
    pad = 0.02 * (hi - lo) if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad

    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)

    for d in range(n):
        fig, ax = plt.subplots(figsize=(6.5, 6), layout="constrained")
        ax.plot([lo, hi], [lo, hi], color="0.7", lw=1, zorder=0)
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


def write_videos(res: dict, channels: list[int], run_dir: Path) -> None:
    fps = float(res["spec"]["fps"])
    run_name = res["name"]
    channel_kinds = [k for k in VIDEO_MAPS if k in ("h", "x")]
    map_kinds = [k for k in VIDEO_MAPS if k in ("cos_h", "cos_x", "norm_h", "l2_h", "h_CH", "x_CH")]
    scatter_kinds = [k for k in VIDEO_MAPS if k in ("scatter_x", "scatter_h")]

    for kind in channel_kinds:
        mean_map = res[f"mean_{kind}"]
        for channel in channels:
            stem = f"{kind}_ch{channel:03d}"
            formula = f"{KIND_TITLES[kind]}  |  channel {channel}"
            frame_dir = run_dir / "frames" / stem
            render_frames(
                mean_map[:, channel],
                frame_dir,
                run_name=run_name,
                formula=formula,
                kind=kind,
            )
            if mean_map.shape[0] <= GRID_MAX_FRAMES:
                save_grid(
                    mean_map[:, channel],
                    run_dir / f"{stem}_grid.png",
                    title=f"{run_name}\n{formula}",
                    kind=kind,
                )
            written = write_video(frame_dir, run_dir / stem, fps=fps, n_frames=mean_map.shape[0])
            print(f"  {stem}: {[p.name for p in written.values()]}")

    for kind in map_kinds:
        maps = map_tensor(res, kind)
        formula = KIND_TITLES[kind]
        frame_dir = run_dir / "frames" / kind
        render_frames(maps, frame_dir, run_name=run_name, formula=formula, kind=kind)
        if maps.shape[0] <= GRID_MAX_FRAMES:
            save_grid(maps, run_dir / f"{kind}_grid.png", title=f"{run_name}\n{formula}", kind=kind)
        written = write_video(frame_dir, run_dir / kind, fps=fps, n_frames=maps.shape[0])
        print(f"  {kind}: {[p.name for p in written.values()]}")

    for kind in scatter_kinds:
        if kind == "scatter_x":
            written = scatter_io_animation(
                res["mean_x"][:-1],
                res["mean_x"][1:],
                title=f"{run_name} | {KIND_TITLES[kind]}",
                frame_dir=run_dir / "frames" / kind,
                out_stem=run_dir / kind,
                xlabel="x_d",
                ylabel="x_{d+1}",
                fps=fps,
            )
        else:
            es = float(res["euler_step"])
            written = scatter_io_animation(
                res["mean_x"],
                res["mean_h"] / max(es, 1e-12),
                title=f"{run_name} | {KIND_TITLES[kind]}",
                frame_dir=run_dir / "frames" / kind,
                out_stem=run_dir / kind,
                xlabel="x_d",
                ylabel=r"$f(x_d)=h_d/\mathrm{ES}$",
                fps=fps,
            )
        print(f"  {kind}: {[p.name for p in written.values()]}")


def resolve_run(spec: dict, *, labels: list[int], class_names: list[str]) -> dict:
    out = dict(spec)
    out["class_id"] = out.get("class_id", CLASS_ID)
    out["max_images"] = out.get("max_images", MAX_IMAGES_PER_CLASS)
    out["batch_size"] = out.get("batch_size", BATCH_SIZE)
    out["fps"] = out.get("fps", FPS)
    out["ignore_top_k_channels"] = out.get("ignore_top_k_channels", IGNORE_TOP_K_CHANNELS)
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


def run_one(model, shared_block, dataset, class_names: list[str], spec: dict) -> dict:
    name = spec["name"]
    run_dir = OUT_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ===")
    save_inputs(spec, dataset, class_names, run_dir)

    D = spec["D"]
    step = spec.get("euler_step")
    if step is None:
        step = model.stage3_length / D
    method = spec.get("method")
    print(f"D={D} euler_step={step:.4g} method={method or 'RK1'} batch={spec['batch_size']} n={len(spec['image_indices'])}")

    res = trajectory_stats(
        model,
        shared_block,
        spec["image_indices"],
        D=D,
        euler_step=step,
        method=method,
        batch_size=spec["batch_size"],
        dataset=dataset,
    )
    res["name"] = name
    res["spec"] = spec

    ignored = apply_ignore_top_k_channels(res, spec["ignore_top_k_channels"])
    if ignored:
        print(f"ignored top-{spec['ignore_top_k_channels']} channels by ||h|| (WxH): {ignored}")

    save_tables_and_config(res, class_names, run_dir)
    save_metric_plots(res, run_dir)

    channel_kinds = [k for k in VIDEO_MAPS if k in ("h", "x")]
    rank_kind = channel_kinds[0] if channel_kinds else "h"
    channels = CHANNELS if CHANNELS is not None else most_active_channels(res[f"mean_{rank_kind}"], N_AUTO_CHANNELS)
    print(f"channels={channels}")
    write_videos(res, channels, run_dir)

    # Static overlays (always). Animations are gated by VIDEO_MAPS via write_videos.
    scatter_io(
        res["mean_x"][:-1],
        res["mean_x"][1:],
        title=f"{name} | {KIND_TITLES['scatter_x']}",
        path=run_dir / "scatter_x.png",
        xlabel="x_d",
        ylabel="x_{d+1}",
        channel=SCATTER_CHANNEL,
        draw_y_equals_x=True,
    )
    scatter_residual_field(res, path=run_dir / "scatter_h.png", channel=SCATTER_CHANNEL)
    print(f"done -> {run_dir}")

    # Drop heavy maps before returning a light copy for overlay.
    light = {
        "name": res["name"],
        "D": res["D"],
        "euler_step": res["euler_step"],
        "mean_x": res["mean_x"],
        "mean_h": res["mean_h"],
        "n_images": res["n_images"],
    }
    del res
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return light


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list-only", action="store_true", help="Print resolved RUNS and exit")
    p.add_argument("--only", nargs="+", default=None, help="Only these run names")
    p.add_argument("--skip-existing", action="store_true", help="Skip runs that already have config.json")
    p.add_argument("--keep-frames", action="store_true", help="Keep PNG frame directories")
    return p.parse_args()


def main() -> None:
    global KEEP_FRAMES
    load_dotenv(_REPO_ROOT / ".env")
    args = parse_args()
    if args.keep_frames:
        KEEP_FRAMES = True

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={OUT_DIR}  checkpoint={CHECKPOINT}")

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
        print(f"  {spec['name']}: {who}, n={len(spec['image_indices'])}, bs={spec['batch_size']}, fps={spec['fps']}, ignore_top_k={spec['ignore_top_k_channels']}")

    if args.list_only:
        return

    if not CHECKPOINT.is_file():
        raise SystemExit(f"checkpoint not found: {CHECKPOINT}")

    model = load_shared_convnext(CHECKPOINT).to(device).eval()
    shared_block = model.deltifiedStage3[0]
    assert shared_block.eulerStep == 1.0

    completed: list[dict] = []
    for spec in runs:
        marker = OUT_DIR / spec["name"] / "config.json"
        if args.skip_existing and marker.is_file():
            print(f"skip existing {spec['name']}")
            continue
        completed.append(run_one(model, shared_block, dataset, class_names, spec))

    if SCATTER_OVERLAY_BY_D and completed:
        by_key: dict[tuple, list[dict]] = {}
        for res in completed:
            key = (tuple(res["mean_x"].shape[1:]), res["n_images"])
            by_key.setdefault(key, []).append(res)
        for group in by_key.values():
            if len(group) < 2:
                continue
            group = sorted(group, key=lambda r: r["D"])
            ds = "-".join(str(r["D"]) for r in group)
            out = OUT_DIR / f"scatter_h_overlay_D{ds}.png"
            scatter_overlay_by_d(group, path=out, channel=SCATTER_CHANNEL)
            print(f"overlay scatter -> {out}")


if __name__ == "__main__":
    main()
