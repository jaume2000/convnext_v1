#!/bin/bash
#SBATCH --job-name=feature_maps
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=logs/feature_map_explorer_%j.out
#SBATCH --error=logs/feature_map_explorer_%j.err

# Stage-3 feature-map trajectories. Config lives in
# scripts/feature_map_explorer.py:
#   BACKBONE = "shared" | "interpoled"
#   INTERPOLED_MODELS = ["convnext", "resnet50", "resnet101", "swin"]
#   INTERPOLED_EXPERIMENTS = [...]
#
# Submit from the repo root:
#   source .env && sbatch --account="$SLURM_ACCOUNT" jobs/feature_map_explorer.sh
#
# Optional overrides:
#   TRAIN_ARGS='--list-only'
#   TRAIN_ARGS='--only resnet50_baseline_R1_ES1_c289_n1 swin_R2_ES0.5_c289_n1'
#   TRAIN_ARGS='--skip-existing'
#   TRAIN_ARGS='--keep-frames'

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ -z "${SLURM_ACCOUNT:-}" ]]; then
    echo "SLURM_ACCOUNT is not set. Add it to ${PROJECT_ROOT}/.env" >&2
    exit 1
  fi
  mkdir -p "${PROJECT_ROOT}/logs"
  exec sbatch --account="${SLURM_ACCOUNT}" "${BASH_SOURCE[0]}" "$@"
fi

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

module purge
module load profile/deeplrn
module load cineca-ai
# Optional: many Cineca images ship ffmpeg as a module; ignore if absent.
module load ffmpeg 2>/dev/null || true

VENV_PATH="${VENV_PATH:-$HOME/venvs/convnext}"
VENV_PATH="${VENV_PATH/#\~/$HOME}"
PY_TAG="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
VENV_SITE="${VENV_PATH}/lib/python${PY_TAG}/site-packages"
if [[ ! -d "${VENV_SITE}" ]]; then
  echo "Venv site-packages not found at '${VENV_SITE}'." >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${VENV_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

HF_HOME="${HF_HOME:-${WORK:+$WORK/huggingface}}"
HF_HOME="${HF_HOME:-${CINECA_SCRATCH:-$HOME}/hf}"
HF_HOME="${HF_HOME/#\~/$HOME}"
export HF_HOME
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# torchvision ImageNet weights (preload on login; compute nodes have no net)
export TORCH_HOME="${TORCH_HOME:-${WORK:+$WORK/torch}}"
export TORCH_HOME="${TORCH_HOME:-${CINECA_SCRATCH:-$HOME}/torch}"
export TORCH_HOME="${TORCH_HOME/#\~/$HOME}"

DATASET_DIR="ILSVRC___imagenet-1k"
if [[ ! -d "${HF_DATASETS_CACHE}/${DATASET_DIR}" ]]; then
  echo "ImageNet cache not found at ${HF_DATASETS_CACHE}/${DATASET_DIR}" >&2
  exit 1
fi

SHARED_CKPT="${PROJECT_ROOT}/outputs/shared_convnextv1_imagenet/weights/last.pth"
INTERPOLED_CKPT="${PROJECT_ROOT}/outputs/convnextv1_imagenet_droppath0/weights/last.pth"
if [[ ! -f "${SHARED_CKPT}" && ! -f "${INTERPOLED_CKPT}" ]]; then
  echo "No feature-map checkpoint found (expected shared and/or interpoled)." >&2
  echo "  shared:     ${SHARED_CKPT}" >&2
  echo "  interpoled: ${INTERPOLED_CKPT}" >&2
  exit 1
fi

SCRIPT="${PROJECT_ROOT}/scripts/feature_map_explorer.py"

echo "Host: $(hostname)"
echo "Project: ${PROJECT_ROOT}"
echo "Python: $(which python)"
echo "HF cache: ${HF_DATASETS_CACHE}"
echo "TORCH_HOME: ${TORCH_HOME}"
echo "Shared ckpt: ${SHARED_CKPT} ($([ -f "${SHARED_CKPT}" ] && echo ok || echo missing))"
echo "Interpoled ckpt: ${INTERPOLED_CKPT} ($([ -f "${INTERPOLED_CKPT}" ] && echo ok || echo missing))"
echo "Script: ${SCRIPT}"
echo "Args: ${TRAIN_ARGS:-}"
echo "ffmpeg: $(command -v ffmpeg || echo missing)"
echo "Start: $(date)"
python -c "import torch, matplotlib; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} mpl={matplotlib.__version__}')"

# shellcheck disable=SC2086
python "${SCRIPT}" ${TRAIN_ARGS:-}

echo "End: $(date)"
