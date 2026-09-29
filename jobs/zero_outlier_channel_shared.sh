#!/bin/bash
#SBATCH --job-name=zero_outlier_ch
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 1 GPU, 16 CPUs on Leonardo booster.
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=logs/zero_outlier_channel_shared_%j.out
#SBATCH --error=logs/zero_outlier_channel_shared_%j.err

# Shared ConvNeXt: detect stage-3 outlier channel (R1/D=9), zero it after every
# residual, then ablate 20 random channels one-by-one. Metrics + ImageNet val
# for R1 (D=9) and R10 (D=90). Histograms of Δloss under
# $WORK/zero_outlier_channel_shared/ (override with ZERO_OUTLIER_ROOT).
#
# Submit from the repo root:
#   source .env && sbatch --account="$SLURM_ACCOUNT" jobs/zero_outlier_channel_shared.sh
#
# Optional overrides:
#   TRAIN_ARGS='--skip-metrics'
#   TRAIN_ARGS='--R 1 --n-random 5'
#   TRAIN_ARGS='--with-random-metrics'
#   ZERO_OUTLIER_ROOT=/custom/path

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

export TORCH_HOME="${TORCH_HOME:-${WORK:+$WORK/torch}}"
export TORCH_HOME="${TORCH_HOME:-${CINECA_SCRATCH:-$HOME}/torch}"
export TORCH_HOME="${TORCH_HOME/#\~/$HOME}"

# Bulky outputs under $WORK (home is ~50G on Leonardo).
if [[ -z "${ZERO_OUTLIER_ROOT:-}" ]]; then
  if [[ -n "${WORK:-}" ]]; then
    ZERO_OUTLIER_ROOT="${WORK}/zero_outlier_channel_shared"
  else
    ZERO_OUTLIER_ROOT="${PROJECT_ROOT}/outputs/zero_outlier_channel_shared"
  fi
fi
ZERO_OUTLIER_ROOT="${ZERO_OUTLIER_ROOT/#\~/$HOME}"
export ZERO_OUTLIER_ROOT
mkdir -p "${ZERO_OUTLIER_ROOT}"

DATASET_DIR="ILSVRC___imagenet-1k"
if [[ ! -d "${HF_DATASETS_CACHE}/${DATASET_DIR}" ]]; then
  echo "ImageNet cache not found at ${HF_DATASETS_CACHE}/${DATASET_DIR}" >&2
  exit 1
fi

SHARED_CKPT="${PROJECT_ROOT}/outputs/shared_convnextv1_imagenet/weights/last.pth"
if [[ ! -f "${SHARED_CKPT}" ]]; then
  echo "Shared checkpoint missing: ${SHARED_CKPT}" >&2
  exit 1
fi

SCRIPT="${PROJECT_ROOT}/scripts/zero_outlier_channel_shared.py"

echo "Host: $(hostname)"
echo "Project: ${PROJECT_ROOT}"
echo "Python: $(which python)"
echo "HF cache: ${HF_DATASETS_CACHE}"
echo "TORCH_HOME: ${TORCH_HOME}"
echo "ZERO_OUTLIER_ROOT: ${ZERO_OUTLIER_ROOT}"
echo "Shared ckpt: ${SHARED_CKPT}"
echo "Script: ${SCRIPT}"
echo "Args: ${TRAIN_ARGS:-}"
echo "Start: $(date)"
python -c "import torch, matplotlib; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} mpl={matplotlib.__version__}')"

# shellcheck disable=SC2086
python "${SCRIPT}" ${TRAIN_ARGS:-}

echo "End: $(date)"
