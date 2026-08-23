#!/bin/bash
#SBATCH --job-name=train_convnext
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Booster nodes are 32 cores / 4 GPUs; asking for fewer just idles dataloader cores.
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=logs/train_convnext_%j.out
#SBATCH --error=logs/train_convnext_%j.err

# Submit from the repo root:
#   source .env && sbatch --account="$SLURM_ACCOUNT" jobs/train_convnext.sh

set -euo pipefail

# Under SLURM, BASH_SOURCE can point at a spool copy of this script.
# SLURM_SUBMIT_DIR is the directory where sbatch was invoked (repo root).
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

# sbatch parses #SBATCH before the script runs, so account must come from CLI.
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

# Use cineca-ai's python/torch/torchvision. Only add the venv site-packages
# (e.g. timm) on PYTHONPATH — do not `source activate`, which hides modules.
VENV_PATH="${VENV_PATH:-$HOME/venvs/convnext}"
VENV_PATH="${VENV_PATH/#\~/$HOME}"
PY_TAG="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
VENV_SITE="${VENV_PATH}/lib/python${PY_TAG}/site-packages"
if [[ ! -d "${VENV_SITE}" ]]; then
  echo "Venv site-packages not found at '${VENV_SITE}'. Create it on a login node:" >&2
  echo "  module load profile/deeplrn && module load cineca-ai" >&2
  echo "  python -m venv --system-site-packages \"\$HOME/venvs/convnext\"" >&2
  echo "  source \"\$HOME/venvs/convnext/bin/activate\" && pip install --force-reinstall --no-deps timm" >&2
  echo "  deactivate && export PYTHONNOUSERSITE=1" >&2
  exit 1
fi

# Ignore ~/.local pip installs (they conflict with cineca-ai torch/torchvision).
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${VENV_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
# The main process only feeds the GPUs; dataloader workers are single-threaded anyway
# (torch pins them to 1 thread), and idle OpenMP threads spin-wait and steal their cores.
# 1 rather than 2: the workers inherit this, and they already fill the node on their own.
export OMP_NUM_THREADS=1

# Hugging Face: compute nodes are offline. Point at the existing cache
# (parent of ILSVRC___imagenet-1k), e.g. $WORK/huggingface on Leonardo.
HF_HOME="${HF_HOME:-${WORK:+$WORK/huggingface}}"
HF_HOME="${HF_HOME:-${CINECA_SCRATCH:-$HOME}/hf}"
HF_HOME="${HF_HOME/#\~/$HOME}"
export HF_HOME
# Dataset arrows live next to hub/ under $WORK/huggingface, not in a datasets/ subdir.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DATASET_DIR="ILSVRC___imagenet-1k"
if [[ ! -d "${HF_DATASETS_CACHE}/${DATASET_DIR}" ]]; then
  echo "ImageNet cache not found at ${HF_DATASETS_CACHE}/${DATASET_DIR}" >&2
  echo "Set HF_HOME / HF_DATASETS_CACHE in .env to the folder that contains ${DATASET_DIR}." >&2
  exit 1
fi

# Shuffled training turns into random ~110 KiB reads, which on GPFS cap the dataloader
# at ~900 img/s while the four GPUs can consume ~4600. Copy the dataset into the node's
# tmpfs first: that copy is sequential, which is exactly what GPFS is good at, and every
# read afterwards comes from RAM. Set STAGE_DATA=0 to train straight from GPFS.
if [[ "${STAGE_DATA:-1}" == "1" ]]; then
  STAGE_ROOT="/dev/shm/hfstage_${SLURM_JOB_ID}"
  needed_kb="$(du -sk "${HF_DATASETS_CACHE}/${DATASET_DIR}" | cut -f1)"
  avail_kb="$(df --output=avail -k /dev/shm | tail -1)"
  margin_kb=$((30 * 1024 * 1024))
  if (( needed_kb + margin_kb > avail_kb )); then
    echo "Skipping staging: need $((needed_kb / 1024 / 1024))G + 30G margin," \
         "/dev/shm has $((avail_kb / 1024 / 1024))G" >&2
  else
    # tmpfs is unreclaimable, so release it even if SLURM kills the job at the time limit.
    trap 'rm -rf "${STAGE_ROOT}"' EXIT TERM INT
    echo "Staging $((needed_kb / 1024 / 1024))G into ${STAGE_ROOT}"
    stage_start="${SECONDS}"
    pushd "${HF_DATASETS_CACHE}" > /dev/null
    find "${DATASET_DIR}" -type d -exec mkdir -p "${STAGE_ROOT}/{}" \;
    find "${DATASET_DIR}" -type f -print0 | xargs -0 -P 8 -I{} cp -p {} "${STAGE_ROOT}/{}"
    popd > /dev/null
    echo "Staged in $((SECONDS - stage_start))s"
    export HF_DATASETS_CACHE="${STAGE_ROOT}"
  fi
fi

echo "Host: $(hostname)"
echo "Project: ${PROJECT_ROOT}"
echo "Python: $(which python)"
echo "Venv site: ${VENV_SITE}"
echo "HF cache: ${HF_DATASETS_CACHE}"
echo "Account: ${SLURM_ACCOUNT:-unset}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start: $(date)"
python -c "import torch, torchvision, timm; print(f'torch={torch.__version__} ({torch.__file__})'); print(f'torchvision={torchvision.__version__} ({torchvision.__file__})'); print(f'timm={timm.__version__} ({timm.__file__})')"

# Override to profile instead of train:
#   TRAIN_SCRIPT=scripts/bench.py TRAIN_ARGS="--skip-gpu --skip-io" \
#     sbatch --time=00:30:00 --account="$SLURM_ACCOUNT" jobs/train_convnext.sh
srun python "${TRAIN_SCRIPT:-scripts/train.py}" ${TRAIN_ARGS:-}

echo "End: $(date)"
