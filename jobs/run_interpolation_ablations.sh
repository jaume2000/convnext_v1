#!/bin/bash
#SBATCH --job-name=interp_ablations
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=logs/run_interpolation_ablations_%j.out
#SBATCH --error=logs/run_interpolation_ablations_%j.err

# Run plain/bilinear R×RK interpolation ablations for every backbone.
# Categories with an existing outputs/<name>/results2.csv are skipped.
#
#   source .env && sbatch --account="$SLURM_ACCOUNT" jobs/run_interpolation_ablations.sh
#   bash jobs/run_interpolation_ablations.sh --dry-run

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

# Allow local dry-run without sbatch.
if [[ "${1:-}" == "--dry-run" ]]; then
  cd "${PROJECT_ROOT}"
  python "${PROJECT_ROOT}/scripts/run_interpolation_ablations.py" --dry-run
  exit 0
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
export TORCH_HOME="${TORCH_HOME:-${WORK:+$WORK/torch}}"
export TORCH_HOME="${TORCH_HOME:-${CINECA_SCRATCH:-$HOME}/torch}"
export TORCH_HOME="${TORCH_HOME/#\~/$HOME}"

HF_HOME="${HF_HOME:-${WORK:+$WORK/huggingface}}"
HF_HOME="${HF_HOME:-${CINECA_SCRATCH:-$HOME}/hf}"
HF_HOME="${HF_HOME/#\~/$HOME}"
export HF_HOME
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DATASET_DIR="ILSVRC___imagenet-1k"
if [[ ! -d "${HF_DATASETS_CACHE}/${DATASET_DIR}" ]]; then
  echo "ImageNet cache not found at ${HF_DATASETS_CACHE}/${DATASET_DIR}" >&2
  exit 1
fi

SCRIPT="${PROJECT_ROOT}/scripts/run_interpolation_ablations.py"

echo "Host: $(hostname)"
echo "Project: ${PROJECT_ROOT}"
echo "Python: $(which python)"
echo "HF cache: ${HF_DATASETS_CACHE}"
echo "TORCH_HOME: ${TORCH_HOME}"
echo "Script: ${SCRIPT}"
echo "Start: $(date)"
python -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')"

python "${SCRIPT}" "$@"

echo "End: $(date)"
