#!/bin/bash
#SBATCH --job-name=train_convnext
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Host: $(hostname)"
echo "Project: ${PROJECT_ROOT}"
echo "Python: $(which python)"
echo "Venv site: ${VENV_SITE}"
echo "Account: ${SLURM_ACCOUNT:-unset}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start: $(date)"
python -c "import torch, torchvision, timm; print(f'torch={torch.__version__} ({torch.__file__})'); print(f'torchvision={torchvision.__version__} ({torchvision.__file__})'); print(f'timm={timm.__version__} ({timm.__file__})')"

srun python scripts/train.py

echo "End: $(date)"
