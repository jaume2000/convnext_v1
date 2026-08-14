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

# Expand ~ / $HOME from .env, then activate the project venv.
VENV_PATH="${VENV_PATH:-$HOME/venvs/convnext}"
VENV_PATH="${VENV_PATH/#\~/$HOME}"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Venv not found at '${VENV_PATH}'. Create it on a login node:" >&2
  echo "  module load profile/deeplrn && module load cineca-ai" >&2
  echo "  python -m venv --system-site-packages \"\$HOME/venvs/convnext\"" >&2
  echo "  source \"\$HOME/venvs/convnext/bin/activate\" && pip install timm" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

# Ignore ~/.local pip installs (they conflict with cineca-ai torch/torchvision).
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Host: $(hostname)"
echo "Project: ${PROJECT_ROOT}"
echo "Python: $(which python)"
echo "Account: ${SLURM_ACCOUNT:-unset}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start: $(date)"
python -c "import torch, torchvision; print(f'torch={torch.__version__} ({torch.__file__})'); print(f'torchvision={torchvision.__version__} ({torchvision.__file__})')"

srun python scripts/train.py

echo "End: $(date)"
