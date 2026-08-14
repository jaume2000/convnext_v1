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

# Use source .env && sbatch --account="$SLURM_ACCOUNT" jobs/train_convnext.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
mkdir -p logs

module purge
module load profile/deeplrn
module load cineca-ai

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Host: $(hostname)"
echo "Account: ${SLURM_ACCOUNT:-unset}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start: $(date)"

srun python scripts/train.py

echo "End: $(date)"
