#!/bin/bash
#SBATCH --job-name=convnext_interp_n
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=logs/convnext_interpolation_normal_%j.out
#SBATCH --error=logs/convnext_interpolation_normal_%j.err

# Alias for jobs/convnext_interpolation.sh (drop-path trained ConvNeXt is the default).
#
#   source .env && sbatch --account="$SLURM_ACCOUNT" jobs/convnext_interpolation_normal.sh

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

exec bash "${PROJECT_ROOT}/jobs/convnext_interpolation.sh" "$@"
