#!/bin/bash
#SBATCH --job-name=shared_ablation
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=logs/shared_convnext_ablation_%j.out
#SBATCH --error=logs/shared_convnext_ablation_%j.err

# Submit from the repo root:
#   source .env && sbatch --account="$SLURM_ACCOUNT" jobs/shared_convnext_ablation.sh
#
# Resume after interruption:
#   RESUME=1 source .env && sbatch --account="$SLURM_ACCOUNT" jobs/shared_convnext_ablation.sh
#
# Smoke test (2 batches per configuration):
#   MAX_BATCHES=2 source .env && sbatch --account="$SLURM_ACCOUNT" jobs/shared_convnext_ablation.sh

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

EXPERIMENT_NAME="${EXPERIMENT_NAME:-sharedConvnextAblation}"
EXPERIMENT="${PROJECT_ROOT}/outputs/${EXPERIMENT_NAME}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/outputs/shared_convnextv1_imagenet/weights/last.pth}"
RESUME="${RESUME:-0}"
MAX_BATCHES="${MAX_BATCHES:-}"
BATCH_SIZE="${BATCH_SIZE:-96}"
NUM_WORKERS="${NUM_WORKERS:-4}"
# Staging copies ImageNet into RAM (/dev/shm); skip for this CPU-heavy sweep.
STAGE_DATA="${STAGE_DATA:-0}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found at ${CHECKPOINT}" >&2
  exit 1
fi

if [[ "${STAGE_DATA}" == "1" ]]; then
  STAGE_ROOT="/dev/shm/hfstage_${SLURM_JOB_ID}"
  needed_kb="$(du -sk "${HF_DATASETS_CACHE}/${DATASET_DIR}" | cut -f1)"
  avail_kb="$(df --output=avail -k /dev/shm | tail -1)"
  margin_kb=$((30 * 1024 * 1024))
  if (( needed_kb + margin_kb > avail_kb )); then
    echo "Skipping staging: need $((needed_kb / 1024 / 1024))G + 30G margin," \
         "/dev/shm has $((avail_kb / 1024 / 1024))G" >&2
  else
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

SCRIPT_ARGS=(
  --checkpoint "${CHECKPOINT}"
  --output-dir "${EXPERIMENT}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
)
if [[ "${RESUME}" == "1" ]]; then
  SCRIPT_ARGS+=(--resume)
fi
if [[ -n "${MAX_BATCHES}" ]]; then
  SCRIPT_ARGS+=(--max-batches "${MAX_BATCHES}")
fi

echo "Host: $(hostname)"
echo "Project: ${PROJECT_ROOT}"
echo "Python: $(which python)"
echo "HF cache: ${HF_DATASETS_CACHE}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Experiment path: ${EXPERIMENT}"
echo "Batch size: ${BATCH_SIZE}"
echo "Num workers: ${NUM_WORKERS}"
echo "Stage data: ${STAGE_DATA}"
echo "Resume: ${RESUME}"
echo "Start: $(date)"

python "${PROJECT_ROOT}/scripts/shared_convnext_ablation.py" "${SCRIPT_ARGS[@]}"

echo "End: $(date)"
echo "Results: ${EXPERIMENT}/results.csv"
