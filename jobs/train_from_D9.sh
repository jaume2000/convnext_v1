#!/bin/bash
#SBATCH --job-name=D9_finetune_stage3
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Booster nodes are 32 cores (1x Xeon 8358) / 4 GPUs, so the 4 ranks get 8 dataloader
# workers each (see available_cpus() in the train script); fewer just idles cores.
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=logs/train_from_D9_%j.out
#SBATCH --error=logs/train_from_D9_%j.err

# Fine-tunes stage3 + head on top of the shared block unrolled to 9 Euler steps.
# Everything up to stage3 stays frozen, including the shared block itself.
#
# Submit from the repo root:
#   source .env && sbatch --account="$SLURM_ACCOUNT" jobs/train_from_D9.sh
#
# Chain 12 h jobs after the first segment (or a time-limit kill):
#   echo 'RETAKE=1' >> .env   # or export RETAKE=1 for one submission
#   source .env && sbatch --account="$SLURM_ACCOUNT" jobs/train_from_D9.sh

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
  echo "Venv site-packages not found at '${VENV_SITE}'. Create it on a login node:" >&2
  echo "  module load profile/deeplrn && module load cineca-ai" >&2
  echo "  python -m venv --system-site-packages \"\$HOME/venvs/convnext\"" >&2
  echo "  source \"\$HOME/venvs/convnext/bin/activate\" && pip install --force-reinstall --no-deps timm" >&2
  echo "  deactivate && export PYTHONNOUSERSITE=1" >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${VENV_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
# 1, not 2: this is inherited by every dataloader worker, and the 32 workers already fill
# the node's 32 cores on their own. The main processes only feed the GPUs, so they have no
# use for a second thread either.
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
  echo "Set HF_HOME / HF_DATASETS_CACHE in .env to the folder that contains ${DATASET_DIR}." >&2
  exit 1
fi

# Assigned after sourcing .env, which overrides it: .env still names the pretrained run,
# and the train script hardcodes its own name for the same reason (the Trainer would
# otherwise write over the last.pth this job reads). Both names must agree or the RETAKE
# precheck below would guard a different experiment than the one that resumes, so this
# has to be edited alongside STAGE3_LENGTH in the train script.
EXPERIMENT_NAME="D128_shared_finetune_stage4"
EXPERIMENT="${PROJECT_ROOT}/outputs/${EXPERIMENT_NAME}"
PRETRAINED="${PROJECT_ROOT}/outputs/shared_convnextv1_imagenet/weights/last.pth"
RETAKE="${RETAKE:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

# Both checks run before the ~150G of staging below, so a missing file costs seconds
# instead of the best part of an hour.
if [[ "${RETAKE}" == "1" ]]; then
  if [[ ! -f "${EXPERIMENT}/weights/last.pth" ]]; then
    echo "RETAKE=1 but checkpoint not found at ${EXPERIMENT}/weights/last.pth" >&2
    echo "The first segment of this experiment must run with RETAKE=0." >&2
    exit 1
  fi
elif [[ ! -f "${PRETRAINED}" ]]; then
  echo "Pretrained shared-only checkpoint not found at ${PRETRAINED}" >&2
  echo "outputs/ is gitignored, so it has to be copied to Leonardo separately." >&2
  exit 1
fi

if [[ "${STAGE_DATA:-1}" == "1" ]]; then
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

TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/9.py}"

echo "Host: $(hostname)"
echo "Project: ${PROJECT_ROOT}"
echo "Python: $(which python)"
echo "Venv site: ${VENV_SITE}"
echo "HF cache: ${HF_DATASETS_CACHE}"
echo "Account: ${SLURM_ACCOUNT:-unset}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "RETAKE: ${RETAKE}"
echo "Train script: ${TRAIN_SCRIPT}"
echo "EXPERIMENT_NAME: ${EXPERIMENT_NAME}"
echo "Experiment path: ${EXPERIMENT}"
echo "Pretrained: ${PRETRAINED}"
echo "nproc_per_node: ${NPROC_PER_NODE}"
echo "Start: $(date)"
python -c "import torch, torchvision, timm; print(f'torch={torch.__version__} ({torch.__file__})'); print(f'torchvision={torchvision.__version__} ({torchvision.__file__})'); print(f'timm={timm.__version__} ({timm.__file__})')"

export RETAKE
export EXPERIMENT_NAME
# DDP needs torchrun so each rank gets LOCAL_RANK / RANK / WORLD_SIZE.
# Do not launch with bare `python`: that keeps a single process on one GPU.
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" ${TRAIN_ARGS:-}

echo "End: $(date)"
