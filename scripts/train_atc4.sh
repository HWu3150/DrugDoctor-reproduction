#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

############################
# Customize here
############################

CUDA_DEVICE=0
MODEL_NAME="AIDrug"

RECORDS_PATH="${REPO_ROOT}/data/output/records_final.pkl"
VOC_PATH="${REPO_ROOT}/data/output/voc_final.pkl"
DDI_CSV_PATH="${REPO_ROOT}/data/output/ddi_adj_matrix_atc4.csv"
SUBSTRUCT_SMILES_PATH="${REPO_ROOT}/data/output/substructure_smiles_atc4.pkl"

# Follow the original training defaults unless you intentionally change them.
LR="5e-4"
EPOCHS=200
TARGET_DDI=0.06
KGLOSS_ALPHA=0.5
BATCH_SIZE=16
EMB_DIM=112
KP=0.05
DIM=64
THRESHOLD=0.4

# WandB
USE_WANDB=true
WANDB_PROJECT="DrugDoctor"
WANDB_ENTITY=""
WANDB_NAME="atc4_baseline"
WANDB_GROUP=""
WANDB_MODE="online"
# export WANDB_API_KEY="your_wandb_key"

############################
# End customization
############################

mkdir -p log "saved/${MODEL_NAME}"

CMD=(
  python src/main.py
  --cuda "${CUDA_DEVICE}"
  --model_name "${MODEL_NAME}"
  --records_path "${RECORDS_PATH}"
  --voc_path "${VOC_PATH}"
  --ddi_csv_path "${DDI_CSV_PATH}"
  --substruct_smiles_path "${SUBSTRUCT_SMILES_PATH}"
  --lr "${LR}"
  --epoch "${EPOCHS}"
  --target_ddi "${TARGET_DDI}"
  --kgloss_alpha "${KGLOSS_ALPHA}"
  --batch_size "${BATCH_SIZE}"
  --emb_dim "${EMB_DIM}"
  --kp "${KP}"
  --dim "${DIM}"
  --threshold "${THRESHOLD}"
)

if [[ "${USE_WANDB}" == "true" ]]; then
  CMD+=(
    --use_wandb
    --wandb_project "${WANDB_PROJECT}"
    --wandb_name "${WANDB_NAME}"
    --wandb_mode "${WANDB_MODE}"
  )

  if [[ -n "${WANDB_ENTITY}" ]]; then
    CMD+=(--wandb_entity "${WANDB_ENTITY}")
  fi

  if [[ -n "${WANDB_GROUP}" ]]; then
    CMD+=(--wandb_group "${WANDB_GROUP}")
  fi
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
