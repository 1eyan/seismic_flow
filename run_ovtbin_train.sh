#!/usr/bin/env bash
set -euo pipefail
# =========================================================================
# OVT SSL 训练脚本 — raw context + grid target
#
# Usage:
#   ./run_ovtbin_train.sh
#
# Env overrides:
#   MODEL_NAME=xxx GPU_IDS=0,1,2,3 ./run_ovtbin_train.sh
# =========================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# ---- Data paths ----
DATA_DIR="${DATA_DIR:-/data/shared/测试数据/h5}"
H5_RAW="${H5_RAW:-${DATA_DIR}/field1031_irregular_ovt.h5}"
H5_GRID="${H5_GRID:-${DATA_DIR}/field1031_test_aligned.h5}"
H5_REGULAR="${H5_REGULAR:-${DATA_DIR}/field1031_label_ovt.h5}"

# ---- OVT SSL params ----
OVT_TARGET_SLOTS="${OVT_TARGET_SLOTS:-32}"
OVT_KDTREE_OFFSET_WEIGHT="${OVT_KDTREE_OFFSET_WEIGHT:-2.0}"
TRACE_PS="${TRACE_PS:-128}"
TIME_PS="${TIME_PS:-1256}"

# ---- Model ----
MODEL_NAME="${MODEL_NAME:-trace_axis}"
MODEL_TYPE="${MODEL_TYPE:-trace_axis}"
SEGY_PROFILE="${SEGY_PROFILE:-field1031}"

# ---- Training hyperparams ----
BATCH_SIZE="${BATCH_SIZE:-11}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-800}"
SEED="${SEED:-42}"
# ---- Flow Matching ----
PATH_TYPE="${PATH_TYPE:-Linear}"
PREDICTION="${PREDICTION:-velocity}"
LOSS_WEIGHT="${LOSS_WEIGHT:-none}"

USE_MULTISCALE_LOSS="${USE_MULTISCALE_LOSS:-true}"
MULTISCALE_LOSS_WEIGHT="${MULTISCALE_LOSS_WEIGHT:-0.1}"
USE_SPECTRAL_LOSS="${USE_SPECTRAL_LOSS:-true}"
SPEC_WEIGHT="${SPEC_WEIGHT:-0.1}"


# ---- Model arch ----
USE_P_SCALE="${USE_P_SCALE:-false}"
USE_MISSING_EMBEDDING="${USE_MISSING_EMBEDDING:-false}"
USE_ENERGY_MLP="${USE_ENERGY_MLP:-false}"
HEADWISE_ATTN_OUTPUT_GATE="${HEADWISE_ATTN_OUTPUT_GATE:-false}"
ELEMENTWISE_ATTN_OUTPUT_GATE="${ELEMENTWISE_ATTN_OUTPUT_GATE:-false}"


# ---- Checkpoint ----
PRETRAINED="${PRETRAINED:-}"
PRETRAINED_STRICT="${PRETRAINED_STRICT:-true}"

# ---- Accelerate ----
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_config.yaml}"

echo "============================================================"
echo "OVT SSL Training: ${MODEL_NAME}"
echo "============================================================"
echo "raw:        ${H5_RAW}"
echo "grid:       ${H5_GRID}"
echo "regular:    ${H5_REGULAR}"
echo "target_slots: ${OVT_TARGET_SLOTS}"
echo "trace_ps:   ${TRACE_PS}"
echo "time_ps:    ${TIME_PS}"
echo "model_type: ${MODEL_TYPE}"
echo "batch_size: ${BATCH_SIZE}"
echo "lr:         ${LR}"
echo "epochs:     ${EPOCHS}"
echo "============================================================"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --main_process_port 29501 train_fpmV3_ddp.py \
    --model_name "${MODEL_NAME}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --epochs "${EPOCHS}" \
    --model_type "${MODEL_TYPE}" \
    --seed "${SEED}" \
    --data_type "${SEGY_PROFILE}" \
    --segy_profile "${SEGY_PROFILE}" \
    --use_p_scale "${USE_P_SCALE}" \
    --use_missing_embedding "${USE_MISSING_EMBEDDING}" \
    --use_energy_mlp "${USE_ENERGY_MLP}" \
    --headwise_attn_output_gate "${HEADWISE_ATTN_OUTPUT_GATE}" \
    --elementwise_attn_output_gate "${ELEMENTWISE_ATTN_OUTPUT_GATE}" \
    --path_type "${PATH_TYPE}" \
    --prediction "${PREDICTION}" \
    --loss_weight "${LOSS_WEIGHT}" \
    --use_multiscale_loss "${USE_MULTISCALE_LOSS}" \
    --multiscale_loss_weight "${MULTISCALE_LOSS_WEIGHT}" \
    --use_spectral_loss "${USE_SPECTRAL_LOSS}" \
    --spec_weight "${SPEC_WEIGHT}" \
    --dataset_mode ovtbin \
    --h5File "${H5_RAW}" \
    --h5File_grid "${H5_GRID}" \
    --h5File_regular "${H5_REGULAR}" \
    --ovt_target_slots "${OVT_TARGET_SLOTS}" \
    --ovt_kdtree_offset_weight "${OVT_KDTREE_OFFSET_WEIGHT}" \
    --trace_ps "${TRACE_PS}" \
    --time_ps "${TIME_PS}" \
    $([ -n "${PRETRAINED}" ] && echo "--pretrained ${PRETRAINED} --pretrained_strict ${PRETRAINED_STRICT}")
