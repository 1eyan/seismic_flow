#!/usr/bin/env bash
set -euo pipefail
# =========================================================================
# Query-Context 训练脚本 — irregular context -> regular query
#
# Usage:
#   ./run_queryctx_train.sh
#
# Env overrides:
#   MODEL_NAME=xxx GPU_IDS=0,1,2,3 ./run_queryctx_train.sh
# =========================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# ---- Data paths ----
DATA_DIR="${DATA_DIR:-/data/shared/测试数据/h5}"
H5_IRREGULAR="${H5_IRREGULAR:-${DATA_DIR}/field1031_irregular.h5}"
H5_REGULAR="${H5_REGULAR:-${DATA_DIR}/field1031_label.h5}"
DATASET_NEIGHBORS="${DATASET_NEIGHBORS:-/data/shared/测试数据/h5/anchor_patch_e2ev2/train_pool_idx_2d_block.npz}"

# ---- Query-Context params ----
TRAIN_NUM_QUERY="${TRAIN_NUM_QUERY:-30}"
TRAIN_CONTEXT_SIZE="${TRAIN_CONTEXT_SIZE:-}"           # empty = auto (trace_ps - num_query)
PATCH_BETA="${PATCH_BETA:-0.3}"
PATCH_METRIC_WEIGHTS="${PATCH_METRIC_WEIGHTS:-1.0,1.0,1.0,1.0}"
TRACE_SORT_KEYS="${TRACE_SORT_KEYS:-offset,azimuth}"
EPOCH_REPEAT="${EPOCH_REPEAT:-3}"
COORD_AUG_SCALE="${COORD_AUG_SCALE:-0.05}"
ALLOW_COORD_STATS_FALLBACK="${ALLOW_COORD_STATS_FALLBACK:-false}"
TARGET_MODE="${TARGET_MODE:-self}"
REGULAR_HOLDOUT_NPZ="${REGULAR_HOLDOUT_NPZ:-/data/shared/测试数据/h5/anchor_patch_e2ev2/train_regular_holdout_query_context.npz}"
REGULAR_TASK_PROB="${REGULAR_TASK_PROB:-0.5}"
TRACE_PS="${TRACE_PS:-128}"
TIME_PS="${TIME_PS:-1256}"

# ---- Model ----
MODEL_NAME="${MODEL_NAME:-trace_axis}"
MODEL_TYPE="${MODEL_TYPE:-trace_axis}"
SEGY_PROFILE="${SEGY_PROFILE:-field1031}"

# ---- Training hyperparams ----
BATCH_SIZE="${BATCH_SIZE:-2}"
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
echo "Query-Context Training: ${MODEL_NAME}"
echo "============================================================"
echo "irregular:   ${H5_IRREGULAR}"
echo "regular:     ${H5_REGULAR}"
echo "neighbors:   ${DATASET_NEIGHBORS}"
echo "num_query:   ${TRAIN_NUM_QUERY}"
echo "patch_beta:  ${PATCH_BETA}"
echo "trace_ps:    ${TRACE_PS}"
echo "time_ps:     ${TIME_PS}"
echo "epoch_repeat:${EPOCH_REPEAT}"
echo "coord_aug:   ${COORD_AUG_SCALE}"
echo "model_type:  ${MODEL_TYPE}"
echo "batch_size:  ${BATCH_SIZE}"
echo "lr:          ${LR}"
echo "epochs:      ${EPOCHS}"
echo "============================================================"

# Build optional --train_context_size arg
CTX_SIZE_ARG=""
if [[ -n "${TRAIN_CONTEXT_SIZE}" ]]; then
    CTX_SIZE_ARG="--train_context_size ${TRAIN_CONTEXT_SIZE}"
fi

accelerate launch --config_file "${ACCELERATE_CONFIG}" --main_process_port 29503 train_fpmV3_ddp.py \
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
    --dataset_mode queryctx_v2 \
    --h5File "${H5_IRREGULAR}" \
    --h5File_regular "${H5_REGULAR}" \
    --dataset_neighbors_train "${DATASET_NEIGHBORS}" \
    --train_num_query "${TRAIN_NUM_QUERY}" \
    ${CTX_SIZE_ARG} \
    --patch_beta "${PATCH_BETA}" \
    --patch_metric_weights "${PATCH_METRIC_WEIGHTS}" \
    --trace_sort_keys_queryctx "${TRACE_SORT_KEYS}" \
    --epoch_repeat "${EPOCH_REPEAT}" \
    --target_mode "${TARGET_MODE}" \
    --coord_aug_scale "${COORD_AUG_SCALE}" \
    --allow_coord_stats_fallback "${ALLOW_COORD_STATS_FALLBACK}" \
    --trace_ps "${TRACE_PS}" \
    $([ -n "${REGULAR_HOLDOUT_NPZ}" ] && echo "--regular_holdout_npz ${REGULAR_HOLDOUT_NPZ} --regular_task_prob ${REGULAR_TASK_PROB}") \
    --time_ps "${TIME_PS}" \
    $([ -n "${PRETRAINED}" ] && echo "--pretrained ${PRETRAINED} --pretrained_strict ${PRETRAINED_STRICT}")
