#!/usr/bin/env bash
set -euo pipefail
# =========================================================================
# Query-Context V1 推理 — supervised mode (gen_infer_5d.py)
#
# Usage:
#   Single-GPU:  ./run_queryctx_infer.sh
#   Multi-GPU:   NPROC_PER_NODE=4 ./run_queryctx_infer.sh
# =========================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DEVICE="${DEVICE:-cuda:0}"
MASTER_PORT="${MASTER_PORT:-29504}"

# ---- Data paths ----
DATA_DIR="${DATA_DIR:-/data/shared/测试数据/h5}"
H5_IRREGULAR="${H5_IRREGULAR:-${DATA_DIR}/field1031_irregular.h5}"
H5_REGULAR="${H5_REGULAR:-${DATA_DIR}/field1031_label.h5}"
DATASET_NEIGHBORS_INFER="${DATASET_NEIGHBORS_INFER:-/data/shared/测试数据/h5/anchor_patch_e2ev1/infer_query_context.npz}"

# ---- SEGY ----
MASK_PATH="${MASK_PATH:-/data/shared/测试数据/mask_from_label.sgy}"
LABEL_SEGY="${LABEL_SEGY:-/data/shared/测试数据/reg_pku_1031/reg_pku_1030/reg5dbin_label1031.sgy}"

# ---- Output ----
CHECKPOINT="${CHECKPOINT:-/home/chengzhitong/5d_regular/seismic_flow_v2/resultsFPM/trace_axis_datatype_field1031_0616_5dsup/checkpoints/model-epoch-430.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/gen_fill_results_queryctx}"
OUTPUT_SEGY="${OUTPUT_SEGY:-${OUTPUT_DIR}/filled_missing.sgy}"
OUTPUT_RESIDUAL_SEGY="${OUTPUT_RESIDUAL_SEGY:-${OUTPUT_DIR}/residual.sgy}"

# ---- Query-Context ----
TRACE_PS="${TRACE_PS:-128}"
TIME_PS="${TIME_PS:-1256}"

# ---- Inference ----
BATCH_SIZE="${BATCH_SIZE:-28}"
SEGY_PROFILE="${SEGY_PROFILE:-field1031}"
MISSING_EPS="${MISSING_EPS:-1e-10}"

# ---- Model arch (must match training) ----
USE_P_SCALE="${USE_P_SCALE:-false}"
USE_MISSING_EMBEDDING="${USE_MISSING_EMBEDDING:-false}"
MLP_RATIO="${MLP_RATIO:-4}"
NUM_BANDS="${NUM_BANDS:-16}"

# ---- Flow Matching (must match training) ----
PATH_TYPE="${PATH_TYPE:-Linear}"
PREDICTION="${PREDICTION:-velocity}"
LOSS_WEIGHT="${LOSS_WEIGHT:-none}"

# ---- Sampling ----
SAMPLING_METHOD="${SAMPLING_METHOD:-ode}"
ODE_NUM_STEPS="${ODE_NUM_STEPS:-50}"
ODE_SAMPLING_METHOD="${ODE_SAMPLING_METHOD:-dopri5}"
ODE_ATOL="${ODE_ATOL:-1e-6}"
ODE_RTOL="${ODE_RTOL:-1e-3}"
SDE_NUM_STEPS="${SDE_NUM_STEPS:-250}"

# ---- Additional ----
FILL_INTERVAL="${FILL_INTERVAL:-1}"
HEADER_MODE="${HEADER_MODE:-fixed}"
VISUALIZE="${VISUALIZE:-true}"
VIS_BATCHES="${VIS_BATCHES:-0}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Query-Context V1 Inference (gen_infer_5d.py)"
echo "============================================================"
echo "checkpoint:     ${CHECKPOINT}"
echo "h5_irregular:   ${H5_IRREGULAR}"
echo "h5_regular:     ${H5_REGULAR}"
echo "neighbors npz:  ${DATASET_NEIGHBORS_INFER}"
echo "mask_segy:      ${MASK_PATH}"
echo "output_segy:    ${OUTPUT_SEGY}"
echo "profile:        ${SEGY_PROFILE}"
echo "use_p_scale:    ${USE_P_SCALE}"
echo "nproc_per_node: ${NPROC_PER_NODE}"
echo "============================================================"

cmd=(
  --checkpoint "${CHECKPOINT}"
  --h5_irregular "${H5_IRREGULAR}"
  --h5_regular "${H5_REGULAR}"
  --h5_mask "${H5_IRREGULAR}"
  --mask_path "${MASK_PATH}"
  --dataset_neighbors_infer "${DATASET_NEIGHBORS_INFER}"
  --output_dir "${OUTPUT_DIR}"
  --output_segy "${OUTPUT_SEGY}"
  --output_residual_segy "${OUTPUT_RESIDUAL_SEGY}"
  --queryctx_variant v1
  --batch_size "${BATCH_SIZE}"
  --time_ps "${TIME_PS}"
  --trace_ps "${TRACE_PS}"
  --missing_eps "${MISSING_EPS}"
  --segy_config "${SEGY_PROFILE}"
  --header_mode "${HEADER_MODE}"
  --use_p_scale "${USE_P_SCALE}"
  --use_missing_embedding "${USE_MISSING_EMBEDDING}"
  --mlp_ratio "${MLP_RATIO}"
  --num_bands "${NUM_BANDS}"
  --path_type "${PATH_TYPE}"
  --prediction "${PREDICTION}"
  --loss_weight "${LOSS_WEIGHT}"
  --sampling_method "${SAMPLING_METHOD}"
  --ode_sampling_method "${ODE_SAMPLING_METHOD}"
  --ode_num_steps "${ODE_NUM_STEPS}"
  --ode_atol "${ODE_ATOL}"
  --ode_rtol "${ODE_RTOL}"
  --sde_num_steps "${SDE_NUM_STEPS}"
  --fill_interval "${FILL_INTERVAL}"
  --allow_coord_stats_fallback true
  --visualize "${VISUALIZE}"
  --vis_batches "${VIS_BATCHES}"
)

if [[ -n "${LABEL_SEGY}" ]]; then
  cmd+=(--label_segy "${LABEL_SEGY}")
fi

if [ "${NPROC_PER_NODE}" -gt 1 ]; then
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" gen_infer_5d.py "${cmd[@]}"
else
    "${PYTHON_BIN}" gen_infer_5d.py "${cmd[@]}"
fi
