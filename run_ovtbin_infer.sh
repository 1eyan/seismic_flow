#!/usr/bin/env bash
set -euo pipefail
# =========================================================================
# OVT SSL 推理 — raw context + grid target
#
# Usage:
#   Single-GPU:  ./run_ovtbin_infer.sh
#   Multi-GPU:   NPROC_PER_NODE=4 ./run_ovtbin_infer.sh
# =========================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DEVICE="${DEVICE:-cuda:0}"
MASTER_PORT="${MASTER_PORT:-29503}"

# ---- Data paths ----
DATA_DIR="${DATA_DIR:-/data/shared/测试数据/h5}"
H5_RAW="${H5_RAW:-${DATA_DIR}/field1031_irregular_ovt.h5}"
H5_GRID="${H5_GRID:-${DATA_DIR}/field1031_test_aligned.h5}"
H5_REGULAR="${H5_REGULAR:-${DATA_DIR}/field1031_label_ovt.h5}"

# ---- SEGY ----
MASK_SEGY="${MASK_SEGY:-/data/shared/测试数据/mask_from_label.sgy}"
LABEL_SEGY="${LABEL_SEGY:-/data/shared/测试数据/reg_pku_1031/reg_pku_1030/reg5dbin_label1031.sgy}"

# ---- Output ----
MODEL_NAME="${MODEL_NAME:-trace_axis}"
CHECKPOINT="${CHECKPOINT:-/home/chengzhitong/5d_regular/seismic_flow_v2/resultsFPM/trace_axis_datatype_field1031_0613_ovt/checkpoints/model-epoch-800.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/gen_fill_results_ovtbin}"
OUTPUT_SEGY="${OUTPUT_SEGY:-${OUTPUT_DIR}/filled_missing.sgy}"
OUTPUT_RESIDUAL_SEGY="${OUTPUT_RESIDUAL_SEGY:-${OUTPUT_DIR}/residual.sgy}"
FILLED_GRID_OUT="${FILLED_GRID_OUT:-${OUTPUT_DIR}/filled_grid.h5}"

# ---- OVT SSL ----
OVT_TARGET_SLOTS="${OVT_TARGET_SLOTS:-30}"
OVT_KDTREE_OFFSET_WEIGHT="${OVT_KDTREE_OFFSET_WEIGHT:-2.0}"
TRACE_PS="${TRACE_PS:-128}"
TIME_PS="${TIME_PS:-1256}"

# ---- Inference ----
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-72}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEGY_PROFILE="${SEGY_PROFILE:-field1031}"
H5_MISSING_EPS="${H5_MISSING_EPS:-1e-10}"

# ---- Model arch (must match training) ----
MODEL_TYPE="${MODEL_TYPE:-trace_axis}"
MLP_RATIO="${MLP_RATIO:-4}"
NUM_BANDS="${NUM_BANDS:-16}"
USE_MISSING_EMBEDDING="${USE_MISSING_EMBEDDING:-false}"
USE_ENERGY_MLP="${USE_ENERGY_MLP:-false}"
HEADWISE_ATTN_OUTPUT_GATE="${HEADWISE_ATTN_OUTPUT_GATE:-false}"
ELEMENTWISE_ATTN_OUTPUT_GATE="${ELEMENTWISE_ATTN_OUTPUT_GATE:-false}"
USE_P_SCALE="${USE_P_SCALE:-false}"
CHUNK_LENGTH_FLOW="${CHUNK_LENGTH_FLOW:-256}"

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
SDE_SAMPLING_METHOD="${SDE_SAMPLING_METHOD:-Euler}"
SDE_NUM_STEPS="${SDE_NUM_STEPS:-250}"

# ---- Additional ----
FULL_COVERAGE="${FULL_COVERAGE:-true}"  # 全量推理：遍历全部网格单元（observed+missing）恰好一次
VISUALIZE="${VISUALIZE:-true}"
VIS_BATCHES="${VIS_BATCHES:-0}"
BACKFILL_INTERVAL="${BACKFILL_INTERVAL:-1}"
HEADER_MODE="${HEADER_MODE:-fixed}"
SORT_OUTPUT="${SORT_OUTPUT:-true}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "OVT SSL Inference"
echo "============================================================"
echo "checkpoint:     ${CHECKPOINT}"
echo "grid (target):  ${H5_GRID}"
echo "raw (context):  ${H5_RAW}"
echo "regular (stats):${H5_REGULAR}"
echo "mask_segy:      ${MASK_SEGY}"
echo "output_segy:    ${OUTPUT_SEGY}"
echo "filled_grid:    ${FILLED_GRID_OUT}"
echo "model_type:     ${MODEL_TYPE}"
echo "use_p_scale:    ${USE_P_SCALE}"
echo "full_coverage:  ${FULL_COVERAGE}"
echo "nproc_per_node: ${NPROC_PER_NODE}"
echo "============================================================"

cmd=(
  --checkpoint "${CHECKPOINT}"
  --h5_regular "${H5_REGULAR}"
  --h5_mask "${H5_GRID}"
  --h5_raw_context "${H5_RAW}"
  --mask_path "${MASK_SEGY}"
  --output_dir "${OUTPUT_DIR}"
  --output_segy "${OUTPUT_SEGY}"
  --output_residual_segy "${OUTPUT_RESIDUAL_SEGY}"
  --filled_grid_out "${FILLED_GRID_OUT}"
  --dataset_mode ovtbin
  --num_workers "${NUM_WORKERS}"
  --segy_profile "${SEGY_PROFILE}"
  --model_type "${MODEL_TYPE}"
  --sampling_method "${SAMPLING_METHOD}"
  --ode_num_steps "${ODE_NUM_STEPS}"
  --ode_sampling_method "${ODE_SAMPLING_METHOD}"
  --ode_atol "${ODE_ATOL}"
  --ode_rtol "${ODE_RTOL}"
  --sde_sampling_method "${SDE_SAMPLING_METHOD}"
  --sde_num_steps "${SDE_NUM_STEPS}"
  --use_missing_embedding "${USE_MISSING_EMBEDDING}"
  --use_energy_mlp "${USE_ENERGY_MLP}"
  --headwise_attn_output_gate "${HEADWISE_ATTN_OUTPUT_GATE}"
  --elementwise_attn_output_gate "${ELEMENTWISE_ATTN_OUTPUT_GATE}"
  --mlp_ratio "${MLP_RATIO}"
  --num_bands "${NUM_BANDS}"
  --use_p_scale "${USE_P_SCALE}"
  --chunk_length_flow "${CHUNK_LENGTH_FLOW}"
  --path_type "${PATH_TYPE}"
  --prediction "${PREDICTION}"
  --loss_weight "${LOSS_WEIGHT}"
  --visualize "${VISUALIZE}"
  --vis_batches "${VIS_BATCHES}"
  --inference_batch_size "${INFERENCE_BATCH_SIZE}"
  --trace_ps "${TRACE_PS}"
  --time_ps "${TIME_PS}"
  --ovt_target_slots "${OVT_TARGET_SLOTS}"
  --ovt_kdtree_offset_weight "${OVT_KDTREE_OFFSET_WEIGHT}"
  --sort_output "${SORT_OUTPUT}"
  --backfill_interval "${BACKFILL_INTERVAL}"
  --header_mode "${HEADER_MODE}"
  --non_strict_load
)

if [[ -n "${LABEL_SEGY}" ]]; then
  cmd+=(--label_segy "${LABEL_SEGY}")
fi

if [[ "${FULL_COVERAGE}" == "true" ]]; then
  cmd+=(--full_coverage)
fi

if [ "${NPROC_PER_NODE}" -gt 1 ]; then
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" gen_infer.py "${cmd[@]}"
else
    "${PYTHON_BIN}" gen_infer.py "${cmd[@]}"
fi
