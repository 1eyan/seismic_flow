#!/usr/bin/env bash
set -euo pipefail

# Launch FPM V3 H5 inference and fill missing traces into a SEGY file.
# Edit the paths below before running, or override them with environment vars:
#   CHECKPOINT=/path/model.pth H5_MASK=/path/mask.h5 ./run_gen_infer_field1031.sh
#
# Single-GPU:  ./run_gen_infer_field1031.sh
# Multi-GPU:   NPROC_PER_NODE=4 ./run_gen_infer_field1031.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"  # >1 enables DDP via torchrun
DEVICE="${DEVICE:-cuda:0}"  # only used in single-GPU mode

CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/resultsFPM/gated_datatype_field1031_0517/checkpoints/model-epoch-200.pth}"
H5_REGULAR="${H5_REGULAR:-/cloud/cloud-s3fs/reg5dbin_label1031.h5}"
H5_MASK="${H5_MASK:-/cloud/cloud-s3fs/reg5dbin_label1031_binning.h5}"
MASK_SEGY="${MASK_SEGY:-/cloud/cloud-ssd2/测试数据/mask_from_label.sgy}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/gen_fill_results_1031field}"
OUTPUT_SEGY="${OUTPUT_SEGY:-${OUTPUT_DIR}/filled_missing.sgy}"
OUTPUT_RESIDUAL_SEGY="${OUTPUT_RESIDUAL_SEGY:-${OUTPUT_DIR}/residual.sgy}"
LABEL_SEGY="${LABEL_SEGY:-/cloud/cloud-ssd2/测试数据/reg_pku_1031/reg_pku_1030/reg5dbin_label1031.sgy}"  # ground truth SEGY for residual computation (optional)

SEGY_PROFILE="${SEGY_PROFILE:-field1031}"

NUM_WORKERS="${NUM_WORKERS:-4}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-1}"  # patches per forward pass
TRACE_PS="${TRACE_PS:-128}"  # patch trace count (must match training)
OVERLAP_RATIO="${OVERLAP_RATIO:-0.25}"  # sliding window overlap (0.5 = 50% overlap, equal-weight averaging)
TIME_PS="${TIME_PS:-1256}"
H5_MISSING_EPS="${H5_MISSING_EPS:-1e-10}"  # trace amplitude threshold for missing detection

MODEL_TYPE="${MODEL_TYPE:-gated}"
SAMPLING_METHOD="${SAMPLING_METHOD:-ode}"
ODE_NUM_STEPS="${ODE_NUM_STEPS:-50}"
ODE_SAMPLING_METHOD="${ODE_SAMPLING_METHOD:-dopri5}"
ODE_ATOL="${ODE_ATOL:-1e-6}"
ODE_RTOL="${ODE_RTOL:-1e-3}"
SDE_SAMPLING_METHOD="${SDE_SAMPLING_METHOD:-Euler}"
SDE_NUM_STEPS="${SDE_NUM_STEPS:-250}"

# Model hyperparams — must match training
USE_MISSING_EMBEDDING="${USE_MISSING_EMBEDDING:-false}"
USE_ENERGY_MLP="${USE_ENERGY_MLP:-false}"
HEADWISE_ATTN_OUTPUT_GATE="${HEADWISE_ATTN_OUTPUT_GATE:-true}"
ELEMENTWISE_ATTN_OUTPUT_GATE="${ELEMENTWISE_ATTN_OUTPUT_GATE:-false}"
GEOM_MODE="${GEOM_MODE:-relative}"
USE_P_SCALE="${USE_P_SCALE:-true}"
CHUNK_LENGTH_FLOW="${CHUNK_LENGTH_FLOW:-256}"

# Visualization toggle
VISUALIZE="${VISUALIZE:-true}"
VIS_BATCHES="${VIS_BATCHES:-0}"

# Output SEGY sorting
SORT_OUTPUT="${SORT_OUTPUT:-true}"  # sort output traces by profile.sort_keys

mkdir -p "${OUTPUT_DIR}"

cmd=(
  --checkpoint "${CHECKPOINT}"
  --h5_regular "${H5_REGULAR}"
  --h5_mask "${H5_MASK}"
  --mask_path "${MASK_SEGY}"
  --output_dir "${OUTPUT_DIR}"
  --output_segy "${OUTPUT_SEGY}"
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
  --geom_mode "${GEOM_MODE}"
  --use_p_scale "${USE_P_SCALE}"
  --chunk_length_flow "${CHUNK_LENGTH_FLOW}"
  --visualize "${VISUALIZE}"
  --vis_batches "${VIS_BATCHES}"
  --output_residual_segy "${OUTPUT_RESIDUAL_SEGY}"
  --inference_batch_size "${INFERENCE_BATCH_SIZE}"
  --trace_ps "${TRACE_PS}"
  --overlap_ratio "${OVERLAP_RATIO}"
  --sort_output "${SORT_OUTPUT}"
)

if [[ -n "${TIME_PS}" ]]; then
  cmd+=(--time_ps "${TIME_PS}")
fi

if [[ -n "${H5_MISSING_EPS}" ]]; then
  cmd+=(--h5_missing_eps "${H5_MISSING_EPS}")
fi

if [[ -n "${LABEL_SEGY}" ]]; then
  cmd+=(--label_segy "${LABEL_SEGY}")
fi

# Build launch command
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  LAUNCHER=(torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port 29502)
else
  LAUNCHER=("${PYTHON_BIN}")
  cmd+=(--device "${DEVICE}")
fi

echo "============================================================"
echo "FPM V3 inference launch (patch-based)"
echo "root_dir      : ${ROOT_DIR}"
echo "launcher      : ${LAUNCHER[*]}"
echo "nproc_per_node: ${NPROC_PER_NODE}"
echo "checkpoint    : ${CHECKPOINT}"
echo "h5_regular    : ${H5_REGULAR}"
echo "h5_mask       : ${H5_MASK}"
echo "mask_segy     : ${MASK_SEGY}"
echo "label_segy    : ${LABEL_SEGY:-<not set>}"
echo "output_segy   : ${OUTPUT_SEGY}"
echo "output_residual: ${OUTPUT_RESIDUAL_SEGY}"
echo "segy_profile  : ${SEGY_PROFILE}"
echo "h5_missing_eps: ${H5_MISSING_EPS}"
echo "model_type    : ${MODEL_TYPE}"
echo "inference_batch_size: ${INFERENCE_BATCH_SIZE}"
echo "trace_ps      : ${TRACE_PS}"
echo "overlap_ratio : ${OVERLAP_RATIO}"
echo "geom_mode     : ${GEOM_MODE}"
echo "use_p_scale   : ${USE_P_SCALE}"
echo "visualize     : ${VISUALIZE}"
echo "sort_output   : ${SORT_OUTPUT}"
echo "============================================================"

"${LAUNCHER[@]}" "${ROOT_DIR}/gen_infer.py" "${cmd[@]}" 2>&1 | tee "${OUTPUT_DIR}/run_gen_infer.stdout.log"
