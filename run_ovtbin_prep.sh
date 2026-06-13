#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/convert_toolV2/ovtbin_h5.py"

# =========================================================================
# OVT binning data preparation — 3 modes:
#   build_grid       label -> OVT + grid_spec
#   add_ovt_fields   irregular -> OVT only (no projection)
#   project_to_grid  test -> aligned to grid (with trace_mask)
#
# Usage:
#   DATA_DIR=/path/to/h5 RUN_MODE=build_grid      ./run_ovtbin_prep.sh
#   DATA_DIR=/path/to/h5 RUN_MODE=add_ovt_fields  ./run_ovtbin_prep.sh
#   DATA_DIR=/path/to/h5 RUN_MODE=project_to_grid ./run_ovtbin_prep.sh
#
# All H5 files and grid_spec live under DATA_DIR.
# =========================================================================

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_MODE="${RUN_MODE:-add_ovt_fields}"
DRY_RUN="${DRY_RUN:-false}"

# ---- Single root for all data ----
DATA_DIR="${DATA_DIR:-/data/shared/测试数据/h5}"
GRID_SPEC="${GRID_SPEC:-${DATA_DIR}/ovtgrid.json}"

# ---- Common parameters ----
GROUP_NAME="${GROUP_NAME:-1551}"
SEGY_PROFILE="${SEGY_PROFILE:-field1031}"
BINNING_MODE="${BINNING_MODE:-expert}"
WAVE_TYPE="${WAVE_TYPE:-PP}"
OFFSET_X_BIN_SIZE="${OFFSET_X_BIN_SIZE:-400}"
OFFSET_Y_BIN_SIZE="${OFFSET_Y_BIN_SIZE:-400}"
OFFSET_X_SHIFT="${OFFSET_X_SHIFT:-}"   # auto: -bin_x/2
OFFSET_Y_SHIFT="${OFFSET_Y_SHIFT:-}"   # auto: -bin_y/2
GAMMA="${GAMMA:-2.0}"
SOURCE_LINE_INTERVAL="${SOURCE_LINE_INTERVAL:-50}"
RECEIVER_LINE_INTERVAL="${RECEIVER_LINE_INTERVAL:-200}"
MISSING_EPS="${MISSING_EPS:-1e-10}"
QC_DIR="${QC_DIR:-${DATA_DIR}}"

# ---- 4D cell projection options (only for project_to_grid) ----
MIDPOINT_BIN_MODE="${MIDPOINT_BIN_MODE:-cmp}"
MIDPOINT_KEY_COLUMNS="${MIDPOINT_KEY_COLUMNS:-cmp_line,cmp}"
INPUT_DUPLICATE_POLICY="${INPUT_DUPLICATE_POLICY:-mean}"
OUTSIDE_GRID_POLICY="${OUTSIDE_GRID_POLICY:-skip}"
ALLOW_OUTSIDE_GRID="${ALLOW_OUTSIDE_GRID:-}"  # per-mode default; set to true/false to override

# ===================================================================
# Resolve mode
# ===================================================================
case "${RUN_MODE}" in
    build_grid)
        INPUT="${INPUT:-${DATA_DIR}/field1031_label.h5}"
        OUTPUT="${OUTPUT:-${DATA_DIR}/field1031_label_ovt.h5}"
        ALIGN_TO_GRID="false"
        GRID_SOURCE=""
        GRID_SPEC_IN=""
        GRID_SPEC_OUT="${GRID_SPEC_OUT:-${GRID_SPEC}}"
        PROJECTION_MODE="trace-key"      # skip cell uniqueness check
        ALLOW_OUTSIDE_GRID="${ALLOW_OUTSIDE_GRID:-false}"
        ;;
    add_ovt_fields)
        INPUT="${INPUT:-${DATA_DIR}/field1031_irregular.h5}"
        OUTPUT="${OUTPUT:-${DATA_DIR}/field1031_irregular_ovt.h5}"
        ALIGN_TO_GRID="false"
        GRID_SOURCE=""
        GRID_SPEC_IN="${GRID_SPEC_IN:-${GRID_SPEC}}"
        GRID_SPEC_OUT=""
        PROJECTION_MODE="trace-key"      # skip cell uniqueness check
        ALLOW_OUTSIDE_GRID="${ALLOW_OUTSIDE_GRID:-true}"
        ;;
    project_to_grid)
        INPUT="${INPUT:-${DATA_DIR}/field1031_irregular.h5}"
        OUTPUT="${OUTPUT:-${DATA_DIR}/field1031_test_aligned.h5}"
        ALIGN_TO_GRID="true"
        GRID_SOURCE="${GRID_SOURCE:-${DATA_DIR}/field1031_label_ovt.h5}"
        GRID_SPEC_IN="${GRID_SPEC_IN:-${GRID_SPEC}}"
        GRID_SPEC_OUT=""
        PROJECTION_MODE="cell"
        ALLOW_OUTSIDE_GRID="${ALLOW_OUTSIDE_GRID:-true}"
        ;;
    *)
        echo "ERROR: RUN_MODE must be build_grid | add_ovt_fields | project_to_grid" >&2
        exit 1
        ;;
esac

if [[ -z "${INPUT}" ]] || [[ -z "${OUTPUT}" ]]; then
    echo "ERROR: INPUT and OUTPUT are required" >&2
    exit 1
fi

# ===================================================================
# Build command
# ===================================================================
cmd=(
    "${PYTHON_BIN}" "${PY_SCRIPT}"
    --input "${INPUT}" --output "${OUTPUT}"
    --input-type h5
    --group-name "${GROUP_NAME}"
    --segy-profile "${SEGY_PROFILE}"
    --wave-type "${WAVE_TYPE}"
    --binning-mode "${BINNING_MODE}"
    --offset-x-bin-size "${OFFSET_X_BIN_SIZE}"
    --offset-y-bin-size "${OFFSET_Y_BIN_SIZE}"
    --gamma "${GAMMA}"
    --missing-eps "${MISSING_EPS}"
    --projection-mode "${PROJECTION_MODE}"
    --overwrite-output --overwrite-fields
    --qc-dir "${QC_DIR}"
)

[[ -n "${OFFSET_X_SHIFT}" ]] && cmd+=(--offset-x-shift "${OFFSET_X_SHIFT}")
[[ -n "${OFFSET_Y_SHIFT}" ]] && cmd+=(--offset-y-shift "${OFFSET_Y_SHIFT}")
[[ -n "${SOURCE_LINE_INTERVAL}" ]] && cmd+=(--source-line-interval "${SOURCE_LINE_INTERVAL}")
[[ -n "${RECEIVER_LINE_INTERVAL}" ]] && cmd+=(--receiver-line-interval "${RECEIVER_LINE_INTERVAL}")

# Cell projection extras
if [[ "${PROJECTION_MODE}" == "cell" ]]; then
    cmd+=(
        --midpoint-bin-mode "${MIDPOINT_BIN_MODE}"
        --midpoint-key-columns "${MIDPOINT_KEY_COLUMNS}"
        --input-duplicate-policy "${INPUT_DUPLICATE_POLICY}"
        --outside-grid-policy "${OUTSIDE_GRID_POLICY}"
    )
fi

[[ -n "${GRID_SOURCE}" ]]   && cmd+=(--grid-source "${GRID_SOURCE}")
[[ -n "${GRID_SPEC_IN}" ]]  && cmd+=(--grid-spec-in "${GRID_SPEC_IN}")
[[ -n "${GRID_SPEC_OUT}" ]] && cmd+=(--grid-spec-out "${GRID_SPEC_OUT}")
[[ "${ALIGN_TO_GRID}" == "true" ]] && cmd+=(--align-to-grid)
[[ "${ALLOW_OUTSIDE_GRID}" == "true" ]] && cmd+=(--allow-outside-grid)

# ===================================================================
echo "============================================================"
echo "OVT binning — ${RUN_MODE}"
echo "============================================================"
echo "input         : ${INPUT}"
echo "output        : ${OUTPUT}"
echo "data_dir      : ${DATA_DIR}"
echo "bin sizes     : ${OFFSET_X_BIN_SIZE} x ${OFFSET_Y_BIN_SIZE}"
echo "align_to_grid : ${ALIGN_TO_GRID}"
echo "============================================================"
echo "${cmd[*]}"
echo ""

[[ "${DRY_RUN}" == "true" ]] && exit 0
exec "${cmd[@]}"
