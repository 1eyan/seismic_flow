#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
PY_SCRIPT="${SCRIPT_DIR}/ovtbin_h5.py"

# Usage:
#   RUN_MODE=build_grid bash convert_tool/run_ovtbin_h5.sh
#   RUN_MODE=project_irregular bash convert_tool/run_ovtbin_h5.sh
#
# Env overrides are supported for every variable below.

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_MODE="${RUN_MODE:-build_grid}"  # build_grid | project_irregular
DRY_RUN="${DRY_RUN:-false}"

LABEL_INPUT="${LABEL_INPUT:-/NAS/czt/mount/chengzhitong/data/测试数据/h5/field1031_label.h5}"
LABEL_OUTPUT="${LABEL_OUTPUT:-/NAS/czt/mount/chengzhitong/data/测试数据/h5/field1031_label_ovt.h5}"
IRREGULAR_INPUT="${IRREGULAR_INPUT:-/NAS/czt/mount/chengzhitong/data/测试数据/h5/field1031_irregular.h5}"
MASK_OUTPUT="${MASK_OUTPUT:-/NAS/czt/mount/chengzhitong/data/测试数据/h5/field1031_mask_ovt.h5}"
GRID_SPEC="${GRID_SPEC:-/NAS/czt/mount/chengzhitong/data/测试数据/ovtgrid.json}"

INPUT_TYPE="${INPUT_TYPE:-h5}"                         # h5 | segy
GROUP_NAME="${GROUP_NAME:-1551}"
SEGY_PROFILE="${SEGY_PROFILE:-field1031}"               # sw06 | field1031 | segc3
BINNING_MODE="${BINNING_MODE:-expert}"                  # expert | beginner
WAVE_TYPE="${WAVE_TYPE:-PP}"                            # PP | PS

# --- Binning parameters (used in expert mode) ---
OFFSET_X_BIN_SIZE="${OFFSET_X_BIN_SIZE:-400}"
OFFSET_Y_BIN_SIZE="${OFFSET_Y_BIN_SIZE:-400}"

# --- Beginner mode parameters ---
SOURCE_LINE_INTERVAL="${SOURCE_LINE_INTERVAL:-50}"
RECEIVER_LINE_INTERVAL="${RECEIVER_LINE_INTERVAL:-200}"
GAMMA="${GAMMA:-2.0}"

# --- Shifts ---
OFFSET_X_SHIFT="${OFFSET_X_SHIFT:-}"   # auto: -bin_x/2
OFFSET_Y_SHIFT="${OFFSET_Y_SHIFT:-}"   # auto: -bin_y/2

# --- 4D cell projection ---
PROJECTION_MODE="${PROJECTION_MODE:-cell}"              # cell | trace-key
MIDPOINT_BIN_MODE="${MIDPOINT_BIN_MODE:-cmp}"           # cmp | coordinate
MIDPOINT_KEY_COLUMNS="${MIDPOINT_KEY_COLUMNS:-cmp_line,cmp}"
INPUT_DUPLICATE_POLICY="${INPUT_DUPLICATE_POLICY:-mean}"  # mean | error | first
OUTSIDE_GRID_POLICY="${OUTSIDE_GRID_POLICY:-skip}"        # skip | error
ALLOW_OUTSIDE_GRID="${ALLOW_OUTSIDE_GRID:-false}"
KEY_COLUMNS="${KEY_COLUMNS:-shot_line,shot_stake,recv_line,recv_stake}"

# --- Output options ---
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-true}"
OVERWRITE_FIELDS="${OVERWRITE_FIELDS:-true}"
QC_DIR="${QC_DIR:-/NAS/czt/mount/chengzhitong/data/测试数据}"
MISSING_EPS="${MISSING_EPS:-1e-10}"

case "${RUN_MODE}" in
    build_grid)
        INPUT="${INPUT:-${LABEL_INPUT}}"
        OUTPUT="${OUTPUT:-${LABEL_OUTPUT}}"
        ALIGN_TO_GRID="false"
        GRID_SOURCE="${GRID_SOURCE:-}"
        GRID_GROUP_NAME="${GRID_GROUP_NAME:-}"
        GRID_SPEC_IN="${GRID_SPEC_IN:-}"
        GRID_SPEC_OUT="${GRID_SPEC_OUT:-${GRID_SPEC}}"
        ;;
    project_irregular)
        INPUT="${INPUT:-${IRREGULAR_INPUT}}"
        OUTPUT="${OUTPUT:-${MASK_OUTPUT}}"
        ALIGN_TO_GRID="true"
        GRID_SOURCE="${GRID_SOURCE:-${LABEL_OUTPUT}}"
        GRID_GROUP_NAME="${GRID_GROUP_NAME:-}"
        GRID_SPEC_IN="${GRID_SPEC_IN:-${GRID_SPEC}}"
        GRID_SPEC_OUT="${GRID_SPEC_OUT:-}"
        ;;
    *)
        echo "ERROR: RUN_MODE must be build_grid or project_irregular, got ${RUN_MODE}" >&2
        exit 1
        ;;
esac

if [[ -z "${INPUT}" ]]; then
    echo "ERROR: INPUT is required." >&2
    exit 1
fi
if [[ -z "${OUTPUT}" ]]; then
    echo "ERROR: OUTPUT is required." >&2
    exit 1
fi

cmd=(
    "${PYTHON_BIN}" "${PY_SCRIPT}"
    --input "${INPUT}"
    --output "${OUTPUT}"
    --input-type "${INPUT_TYPE}"
    --group-name "${GROUP_NAME}"
    --segy-profile "${SEGY_PROFILE}"
    --wave-type "${WAVE_TYPE}"
    --binning-mode "${BINNING_MODE}"
    --gamma "${GAMMA}"
    --missing-eps "${MISSING_EPS}"
    --projection-mode "${PROJECTION_MODE}"
    --midpoint-bin-mode "${MIDPOINT_BIN_MODE}"
    --midpoint-key-columns "${MIDPOINT_KEY_COLUMNS}"
    --input-duplicate-policy "${INPUT_DUPLICATE_POLICY}"
    --outside-grid-policy "${OUTSIDE_GRID_POLICY}"
)

[[ -n "${OFFSET_X_SHIFT}" ]] && cmd+=(--offset-x-shift "${OFFSET_X_SHIFT}")
[[ -n "${OFFSET_Y_SHIFT}" ]] && cmd+=(--offset-y-shift "${OFFSET_Y_SHIFT}")

if [[ "${PROJECTION_MODE}" == "trace-key" && -n "${KEY_COLUMNS}" ]]; then
    cmd+=(--key-columns "${KEY_COLUMNS}")
fi

if [[ -n "${OFFSET_X_BIN_SIZE}" ]]; then
    cmd+=(--offset-x-bin-size "${OFFSET_X_BIN_SIZE}")
fi
if [[ -n "${OFFSET_Y_BIN_SIZE}" ]]; then
    cmd+=(--offset-y-bin-size "${OFFSET_Y_BIN_SIZE}")
fi
if [[ -n "${SOURCE_LINE_INTERVAL}" ]]; then
    cmd+=(--source-line-interval "${SOURCE_LINE_INTERVAL}")
fi
if [[ -n "${RECEIVER_LINE_INTERVAL}" ]]; then
    cmd+=(--receiver-line-interval "${RECEIVER_LINE_INTERVAL}")
fi

if [[ -n "${GRID_SOURCE}" ]]; then
    cmd+=(--grid-source "${GRID_SOURCE}")
fi
if [[ -n "${GRID_GROUP_NAME}" ]]; then
    cmd+=(--grid-group-name "${GRID_GROUP_NAME}")
fi
if [[ -n "${GRID_SPEC_IN}" ]]; then
    cmd+=(--grid-spec-in "${GRID_SPEC_IN}")
fi
if [[ -n "${GRID_SPEC_OUT}" ]]; then
    cmd+=(--grid-spec-out "${GRID_SPEC_OUT}")
fi
if [[ "${ALIGN_TO_GRID}" == "true" ]]; then
    cmd+=(--align-to-grid)
fi
if [[ "${ALLOW_OUTSIDE_GRID}" == "true" ]]; then
    cmd+=(--allow-outside-grid)
fi

if [[ "${OVERWRITE_OUTPUT}" == "true" ]]; then
    cmd+=(--overwrite-output)
fi
if [[ "${OVERWRITE_FIELDS}" == "true" ]]; then
    cmd+=(--overwrite-fields)
fi
if [[ -n "${QC_DIR}" ]]; then
    cmd+=(--qc-dir "${QC_DIR}")
fi

echo "RUN_MODE: ${RUN_MODE}"
echo "Executing: ${cmd[*]}"
if [[ "${DRY_RUN}" == "true" ]]; then
    exit 0
fi
exec "${cmd[@]}"
