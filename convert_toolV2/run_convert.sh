#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/Segy2H5.py"

# Override with env vars or edit here:
#   IRR_SEGY=/path/irr.sgy MASK_SEGY=/path/mask.sgy LABEL_SEGY=/path/label.sgy ./run_convert.sh

IRR_SEGY="${IRR_SEGY:-/NAS/czt/mount/chengzhitong/data/测试数据/raw5d_data_updated/raw5d_data1104.sgy}"
MASK_SEGY="${MASK_SEGY:-/NAS/czt/mount/chengzhitong/data/测试数据/mask_from_label.sgy}"
LABEL_SEGY="${LABEL_SEGY:-/NAS/czt/mount/chengzhitong/data/测试数据/reg_pku_1031/reg_pku_1030/reg5dbin_label1031.sgy}"
#IRR_SEGY="${IRR_SEGY:-/cloud/cloud-s3fs/dongfang_syn_reg/004-sw06-Sj5-label_irr30.sgy}"
#MASK_SEGY="${MASK_SEGY:-/cloud/cloud-s3fs/dongfang_syn_reg/mask_miss30pct_004-sw06-Sj5-label.sgy}"
#LABEL_SEGY="${LABEL_SEGY:-/cloud/cloud-s3fs/dongfang_syn_reg/004-sw06-Sj5-label.sgy}"


SEGY_PROFILE="${SEGY_PROFILE:-field1031}"  # sw06 | field1031 | segc3
MODE="${MODE:-fixed}"                      # fixed | self_computed
DATASET_NAME="${DATASET_NAME:-field1031}"
GROUP_NAME="${GROUP_NAME:-1551}"

python "${PY_SCRIPT}" \
    --irr "${IRR_SEGY}" \
    --mask "${MASK_SEGY}" \
    --label "${LABEL_SEGY}" \
    --dataset-name "${DATASET_NAME}" \
    --mode "${MODE}" \
    --group-name "${GROUP_NAME}" \
    --segy_profile "${SEGY_PROFILE}"
