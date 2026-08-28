#!/usr/bin/env bash
set -euo pipefail

WKDIR="/sharedata/public/LOL_Project/GameHyperscanning2022/process/r_Game2"
PAIRS_FILE="${WKDIR}/pairs.txt"
RESULT_DIR="${WKDIR}/r_results/paired_whole_run_residual_rmMotionCSFWM"
MASTER="${WKDIR}/scripts/MNI152_2.5mm_brain.nii.gz"

if [[ ! -f "${PAIRS_FILE}" ]]; then
    echo "ERROR: missing interacting-dyad list: ${PAIRS_FILE}" >&2
    exit 1
fi

while IFS= read -r pair_id || [[ -n "${pair_id}" ]]; do
    pair_id="${pair_id//$'\r'/}"
    [[ -z "${pair_id}" ]] && continue

    r_file="${RESULT_DIR}/${pair_id}_r.txt"
    out_file="${RESULT_DIR}/${pair_id}_r.nii"
    if [[ ! -f "${r_file}" ]]; then
        echo "ERROR: missing correlation file: ${r_file}" >&2
        exit 1
    fi

    echo "Reconstructing interacting dyad ${pair_id}"
    3dUndump         -overwrite         -prefix "${out_file}"         -master "${MASTER}"         -datum float         -ijk "${r_file}"
done < "${PAIRS_FILE}"
