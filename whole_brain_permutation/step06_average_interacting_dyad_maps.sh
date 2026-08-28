#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

WKDIR="/sharedata/public/LOL_Project/GameHyperscanning2022/process/r_Game2"
RESULT_DIR="${WKDIR}/r_results/paired_whole_run_residual_rmMotionCSFWM"
MASTER="${WKDIR}/scripts/MNI152_2.5mm_brain.nii.gz"
map_files=("${RESULT_DIR}"/*_r.nii)

if (( ${#map_files[@]} == 0 )); then
    echo "ERROR: no interacting-dyad r maps found in ${RESULT_DIR}" >&2
    exit 1
fi

echo "Averaging ${#map_files[@]} interacting-dyad maps"
3dMean     -overwrite     -prefix "${RESULT_DIR}/paired_meanr.nii"     -datum float     "${map_files[@]}"

3dmaskdump     -mask "${MASTER}"     "${RESULT_DIR}/paired_meanr.nii" > "${RESULT_DIR}/paired_meanr.txt"
