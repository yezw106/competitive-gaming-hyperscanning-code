#!/usr/bin/env bash
set -euo pipefail

WKDIR="/sharedata/public/LOL_Project/GameHyperscanning2022/process/r_Game2"
RESULT_ROOT="${WKDIR}/r_results/permutation_whole_run_residual_rmMotionCSFWM"
MASTER="${WKDIR}/scripts/MNI152_2.5mm_brain.nii.gz"
INPUT_TXT="${RESULT_ROOT}/permutation_1000_p_fit_threshold0.001_scale10000.txt"
OUTPUT_MAP="${RESULT_ROOT}/permutation_1000_p_fit_threshold0.001_scale10000.nii"

if [[ ! -f "${INPUT_TXT}" ]]; then
    echo "ERROR: missing fitted and thresholded CDF file: ${INPUT_TXT}" >&2
    exit 1
fi

3dUndump     -overwrite     -prefix "${OUTPUT_MAP}"     -master "${MASTER}"     -datum float     -ijk "${INPUT_TXT}"

echo "Created ${OUTPUT_MAP}"
