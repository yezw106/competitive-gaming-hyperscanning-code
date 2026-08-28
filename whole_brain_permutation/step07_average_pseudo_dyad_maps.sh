#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

WKDIR="/sharedata/public/LOL_Project/GameHyperscanning2022/process/r_Game2"
RESULT_ROOT="${WKDIR}/r_results/permutation_whole_run_residual_rmMotionCSFWM"
MASTER="${WKDIR}/scripts/MNI152_2.5mm_brain.nii.gz"

for i in $(seq 1 1000); do
    result_dir="${RESULT_ROOT}/randpairs_${i}"
    map_files=("${result_dir}"/*_r.nii)

    if (( ${#map_files[@]} == 0 )); then
        echo "ERROR: no pseudo-dyad r maps found in ${result_dir}" >&2
        exit 1
    fi

    echo "Averaging pseudo-dyad group ${i}/1000 (${#map_files[@]} maps)"
    3dMean         -overwrite         -prefix "${result_dir}/randpair_meanr_${i}.nii"         -datum float         "${map_files[@]}"

    3dmaskdump         -mask "${MASTER}"         "${result_dir}/randpair_meanr_${i}.nii"         > "${result_dir}/randpair_meanr_${i}.txt"
done
