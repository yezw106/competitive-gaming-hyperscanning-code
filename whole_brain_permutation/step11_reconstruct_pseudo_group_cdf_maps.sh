#!/usr/bin/env bash
set -euo pipefail

# Reconstruct the empirical CDF map for each Game 2 pseudo-dyad group.
WKDIR="/sharedata/public/LOL_Project/GameHyperscanning2022/process/r_Game2"
RESULT_ROOT="${WKDIR}/r_results/permutation_whole_run_residual_rmMotionCSFWM"
MASTER="${WKDIR}/scripts/MNI152_2.5mm_brain.nii.gz"

for i in $(seq 1 1000); do
    result_dir="${RESULT_ROOT}/randpairs_${i}"
    cdf_file="${result_dir}/permutation_1000_p.txt"
    out_file="${result_dir}/permutation_1000_p.nii"

    if [[ ! -f "${cdf_file}" ]]; then
        echo "ERROR: missing empirical CDF file: ${cdf_file}" >&2
        exit 1
    fi

    echo "Reconstructing null CDF map ${i}/1000"
    3dUndump \
        -overwrite \
        -prefix "${out_file}" \
        -master "${MASTER}" \
        -datum float \
        -ijk "${cdf_file}"
done
