#!/usr/bin/env bash
set -euo pipefail

# Obtain the maximum suprathreshold cluster extent for each Game 2 null map.
WKDIR="/sharedata/public/LOL_Project/GameHyperscanning2022/process/r_Game2"
RESULT_ROOT="${WKDIR}/r_results/permutation_whole_run_residual_rmMotionCSFWM"

for i in $(seq 1 1000); do
    result_dir="${RESULT_ROOT}/randpairs_${i}"
    p_file="${result_dir}/permutation_1000_p.nii"
    out_file="${result_dir}/p0001_clusterize.txt"

    if [[ ! -f "${p_file}" ]]; then
        echo "ERROR: missing null CDF map: ${p_file}" >&2
        exit 1
    fi

    echo "Clustering permutation group ${i}/1000"
    3dClusterize \
        -inset "${p_file}" \
        -ithr 0 \
        -1sided RIGHT 0.9990 \
        -NN 2 \
        -clust_nvox 2 \
        -1Dformat > "${out_file}"
done
