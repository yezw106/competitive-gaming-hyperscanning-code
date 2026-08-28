#!/usr/bin/env bash
set -euo pipefail

# Reconstruct all Game 2 pseudo-dyad voxel-wise correlation maps.
WKDIR="/sharedata/public/LOL_Project/GameHyperscanning2022/process/r_Game2"
RESULT_ROOT="${WKDIR}/r_results/permutation_whole_run_residual_rmMotionCSFWM"
PAIR_LIST_DIR="${WKDIR}/randpairLists"
MASTER="${WKDIR}/scripts/MNI152_2.5mm_brain.nii.gz"

for i in $(seq 1 1000); do
    pair_file="${PAIR_LIST_DIR}/randpairs_${i}.txt"
    result_dir="${RESULT_ROOT}/randpairs_${i}"

    if [[ ! -f "${pair_file}" ]]; then
        echo "ERROR: missing pseudo-dyad list: ${pair_file}" >&2
        exit 1
    fi
    if [[ ! -d "${result_dir}" ]]; then
        echo "ERROR: missing result directory: ${result_dir}" >&2
        exit 1
    fi

    echo "Reconstructing permutation group ${i}/1000"
    while IFS= read -r pair_id || [[ -n "${pair_id}" ]]; do
        pair_id="${pair_id//$'\r'/}"
        [[ -z "${pair_id}" ]] && continue

        r_file="${result_dir}/${pair_id}_r.txt"
        out_file="${result_dir}/${pair_id}_r.nii"
        if [[ ! -f "${r_file}" ]]; then
            echo "ERROR: missing correlation file: ${r_file}" >&2
            exit 1
        fi

        3dUndump \
            -overwrite \
            -prefix "${out_file}" \
            -master "${MASTER}" \
            -datum float \
            -ijk "${r_file}"
    done < "${pair_file}"
done
