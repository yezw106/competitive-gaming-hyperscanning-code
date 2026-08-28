#!/usr/bin/env bash
set -euo pipefail

WKDIR="/sharedata/public/LOL_Project/GameHyperscanning2022/process/r_Game2"
RESULT_ROOT="${WKDIR}/r_results/permutation_whole_run_residual_rmMotionCSFWM"
INPUT_MAP="${RESULT_ROOT}/permutation_1000_p_fit_threshold0.001_scale10000.nii"
THRESHOLD_FILE="${RESULT_ROOT}/cluster_extent_threshold_p05.txt"
VOXEL_MASK="${RESULT_ROOT}/permutation_cdf_gt0999_binary.nii.gz"
CLUSTER_LABELS="${RESULT_ROOT}/permutation_cdf_gt0999_cluster_labels.nii.gz"
FINAL_MAP="${RESULT_ROOT}/permutation_1000_p_fit_threshold0.001_scale10000_cluster_corrected.nii.gz"
REPORT="${RESULT_ROOT}/permutation_cdf_gt0999_cluster_report.txt"

if [[ ! -f "${INPUT_MAP}" ]]; then
    echo "ERROR: missing real fitted-CDF map: ${INPUT_MAP}" >&2
    exit 1
fi
if [[ ! -f "${THRESHOLD_FILE}" ]]; then
    echo "ERROR: missing cluster threshold: ${THRESHOLD_FILE}" >&2
    exit 1
fi

cluster_nvox=$(tr -d '[:space:]' < "${THRESHOLD_FILE}")
if [[ ! "${cluster_nvox}" =~ ^[0-9]+$ ]] || (( cluster_nvox < 1 )); then
    echo "ERROR: invalid cluster threshold: ${cluster_nvox}" >&2
    exit 1
fi

# The fitted map is already zero outside CDF > .999. Binarize it before
# applying the NN=2 cluster-extent threshold.
3dcalc     -overwrite     -a "${INPUT_MAP}"     -expr 'step(a)'     -prefix "${VOXEL_MASK}"

3dClusterize     -overwrite     -inset "${VOXEL_MASK}"     -ithr 0     -1sided RIGHT 0.5     -NN 2     -clust_nvox "${cluster_nvox}"     -pref_map "${CLUSTER_LABELS}"     -1Dformat > "${REPORT}"

3dcalc     -overwrite     -a "${INPUT_MAP}"     -b "${CLUSTER_LABELS}"     -expr 'a*step(b)'     -prefix "${FINAL_MAP}"

echo "Cluster threshold: ${cluster_nvox} voxels (NN=2)"
echo "Final corrected map: ${FINAL_MAP}"
