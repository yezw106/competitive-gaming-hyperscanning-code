#!/usr/bin/env bash
set -euo pipefail

PROCESS_ROOT="/sharedata/public/LOL_Project/GameHyperscanning2022/process"
GAME2_ROOT="${PROCESS_ROOT}/r_Game2"
SUBJECTS_FILE="${GAME2_ROOT}/subjects.txt"
MASTER_MASK="${GAME2_ROOT}/scripts/MNI152_2.5mm_brain.nii.gz"
OUTPUT_DIR="${GAME2_ROOT}/whole_run_residual_rmMotionCSFWM"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${SUBJECTS_FILE}" ]]; then
    echo "ERROR: missing subject list: ${SUBJECTS_FILE}" >&2
    exit 1
fi
if [[ ! -f "${MASTER_MASK}" ]]; then
    echo "ERROR: missing analysis mask: ${MASTER_MASK}" >&2
    exit 1
fi

while IFS= read -r subject_id || [[ -n "${subject_id}" ]]; do
    subject_id="${subject_id//$'\r'/}"
    [[ -z "${subject_id}" ]] && continue

    input_file="${PROCESS_ROOT}/${subject_id}/residual_Game2_rmMotionCSFWM.nii.gz"
    output_file="${OUTPUT_DIR}/${subject_id}.txt"

    if [[ ! -f "${input_file}" ]]; then
        echo "ERROR: missing preprocessed Game 2 data: ${input_file}" >&2
        exit 1
    fi

    echo "Extracting ${subject_id}"
    3dmaskdump         -mask "${MASTER_MASK}"         "${input_file}" > "${output_file}"
done < "${SUBJECTS_FILE}"
