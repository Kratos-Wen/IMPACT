#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-test}"
SPLIT_INDEX="${SPLIT_INDEX:-1}"
POOLING="${POOLING:-mean}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
EXO_VIEWS_CSV="${EXO_VIEWS_CSV:-front,left,right,top}"
FEATURE_TYPES_CSV="${FEATURE_TYPES_CSV:-i3d,videomaev2,mvitv2}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
ANNOTATION_ROOT="${ANNOTATION_ROOT:-${ROOT_DIR}/dataset/CV/annotations_CAS}"
SPLIT_DIR="${SPLIT_DIR:-${ROOT_DIR}/dataset/CV/splits_CAS}"
FEATURE_BASE_DIR="${FEATURE_BASE_DIR:-${ROOT_DIR}/features/cv}"
FEATURE_ROOT="${FEATURE_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/cv_ta/cosine_knn/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/cv_ta/cosine_knn/${RUN_TAG}}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

IFS=',' read -r -a FEATURE_TYPES <<< "${FEATURE_TYPES_CSV}"

if [[ ${#FEATURE_TYPES[@]} -eq 0 ]]; then
  echo "No feature types provided via FEATURE_TYPES_CSV." >&2
  exit 1
fi

if ! [[ "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer, got: ${MAX_PARALLEL}" >&2
  exit 1
fi

failed=0

await_slot() {
  while (( $(jobs -pr | wc -l) >= MAX_PARALLEL )); do
    if ! wait -n; then
      failed=1
    fi
  done
}

launch_job() {
  local name="$1"
  local protocol="$2"
  local source_views="$3"
  local target_views="$4"
  local feature_type="$5"

  local json_path="${OUTPUT_ROOT}/${name}.json"
  local log_path="${LOG_ROOT}/${name}.log"
  local resolved_feature_root="${FEATURE_ROOT:-${FEATURE_BASE_DIR%/}/${feature_type}}"

  await_slot
  echo "[launch] ${name}"
  (
    cmd=(
      "${PYTHON_BIN}" "${ROOT_DIR}/tasks/CV-TA/cosine_knn/cv_ta_retrieval.py"
      --protocol "${protocol}"
      --split "${SPLIT}"
      --split-index "${SPLIT_INDEX}"
      --source-views "${source_views}"
      --target-views "${target_views}"
      --feature-type "${feature_type}"
      --feature-root "${resolved_feature_root}"
      --pooling "${POOLING}"
      --annotation-root "${ANNOTATION_ROOT}"
      --split-dir "${SPLIT_DIR}"
      --output-json "${json_path}"
      --quiet
    )
    "${cmd[@]}"
  ) >"${log_path}" 2>&1 &
}

for feature_type in "${FEATURE_TYPES[@]}"; do
  feature_type="${feature_type// /}"
  if [[ -z "${feature_type}" ]]; then
    continue
  fi

  # One exo-exo run covers all directed exo pairs because cv_ta_retrieval.py
  # expands source_views x target_views internally and skips same-view pairs.
  launch_job "local_exo_exo_${feature_type}" \
    "local" \
    "${EXO_VIEWS_CSV}" \
    "${EXO_VIEWS_CSV}" \
    "${feature_type}"

  launch_job "global_exo_exo_${feature_type}" \
    "global" \
    "${EXO_VIEWS_CSV}" \
    "${EXO_VIEWS_CSV}" \
    "${feature_type}"

  # exo2ego covers all exo -> ego diagnostic pairs in one run.
  launch_job "exo2ego_${feature_type}" \
    "exo2ego" \
    "${EXO_VIEWS_CSV}" \
    "ego" \
    "${feature_type}"
done

while (( $(jobs -pr | wc -l) > 0 )); do
  if ! wait -n; then
    failed=1
  fi
done

cat <<EOF
[done] CV-TA evaluation jobs finished.
output_root: ${OUTPUT_ROOT}
log_root: ${LOG_ROOT}
feature_types: ${FEATURE_TYPES_CSV}
split: ${SPLIT}
split_index: ${SPLIT_INDEX}
exo_views: ${EXO_VIEWS_CSV}
EOF

if (( failed != 0 )); then
  echo "One or more jobs failed. Check logs under ${LOG_ROOT}." >&2
  exit 1
fi
