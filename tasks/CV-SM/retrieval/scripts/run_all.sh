#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-test}"
SPLITS_CSV="${SPLITS_CSV:-1,2,3,4}"
FEATURE_TYPES_CSV="${FEATURE_TYPES_CSV:-i3d,videomaev2,mvitv2}"
QUERY_VIEWS_CSV="${QUERY_VIEWS_CSV:-ego,front,left,right,top}"
GALLERY_VIEWS_CSV="${GALLERY_VIEWS_CSV:-ego,front,left,right,top}"
POOLING="${POOLING:-mean}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
ANNOTATION_ROOT="${ANNOTATION_ROOT:-${ROOT_DIR}/dataset/CV/annotations_CAS}"
SPLIT_DIR="${SPLIT_DIR:-${ROOT_DIR}/dataset/CV/splits_CAS}"
FEATURE_BASE_DIR="${FEATURE_BASE_DIR:-${ROOT_DIR}/features/cv}"
FEATURE_ROOT="${FEATURE_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/cv_sm/retrieval/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/cv_sm/retrieval/${RUN_TAG}}"
QUIET="${QUIET:-0}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

IFS=',' read -r -a SPLITS <<< "${SPLITS_CSV}"
IFS=',' read -r -a FEATURE_TYPES <<< "${FEATURE_TYPES_CSV}"

if [[ ${#SPLITS[@]} -eq 0 ]]; then
  echo "No splits provided via SPLITS_CSV." >&2
  exit 1
fi

if [[ ${#FEATURE_TYPES[@]} -eq 0 ]]; then
  echo "No feature types provided via FEATURE_TYPES_CSV." >&2
  exit 1
fi

if ! [[ "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer, got: ${MAX_PARALLEL}" >&2
  exit 1
fi

if ! [[ "${QUIET}" =~ ^(0|1)$ ]]; then
  echo "QUIET must be 0 or 1, got: ${QUIET}" >&2
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
  local split_index="$1"
  local feature_type="$2"

  local name="${feature_type}_split${split_index}"
  local json_path="${OUTPUT_ROOT}/${name}.json"
  local log_path="${LOG_ROOT}/${name}.log"
  local resolved_feature_root="${FEATURE_ROOT:-${FEATURE_BASE_DIR%/}/${feature_type}}"

  await_slot
  echo "[launch] ${name}"
  (
    cmd=(
      "${PYTHON_BIN}" "${ROOT_DIR}/tasks/CV-SM/retrieval/cv_smr_retrieval.py"
      --split "${SPLIT}"
      --split-index "${split_index}"
      --feature-type "${feature_type}"
      --feature-root "${resolved_feature_root}"
      --pooling "${POOLING}"
      --query-views "${QUERY_VIEWS_CSV}"
      --gallery-views "${GALLERY_VIEWS_CSV}"
      --annotation-root "${ANNOTATION_ROOT}"
      --split-dir "${SPLIT_DIR}"
      --output-json "${json_path}"
    )

    if [[ "${QUIET}" == "1" ]]; then
      cmd+=(--quiet)
    fi

    "${cmd[@]}"
  ) >"${log_path}" 2>&1 &
}

for split_index in "${SPLITS[@]}"; do
  split_index="${split_index// /}"
  if [[ -z "${split_index}" ]]; then
    continue
  fi
  if ! [[ "${split_index}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid split index: ${split_index}" >&2
    exit 1
  fi

  for feature_type in "${FEATURE_TYPES[@]}"; do
    feature_type="${feature_type// /}"
    if [[ -z "${feature_type}" ]]; then
      continue
    fi
    launch_job "${split_index}" "${feature_type}"
  done
done

while (( $(jobs -pr | wc -l) > 0 )); do
  if ! wait -n; then
    failed=1
  fi
done

cat <<INFO
[done] CV-SMR evaluation jobs finished.
output_root: ${OUTPUT_ROOT}
log_root: ${LOG_ROOT}
split: ${SPLIT}
splits: ${SPLITS_CSV}
feature_types: ${FEATURE_TYPES_CSV}
query_views: ${QUERY_VIEWS_CSV}
gallery_views: ${GALLERY_VIEWS_CSV}
pooling: ${POOLING}
max_parallel: ${MAX_PARALLEL}
quiet: ${QUIET}
INFO

if (( failed != 0 )); then
  echo "One or more jobs failed. Check logs under ${LOG_ROOT}." >&2
  exit 1
fi
