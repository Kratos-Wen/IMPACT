#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
TEST_SPLIT="${TEST_SPLIT:-test}"
SPLITS_CSV="${SPLITS_CSV:-1,2,3,4}"
FEATURE_TYPES_CSV="${FEATURE_TYPES_CSV:-i3d,videomaev2,mvitv2}"
LABEL_MODES_CSV="${LABEL_MODES_CSV:-coarse,verb,noun,verb_noun}"
TRAIN_VIEWS_CSV="${TRAIN_VIEWS_CSV:-ego,front,left,right,top}"
TEST_VIEWS_CSV="${TEST_VIEWS_CSV:-ego,front,left,right,top}"
POOLING="${POOLING:-mean}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-1e-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SEED="${SEED:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
ANNOTATION_ROOT="${ANNOTATION_ROOT:-${ROOT_DIR}/dataset/CV/annotations_CAS}"
SPLIT_DIR="${SPLIT_DIR:-${ROOT_DIR}/dataset/CV/splits_CAS}"
FEATURE_BASE_DIR="${FEATURE_BASE_DIR:-${ROOT_DIR}/features/cv}"
FEATURE_ROOT="${FEATURE_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/cv_sm/classification/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/cv_sm/classification/${RUN_TAG}}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

IFS=',' read -r -a SPLITS <<< "${SPLITS_CSV}"
IFS=',' read -r -a FEATURE_TYPES <<< "${FEATURE_TYPES_CSV}"
IFS=',' read -r -a LABEL_MODES <<< "${LABEL_MODES_CSV}"

if [[ ${#SPLITS[@]} -eq 0 ]]; then
  echo "No splits provided via SPLITS_CSV." >&2
  exit 1
fi

if [[ ${#FEATURE_TYPES[@]} -eq 0 ]]; then
  echo "No feature types provided via FEATURE_TYPES_CSV." >&2
  exit 1
fi

if [[ ${#LABEL_MODES[@]} -eq 0 ]]; then
  echo "No label modes provided via LABEL_MODES_CSV." >&2
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
  local split_index="$1"
  local feature_type="$2"
  local label_mode="$3"

  local name="${feature_type}_split${split_index}_${label_mode}"
  local json_path="${OUTPUT_ROOT}/${name}.json"
  local log_path="${LOG_ROOT}/${name}.log"
  local resolved_feature_root="${FEATURE_ROOT:-${FEATURE_BASE_DIR%/}/${feature_type}}"

  await_slot
  echo "[launch] ${name}"
  (
    cmd=(
      "${PYTHON_BIN}" "${ROOT_DIR}/tasks/CV-SM/classification/cv_smc_classification.py"
      --split-index "${split_index}"
      --train-split "${TRAIN_SPLIT}"
      --test-split "${TEST_SPLIT}"
      --train-views "${TRAIN_VIEWS_CSV}"
      --test-views "${TEST_VIEWS_CSV}"
      --feature-type "${feature_type}"
      --feature-root "${resolved_feature_root}"
      --pooling "${POOLING}"
      --label-mode "${label_mode}"
      --epochs "${EPOCHS}"
      --batch-size "${BATCH_SIZE}"
      --learning-rate "${LEARNING_RATE}"
      --weight-decay "${WEIGHT_DECAY}"
      --seed "${SEED}"
      --device "${DEVICE}"
      --annotation-root "${ANNOTATION_ROOT}"
      --split-dir "${SPLIT_DIR}"
      --output-json "${json_path}"
      --quiet
    )
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

    for label_mode in "${LABEL_MODES[@]}"; do
      label_mode="${label_mode// /}"
      if [[ -z "${label_mode}" ]]; then
        continue
      fi
      launch_job "${split_index}" "${feature_type}" "${label_mode}"
    done
  done
done

while (( $(jobs -pr | wc -l) > 0 )); do
  if ! wait -n; then
    failed=1
  fi
done

cat <<EOF
[done] CV-SMC evaluation jobs finished.
output_root: ${OUTPUT_ROOT}
log_root: ${LOG_ROOT}
train_split: ${TRAIN_SPLIT}
test_split: ${TEST_SPLIT}
splits: ${SPLITS_CSV}
feature_types: ${FEATURE_TYPES_CSV}
label_modes: ${LABEL_MODES_CSV}
train_views: ${TRAIN_VIEWS_CSV}
test_views: ${TEST_VIEWS_CSV}
pooling: ${POOLING}
device: ${DEVICE}
epochs: ${EPOCHS}
batch_size: ${BATCH_SIZE}
max_parallel: ${MAX_PARALLEL}
EOF

if (( failed != 0 )); then
  echo "One or more jobs failed. Check logs under ${LOG_ROOT}." >&2
  exit 1
fi
