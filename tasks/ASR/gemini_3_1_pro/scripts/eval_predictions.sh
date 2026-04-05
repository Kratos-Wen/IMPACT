#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PRED_DIR="${1:?Usage: $0 <PRED_DIR> [BUNDLE_SPLIT] [SPLIT_ID] [CAMERA] [ANNOTATION_DIR] [SPLIT_DIR]}"
BUNDLE_SPLIT="${2:-test}"
SPLIT_ID="${3:-split1}"
CAMERA="${4:-front}"
ANNOTATION_DIR="${5:-$ROOT_DIR/dataset/ASR/annotations}"
SPLIT_DIR="${6:-$ROOT_DIR/dataset/ASR/splits_front_only_v1}"

"${PYTHON_BIN}" "${ROOT_DIR}/tasks/ASR/gemini_3_1_pro/evaluate_asr.py" \
  --pred_dir "${PRED_DIR}" \
  --bundle_split "${BUNDLE_SPLIT}" \
  --impact_split "${SPLIT_ID}" \
  --camera "${CAMERA}" \
  --annotation_dir "${ANNOTATION_DIR}" \
  --split_dir "${SPLIT_DIR}"
