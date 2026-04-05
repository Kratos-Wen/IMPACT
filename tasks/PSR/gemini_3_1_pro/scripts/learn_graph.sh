#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ANNOTATION_DIR="${1:-$ROOT_DIR/dataset/ASR/annotations}"
OUTPUT_PATH="${2:-$ROOT_DIR/tasks/PSR/gemini_3_1_pro/configs/procedure_graph.json}"
ALIAS_MAP="${3:-$ROOT_DIR/tasks/PSR/gemini_3_1_pro/configs/component_alias.json}"

"${PYTHON_BIN}" "${ROOT_DIR}/tasks/PSR/gemini_3_1_pro/learn_procedure_graph.py" \
  --asr_dir "${ANNOTATION_DIR}" \
  --out "${OUTPUT_PATH}" \
  --alias_map "${ALIAS_MAP}"

echo "procedure_graph: ${OUTPUT_PATH}"
