#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PRED_DIR="${1:?Usage: $0 <PRED_DIR> [BUNDLE_SPLIT] [GT_DIR] [PROCEDURE_GRAPH] [COMPONENT_ALIAS]}"
BUNDLE_SPLIT="${2:-test}"
GT_DIR="${3:-$ROOT_DIR/dataset/PSR/labels_front_only_v1}"
PROCEDURE_GRAPH="${4:-$ROOT_DIR/tasks/PSR/gemini_3_1_pro/configs/procedure_graph.json}"
COMPONENT_ALIAS="${5:-$ROOT_DIR/tasks/PSR/gemini_3_1_pro/configs/component_alias.json}"

"${PYTHON_BIN}" "${ROOT_DIR}/tasks/PSR/gemini_3_1_pro/evaluate_psr.py" \
  --pred_dir "${PRED_DIR}" \
  --bundle_split "${BUNDLE_SPLIT}" \
  --gt_dir "${GT_DIR}" \
  --procedure_graph "${PROCEDURE_GRAPH}" \
  --component_alias "${COMPONENT_ALIAS}"
