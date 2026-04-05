#!/usr/bin/env bash
set -euo pipefail

SPLIT_ID="${1:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
OUTPUT_PATH="${2:-$ROOT_DIR/outputs/psr/ms_tcn2/procedure_graph_split${SPLIT_ID#split}.json}"
ALIAS_MAP="${3:-$ROOT_DIR/tasks/PSR/ms_tcn2/configs/component_alias.json}"
ANNOTATION_DIR="${4:-$ROOT_DIR/dataset/ASR/annotations}"
SPLIT_DIR="${5:-$ROOT_DIR/dataset/ASR/splits_front_only_v1}"
LOG_BASE="${6:-$ROOT_DIR/logs/psr/ms_tcn2}"

SPLIT_ID="${SPLIT_ID#split}"
IMPACT_SPLIT="split${SPLIT_ID}"
LOG_PATH="${LOG_BASE%/}/learn_graph_${IMPACT_SPLIT}.log"

mkdir -p "$LOG_BASE" "$(dirname "$OUTPUT_PATH")"

cd "$ROOT_DIR/third_party/ms_tcn2"
PYTHONUNBUFFERED=1 \
  python learn_procedure_graph.py \
    --out "$OUTPUT_PATH" \
    --alias_map "$ALIAS_MAP" \
    --split_dir "$SPLIT_DIR" \
    --impact_split "$IMPACT_SPLIT" \
    --bundle_split train \
    --annotation_dir "$ANNOTATION_DIR" \
    --camera front \
  > "$LOG_PATH" 2>&1

echo "[Done] Procedure graph saved to $OUTPUT_PATH"
