#!/usr/bin/env bash
set -euo pipefail

SPLIT_ID="${1:-1}"
FEATURE_DIR="${2:?FEATURE_DIR is required}"
CHECKPOINT_NAME="${3:?CHECKPOINT_NAME is required}"
GPU="${4:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
GRAPH_PATH="${5:-$ROOT_DIR/outputs/psr/ms_tcn2/procedure_graph_split${SPLIT_ID#split}.json}"
ANNOTATION_DIR="${6:-$ROOT_DIR/dataset/ASR/annotations}"
SPLIT_DIR="${7:-$ROOT_DIR/dataset/ASR/splits_front_only_v1}"
ALIAS_MAP="${8:-$ROOT_DIR/tasks/PSR/ms_tcn2/configs/component_alias.json}"
LOG_BASE="${9:-$ROOT_DIR/logs/psr/ms_tcn2}"

SPLIT_ID="${SPLIT_ID#split}"
IMPACT_SPLIT="split${SPLIT_ID}"
LOG_PATH="${LOG_BASE%/}/eval_${IMPACT_SPLIT}_$(basename "$CHECKPOINT_NAME").log"

mkdir -p "$LOG_BASE"

cd "$ROOT_DIR/third_party/ASR-PSR-Experiment"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  python eval.py \
    --dataset IMPACT \
    --experiment ms_tcn2 \
    --split "$SPLIT_ID" \
    --checkpoint "$CHECKPOINT_NAME" \
    --procedure_graph "$GRAPH_PATH" \
    --component_alias "$ALIAS_MAP" \
    --split_dir "$SPLIT_DIR" \
    --impact_split "$IMPACT_SPLIT" \
    --bundle_split test \
    --camera front \
    --annotation_dir "$ANNOTATION_DIR" \
    --feature_dir "$FEATURE_DIR" \
  > "$LOG_PATH" 2>&1

echo "[Done] MS-TCN++ PSR evaluation finished. Log: $LOG_PATH"
