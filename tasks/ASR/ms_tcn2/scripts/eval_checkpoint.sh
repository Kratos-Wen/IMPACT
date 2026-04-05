#!/usr/bin/env bash
set -euo pipefail

SPLIT_ID="${1:-1}"
FEATURE_DIR="${2:?FEATURE_DIR is required}"
CHECKPOINT_NAME="${3:?CHECKPOINT_NAME is required}"
GPU="${4:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ANNOTATION_DIR="${5:-$ROOT_DIR/dataset/ASR/annotations}"
SPLIT_DIR="${6:-$ROOT_DIR/dataset/ASR/splits_front_only_v1}"
LOG_BASE="${7:-$ROOT_DIR/logs/asr/ms_tcn2}"

SPLIT_ID="${SPLIT_ID#split}"
IMPACT_SPLIT="split${SPLIT_ID}"
LOG_PATH="${LOG_BASE%/}/eval_${IMPACT_SPLIT}_$(basename "$CHECKPOINT_NAME").log"

mkdir -p "$LOG_BASE"

cd "$ROOT_DIR/third_party/ms_tcn2"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  python eval.py \
    --dataset IMPACT \
    --experiment ms_tcn2 \
    --split "$SPLIT_ID" \
    --checkpoint "$CHECKPOINT_NAME" \
    --split_dir "$SPLIT_DIR" \
    --impact_split "$IMPACT_SPLIT" \
    --bundle_split test \
    --camera front \
    --annotation_dir "$ANNOTATION_DIR" \
    --feature_dir "$FEATURE_DIR" \
  > "$LOG_PATH" 2>&1

echo "[Done] MS-TCN++ ASR evaluation finished. Log: $LOG_PATH"
