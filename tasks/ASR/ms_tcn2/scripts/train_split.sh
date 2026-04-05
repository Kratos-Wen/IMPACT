#!/usr/bin/env bash
set -euo pipefail

SPLIT_ID="${1:-1}"
FEATURE_DIR="${2:?FEATURE_DIR is required}"
GPU="${3:-0}"
NUM_EPOCHS="${4:-100}"
BATCH_SIZE="${5:-6}"
LEARNING_RATE="${6:-0.0005}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ANNOTATION_DIR="${7:-$ROOT_DIR/dataset/ASR/annotations}"
SPLIT_DIR="${8:-$ROOT_DIR/dataset/ASR/splits_front_only_v1}"
LOG_BASE="${9:-$ROOT_DIR/logs/asr/ms_tcn2}"

SPLIT_ID="${SPLIT_ID#split}"
IMPACT_SPLIT="split${SPLIT_ID}"
LOG_PATH="${LOG_BASE%/}/train_${IMPACT_SPLIT}.log"

mkdir -p "$LOG_BASE"

cd "$ROOT_DIR/third_party/ms_tcn2"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  python main.py \
    --action train \
    --dataset IMPACT \
    --experiment ms_tcn2 \
    --split "$SPLIT_ID" \
    --num_epochs "$NUM_EPOCHS" \
    --bz "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --split_dir "$SPLIT_DIR" \
    --impact_split "$IMPACT_SPLIT" \
    --bundle_split train \
    --camera front \
    --annotation_dir "$ANNOTATION_DIR" \
    --feature_dir "$FEATURE_DIR" \
  > "$LOG_PATH" 2>&1

echo "[Done] MS-TCN++ ASR training finished. Log: $LOG_PATH"
