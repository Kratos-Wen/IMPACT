#!/usr/bin/env bash
set -euo pipefail

SPLIT_ID="${1:-1}"
RUN_NAME="${2:?RUN_NAME is required}"
CHECKPOINT_NAME="${3:-best_model}"
SUBSET="${4:-test}"
FEATURE_DIR="${5:?FEATURE_DIR is required}"
GPU="${6:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
LABEL_DIR="${7:-$ROOT_DIR/dataset/PSR/labels_front_only_v1}"
SPLIT_DIR="${8:-$ROOT_DIR/dataset/ASR/splits_front_only_v1}"
LOG_ROOT="${9:-$ROOT_DIR/logs/psr/storm_psr/runs}"

SPLIT_ID="${SPLIT_ID#split}"
IMPACT_SPLIT="split${SPLIT_ID}"
LOG_PATH="${LOG_ROOT%/}/${RUN_NAME}.${SUBSET}.test.log"

mkdir -p "$LOG_ROOT"

cd "$ROOT_DIR/third_party/storm_psr/temporal_stream/train_spatial_temporal"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  python test.py \
    --run_name "$RUN_NAME" \
    --checkpoint "$CHECKPOINT_NAME" \
    --split "$SUBSET" \
    --data_dir "$FEATURE_DIR" \
    --psr_label_path "$LABEL_DIR" \
    --split_dir "$SPLIT_DIR" \
    --impact_split "$IMPACT_SPLIT" \
    --log_path "$LOG_ROOT" \
    --dtype embedding \
  > "$LOG_PATH" 2>&1

echo "[Done] STORM-PSR inference finished. Log: $LOG_PATH"
