#!/usr/bin/env bash
set -euo pipefail

SPLIT_ID="${1:-1}"
FEATURE_DIR="${2:?FEATURE_DIR is required}"
GPU="${3:-0}"
RUN_NAME="${4:-storm_psr_split${SPLIT_ID#split}_$(date +%Y%m%d_%H%M%S)}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
LABEL_DIR="${5:-$ROOT_DIR/dataset/PSR/labels_front_only_v1}"
SPLIT_DIR="${6:-$ROOT_DIR/dataset/ASR/splits_front_only_v1}"
LOG_ROOT="${7:-$ROOT_DIR/logs/psr/storm_psr/runs}"
CKPT_ROOT="${8:-$ROOT_DIR/outputs/psr/storm_psr/checkpoints}"
CFG_PATH="${9:-$ROOT_DIR/tasks/PSR/storm_psr/configs/temporal_f65_dim1408.yaml}"

SPLIT_ID="${SPLIT_ID#split}"
IMPACT_SPLIT="split${SPLIT_ID}"
LOG_PATH="${LOG_ROOT%/}/${RUN_NAME}.train.log"

mkdir -p "$LOG_ROOT" "$CKPT_ROOT"

cd "$ROOT_DIR/third_party/STORM-PSR/temporal_stream/train_spatial_temporal"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  python train.py \
    --data_dir "$FEATURE_DIR" \
    --psr_label_path "$LABEL_DIR" \
    --split_dir "$SPLIT_DIR" \
    --impact_split "$IMPACT_SPLIT" \
    --log_path "$LOG_ROOT" \
    --ckpt_dir "$CKPT_ROOT" \
    --run_name "$RUN_NAME" \
    --config "$CFG_PATH" \
    --dtype embedding \
    --job_file_mode \
  > "$LOG_PATH" 2>&1

echo "[Done] STORM-PSR training finished. Log: $LOG_PATH"
