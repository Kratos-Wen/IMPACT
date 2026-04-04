#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?RUN_NAME is required}"
CHECKPOINT_NAME="${2:-best_model}"
SUBSET="${3:-test}"
VIDEO_DIR="${4:?VIDEO_DIR is required}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
LABEL_DIR="${5:-$ROOT_DIR/dataset/PSR/labels_front_only_v1}"
LOG_ROOT="${6:-$ROOT_DIR/logs/psr/storm_psr/runs}"
PROCEDURE_INFO="${7:-$LABEL_DIR/procedure_info_IMPACT.json}"

LOG_PATH="${LOG_ROOT%/}/${RUN_NAME}.${SUBSET}.eval.log"

mkdir -p "$LOG_ROOT"

cd "$ROOT_DIR/third_party/STORM-PSR/evaluation"
PYTHONUNBUFFERED=1 \
  python evaluate_TemporalStream.py \
    --run_path "$LOG_ROOT/$RUN_NAME" \
    --rec_path "$LABEL_DIR" \
    --psr_label_path "$LABEL_DIR" \
    --video_dir "$VIDEO_DIR" \
    --procedure_info "$PROCEDURE_INFO" \
    --split "$SUBSET" \
    --checkpoint "$CHECKPOINT_NAME" \
  > "$LOG_PATH" 2>&1

echo "[Done] STORM-PSR evaluation finished. Log: $LOG_PATH"
