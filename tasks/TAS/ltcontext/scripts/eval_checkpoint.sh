#!/usr/bin/env bash
set -euo pipefail

TASK_MODE="${1:?TASK_MODE is required}"
FEATURE_TYPE="${2:?FEATURE_TYPE is required}"
SPLIT="${3:?SPLIT is required}"
GPU="${4:-0}"
CKPT_PATH="${5:?CKPT_PATH is required}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
IMPACT_ROOT="${6:-$ROOT_DIR/dataset/TAS}"
SAVE_ROOT="${7:-$ROOT_DIR/outputs/tas/ltcontext_eval}"

TASK_MODE="$(echo "$TASK_MODE" | tr '[:lower:]' '[:upper:]')"
FEATURE_TYPE="$(echo "$FEATURE_TYPE" | tr '[:upper:]' '[:lower:]')"
CFG_SLUG="$(echo "${TASK_MODE}_${FEATURE_TYPE}" | tr '[:upper:]' '[:lower:]')"
CFG_PATH="$ROOT_DIR/tasks/TAS/ltcontext/configs/${CFG_SLUG}.yaml"
SAVE_DIR="${SAVE_ROOT%/}/${CFG_SLUG}_split${SPLIT}"
LOG_PATH="$SAVE_DIR/eval.log"

mkdir -p "$SAVE_DIR"

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  python "$ROOT_DIR/third_party/ltcontext/run_net.py" \
    --cfg "$CFG_PATH" \
    --impact-root "$IMPACT_ROOT" \
    --impact-label-mode "$TASK_MODE" \
    --impact-feature-type "$FEATURE_TYPE" \
    --impact-split "$SPLIT" \
    TRAIN.ENABLE False \
    TEST.ENABLE True \
    TRAIN.EVAL_SPLIT test \
    TEST.CHECKPOINT_PATH "$CKPT_PATH" \
    TEST.SAVE_PREDICTIONS False \
    TEST.SAVE_RESULT_PATH "$SAVE_DIR" \
  > "$LOG_PATH" 2>&1

echo "[Done] LTContext evaluation finished. Results: $SAVE_DIR"
