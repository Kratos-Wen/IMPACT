#!/usr/bin/env bash
set -euo pipefail

TASK_MODE="${1:?TASK_MODE is required}"
FEATURE_TYPE="${2:?FEATURE_TYPE is required}"
SPLIT="${3:?SPLIT is required}"
GPU="${4:-0}"
CKPT_PATH="${5:?CKPT_PATH is required}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
IMPACT_ROOT="${6:-$ROOT_DIR/dataset/TAS}"
SAVE_ROOT="${7:-$ROOT_DIR/outputs/tas/fact_eval}"

TASK_MODE="$(echo "$TASK_MODE" | tr '[:lower:]' '[:upper:]')"
FEATURE_TYPE="$(echo "$FEATURE_TYPE" | tr '[:upper:]' '[:lower:]')"

case "$TASK_MODE" in
  TAS-S)
    TASK_MODE="CAS"
    ;;
  TAS-BL)
    TASK_MODE="FAS_L"
    ;;
  TAS-BR)
    TASK_MODE="FAS_R"
    ;;
  CAS|FAS_L|FAS_R)
    ;;
  TAS-B)
    echo "Unsupported TASK_MODE: $TASK_MODE"
    echo "Use TAS-BL or TAS-BR for the bimanual protocols."
    exit 1
    ;;
  *)
    echo "Unsupported TASK_MODE: $TASK_MODE"
    echo "Expected one of: TAS-S, TAS-BL, TAS-BR"
    exit 1
    ;;
esac

CFG_SLUG="$(echo "${TASK_MODE}_${FEATURE_TYPE}" | tr '[:upper:]' '[:lower:]')"
CFG_PATH="$ROOT_DIR/tasks/TAS/fact/configs/${CFG_SLUG}.yaml"
SAVE_DIR="${SAVE_ROOT%/}/${CFG_SLUG}_split${SPLIT}"
LOG_PATH="$SAVE_DIR/eval.log"

mkdir -p "$SAVE_DIR"

cd "$ROOT_DIR/third_party/fact"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  python3 -m src.eval_checkpoint \
    --cfg "$CFG_PATH" \
    --ckpt "$CKPT_PATH" \
    --split "split${SPLIT}" \
    --gpu 0 \
    --impact-root "$IMPACT_ROOT" \
    --impact-task "$TASK_MODE" \
    --impact-feature-type "$FEATURE_TYPE" \
    --save-json "$SAVE_DIR/metrics.json" \
  > "$LOG_PATH" 2>&1

echo "[Done] FACT evaluation finished. Results: $SAVE_DIR"
