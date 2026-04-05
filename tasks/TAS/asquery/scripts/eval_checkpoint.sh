#!/usr/bin/env bash
set -euo pipefail

TASK_MODE="${1:?TASK_MODE is required}"
FEATURE_TYPE="${2:?FEATURE_TYPE is required}"
SPLIT="${3:?SPLIT is required}"
GPU="${4:-0}"
CKPT_PATH="${5:?CKPT_PATH is required}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
LOG_ROOT="${6:-$ROOT_DIR/logs/tas/asquery_eval}"

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
BASE_CFG="$ROOT_DIR/tasks/TAS/asquery/configs/${CFG_SLUG}.yaml"
TMP_CFG="$(mktemp "${TMPDIR:-/tmp}/asquery_eval_XXXX.yaml")"
LOG_DIR="${LOG_ROOT%/}/${CFG_SLUG}_split${SPLIT}"
LOG_PATH="$LOG_DIR/eval.log"

mkdir -p "$LOG_DIR"
cp "$BASE_CFG" "$TMP_CFG"
perl -0pi -e "s/val_split: \\['[^']+', '\\d+'\\]/val_split: ['test', '${SPLIT}']/g" "$TMP_CFG"

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  python "$ROOT_DIR/third_party/asquery/eval.py" \
    "$TMP_CFG" \
    "$CKPT_PATH" \
  > "$LOG_PATH" 2>&1

rm -f "$TMP_CFG"
echo "[Done] ASQuery evaluation finished. Log: $LOG_PATH"
