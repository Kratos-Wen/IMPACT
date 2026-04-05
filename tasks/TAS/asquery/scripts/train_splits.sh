#!/usr/bin/env bash
set -euo pipefail

TASK_MODE="${1:-TAS-S}"
FEATURE_TYPE="${2:-videomaev2}"
GPU_LIST="${3:-0,1,2,3}"
RUN_TAG="${4:-$(date +%Y%m%d_%H%M%S)}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
IMPACT_ROOT="${5:-$ROOT_DIR/dataset/TAS}"
LOG_BASE="${6:-$ROOT_DIR/logs/tas/asquery}"

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

case "$FEATURE_TYPE" in
  i3d|videomaev2)
    ;;
  *)
    echo "Unsupported FEATURE_TYPE: $FEATURE_TYPE"
    echo "Expected one of: i3d, videomaev2"
    exit 1
    ;;
esac

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [ "${#GPUS[@]}" -ne 4 ]; then
  echo "Usage: $0 [TASK_MODE] [FEATURE_TYPE] [gpu0,gpu1,gpu2,gpu3] [run_tag] [impact_root] [log_base]"
  exit 1
fi

CFG_SLUG="$(echo "${TASK_MODE}_${FEATURE_TYPE}" | tr '[:upper:]' '[:lower:]')"
CFG_PATH="$ROOT_DIR/tasks/TAS/asquery/configs/${CFG_SLUG}.yaml"
RUN_LOG_BASE="${LOG_BASE%/}/${CFG_SLUG}_${RUN_TAG}"

mkdir -p "$RUN_LOG_BASE"

pids=()
for idx in 0 1 2 3; do
  split=$((idx + 1))
  gpu="${GPUS[$idx]}"
  out_name="${CFG_SLUG}_split${split}_${RUN_TAG}"
  log_path="$RUN_LOG_BASE/split${split}.log"

  (
    cd "$ROOT_DIR"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
      python "$ROOT_DIR/third_party/asquery/main.py" \
        "$CFG_PATH" \
        --impact-root "$IMPACT_ROOT" \
        --impact-label-mode "$TASK_MODE" \
        --impact-feature-type "$FEATURE_TYPE" \
        --impact-split "$split" \
        --output "$out_name" \
      > "$log_path" 2>&1
  ) &
  pids+=("$!")
done

has_error=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    has_error=1
  fi
done

if [ "$has_error" -ne 0 ]; then
  echo "[Error] One or more ASQuery splits failed. Check $RUN_LOG_BASE"
  exit 1
fi

echo "[Done] ASQuery ${TASK_MODE}+${FEATURE_TYPE} splits finished."
