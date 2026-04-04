#!/usr/bin/env bash
set -euo pipefail

TASK_MODE="${1:-CAS}"
FEATURE_TYPE="${2:-videomaev2}"
GPU_LIST="${3:-0,1,2,3}"
RUN_TAG="${4:-$(date +%Y%m%d_%H%M%S)}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
IMPACT_ROOT="${5:-$ROOT_DIR/dataset/TAS}"
LOG_BASE="${6:-$ROOT_DIR/logs/tas/fact}"

TASK_MODE="$(echo "$TASK_MODE" | tr '[:lower:]' '[:upper:]')"
FEATURE_TYPE="$(echo "$FEATURE_TYPE" | tr '[:upper:]' '[:lower:]')"

case "$TASK_MODE" in
  CAS|FAS_L|FAS_R)
    ;;
  *)
    echo "Unsupported TASK_MODE: $TASK_MODE"
    echo "Expected one of: CAS, FAS_L, FAS_R"
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
CFG_PATH="$ROOT_DIR/tasks/TAS/fact/configs/${CFG_SLUG}.yaml"
RUN_LOG_BASE="${LOG_BASE%/}/${CFG_SLUG}_${RUN_TAG}"

mkdir -p "$RUN_LOG_BASE"

pids=()
for idx in 0 1 2 3; do
  split=$((idx + 1))
  gpu="${GPUS[$idx]}"
  run_name="${CFG_SLUG}_split${split}_${RUN_TAG}"
  log_path="$RUN_LOG_BASE/split${split}.log"

  (
    cd "$ROOT_DIR/third_party/fact"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
      python3 -m src.train \
        --cfg "$CFG_PATH" \
        --set aux.gpu 0 split "split${split}" aux.mark "${run_name}" impact_root "$IMPACT_ROOT" impact_task "$TASK_MODE" impact_feature_type "$FEATURE_TYPE" \
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
  echo "[Error] One or more FACT splits failed. Check $RUN_LOG_BASE"
  exit 1
fi

echo "[Done] FACT ${TASK_MODE}+${FEATURE_TYPE} splits finished."
