#!/usr/bin/env bash
set -euo pipefail

TASK_MODE="${1:-PPR_L}"
FEATURE_TYPE="${2:-videomaev2}"
GPU_LIST="${3:-0,1,2,3}"
RUN_TAG="${4:-$(date +%Y%m%d_%H%M%S)}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
IMPACT_ROOT="${5:-$ROOT_DIR/dataset/PPR}"
OUTPUT_BASE="${6:-$ROOT_DIR/outputs/ppr/ltcontext}"
LOG_BASE="${7:-$ROOT_DIR/logs/ppr/ltcontext}"

TASK_MODE="$(echo "$TASK_MODE" | tr '[:lower:]' '[:upper:]')"
FEATURE_TYPE="$(echo "$FEATURE_TYPE" | tr '[:upper:]' '[:lower:]')"

case "$TASK_MODE" in
  PPR_L|PPR_R)
    ;;
  *)
    echo "Unsupported TASK_MODE: $TASK_MODE"
    echo "Expected one of: PPR_L, PPR_R"
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
  echo "Usage: $0 [TASK_MODE] [FEATURE_TYPE] [gpu0,gpu1,gpu2,gpu3] [run_tag] [impact_root] [output_base] [log_base]"
  exit 1
fi

CFG_SLUG="$(echo "${TASK_MODE}_${FEATURE_TYPE}" | tr '[:upper:]' '[:lower:]')"
CFG_PATH="$ROOT_DIR/tasks/PPR/ltcontext/configs/${CFG_SLUG}.yaml"
RUN_ROOT="${CFG_SLUG}_${RUN_TAG}"
RUN_OUTPUT_BASE="${OUTPUT_BASE%/}/${RUN_ROOT}"
RUN_LOG_BASE="${LOG_BASE%/}/${RUN_ROOT}"

mkdir -p "$RUN_OUTPUT_BASE" "$RUN_LOG_BASE"

pids=()
for idx in 0 1 2 3; do
  split=$((idx + 1))
  gpu="${GPUS[$idx]}"
  log_path="$RUN_LOG_BASE/split${split}.log"
  out_dir="$RUN_OUTPUT_BASE/split${split}"

  (
    cd "$ROOT_DIR"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
      python "$ROOT_DIR/third_party/LTContext/run_net.py" \
        --cfg "$CFG_PATH" \
        --impact-root "$IMPACT_ROOT" \
        --impact-label-mode "$TASK_MODE" \
        --impact-feature-type "$FEATURE_TYPE" \
        --impact-split "$split" \
        OUTPUT_DIR "$out_dir" \
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
  echo "[Error] One or more LTContext splits failed. Check $RUN_LOG_BASE"
  exit 1
fi

echo "[Done] LTContext ${TASK_MODE}+${FEATURE_TYPE} splits finished."
