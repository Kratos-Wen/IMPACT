#!/usr/bin/env bash
set -euo pipefail

VIDEO_ROOT="${1:?VIDEO_ROOT is required}"
VIEW="${2:?VIEW is required}"
SPLIT="${3:-test}"
GPU_LIST="${4:-0}"
RUN_TAG="${5:-qwen3vl_8b_${VIEW}_${SPLIT}}"
SPLIT_NAME="${6:-split1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ANNOTATION_DIR="${7:-$ROOT_DIR/dataset/AF-S/Annotation}"
OUTPUT_ROOT="${8:-$ROOT_DIR/outputs/af_s/qwen3_vl_8b}"
LOG_ROOT="${9:-$ROOT_DIR/logs/af_s/qwen3_vl_8b}"
SOURCE_DIR="$ROOT_DIR/third_party/qwen3_vl_8b"

RUN_OUTPUT_DIR="${OUTPUT_ROOT%/}/$RUN_TAG"
LOG_PATH="${LOG_ROOT%/}/eval_${RUN_TAG}.log"
NUM_PROCS="$(awk -F',' '{print NF}' <<<"$GPU_LIST")"

mkdir -p "$RUN_OUTPUT_DIR" "$LOG_ROOT"

cd "$SOURCE_DIR"
CUDA_VISIBLE_DEVICES="$GPU_LIST" PYTHONUNBUFFERED=1 \
  torchrun --standalone --nproc_per_node "$NUM_PROCS" test_qwen_anticipation.py \
    --annotation-dir "$ANNOTATION_DIR" \
    --video-root "$VIDEO_ROOT" \
    --view "$VIEW" \
    --split "$SPLIT" \
    --split-name "$SPLIT_NAME" \
    --output-dir "$RUN_OUTPUT_DIR" \
  > "$LOG_PATH" 2>&1

echo "[Done] Qwen3VL-8B AF-S evaluation finished. Log: $LOG_PATH"
