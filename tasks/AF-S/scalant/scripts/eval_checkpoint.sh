#!/usr/bin/env bash
set -euo pipefail

FEATURE_TYPE="${1:?FEATURE_TYPE is required}"
VIEW="${2:?VIEW is required}"
FEATURE_DIR="${3:?FEATURE_DIR is required}"
CKPT_PATH="${4:?CKPT_PATH is required}"
GPU="${5:-0}"
SPLIT_NAME="${6:-split1}"
CKPT_STEM="$(basename "$CKPT_PATH")"
CKPT_STEM="${CKPT_STEM%.*}"
RUN_TAG="${7:-scalant_${FEATURE_TYPE}_${VIEW}_${SPLIT_NAME}_${CKPT_STEM}}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ANNOTATION_DIR="${8:-$ROOT_DIR/dataset/AF-S/Annotation}"
OUTPUT_ROOT="${9:-$ROOT_DIR/outputs/af_s/scalant_eval}"
LOG_ROOT="${10:-$ROOT_DIR/logs/af_s/scalant}"
SOURCE_DIR="$ROOT_DIR/third_party/scalant"

RUN_OUTPUT_DIR="${OUTPUT_ROOT%/}/$RUN_TAG"
LOG_PATH="${LOG_ROOT%/}/eval_${RUN_TAG}.log"

mkdir -p "$RUN_OUTPUT_DIR" "$LOG_ROOT"

cd "$SOURCE_DIR"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  python test_lightning.py \
    --model-name sca \
    --model-path "$CKPT_PATH" \
    --annotation-dir "$ANNOTATION_DIR" \
    --feature-dir "$FEATURE_DIR" \
    --feature-type "$FEATURE_TYPE" \
    --view "$VIEW" \
    --split-name "$SPLIT_NAME" \
    --devices 1 \
    --output-dir "$RUN_OUTPUT_DIR" \
  > "$LOG_PATH" 2>&1

echo "[Done] ScalAnt AF-S evaluation finished. Log: $LOG_PATH"
