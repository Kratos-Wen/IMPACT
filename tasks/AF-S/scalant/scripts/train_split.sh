#!/usr/bin/env bash
set -euo pipefail

FEATURE_TYPE="${1:?FEATURE_TYPE is required}"
VIEW="${2:?VIEW is required}"
FEATURE_DIR="${3:?FEATURE_DIR is required}"
GPU_LIST="${4:-0}"
SPLIT_NAME="${5:-split1}"
RUN_TAG="${6:-scalant_${FEATURE_TYPE}_${VIEW}_${SPLIT_NAME}}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ANNOTATION_DIR="${7:-$ROOT_DIR/dataset/AF-S/Annotation}"
OUTPUT_ROOT="${8:-$ROOT_DIR/outputs/af_s/scalant}"
LOG_ROOT="${9:-$ROOT_DIR/logs/af_s/scalant}"
SOURCE_DIR="$ROOT_DIR/third_party/scalant"

RUN_OUTPUT_DIR="${OUTPUT_ROOT%/}/$RUN_TAG"
LOG_PATH="${LOG_ROOT%/}/train_${RUN_TAG}.log"
NUM_DEVICES="$(awk -F',' '{print NF}' <<<"$GPU_LIST")"

mkdir -p "$RUN_OUTPUT_DIR" "$LOG_ROOT"

cd "$SOURCE_DIR"
CUDA_VISIBLE_DEVICES="$GPU_LIST" PYTHONUNBUFFERED=1 \
  python train_lightning.py \
    --model-name sca \
    --annotation-dir "$ANNOTATION_DIR" \
    --feature-dir "$FEATURE_DIR" \
    --feature-type "$FEATURE_TYPE" \
    --view "$VIEW" \
    --split-name "$SPLIT_NAME" \
    --devices "$NUM_DEVICES" \
    --output-dir "$RUN_OUTPUT_DIR" \
  > "$LOG_PATH" 2>&1

echo "[Done] ScalAnt AF-S training finished. Log: $LOG_PATH"
