#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VIDEO_DIR="${1:?Usage: $0 <VIDEO_DIR> [ASR_JSON_DIR] [RUN_TAG] [MODEL_NAME] [OUTPUT_ROOT] [LOG_ROOT]}"
ASR_JSON_DIR="${2:-$ROOT_DIR/dataset/ASR/annotations}"
RUN_TAG="${3:-$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME="${4:-gemini-3.1-pro-preview}"
OUTPUT_ROOT="${5:-$ROOT_DIR/outputs/psr/gemini_3_1_pro/${RUN_TAG}/predictions}"
LOG_ROOT="${6:-$ROOT_DIR/logs/psr/gemini_3_1_pro/${RUN_TAG}}"
VIDEO_GLOB="${VIDEO_GLOB:-*.mp4}"
MAX_FRAMES="${MAX_FRAMES:-0}"
TEMPERATURE="${TEMPERATURE:-0.0}"
BASE_FPS="${BASE_FPS:-0.5}"

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "GEMINI_API_KEY is required." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

"${PYTHON_BIN}" -c 'import importlib; missing=[]; modules={"PIL":"Pillow","moviepy":"moviepy","google.genai":"google-genai"}; 
for module, package in modules.items():
    try:
        importlib.import_module(module)
    except Exception:
        missing.append(package)
raise SystemExit("Missing Python packages: " + ", ".join(missing) if missing else 0)'

"${PYTHON_BIN}" "${ROOT_DIR}/tasks/PSR/gemini_3_1_pro/predict_states_and_steps.py" \
  --video_dir "${VIDEO_DIR}" \
  --asr_dir "${ASR_JSON_DIR}" \
  --procedure_graph "${ROOT_DIR}/tasks/PSR/gemini_3_1_pro/configs/procedure_graph.json" \
  --component_alias "${ROOT_DIR}/tasks/PSR/gemini_3_1_pro/configs/component_alias.json" \
  --out_dir "${OUTPUT_ROOT}" \
  --model "${MODEL_NAME}" \
  --video_glob "${VIDEO_GLOB}" \
  --max_frames "${MAX_FRAMES}" \
  --base_fps "${BASE_FPS}" \
  --temperature "${TEMPERATURE}" \
  > "${LOG_ROOT}/inference.log" 2>&1

echo "predictions: ${OUTPUT_ROOT}"
echo "log: ${LOG_ROOT}/inference.log"
