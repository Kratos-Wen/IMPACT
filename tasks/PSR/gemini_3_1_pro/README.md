# Gemini 3.1 Pro for PSR

This directory provides the released IMPACT `PSR` wrapper for `Gemini 3.1 Pro`.

## Path Convention

- annotation assets: `dataset/ASR/annotations`
- PSR labels: `dataset/PSR/labels_front_only_v1`
- method configs: `tasks/PSR/gemini_3_1_pro/configs`
- default outputs: `outputs/psr/gemini_3_1_pro/<run_tag>/predictions`
- default logs: `logs/psr/gemini_3_1_pro/<run_tag>`
- raw videos: external

## Common Arguments

- `VIDEO_DIR`: required directory containing the released evaluation videos
- `ASR_JSON_DIR`: defaults to `dataset/ASR/annotations` and provides the released component metadata and key query times
- `RUN_TAG`: run identifier used to separate outputs and logs
- `MODEL_NAME`: Gemini model name, default `gemini-3.1-pro-preview`
- `BASE_FPS`: sampling fps outside the dense key-transition windows, default `0.5`
- `PRED_DIR`: directory containing saved Gemini prediction JSON files
- `BUNDLE_SPLIT`: evaluation split, usually `test`

## Scripts

- `scripts/learn_graph.sh`: regenerates the released procedure graph from ASR annotations
- `scripts/run_batch_inference.sh`: runs the released batch prompting pipeline and writes prediction JSON files
- `scripts/eval_predictions.sh`: evaluates saved prediction JSON files as PSR

## Examples

```bash
bash tasks/PSR/gemini_3_1_pro/scripts/learn_graph.sh
```

```bash
GEMINI_API_KEY=... \
bash tasks/PSR/gemini_3_1_pro/scripts/run_batch_inference.sh /path/to/videos
```

```bash
bash tasks/PSR/gemini_3_1_pro/scripts/eval_predictions.sh \
  outputs/psr/gemini_3_1_pro/<run_tag>/predictions test
```

## Notes

- The released prompting pipeline jointly predicts `states_over_time` and completed step lists; the PSR evaluator uses the completed-step output.
- The prompt is release-appropriate for `PSR`: it constrains the step vocabulary, conditions on the released procedure graph, and asks for cumulative completed-step lists at the official query times.
- Raw videos are not distributed in this repository, so inference requires an external `VIDEO_DIR`.
- Inference requires the `GEMINI_API_KEY` environment variable.
- Inference also requires Python packages `google-genai`, `moviepy`, and `Pillow`.
