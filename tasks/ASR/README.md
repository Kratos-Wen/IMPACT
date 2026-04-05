# Assembly State Recognition

This directory contains the IMPACT Assembly State Recognition benchmark release.

## Path Convention

- Run all commands from the repository root.
- Repository-managed annotation root: `dataset/ASR/annotations/`
- Repository-managed split root: `dataset/ASR/splits_front_only_v1/`
- Repository-managed runtime root: `logs/asr/`
- User-managed inputs: `FEATURE_DIR`, `VIDEO_DIR`, and Gemini prediction directories

## Common Arguments

- `SPLIT_ID`: split index such as `1`
- `FEATURE_DIR`: path to extracted front-view features
- `CHECKPOINT_NAME`: checkpoint file name under the upstream model directory, for example `epoch-100.model`
- `VIDEO_DIR`: raw-video directory used by the `Gemini 3.1 Pro` wrapper
- `PRED_DIR`: saved `Gemini 3.1 Pro` prediction directory

Current public protocol:
- front-view bundle split `split1`

## Scripts

| Method | Script | Purpose | Usage |
| --- | --- | --- | --- |
| `MS-TCN++` | `tasks/ASR/ms_tcn2/scripts/train_split.sh` | train one released split | `bash tasks/ASR/ms_tcn2/scripts/train_split.sh <SPLIT_ID> <FEATURE_DIR> [GPU] [NUM_EPOCHS] [BATCH_SIZE] [LR] [ANNOTATION_DIR] [SPLIT_DIR] [LOG_ROOT]` |
| `MS-TCN++` | `tasks/ASR/ms_tcn2/scripts/eval_checkpoint.sh` | evaluate one checkpoint | `bash tasks/ASR/ms_tcn2/scripts/eval_checkpoint.sh <SPLIT_ID> <FEATURE_DIR> <CHECKPOINT_NAME> [GPU] [ANNOTATION_DIR] [SPLIT_DIR] [LOG_ROOT]` |
| `VideoMAE v2+Head` | `tasks/ASR/videomae_v2_head/scripts/train_split.sh` | train one released split | `bash tasks/ASR/videomae_v2_head/scripts/train_split.sh <SPLIT_ID> <FEATURE_DIR> [GPU] [NUM_EPOCHS] [BATCH_SIZE] [LR] [ANNOTATION_DIR] [SPLIT_DIR] [LOG_ROOT]` |
| `VideoMAE v2+Head` | `tasks/ASR/videomae_v2_head/scripts/eval_checkpoint.sh` | evaluate one checkpoint | `bash tasks/ASR/videomae_v2_head/scripts/eval_checkpoint.sh <SPLIT_ID> <FEATURE_DIR> <CHECKPOINT_NAME> [GPU] [ANNOTATION_DIR] [SPLIT_DIR] [LOG_ROOT]` |
| `Gemini 3.1 Pro` | `tasks/ASR/gemini_3_1_pro/scripts/run_batch_inference.sh` | run the released batch prompting pipeline | `GEMINI_API_KEY=... bash tasks/ASR/gemini_3_1_pro/scripts/run_batch_inference.sh <VIDEO_DIR> [ASR_JSON_DIR] [RUN_TAG] [MODEL_NAME] [OUTPUT_ROOT] [LOG_ROOT]` |
| `Gemini 3.1 Pro` | `tasks/ASR/gemini_3_1_pro/scripts/eval_predictions.sh` | evaluate saved prediction JSON files | `bash tasks/ASR/gemini_3_1_pro/scripts/eval_predictions.sh <PRED_DIR> [BUNDLE_SPLIT] [SPLIT_ID] [CAMERA] [ANNOTATION_DIR] [SPLIT_DIR]` |

## Examples

```bash
bash tasks/ASR/ms_tcn2/scripts/train_split.sh 1 /path/to/IMPACT_i3d_front/features 0
bash tasks/ASR/ms_tcn2/scripts/eval_checkpoint.sh 1 /path/to/IMPACT_i3d_front/features epoch-100.model 0

bash tasks/ASR/videomae_v2_head/scripts/train_split.sh 1 /path/to/IMPACT_front/features 0
bash tasks/ASR/videomae_v2_head/scripts/eval_checkpoint.sh 1 /path/to/IMPACT_front/features epoch-100.model 0

GEMINI_API_KEY=... bash tasks/ASR/gemini_3_1_pro/scripts/run_batch_inference.sh /path/to/videos
bash tasks/ASR/gemini_3_1_pro/scripts/eval_predictions.sh outputs/asr/gemini_3_1_pro/<run_tag>/predictions test split1 front
```

## Notes

- The released baseline order follows Sec. 5.1 of the paper: `MS-TCN++`, `VideoMAE v2+Head`, `Gemini 3.1 Pro`.
- The public release currently targets the front-view protocol only.
- Evaluation uses `CHECKPOINT_NAME`, not an absolute checkpoint path, because the upstream evaluator resolves checkpoints from its own model directory.
- Method-local `README.md` files document method-specific defaults.
