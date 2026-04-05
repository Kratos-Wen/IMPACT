# Assembly State Recognition

This directory contains the IMPACT Assembly State Recognition benchmark release.

## Path Convention

- Run all commands from the repository root.
- Repository-managed annotation root: `dataset/ASR/annotations/`
- Repository-managed split root: `dataset/ASR/splits_front_only_v1/`
- Repository-managed runtime root: `logs/asr/`
- User-managed input: `FEATURE_DIR`

## Common Arguments

- `SPLIT_ID`: split index such as `1`
- `FEATURE_DIR`: path to extracted front-view features
- `CHECKPOINT_NAME`: checkpoint file name under the upstream model directory, for example `epoch-100.model`

Current public protocol:
- front-view bundle split `split1`

## Scripts

| Method | Script | Purpose | Usage |
| --- | --- | --- | --- |
| `MS-TCN++` | `tasks/ASR/ms_tcn2/scripts/train_split.sh` | train one released split | `bash tasks/ASR/ms_tcn2/scripts/train_split.sh <SPLIT_ID> <FEATURE_DIR> [GPU] [NUM_EPOCHS] [BATCH_SIZE] [LR] [ANNOTATION_DIR] [SPLIT_DIR] [LOG_ROOT]` |
| `MS-TCN++` | `tasks/ASR/ms_tcn2/scripts/eval_checkpoint.sh` | evaluate one checkpoint | `bash tasks/ASR/ms_tcn2/scripts/eval_checkpoint.sh <SPLIT_ID> <FEATURE_DIR> <CHECKPOINT_NAME> [GPU] [ANNOTATION_DIR] [SPLIT_DIR] [LOG_ROOT]` |
| `VideoMAE v2+Head` | `tasks/ASR/videomae_v2_head/scripts/train_split.sh` | train one released split | `bash tasks/ASR/videomae_v2_head/scripts/train_split.sh <SPLIT_ID> <FEATURE_DIR> [GPU] [NUM_EPOCHS] [BATCH_SIZE] [LR] [ANNOTATION_DIR] [SPLIT_DIR] [LOG_ROOT]` |
| `VideoMAE v2+Head` | `tasks/ASR/videomae_v2_head/scripts/eval_checkpoint.sh` | evaluate one checkpoint | `bash tasks/ASR/videomae_v2_head/scripts/eval_checkpoint.sh <SPLIT_ID> <FEATURE_DIR> <CHECKPOINT_NAME> [GPU] [ANNOTATION_DIR] [SPLIT_DIR] [LOG_ROOT]` |

## Examples

```bash
bash tasks/ASR/ms_tcn2/scripts/train_split.sh 1 /path/to/IMPACT_i3d_front/features 0
bash tasks/ASR/ms_tcn2/scripts/eval_checkpoint.sh 1 /path/to/IMPACT_i3d_front/features epoch-100.model 0

bash tasks/ASR/videomae_v2_head/scripts/train_split.sh 1 /path/to/IMPACT_front/features 0
bash tasks/ASR/videomae_v2_head/scripts/eval_checkpoint.sh 1 /path/to/IMPACT_front/features epoch-100.model 0
```

## Notes

- The released baseline order follows the non-VLM methods described in Sec. 5.1 of the paper.
- The public release currently targets the front-view protocol only.
- Evaluation uses `CHECKPOINT_NAME`, not an absolute checkpoint path, because the upstream evaluator resolves checkpoints from its own model directory.
- Method-local `README.md` files document method-specific defaults.
