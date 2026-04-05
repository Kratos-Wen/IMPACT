# Short-Term Action Anticipation

This directory contains the IMPACT `AF-S` benchmark release.

## Path Convention

- Run all commands from the repository root.
- Repository-managed dataset root: `dataset/AF-S/`
- Repository-managed runtime roots: `logs/af_s/` and `outputs/af_s/`
- User-managed inputs: extracted features for supervised baselines and raw videos for `Qwen3VL-8B`

## Common Arguments

- `FEATURE_TYPE`: `vmae` or `i3d`
- `VIEW`: `front`, `left`, `right`, `top`, `ego`, `all`, or `ego-exclude`
- `FEATURE_DIR`: external feature root for `AVT` and `ScalAnt`
- `VIDEO_ROOT`: raw video root for `Qwen3VL-8B`
- `SPLIT_NAME`: official split id; the current release provides `split1`
- `GPU_LIST`: comma-separated GPU ids for training or multi-process zero-shot evaluation
- `GPU`: single GPU id for checkpoint evaluation

## Scripts

| Method | Script | Purpose | Usage |
| --- | --- | --- | --- |
| `AVT` | `tasks/AF-S/avt/scripts/train_split.sh` | train one official split | `bash tasks/AF-S/avt/scripts/train_split.sh <FEATURE_TYPE> <VIEW> <FEATURE_DIR> [GPU_LIST] [SPLIT_NAME] [RUN_TAG] [ANNOTATION_DIR] [OUTPUT_ROOT] [LOG_ROOT]` |
| `AVT` | `tasks/AF-S/avt/scripts/eval_checkpoint.sh` | evaluate one checkpoint | `bash tasks/AF-S/avt/scripts/eval_checkpoint.sh <FEATURE_TYPE> <VIEW> <FEATURE_DIR> <CKPT_PATH> [GPU] [SPLIT_NAME] [RUN_TAG] [ANNOTATION_DIR] [OUTPUT_ROOT] [LOG_ROOT]` |
| `ScalAnt` | `tasks/AF-S/scalant/scripts/train_split.sh` | train one official split | `bash tasks/AF-S/scalant/scripts/train_split.sh <FEATURE_TYPE> <VIEW> <FEATURE_DIR> [GPU_LIST] [SPLIT_NAME] [RUN_TAG] [ANNOTATION_DIR] [OUTPUT_ROOT] [LOG_ROOT]` |
| `ScalAnt` | `tasks/AF-S/scalant/scripts/eval_checkpoint.sh` | evaluate one checkpoint | `bash tasks/AF-S/scalant/scripts/eval_checkpoint.sh <FEATURE_TYPE> <VIEW> <FEATURE_DIR> <CKPT_PATH> [GPU] [SPLIT_NAME] [RUN_TAG] [ANNOTATION_DIR] [OUTPUT_ROOT] [LOG_ROOT]` |
| `Qwen3VL-8B` | `tasks/AF-S/qwen3_vl_8b/scripts/run_eval.sh` | run zero-shot evaluation on one split | `bash tasks/AF-S/qwen3_vl_8b/scripts/run_eval.sh <VIDEO_ROOT> <VIEW> [SPLIT] [GPU_LIST] [RUN_TAG] [SPLIT_NAME] [ANNOTATION_DIR] [OUTPUT_ROOT] [LOG_ROOT]` |

## Examples

```bash
bash tasks/AF-S/avt/scripts/train_split.sh vmae ego-exclude /path/to/features 0 split1 afs_avt_vmae
bash tasks/AF-S/avt/scripts/eval_checkpoint.sh vmae ego-exclude /path/to/features /path/to/best.ckpt 0 split1 afs_avt_eval

bash tasks/AF-S/scalant/scripts/train_split.sh i3d all /path/to/features 0 split1 afs_scalant_i3d
bash tasks/AF-S/scalant/scripts/eval_checkpoint.sh i3d all /path/to/features /path/to/best.ckpt 0 split1 afs_scalant_eval

bash tasks/AF-S/qwen3_vl_8b/scripts/run_eval.sh /path/to/videos ego test 0 afs_qwen3vl_eval split1
```

## Notes

- Method order follows the `AF-S` baseline description in Sec. 4.3 of the paper.
- `AF-L` is reserved separately under `tasks/AF-L/`, but its runnable baselines are not released yet.
- Method-local `README.md` files provide method-specific execution notes and the shared source snapshot is summarized in `tasks/AF-S/UPSTREAM_DIFF.md`.
