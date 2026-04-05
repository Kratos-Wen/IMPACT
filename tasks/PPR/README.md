# Procedural Phase Recognition

This directory contains the IMPACT Procedural Phase Recognition benchmark release.

## Path Convention

- Run all commands from the repository root.
- Repository-managed dataset root: `dataset/PPR/`
- Repository-managed runtime roots: `logs/ppr/` and `outputs/ppr/`
- User-managed inputs: checkpoint paths when a method provides evaluation

## Common Arguments

- `TASK_MODE`: `PPR_L` or `PPR_R`
- `FEATURE_TYPE`: `i3d` or `videomaev2`
- `GPU_LIST`: four comma-separated GPU ids such as `0,1,2,3`
- `SPLIT`: one of `1`, `2`, `3`, or `4`

Protocol mapping:
- `PPR_L` corresponds to `PPR-L`
- `PPR_R` corresponds to `PPR-R`

## Scripts

| Method | Script | Purpose | Usage |
| --- | --- | --- | --- |
| `ASQuery` | `tasks/PPR/asquery/scripts/train_splits.sh` | train all four splits | `bash tasks/PPR/asquery/scripts/train_splits.sh <TASK_MODE> <FEATURE_TYPE> <GPU_LIST> [RUN_TAG] [DATASET_ROOT] [LOG_ROOT]` |
| `ASQuery` | `tasks/PPR/asquery/scripts/eval_checkpoint.sh` | evaluate one checkpoint on one split | `bash tasks/PPR/asquery/scripts/eval_checkpoint.sh <TASK_MODE> <FEATURE_TYPE> <SPLIT> [GPU] <CKPT_PATH> [LOG_ROOT]` |
| `DiffAct` | `tasks/PPR/diffact/scripts/train_splits.sh` | train all four splits | `bash tasks/PPR/diffact/scripts/train_splits.sh <TASK_MODE> <FEATURE_TYPE> <GPU_LIST> [RUN_TAG] [DATASET_ROOT] [OUTPUT_ROOT] [LOG_ROOT]` |
| `LTContext` | `tasks/PPR/ltcontext/scripts/train_splits.sh` | train all four splits | `bash tasks/PPR/ltcontext/scripts/train_splits.sh <TASK_MODE> <FEATURE_TYPE> <GPU_LIST> [RUN_TAG] [DATASET_ROOT] [OUTPUT_ROOT] [LOG_ROOT]` |
| `LTContext` | `tasks/PPR/ltcontext/scripts/eval_checkpoint.sh` | evaluate one checkpoint on one split | `bash tasks/PPR/ltcontext/scripts/eval_checkpoint.sh <TASK_MODE> <FEATURE_TYPE> <SPLIT> [GPU] <CKPT_PATH> [DATASET_ROOT] [SAVE_ROOT]` |
| `FACT` | `tasks/PPR/fact/scripts/train_splits.sh` | train all four splits | `bash tasks/PPR/fact/scripts/train_splits.sh <TASK_MODE> <FEATURE_TYPE> <GPU_LIST> [RUN_TAG] [DATASET_ROOT] [LOG_ROOT]` |
| `FACT` | `tasks/PPR/fact/scripts/eval_checkpoint.sh` | evaluate one checkpoint on one split | `bash tasks/PPR/fact/scripts/eval_checkpoint.sh <TASK_MODE> <FEATURE_TYPE> <SPLIT> [GPU] <CKPT_PATH> [DATASET_ROOT] [SAVE_ROOT]` |

## Examples

```bash
bash tasks/PPR/asquery/scripts/train_splits.sh PPR_L videomaev2 0,1,2,3 ppr_asquery
bash tasks/PPR/asquery/scripts/eval_checkpoint.sh PPR_L videomaev2 1 0 /path/to/checkpoint.pt

bash tasks/PPR/diffact/scripts/train_splits.sh PPR_R i3d 0,1,2,3 ppr_diffact

bash tasks/PPR/ltcontext/scripts/train_splits.sh PPR_L videomaev2 0,1,2,3 ppr_ltcontext
bash tasks/PPR/ltcontext/scripts/eval_checkpoint.sh PPR_L videomaev2 1 0 /path/to/checkpoint.pyth

bash tasks/PPR/fact/scripts/train_splits.sh PPR_R videomaev2 0,1,2,3 ppr_fact
bash tasks/PPR/fact/scripts/eval_checkpoint.sh PPR_R videomaev2 1 0 /path/to/network.iter-XXXXX.net
```

## Notes

- The method order follows Table 5(b) in the paper for the released non-VLM baselines.
- `DiffAct` is released with the stable training entrypoint only.
- Default dataset and runtime paths already match the released repository layout.
