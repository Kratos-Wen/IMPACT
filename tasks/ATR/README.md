# Anomaly Type Recognition

This directory contains the IMPACT Anomaly Type Recognition benchmark release.

## Path Convention

- Run all commands from the repository root.
- Repository-managed dataset root: `dataset/ATR/`
- Repository-managed runtime roots: `logs/atr/` and `outputs/atr/`
- User-managed input: checkpoint paths when a method provides evaluation

## Common Arguments

- `TASK_MODE`: `ATR_L` or `ATR_R`
- `FEATURE_TYPE`: `i3d` or `videomaev2`
- `GPU_LIST`: four comma-separated GPU ids such as `0,1,2,3`
- `SPLIT`: one of `1`, `2`, `3`, or `4`

Protocol mapping:
- `ATR_L` corresponds to `ATR-L`
- `ATR_R` corresponds to `ATR-R`

## Scripts

| Method | Script | Purpose | Usage |
| --- | --- | --- | --- |
| `ltcontext` | `tasks/ATR/ltcontext/scripts/train_splits.sh` | train all four splits | `bash tasks/ATR/ltcontext/scripts/train_splits.sh <TASK_MODE> <FEATURE_TYPE> <GPU_LIST> [RUN_TAG] [DATASET_ROOT] [OUTPUT_ROOT] [LOG_ROOT]` |
| `ltcontext` | `tasks/ATR/ltcontext/scripts/eval_checkpoint.sh` | evaluate one checkpoint on one split | `bash tasks/ATR/ltcontext/scripts/eval_checkpoint.sh <TASK_MODE> <FEATURE_TYPE> <SPLIT> [GPU] <CKPT_PATH> [DATASET_ROOT] [SAVE_ROOT]` |
| `fact` | `tasks/ATR/fact/scripts/train_splits.sh` | train all four splits | `bash tasks/ATR/fact/scripts/train_splits.sh <TASK_MODE> <FEATURE_TYPE> <GPU_LIST> [RUN_TAG] [DATASET_ROOT] [LOG_ROOT]` |

## Examples

```bash
bash tasks/ATR/ltcontext/scripts/train_splits.sh ATR_L videomaev2 0,1,2,3 atr_ltcontext
bash tasks/ATR/ltcontext/scripts/eval_checkpoint.sh ATR_L videomaev2 1 0 /path/to/checkpoint.pyth

bash tasks/ATR/fact/scripts/train_splits.sh ATR_R videomaev2 0,1,2,3 atr_fact
```

## Notes

- `ltcontext` exposes train and standalone checkpoint evaluation entrypoints.
- `fact` exposes the stable training entrypoint only. The bundled `third_party/fact/src/eval_checkpoint.py` explicitly excludes ATR scoring, so no standalone ATR evaluation wrapper is released here.
