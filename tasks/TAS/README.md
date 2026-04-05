# Temporal Action Segmentation

This directory contains the IMPACT Temporal Action Segmentation benchmark release.

## Path Convention

- Run all commands from the repository root.
- Repository-managed dataset root: `dataset/TAS/`
- Repository-managed runtime roots: `logs/tas/` and `outputs/tas/`
- User-managed inputs: extracted feature files and checkpoint paths

## Common Arguments

- `TASK_MODE`: `TAS-S`, `TAS-BL`, or `TAS-BR`
- `FEATURE_TYPE`: `i3d` or `videomaev2`
- `GPU_LIST`: four comma-separated GPU ids such as `0,1,2,3`
- `SPLIT`: one of `1`, `2`, `3`, or `4`

Protocol mapping:
- `TAS-S` is the step-level segmentation task
- `TAS-B` is the bimanual family, with `TAS-BL` and `TAS-BR` as the released left-hand and right-hand protocols
- Legacy aliases `CAS`, `FAS_L`, and `FAS_R` remain accepted for backward compatibility

## Scripts

| Method | Script | Purpose | Usage |
| --- | --- | --- | --- |
| `LTContext` | `tasks/TAS/ltcontext/scripts/train_splits.sh` | train all four splits | `bash tasks/TAS/ltcontext/scripts/train_splits.sh <TASK_MODE> <FEATURE_TYPE> <GPU_LIST> [RUN_TAG] [DATASET_ROOT] [OUTPUT_ROOT] [LOG_ROOT]` |
| `LTContext` | `tasks/TAS/ltcontext/scripts/eval_checkpoint.sh` | evaluate one checkpoint on one split | `bash tasks/TAS/ltcontext/scripts/eval_checkpoint.sh <TASK_MODE> <FEATURE_TYPE> <SPLIT> [GPU] <CKPT_PATH> [DATASET_ROOT] [SAVE_ROOT]` |
| `ASQuery` | `tasks/TAS/asquery/scripts/train_splits.sh` | train all four splits | `bash tasks/TAS/asquery/scripts/train_splits.sh <TASK_MODE> <FEATURE_TYPE> <GPU_LIST> [RUN_TAG] [DATASET_ROOT] [LOG_ROOT]` |
| `ASQuery` | `tasks/TAS/asquery/scripts/eval_checkpoint.sh` | evaluate one checkpoint on one split | `bash tasks/TAS/asquery/scripts/eval_checkpoint.sh <TASK_MODE> <FEATURE_TYPE> <SPLIT> [GPU] <CKPT_PATH> [LOG_ROOT]` |
| `DiffAct` | `tasks/TAS/diffact/scripts/train_splits.sh` | train all four splits | `bash tasks/TAS/diffact/scripts/train_splits.sh <TASK_MODE> <FEATURE_TYPE> <GPU_LIST> [RUN_TAG] [DATASET_ROOT] [OUTPUT_ROOT] [LOG_ROOT]` |
| `FACT` | `tasks/TAS/fact/scripts/train_splits.sh` | train all four splits | `bash tasks/TAS/fact/scripts/train_splits.sh <TASK_MODE> <FEATURE_TYPE> <GPU_LIST> [RUN_TAG] [DATASET_ROOT] [LOG_ROOT]` |
| `FACT` | `tasks/TAS/fact/scripts/eval_checkpoint.sh` | evaluate one checkpoint on one split | `bash tasks/TAS/fact/scripts/eval_checkpoint.sh <TASK_MODE> <FEATURE_TYPE> <SPLIT> [GPU] <CKPT_PATH> [DATASET_ROOT] [SAVE_ROOT]` |

## Examples

```bash
bash tasks/TAS/ltcontext/scripts/train_splits.sh TAS-S videomaev2 0,1,2,3 tas_ltcontext
bash tasks/TAS/ltcontext/scripts/eval_checkpoint.sh TAS-S videomaev2 1 0 /path/to/checkpoint.pyth

bash tasks/TAS/asquery/scripts/train_splits.sh TAS-BR videomaev2 0,1,2,3 tas_asquery
bash tasks/TAS/asquery/scripts/eval_checkpoint.sh TAS-BR videomaev2 1 0 /path/to/checkpoint.pt

bash tasks/TAS/diffact/scripts/train_splits.sh TAS-BL i3d 0,1,2,3 tas_diffact

bash tasks/TAS/fact/scripts/train_splits.sh TAS-S videomaev2 0,1,2,3 tas_fact
bash tasks/TAS/fact/scripts/eval_checkpoint.sh TAS-S videomaev2 1 0 /path/to/network.iter-XXXXX.net
```

## Notes

- The method order follows Table 2 in the paper.
- `DiffAct` is released with the stable training entrypoint only.
- Default dataset and runtime paths are already wired into the scripts; override them only when you intentionally move the released structure.
- Method-local `README.md` files contain method-specific notes and upstream provenance.
