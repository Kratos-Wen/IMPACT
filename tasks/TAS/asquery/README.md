# ASQuery for TAS

This directory provides the released IMPACT `TAS` wrapper for `ASQuery`.

## Path Convention

- dataset root: `dataset/TAS`
- configs: `tasks/TAS/asquery/configs`
- default logs: `logs/tas/asquery`
- default evaluation logs: `logs/tas/asquery_eval`
- source snapshot: `third_party/asquery`
- IMPACT-specific changes: `tasks/TAS/asquery/UPSTREAM_DIFF.md`

## Common Arguments

- `TASK_MODE`: `CAS`, `FAS_L`, or `FAS_R`
- `FEATURE_TYPE`: `videomaev2` or `i3d`
- `GPU_LIST`: four comma-separated GPU ids for split-wise training
- `RUN_TAG`: run identifier appended to training outputs
- `IMPACT_ROOT`: dataset override, default `dataset/TAS`
- `SPLIT`: split id used by checkpoint evaluation
- `GPU`: CUDA device used by evaluation
- `CKPT_PATH`: checkpoint path passed to the upstream evaluator
- `LOG_BASE` or `LOG_ROOT`: optional log directory override

## Scripts

- `scripts/train_splits.sh`: launches four split-wise training jobs
- `scripts/eval_checkpoint.sh`: evaluates a checkpoint on the requested test split

## Examples

```bash
bash tasks/TAS/asquery/scripts/train_splits.sh CAS videomaev2 0,1,2,3 exp_tas_asquery
```

```bash
bash tasks/TAS/asquery/scripts/eval_checkpoint.sh CAS videomaev2 1 0 /path/to/checkpoint.pt
```

## Notes

- The evaluation wrapper creates a temporary config with `val_split` rewritten to the requested `test` split.
