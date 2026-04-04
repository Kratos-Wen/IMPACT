# LTContext for TAS

This directory provides the released IMPACT `TAS` wrapper for `LTContext`.

## Path Convention

- dataset root: `dataset/TAS`
- configs: `tasks/TAS/ltcontext/configs`
- default outputs: `outputs/tas/ltcontext`
- default logs: `logs/tas/ltcontext`
- default evaluation outputs: `outputs/tas/ltcontext_eval`
- source snapshot: `third_party/ltcontext`
- IMPACT-specific changes: `tasks/TAS/ltcontext/UPSTREAM_DIFF.md`

## Common Arguments

- `TASK_MODE`: `CAS`, `FAS_L`, or `FAS_R`
- `FEATURE_TYPE`: `videomaev2` or `i3d`
- `GPU_LIST`: four comma-separated GPU ids for split-wise training
- `RUN_TAG`: run identifier appended to output folders
- `IMPACT_ROOT`: dataset override, default `dataset/TAS`
- `OUTPUT_BASE`, `LOG_BASE`, or `SAVE_ROOT`: optional runtime directory overrides
- `SPLIT`: split id used by checkpoint evaluation
- `GPU`: CUDA device used by evaluation
- `CKPT_PATH`: checkpoint path, typically a `.pyth` file

## Scripts

- `scripts/train_splits.sh`: launches four split-wise training jobs
- `scripts/eval_checkpoint.sh`: evaluates a single checkpoint on the requested split

## Examples

```bash
bash tasks/TAS/ltcontext/scripts/train_splits.sh CAS videomaev2 0,1,2,3 exp_tas_ltcontext
```

```bash
bash tasks/TAS/ltcontext/scripts/eval_checkpoint.sh CAS videomaev2 1 0 /path/to/checkpoint.pyth
```

## Notes

- Evaluation writes metrics to the requested save directory and disables prediction dumping by default.
