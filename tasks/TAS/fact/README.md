# FACT for TAS

This directory provides the released IMPACT `TAS` wrapper for `FACT`.

## Path Convention

- dataset root: `dataset/TAS`
- configs: `tasks/TAS/fact/configs`
- default logs: `logs/tas/fact`
- default evaluation outputs: `outputs/tas/fact_eval`
- source snapshot: `third_party/fact`
- IMPACT-specific changes: `tasks/TAS/fact/UPSTREAM_DIFF.md`

## Common Arguments

- `TASK_MODE`: `CAS`, `FAS_L`, or `FAS_R`
- `FEATURE_TYPE`: `videomaev2` or `i3d`
- `GPU_LIST`: four comma-separated GPU ids for split-wise training
- `RUN_TAG`: run identifier appended to split markers
- `IMPACT_ROOT`: dataset override, default `dataset/TAS`
- `SPLIT`: split id used by checkpoint evaluation
- `GPU`: CUDA device used by evaluation
- `CKPT_PATH`: checkpoint path, typically a `network.iter-XXXXX.net` file
- `LOG_BASE` or `SAVE_ROOT`: optional runtime directory overrides

## Scripts

- `scripts/train_splits.sh`: launches four split-wise training jobs
- `scripts/eval_checkpoint.sh`: evaluates a single checkpoint on the requested split

## Examples

```bash
bash tasks/TAS/fact/scripts/train_splits.sh CAS videomaev2 0,1,2,3 exp_tas_fact
```

```bash
bash tasks/TAS/fact/scripts/eval_checkpoint.sh CAS videomaev2 1 0 /path/to/network.iter-100000.net
```

## Notes

- The evaluation wrapper calls `third_party/fact/src/eval_checkpoint.py` and stores metrics under the requested save root.
