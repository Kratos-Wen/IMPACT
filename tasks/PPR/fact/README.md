# FACT for PPR

This directory provides the released IMPACT `PPR` wrapper for `FACT`.

## Path Convention

- dataset root: `dataset/PPR`
- configs: `tasks/PPR/fact/configs`
- default logs: `logs/ppr/fact`
- default evaluation outputs: `outputs/ppr/fact_eval`
- source snapshot: `third_party/fact`
- IMPACT-specific changes: `tasks/TAS/fact/UPSTREAM_DIFF.md`

## Common Arguments

- `TASK_MODE`: `PPR_L` or `PPR_R`
- `FEATURE_TYPE`: `videomaev2` or `i3d`
- `GPU_LIST`: four comma-separated GPU ids for split-wise training
- `RUN_TAG`: run identifier appended to split markers
- `IMPACT_ROOT`: dataset override, default `dataset/PPR`
- `SPLIT`: split id used by checkpoint evaluation
- `GPU`: CUDA device used by evaluation
- `CKPT_PATH`: checkpoint path, typically a `network.iter-XXXXX.net` file
- `LOG_BASE` or `SAVE_ROOT`: optional runtime directory overrides

## Scripts

- `scripts/train_splits.sh`: launches four split-wise training jobs
- `scripts/eval_checkpoint.sh`: evaluates a single checkpoint on the requested split

## Examples

```bash
bash tasks/PPR/fact/scripts/train_splits.sh PPR_L videomaev2 0,1,2,3 exp_ppr_fact
```

```bash
bash tasks/PPR/fact/scripts/eval_checkpoint.sh PPR_L videomaev2 1 0 /path/to/network.iter-100000.net
```

## Notes

- The evaluation wrapper calls `third_party/fact/src/eval_checkpoint.py` and stores metrics under the requested save root.
