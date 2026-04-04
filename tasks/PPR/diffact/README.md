# DiffAct for PPR

This directory provides the released IMPACT `PPR` wrapper for `DiffAct`.

## Path Convention

- dataset root: `dataset/PPR`
- configs: `tasks/PPR/diffact/configs`
- default outputs: `outputs/ppr/diffact`
- default logs: `logs/ppr/diffact`
- source snapshot: `third_party/diffact`
- IMPACT-specific changes: `tasks/TAS/diffact/UPSTREAM_DIFF.md`

## Common Arguments

- `TASK_MODE`: `PPR_L` or `PPR_R`
- `FEATURE_TYPE`: `videomaev2` or `i3d`
- `GPU_LIST`: four comma-separated GPU ids for split-wise training
- `RUN_TAG`: run identifier appended to generated configs and logs
- `IMPACT_ROOT`: dataset override, default `dataset/PPR`
- `OUTPUT_BASE`: output root for generated configs and result files
- `LOG_BASE`: log directory override

## Scripts

- `scripts/train_splits.sh`: launches four split-wise train-and-test runs

## Examples

```bash
bash tasks/PPR/diffact/scripts/train_splits.sh PPR_L videomaev2 0,1,2,3 exp_ppr_diffact
```

## Notes

- `DiffAct` reports final test metrics as part of the training workflow, so the public release does not expose a separate checkpoint evaluator.
