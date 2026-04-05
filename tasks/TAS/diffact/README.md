# DiffAct for TAS

This directory provides the released IMPACT `TAS` wrapper for `DiffAct`.

## Path Convention

- dataset root: `dataset/TAS`
- configs: `tasks/TAS/diffact/configs`
- default outputs: `outputs/tas/diffact`
- default logs: `logs/tas/diffact`
- source snapshot: `third_party/diffact`
- IMPACT-specific changes: `tasks/TAS/diffact/UPSTREAM_DIFF.md`

## Common Arguments

- `TASK_MODE`: `TAS-S`, `TAS-BL`, or `TAS-BR`
- `FEATURE_TYPE`: `videomaev2` or `i3d`
- `GPU_LIST`: four comma-separated GPU ids for split-wise training
- `RUN_TAG`: run identifier appended to generated configs and logs
- `IMPACT_ROOT`: dataset override, default `dataset/TAS`
- `OUTPUT_BASE`: output root for generated configs and result files
- `LOG_BASE`: log directory override
- Legacy aliases `CAS`, `FAS_L`, and `FAS_R` remain accepted for backward compatibility

## Scripts

- `scripts/train_splits.sh`: launches four split-wise train-and-test runs

## Examples

```bash
bash tasks/TAS/diffact/scripts/train_splits.sh TAS-S videomaev2 0,1,2,3 exp_tas_diffact
```

## Notes

- `DiffAct` reports final test metrics as part of the training workflow, so the public release does not expose a separate checkpoint evaluator.
