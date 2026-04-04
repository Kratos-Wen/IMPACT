# FACT for ATR

This directory provides the released IMPACT `ATR` wrapper for `FACT`.

## Path Convention

- dataset root: `dataset/ATR`
- configs: `tasks/ATR/fact/configs`
- default logs: `logs/atr/fact`
- source snapshot: `third_party/fact`
- IMPACT-specific changes: `tasks/TAS/fact/UPSTREAM_DIFF.md`

## Common Arguments

- `TASK_MODE`: `ATR_L` or `ATR_R`
- `FEATURE_TYPE`: `videomaev2` or `i3d`
- `GPU_LIST`: four comma-separated GPU ids for split-wise training
- `RUN_TAG`: run identifier appended to split markers
- `IMPACT_ROOT`: dataset override, default `dataset/ATR`
- `LOG_BASE`: log directory override

## Scripts

- `scripts/train_splits.sh`: launches four split-wise training jobs

## Examples

```bash
bash tasks/ATR/fact/scripts/train_splits.sh ATR_L videomaev2 0,1,2,3 exp_atr_fact
```

## Notes

- The bundled `third_party/fact/src/eval_checkpoint.py` does not support `ATR`, so the public release exposes only the stable training entrypoint.
- Training already reports the relevant validation metrics during runtime.
