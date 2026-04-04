# FACT on IMPACT ATR

This directory provides the IMPACT ATR training wrapper for `FACT`.

Supported label protocols:
- `ATR_L`
- `ATR_R`

Supported feature types:
- `i3d`
- `videomaev2`

## Train

```bash
bash tasks/ATR/fact/scripts/train_splits.sh ATR_L videomaev2 0,1,2,3 exp_atr_fact
```

Release note:
- the bundled `third_party/FACT/src/eval_checkpoint.py` explicitly excludes ATR
- the public release therefore exposes the stable ATR training entrypoint only, which already reports ATR validation metrics during training

Default paths:
- dataset root: `dataset/ATR/`
- logs: `logs/atr/fact/`

Implementation provenance:
- source snapshot: `third_party/FACT/`
- repository-specific changes: `tasks/TAS/fact/UPSTREAM_DIFF.md`
