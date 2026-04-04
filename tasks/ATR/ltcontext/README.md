# LTContext on IMPACT ATR

This directory provides the IMPACT ATR configurations and launch scripts for `LTContext`.

Supported label protocols:
- `ATR_L`
- `ATR_R`

Supported feature types:
- `i3d`
- `videomaev2`

## Train

```bash
bash tasks/ATR/ltcontext/scripts/train_splits.sh ATR_L videomaev2 0,1,2,3 exp_atr_ltcontext
```

## Evaluate a Checkpoint

```bash
bash tasks/ATR/ltcontext/scripts/eval_checkpoint.sh ATR_L videomaev2 1 0 /path/to/checkpoint.pyth
```

Default paths:
- dataset root: `dataset/ATR/`
- outputs: `outputs/atr/ltcontext/`
- logs: `logs/atr/ltcontext/`

Implementation provenance:
- source snapshot: `third_party/ltcontext/`
- repository-specific changes: `tasks/TAS/ltcontext/UPSTREAM_DIFF.md`
