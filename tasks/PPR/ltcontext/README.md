# LTContext on IMPACT PPR

This directory provides the IMPACT PPR configurations and launch scripts for `LTContext`.

Supported label protocols:
- `PPR_L`
- `PPR_R`

Supported feature types:
- `i3d`
- `videomaev2`

## Train

```bash
bash tasks/PPR/ltcontext/scripts/train_splits.sh PPR_L videomaev2 0,1,2,3 exp_ppr_ltcontext
```

## Evaluate a Checkpoint

```bash
bash tasks/PPR/ltcontext/scripts/eval_checkpoint.sh PPR_L videomaev2 1 0 /path/to/checkpoint.pyth
```

Default paths:
- dataset root: `dataset/PPR/`
- outputs: `outputs/ppr/ltcontext/`
- logs: `logs/ppr/ltcontext/`

Implementation provenance:
- source snapshot: `third_party/ltcontext/`
- repository-specific changes: `tasks/TAS/ltcontext/UPSTREAM_DIFF.md`
