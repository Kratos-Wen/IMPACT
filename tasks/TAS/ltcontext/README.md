# LTContext on IMPACT TAS

This directory provides the IMPACT TAS configurations and launch scripts for `LTContext`.

Supported label protocols:
- `CAS`
- `FAS_L`
- `FAS_R`

Supported feature types:
- `i3d`
- `videomaev2`

## Train

```bash
bash tasks/TAS/ltcontext/scripts/train_splits.sh CAS videomaev2 0,1,2,3 exp_tas_ltcontext
```

## Evaluate a Checkpoint

```bash
bash tasks/TAS/ltcontext/scripts/eval_checkpoint.sh CAS videomaev2 1 0 /path/to/checkpoint.pyth
```

Default paths:
- dataset root: `dataset/TAS/`
- outputs: `outputs/tas/ltcontext/`
- logs: `logs/tas/ltcontext/`

Implementation provenance:
- source snapshot: `third_party/LTContext/`
- repository-specific changes: `UPSTREAM_DIFF.md`
