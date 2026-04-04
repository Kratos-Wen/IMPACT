# DiffAct on IMPACT TAS

This directory provides the IMPACT TAS configurations and launch scripts for `DiffAct`.

Supported label protocols:
- `CAS`
- `FAS_L`
- `FAS_R`

Supported feature types:
- `i3d`
- `videomaev2`

## Train

```bash
bash tasks/TAS/diffact/scripts/train_splits.sh CAS videomaev2 0,1,2,3 exp_tas_diffact
```

`DiffAct` reports final test metrics as part of its train-and-test workflow. For this reason, the TAS release provides a single training entrypoint rather than a separate checkpoint evaluator.

Default paths:
- dataset root: `dataset/TAS/`
- outputs: `outputs/tas/diffact/`
- logs: `logs/tas/diffact/`

Implementation provenance:
- source snapshot: `third_party/DiffAct/`
- repository-specific changes: `UPSTREAM_DIFF.md`
