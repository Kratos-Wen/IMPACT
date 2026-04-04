# DiffAct on IMPACT PPR

This directory provides the IMPACT PPR configurations and launch scripts for `DiffAct`.

Supported label protocols:
- `PPR_L`
- `PPR_R`

Supported feature types:
- `i3d`
- `videomaev2`

## Train

```bash
bash tasks/PPR/diffact/scripts/train_splits.sh PPR_L videomaev2 0,1,2,3 exp_ppr_diffact
```

Default paths:
- dataset root: `dataset/PPR/`
- outputs: `outputs/ppr/diffact/`
- logs: `logs/ppr/diffact/`

Implementation provenance:
- source snapshot: `third_party/diffact/`
- repository-specific changes: `tasks/TAS/diffact/UPSTREAM_DIFF.md`
