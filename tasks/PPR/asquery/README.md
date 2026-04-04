# ASQuery on IMPACT PPR

This directory provides the IMPACT PPR configurations and launch scripts for `ASQuery`.

Supported label protocols:
- `PPR_L`
- `PPR_R`

Supported feature types:
- `i3d`
- `videomaev2`

## Train

```bash
bash tasks/PPR/asquery/scripts/train_splits.sh PPR_L videomaev2 0,1,2,3 exp_ppr_asquery
```

## Evaluate a Checkpoint

```bash
bash tasks/PPR/asquery/scripts/eval_checkpoint.sh PPR_L videomaev2 1 0 /path/to/checkpoint.pt
```

Default paths:
- dataset root: `dataset/PPR/`
- logs: `logs/ppr/asquery/`

Implementation provenance:
- source snapshot: `third_party/asquery/`
- repository-specific changes: `tasks/TAS/asquery/UPSTREAM_DIFF.md`
