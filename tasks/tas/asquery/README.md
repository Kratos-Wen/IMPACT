# ASQuery on IMPACT TAS

This directory provides the IMPACT TAS configurations and launch scripts for `ASQuery`.

Supported label protocols:
- `CAS`
- `FAS_L`
- `FAS_R`

Supported feature types:
- `i3d`
- `videomaev2`

## Train

```bash
bash tasks/tas/asquery/scripts/train_splits.sh CAS videomaev2 0,1,2,3 exp_tas_asquery
```

## Evaluate a Checkpoint

```bash
bash tasks/tas/asquery/scripts/eval_checkpoint.sh CAS videomaev2 1 0 /path/to/ckpt_dir_or_file
```

The evaluation entrypoint generates a temporary evaluation config with `val_split` switched to `test`.

Default paths:
- dataset root: `dataset/tas/`
- outputs: `outputs/tas/asquery/`
- logs: `logs/tas/asquery/`

Implementation provenance:
- source snapshot: `third_party/ASQuery/`
- repository-specific changes: `UPSTREAM_DIFF.md`
