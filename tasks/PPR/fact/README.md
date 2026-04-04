# FACT on IMPACT PPR

This directory provides the IMPACT PPR configurations and launch scripts for `FACT`.

Supported label protocols:
- `PPR_L`
- `PPR_R`

Supported feature types:
- `i3d`
- `videomaev2`

## Train

```bash
bash tasks/PPR/fact/scripts/train_splits.sh PPR_L videomaev2 0,1,2,3 exp_ppr_fact
```

## Evaluate a Checkpoint

```bash
bash tasks/PPR/fact/scripts/eval_checkpoint.sh PPR_L videomaev2 1 0 /path/to/network.iter-XXXXX.net
```

Default paths:
- dataset root: `dataset/PPR/`
- outputs: `outputs/ppr/fact/`
- logs: `logs/ppr/fact/`

Implementation provenance:
- source snapshot: `third_party/fact/`
- repository-specific changes: `tasks/TAS/fact/UPSTREAM_DIFF.md`
