# FACT on IMPACT TAS

This directory provides the IMPACT TAS configurations and launch scripts for `FACT`.

Supported label protocols:
- `CAS`
- `FAS_L`
- `FAS_R`

Supported feature types:
- `i3d`
- `videomaev2`

## Train

```bash
bash tasks/tas/fact/scripts/train_splits.sh CAS videomaev2 0,1,2,3 exp_tas_fact
```

## Evaluate a Checkpoint

```bash
bash tasks/tas/fact/scripts/eval_checkpoint.sh CAS videomaev2 1 0 /path/to/network.iter-XXXXX.net
```

The TAS evaluation entrypoint uses `third_party/FACT/src/eval_checkpoint.py` to score a single checkpoint on the requested split.

Default paths:
- dataset root: `dataset/tas/`
- outputs: `outputs/tas/fact/`
- logs: `logs/tas/fact/`

Implementation provenance:
- source snapshot: `third_party/FACT/`
- repository-specific changes: `UPSTREAM_DIFF.md`
