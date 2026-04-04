# classification for CV-SM

This directory exposes the public `CV-SMC` launcher.

## Script

- `scripts/run_all.sh`: runs the linear-probe classification benchmark for the requested split indices, feature families, and label spaces

## Usage

```bash
bash tasks/CV-SM/classification/scripts/run_all.sh
```

```bash
RUN_TAG=cv_smc_videomaev2 FEATURE_TYPES_CSV=videomaev2 LABEL_MODES_CSV=coarse,verb,noun,verb_noun \
bash tasks/CV-SM/classification/scripts/run_all.sh
```
