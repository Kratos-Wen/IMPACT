# retrieval for CV-SM

This directory exposes the public `CV-SMR` launcher.

## Script

- `scripts/run_all.sh`: runs cross-view semantic retrieval across the requested split indices and feature families

## Usage

```bash
bash tasks/CV-SM/retrieval/scripts/run_all.sh
```

```bash
RUN_TAG=cv_smr_videomaev2 FEATURE_TYPES_CSV=videomaev2 SPLITS_CSV=1,2,3,4 \
bash tasks/CV-SM/retrieval/scripts/run_all.sh
```
