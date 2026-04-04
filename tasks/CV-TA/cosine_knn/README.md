# cosine_knn for CV-TA

This directory exposes the public `CV-TA` launcher based on cosine-similarity retrieval over frozen segment features.

## Script

- `scripts/run_all.sh`: runs `local`, `global`, and `exo2ego` for the selected split and feature families

## Usage

```bash
bash tasks/CV-TA/cosine_knn/scripts/run_all.sh
```

```bash
RUN_TAG=cv_ta_s4 FEATURE_TYPES_CSV=videomaev2 SPLIT_INDEX=4 \
bash tasks/CV-TA/cosine_knn/scripts/run_all.sh
```
