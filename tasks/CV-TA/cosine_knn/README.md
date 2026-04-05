# Cosine kNN for CV-TA

This directory provides the released `CV-TA` baseline based on cosine-similarity retrieval over frozen segment features.

## Path Convention

- step-level (`TAS-S`) annotations: `dataset/CV/annotations_CAS`
- split bundles: `dataset/CV/splits_CAS`
- default feature base: `features/cv`
- default outputs: `outputs/cv_ta/cosine_knn/<run_tag>`
- default logs: `logs/cv_ta/cosine_knn/<run_tag>`
- evaluator: `tasks/CV-TA/cosine_knn/cv_ta_retrieval.py`

## Common Arguments

- `RUN_TAG`: run identifier used to separate outputs and logs
- `SPLIT`: split name passed to the evaluator, usually `test`
- `SPLIT_INDEX`: split bundle index
- `FEATURE_TYPES_CSV`: comma-separated feature families, typically `i3d,videomaev2,mvitv2`
- `EXO_VIEWS_CSV`: exocentric views used by `local`, `global`, and `exo2ego`
- `FEATURE_BASE_DIR`: base directory that contains one subdirectory per feature family
- `FEATURE_ROOT`: optional explicit feature directory for single-backbone runs
- `ANNOTATION_ROOT`, `SPLIT_DIR`, `OUTPUT_ROOT`, `LOG_ROOT`: optional runtime overrides

## Scripts

- `scripts/run_all.sh`: runs `local`, `global`, and `exo2ego` for the selected split and feature families

## Examples

```bash
bash tasks/CV-TA/cosine_knn/scripts/run_all.sh
```

```bash
RUN_TAG=cv_ta_s4 FEATURE_TYPES_CSV=videomaev2 SPLIT_INDEX=4 \
bash tasks/CV-TA/cosine_knn/scripts/run_all.sh
```

## Notes

- One launcher call emits one JSON file per protocol and feature family.
- The released backbone set follows the paper: `i3d`, `videomaev2`, and `mvitv2`.
