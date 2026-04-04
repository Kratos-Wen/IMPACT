# CV-SMR for CV-SM

This directory provides the released `CV-SMR` launcher.

## Path Convention

- annotations: `dataset/CV/annotations_CAS`
- split bundles: `dataset/CV/splits_CAS`
- default feature base: `features/cv`
- default outputs: `outputs/cv_sm/retrieval/<run_tag>`
- default logs: `logs/cv_sm/retrieval/<run_tag>`
- evaluator: `tools/cross_view/cv_smr_retrieval.py`

## Common Arguments

- `RUN_TAG`: run identifier used to separate outputs and logs
- `SPLIT`: split name passed to the evaluator, usually `test`
- `SPLITS_CSV`: comma-separated split bundle indices
- `FEATURE_TYPES_CSV`: comma-separated feature families
- `QUERY_VIEWS_CSV` and `GALLERY_VIEWS_CSV`: comma-separated view lists
- `FEATURE_BASE_DIR`: base directory that contains one subdirectory per feature family
- `FEATURE_ROOT`: optional explicit feature directory for single-backbone runs
- `ANNOTATION_ROOT`, `SPLIT_DIR`, `OUTPUT_ROOT`, `LOG_ROOT`: optional runtime overrides

## Scripts

- `scripts/run_all.sh`: runs cross-view semantic retrieval across the requested split indices and feature families

## Examples

```bash
bash tasks/CV-SM/retrieval/scripts/run_all.sh
```

```bash
RUN_TAG=cv_smr_videomaev2 FEATURE_TYPES_CSV=videomaev2 SPLITS_CSV=1,2,3,4 \
bash tasks/CV-SM/retrieval/scripts/run_all.sh
```

## Notes

- One launcher call emits one JSON file per split and feature family.
