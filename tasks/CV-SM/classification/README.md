# CV-SMC for CV-SM

This directory provides the released `CV-SMC` launcher.

## Path Convention

- annotations: `dataset/CV/annotations_CAS`
- split bundles: `dataset/CV/splits_CAS`
- default feature base: `features/cv`
- default outputs: `outputs/cv_sm/classification/<run_tag>`
- default logs: `logs/cv_sm/classification/<run_tag>`
- evaluator: `tasks/CV-SM/classification/cv_smc_classification.py`

## Common Arguments

- `RUN_TAG`: run identifier used to separate outputs and logs
- `TRAIN_SPLIT` and `TEST_SPLIT`: split names used by the evaluator
- `SPLITS_CSV`: comma-separated split bundle indices
- `FEATURE_TYPES_CSV`: comma-separated feature families, typically `i3d,videomaev2,mvitv2`
- `LABEL_MODES_CSV`: comma-separated label spaces such as `coarse,verb,noun,verb_noun`
- `TRAIN_VIEWS_CSV` and `TEST_VIEWS_CSV`: comma-separated view lists
- `FEATURE_BASE_DIR`: base directory that contains one subdirectory per feature family
- `FEATURE_ROOT`: optional explicit feature directory for single-backbone runs
- `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `WEIGHT_DECAY`, `DEVICE`, `SEED`: optional linear-probe settings
- `ANNOTATION_ROOT`, `SPLIT_DIR`, `OUTPUT_ROOT`, `LOG_ROOT`: optional runtime overrides

## Scripts

- `scripts/run_all.sh`: runs the linear-probe benchmark for the requested split indices, feature families, and label spaces

## Examples

```bash
bash tasks/CV-SM/classification/scripts/run_all.sh
```

```bash
RUN_TAG=cv_smc_videomaev2 FEATURE_TYPES_CSV=videomaev2 LABEL_MODES_CSV=coarse,verb,noun,verb_noun \
bash tasks/CV-SM/classification/scripts/run_all.sh
```

## Notes

- One launcher call emits one JSON file per split, feature family, and label space.
- The released backbone set follows the paper: `i3d`, `videomaev2`, and `mvitv2`.
