# CV-TA

`CV-TA` is the IMPACT benchmark for Cross-View Temporal Alignment.

## Path Convention

- step-level (`TAS-S`) annotations: `dataset/CV/annotations_CAS`
- split bundles: `dataset/CV/splits_CAS`
- default feature roots: `features/cv/videomaev2`, `features/cv/i3d`, `features/cv/mvitv2`
- default outputs: `outputs/cv_ta/cosine_knn/<run_tag>`
- default logs: `logs/cv_ta/cosine_knn/<run_tag>`

## Common Arguments

- `RUN_TAG`: run identifier used to separate outputs and logs
- `SPLIT`: split name passed to the evaluator, usually `test`
- `SPLIT_INDEX`: split bundle index, matching paper splits `S1` to `S4`
- `FEATURE_TYPES_CSV`: comma-separated feature families, supported values are `videomaev2`, `i3d`, `mvitv2`
- `EXO_VIEWS_CSV`: exocentric views used in `local`, `global`, and `exo2ego`
- `POOLING`: feature pooling mode, normally `mean`
- `ANNOTATION_ROOT`: override for the step-level (`TAS-S`) annotation root
- `SPLIT_DIR`: override for the split bundle directory
- `FEATURE_BASE_DIR`: base directory that contains one subdirectory per feature family
- `FEATURE_ROOT`: optional explicit feature directory, mainly for single-backbone runs
- `OUTPUT_ROOT`: override for result JSON files
- `LOG_ROOT`: override for launcher logs

## Scripts

- `Cosine kNN`: `cosine_knn/scripts/run_all.sh` runs the released `CV-TA` baseline for `local`, `global`, and `exo2ego`

## Examples

```bash
bash tasks/CV-TA/cosine_knn/scripts/run_all.sh
```

```bash
FEATURE_TYPES_CSV=videomaev2 SPLIT_INDEX=4 \
bash tasks/CV-TA/cosine_knn/scripts/run_all.sh
```

```bash
FEATURE_BASE_DIR=/path/to/features/cv \
FEATURE_TYPES_CSV=videomaev2 EXO_VIEWS_CSV=front,left,right,top \
bash tasks/CV-TA/cosine_knn/scripts/run_all.sh
```

## Notes

- The public baseline is a cosine-similarity retrieval evaluator over frozen segment features.
- The released backbone set follows the paper: `i3d`, `videomaev2`, and `mvitv2`.
- Per the paper, `local` and `global` are reported on `S1` to `S3`, while `exo2ego` is the `S4` diagnostic.
