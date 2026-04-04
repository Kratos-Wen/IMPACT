# CV-SM

`CV-SM` is the IMPACT benchmark for Cross-View Semantic Matching.

The public release provides two benchmark variants:
- `CV-SMR`: Cross-View Semantic Matching via Retrieval
- `CV-SMC`: Cross-View Semantic Matching via Classification

## Path Convention

- annotations: `dataset/CV/annotations_CAS`
- split bundles: `dataset/CV/splits_CAS`
- default feature roots: `features/cv/videomaev2`, `features/cv/i3d`, `features/cv/mvitv2`
- default outputs: `outputs/cv_sm/<method>/<run_tag>`
- default logs: `logs/cv_sm/<method>/<run_tag>`

## Common Arguments

- `RUN_TAG`: run identifier used to separate outputs and logs
- `SPLITS_CSV`: comma-separated split bundle indices, usually `1,2,3,4`
- `FEATURE_TYPES_CSV`: comma-separated feature families, supported values are `videomaev2`, `i3d`, `mvitv2`
- `ANNOTATION_ROOT`: override for the CAS annotation root
- `SPLIT_DIR`: override for the split bundle directory
- `FEATURE_BASE_DIR`: base directory that contains one subdirectory per feature family
- `FEATURE_ROOT`: optional explicit feature directory, mainly for single-backbone runs
- `OUTPUT_ROOT`: override for result JSON files
- `LOG_ROOT`: override for launcher logs

## Scripts

- `retrieval/scripts/run_all.sh`: runs `CV-SMR`
- `classification/scripts/run_all.sh`: runs `CV-SMC`

## Examples

```bash
bash tasks/CV-SM/retrieval/scripts/run_all.sh
```

```bash
FEATURE_TYPES_CSV=videomaev2 LABEL_MODES_CSV=coarse,verb,noun,verb_noun \
bash tasks/CV-SM/classification/scripts/run_all.sh
```

## Notes

- `CV-SMR` reports retrieval metrics over cross-trial, cross-view semantic matches.
- `CV-SMC` trains a linear probe on frozen features and reports classification metrics on the held-out split.
- Per the paper, `CV-SMR` and `CV-SMC` report all four splits `S1` to `S4`.
