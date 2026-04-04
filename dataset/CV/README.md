# Cross-View Assets

This directory stores the released protocol assets for the cross-view benchmarks in IMPACT.

Contents:
- `annotations_CAS/`: synchronized CAS annotations for `ego`, `front`, `left`, `right`, and `top`
- `splits_CAS/`: official split bundles for `train`, `val`, and `test` across splits `S1` to `S4`

Task usage:
- `tasks/CV-TA/` uses these assets for `local`, `global`, and `exo2ego` retrieval protocols
- `tasks/CV-SM/` uses the same assets for `CV-SMR` and `CV-SMC`

Not included:
- raw videos
- extracted features
- model checkpoints

Default feature convention used by the public launchers:
- `FEATURE_BASE_DIR=features/cv`
- `features/cv/videomaev2`
- `features/cv/i3d`
- `features/cv/mvitv2`

Dataset assets in this directory are covered by [LICENSE-DATA](../../LICENSE-DATA), unless otherwise noted.
