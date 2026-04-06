# Cross-View Assets

This directory stores the released protocol assets for the cross-view benchmarks in IMPACT.

Contents:
- `annotations_CAS/`: synchronized step-level (`TAS-S`) annotations for `ego`, `front`, `left`, `right`, and `top`
- `splits_CAS/`: official split bundles for `train`, `val`, and `test` across splits `S1` to `S4`

Task usage:
- `tasks/CV-TA/` uses these assets for `local`, `global`, and `exo2ego` retrieval protocols
- `tasks/CV-SM/` uses the same assets for `CV-SMR` and `CV-SMC`

Stored outside this directory:
- raw videos are distributed through the Google Drive release bundle
- extracted features are distributed through the Google Drive release bundle
- model checkpoints are distributed through the Google Drive release bundle

This subdirectory stores only the lightweight protocol assets required by the public cross-view launchers.

Default feature convention used by the public launchers:
- `FEATURE_BASE_DIR=features/cv`
- `features/cv/videomaev2`
- `features/cv/i3d`
- `features/cv/mvitv2`

Dataset assets in this directory are covered by [LICENSE-DATA](../../LICENSE-DATA), unless otherwise noted.
