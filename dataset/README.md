# Dataset Assets

This directory stores lightweight benchmark assets that are required to reproduce the IMPACT evaluation protocol.

Included task assets:
- `TAS/`: mappings, official splits, and frame-level labels for `TAS-S` and `TAS-B` (`TAS-BL`, `TAS-BR`)
- `CV/`: synchronized step-level (`TAS-S`) annotations and official split bundles for `CV-TA`, `CV-SMR`, and `CV-SMC`
- `AF-S/`: multi-view anticipation annotations and official split bundles for `AF-S`
- `AF-L/`: placeholder directory reserved for long-horizon forecasting assets
- `PSR/`: STORM-style step labels, component names, and `procedure_info_IMPACT.json`
- `ASR/`: frame-level assembly-state annotations and the released front-view split protocol
- `PPR/`: mappings, official splits, and frame-level labels for `PPR-L` and `PPR-R`
- `ATR/`: mappings, official splits, frame-level labels, anomaly masks, and segment manifests for `ATR-L` and `ATR-R`

Implementation mapping:
- `PPR_L` corresponds to `PPR-L`
- `PPR_R` corresponds to `PPR-R`
- `ATR_L` corresponds to `ATR-L`
- `ATR_R` corresponds to `ATR-R`

Current protocol scope:
- `CV/` exposes the shared cross-view assets used by `tasks/CV-TA/` and `tasks/CV-SM/`
- `AF-S/` exposes the anticipation annotations and split bundles used by `tasks/AF-S/`
- `AF-L/` currently documents the pending long-horizon release only
- `PSR/` and `ASR/` currently release the front-view protocol assets used by the benchmark wrappers in `tasks/PSR/` and `tasks/ASR/`
- `PPR/` and `ATR/` reuse the same task families as `TAS`, but expose task-specific mappings, splits, and labels
- raw media, feature packs, checkpoints, and supplementary release bundles are distributed through Google Drive alongside the repository assets

Task defaults used by the public scripts:
- `tasks/TAS/` uses `dataset/TAS/`
- `tasks/CV-TA/` uses `dataset/CV/`
- `tasks/CV-SM/` uses `dataset/CV/`
- `tasks/AF-S/` uses `dataset/AF-S/`
- `tasks/AF-L/` will use `dataset/AF-L/`
- `tasks/PSR/` uses `dataset/ASR/` and `dataset/PSR/`
- `tasks/ASR/` uses `dataset/ASR/`
- `tasks/PPR/` uses `dataset/PPR/`
- `tasks/ATR/` uses `dataset/ATR/`

Licensing:
- dataset assets in this directory are covered by [LICENSE-DATA](../LICENSE-DATA), unless otherwise noted
