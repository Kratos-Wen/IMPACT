# Dataset Assets

This directory stores lightweight benchmark assets that are required to reproduce the IMPACT evaluation protocol.

Included task assets:
- `TAS/`: mappings, official splits, and frame-level labels for `TAS-S`, `TAS-BL`, and `TAS-BR`
- `ASR/`: frame-level assembly-state annotations and the released front-view split protocol
- `PSR/`: STORM-style step labels, component names, and `procedure_info_IMPACT.json`
- `PPR/`: mappings, official splits, and frame-level labels for `PPR-L` and `PPR-R`
- `ATR/`: mappings, official splits, frame-level labels, anomaly masks, and segment manifests for `ATR-L` and `ATR-R`

Implementation mapping:
- `CAS` corresponds to `TAS-S`
- `FAS_L` corresponds to `TAS-BL`
- `FAS_R` corresponds to `TAS-BR`
- `PPR_L` corresponds to `PPR-L`
- `PPR_R` corresponds to `PPR-R`
- `ATR_L` corresponds to `ATR-L`
- `ATR_R` corresponds to `ATR-R`

Current protocol scope:
- `ASR/` and `PSR/` currently release the front-view protocol assets used by the benchmark wrappers in `tasks/ASR/` and `tasks/PSR/`
- `PPR/` and `ATR/` reuse the same task families as `TAS`, but expose task-specific mappings, splits, and labels

The repository does not include:
- raw videos
- extracted features
- model predictions or checkpoints

Task defaults used by the public scripts:
- `tasks/TAS/` uses `dataset/TAS/`
- `tasks/ASR/` uses `dataset/ASR/`
- `tasks/PSR/` uses `dataset/ASR/` and `dataset/PSR/`
- `tasks/PPR/` uses `dataset/PPR/`
- `tasks/ATR/` uses `dataset/ATR/`

Licensing:
- dataset assets in this directory are covered by [LICENSE-DATA](../LICENSE-DATA), unless otherwise noted
