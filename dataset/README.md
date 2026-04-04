# Dataset Assets

This directory stores lightweight benchmark assets that are required to reproduce the IMPACT evaluation protocol.

Current release status:
- `TAS/` is included in this repository
- `ASR/`, `PSR/`, `PPR/`, and `ATR/` are reserved for subsequent releases

The TAS release currently provides:
- official split files
- label mappings
- frame-level ground-truth annotations for the three TAS benchmarks in the paper:
  `TAS-S`, `TAS-BL`, and `TAS-BR`

Implementation mapping:
- `CAS` corresponds to `TAS-S`
- `FAS_L` corresponds to `TAS-BL`
- `FAS_R` corresponds to `TAS-BR`

The repository does not include:
- raw videos
- extracted features
- model predictions or checkpoints

All TAS method entrypoints under `tasks/TAS/` use `dataset/TAS/` as the default annotation root.

Licensing:
- dataset assets in this directory are covered by [LICENSE-DATA](/hkfs/work/workspace/scratch/jv8319-HAS/IMPACT_open_source/LICENSE-DATA), unless otherwise noted
