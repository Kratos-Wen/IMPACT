# Dataset Assets

This directory stores lightweight benchmark assets that are required to reproduce the IMPACT evaluation protocol.

Current release status:
- `tas/` is included in this repository
- `asr/`, `psr/`, `ppr/`, and `atr/` are reserved for subsequent releases

The TAS release currently provides:
- official split files
- label mappings
- frame-level ground-truth annotations for `CAS`, `FAS_L`, and `FAS_R`

The repository does not include:
- raw videos
- extracted features
- model predictions or checkpoints

All TAS method entrypoints under `tasks/tas/` use `dataset/tas/` as the default annotation root.
