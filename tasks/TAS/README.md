# Temporal Action Segmentation

This directory contains the IMPACT Temporal Action Segmentation benchmark release.

Included reference implementations:
- `ltcontext`
- `diffact`
- `asquery`
- `fact`

Benchmarks defined in the paper:
- `TAS-S`
- `TAS-BL`
- `TAS-BR`

Launcher mapping used in this repository:
- `CAS` for `TAS-S`
- `FAS_L` for `TAS-BL`
- `FAS_R` for `TAS-BR`

Each method directory provides:
- IMPACT-specific configuration files
- standardized training and evaluation entrypoints
- a short provenance note describing method-side changes relative to the upstream implementation
