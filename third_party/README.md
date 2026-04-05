# Third-Party Implementations

This directory contains the method implementations used by the IMPACT benchmark release.

Bundled sources:
- `ltcontext`
- `diffact`
- `asquery`
- `fact`
- `ms_tcn2`
- `videomae_v2_head`
- `storm_psr`
- `avt`
- `scalant`
- `qwen3_vl_8b`

Task mapping:
- `ltcontext`, `diffact`, `asquery`, and `fact` back the released `TAS`, `PPR`, and `ATR` wrappers
- `ms_tcn2` backs the released `ASR` and indirect `PSR` wrappers for `MS-TCN++`
- `videomae_v2_head` backs the released `ASR` and indirect `PSR` wrappers for `VideoMAE v2+Head`
- `storm_psr` backs the released direct `PSR` temporal-stream wrapper
- `avt`, `scalant`, and `qwen3_vl_8b` back the released `AF-S` wrappers

Repository-authored benchmark wrappers live under `tasks/`. Upstream licenses remain inside the corresponding source snapshots.

For the TAS methods, repository-specific changes relative to upstream are summarized in the method-local `UPSTREAM_DIFF.md` files under `tasks/TAS/`.
