# Third-Party Implementations

This directory contains the method implementations used by the IMPACT benchmark release.

Bundled sources:
- `ltcontext`
- `diffact`
- `asquery`
- `fact`
- `asr_psr_experiment`
- `storm_psr`

Task mapping:
- `ltcontext`, `diffact`, `asquery`, and `fact` back the released `TAS`, `PPR`, and `ATR` wrappers
- `asr_psr_experiment` backs the released `ASR` wrappers and the indirect `PSR` pipelines
- `storm_psr` backs the released direct `PSR` temporal-stream wrapper

Repository-authored benchmark wrappers live under `tasks/`. Upstream licenses remain inside the corresponding source snapshots.

For the TAS methods, repository-specific changes relative to upstream are summarized in the method-local `UPSTREAM_DIFF.md` files under `tasks/TAS/`.
