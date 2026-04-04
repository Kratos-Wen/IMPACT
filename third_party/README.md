# Third-Party Implementations

This directory contains the method implementations used by the IMPACT benchmark release.

Bundled sources:
- `LTContext`
- `DiffAct`
- `ASQuery`
- `FACT`
- `ASR-PSR-Experiment`
- `STORM-PSR`

Task mapping:
- `LTContext`, `DiffAct`, `ASQuery`, and `FACT` back the released `TAS`, `PPR`, and `ATR` wrappers
- `ASR-PSR-Experiment` backs the released `ASR` wrappers and the indirect `PSR` pipelines
- `STORM-PSR` backs the released direct `PSR` temporal-stream wrapper

Repository-authored benchmark wrappers live under `tasks/`. Upstream licenses remain inside the corresponding source snapshots.

For the TAS methods, repository-specific changes relative to upstream are summarized in the method-local `UPSTREAM_DIFF.md` files under `tasks/TAS/`.
