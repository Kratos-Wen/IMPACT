# DiffAct: Repository-Specific Modifications

The source snapshot used for this release is located in `third_party/diffact/`.

Relative to the upstream `DiffAct` release, the IMPACT benchmark version introduces the following method-side changes:
- `main.py`: IMPACT root, split, label-mode, and feature-type overrides
- `dataset.py`: IMPACT feature-name resolution and clip-to-frame alignment handling
- `utils.py`: phase metrics for `normal`, `anomaly`, and `recovery`

Benchmark-level TAS configurations and launch scripts are maintained separately under `tasks/TAS/diffact/`.
