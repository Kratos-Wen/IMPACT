# ASQuery: Repository-Specific Modifications

The source snapshot used for this release is located in `third_party/ASQuery/`.

Relative to the upstream `ASQuery` release, the IMPACT benchmark version introduces the following method-side changes:
- `main.py`: IMPACT root, split, label-mode, and feature-type overrides
- `eval.py`: TAS metric reporting adjustments
- `libs/datasets/breakfast.py`: IMPACT dataset registration and feature resolution logic
- `libs/utils/metrics.py`: phase metrics for `normal`, `anomaly`, and `recovery`

Benchmark-level TAS configurations and launch scripts are maintained separately under `tasks/tas/asquery/`.
