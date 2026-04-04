# LTContext: Repository-Specific Modifications

The source snapshot used for this release is located in `third_party/LTContext/`.

Relative to the upstream `LTContext` release, the IMPACT benchmark version introduces the following method-side changes:
- `run_net.py`: IMPACT task switches, feature-type switches, and split override support
- `ltc/dataset/impact.py`: IMPACT dataset registration
- `ltc/train_net.py`: IMPACT metric selection support
- `ltc/test_net.py`: IMPACT evaluation path handling and ATR evaluation hook
- `ltc/utils/atr.py`: ATR metrics for the IMPACT setup
- `ltc/utils/metrics/__init__.py`: phase metrics for `normal`, `anomaly`, and `recovery`
- `ltc/utils/meters.py`: logging and model-selection updates for the additional metrics

Benchmark-level TAS configurations and launch scripts are maintained separately under `tasks/tas/ltcontext/`.
