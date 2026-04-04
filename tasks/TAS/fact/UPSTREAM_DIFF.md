# FACT: Repository-Specific Modifications

The source snapshot used for this release is located in `third_party/fact/`.

Relative to the upstream `FACT` release, the IMPACT benchmark version introduces the following method-side changes:
- `src/train.py`: IMPACT-aware training, validation, and metric-based model selection
- `src/eval_checkpoint.py`: single-checkpoint evaluation helper
- `src/configs/default.py`: IMPACT-specific convenience switches
- `src/utils/dataset.py`: IMPACT dataset routing, file layout, and feature-type handling
- `src/utils/evaluate.py`: phase metrics for `normal`, `anomaly`, and `recovery`
- `src/utils/atr.py`: ATR metrics used by the broader IMPACT codebase
- `src/models/blocks.py`: ATR-task branching support

Benchmark-level TAS configurations and launch scripts are maintained separately under `tasks/TAS/fact/`.
