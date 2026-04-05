# AF-S Source Release Notes

The released `AF-S` implementation is organized into method-named source snapshots under `third_party/avt`, `third_party/scalant`, and `third_party/qwen3_vl_8b`.

Released source scope:
- supervised baselines: `AVT`, `ScalAnt`
- zero-shot baseline: `Qwen3VL-8B`
- shared data pipeline: `FeatureDataset`, Lightning data module, and evaluation utilities

Release-specific changes:
- removed local `data/`, `expts/`, `.git`, caches, and generated split reports
- removed the unreleased `impact_mlp` baseline
- split the former internal integrated codebase into method-named public source snapshots
- replaced hardcoded filesystem defaults with repository-relative `dataset/AF-S`, `features/af_s`, and `outputs/af_s`
- removed mandatory Weights & Biases logging from the public training entrypoint
- exposed benchmark wrappers under `tasks/AF-S/`

Not changed:
- model implementations for `AVT`, `ScalAnt`, and `Qwen3VL-8B`
- split-bundle protocol logic and metric computation
