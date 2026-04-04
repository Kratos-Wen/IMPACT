# Release Scope

This repository release focuses on the Temporal Action Segmentation benchmark in IMPACT.

Included assets:
- official TAS split files
- TAS label mappings
- frame-level TAS annotations for `CAS`, `FAS_L`, and `FAS_R`
- method-specific TAS configurations and launch scripts for `LTContext`, `DiffAct`, `ASQuery`, and `FACT`
- source snapshots of the corresponding method implementations

Excluded assets:
- raw videos
- extracted features
- pretrained weights
- runtime logs
- intermediate predictions
- experiment-specific utility scripts

Repository conventions:
- task-level entrypoints live under `tasks/<task>/<method>/`
- source snapshots live under `third_party/`
- dataset assets live under `dataset/`
- generated outputs are written to `outputs/`
- runtime logs are written to `logs/`
