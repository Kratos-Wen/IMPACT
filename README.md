# IMPACT

IMPACT provides the benchmark codebase for the IMPACT dataset and evaluation suite. The repository is organized first by task and then by method so that each benchmark setting exposes a consistent configuration and execution interface.

The current public release covers Temporal Action Segmentation (TAS) with four reference implementations:
- `LTContext`
- `DiffAct`
- `ASQuery`
- `FACT`

Included in this release:
- TAS protocol assets in `dataset/tas/`, including label mappings, official splits, and frame-level ground truth
- IMPACT-specific TAS configurations and launch scripts in `tasks/tas/`
- method source snapshots in `third_party/`, with repository-specific changes documented per method

Not included in this release:
- raw videos
- extracted features
- pretrained checkpoints
- runtime logs and intermediate outputs
- experiment-specific helper scripts

## Repository Layout

```text
IMPACT/
├── dataset/
├── tasks/
│   ├── tas/
│   ├── asr/
│   ├── psr/
│   ├── ppr/
│   └── atr/
├── third_party/
└── docs/
```

## Current TAS Release

The TAS release provides standardized entrypoints for:
- `CAS`
- `FAS_L`
- `FAS_R`

Each method directory under `tasks/tas/` contains:
- `configs/`: IMPACT TAS configurations
- `scripts/`: training and evaluation entrypoints
- `README.md`: method-specific usage notes
- `UPSTREAM_DIFF.md`: repository-specific changes relative to the upstream implementation

Public TAS scripts use the following defaults:
- annotation root: `dataset/tas/`
- output root: `outputs/`
- log root: `logs/`

Feature files are intentionally excluded from version control and should be prepared separately.

## Release Roadmap

Directories for `ASR`, `PSR`, `PPR`, and `ATR` are included to keep the repository layout stable. These tasks will be populated in subsequent releases.
