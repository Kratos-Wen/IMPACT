# IMPACT

Official implementation of "IMPACT: A Dataset for Multi-Granularity Human Procedural Action Understanding in Industrial Assembly".

IMPACT provides the benchmark codebase for the IMPACT dataset and evaluation suite. The repository is organized first by task and then by method so that each benchmark setting exposes a consistent configuration and execution interface.

The current public release covers Temporal Action Segmentation (TAS) with four reference implementations:
- `LTContext`
- `DiffAct`
- `ASQuery`
- `FACT`

Included in this release:
- TAS protocol assets in `dataset/TAS/`, including label mappings, official splits, and frame-level ground truth
- IMPACT-specific TAS configurations and launch scripts in `tasks/TAS/`
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
│   └── TAS/
├── tasks/
│   ├── TAS/
│   ├── ASR/
│   ├── PSR/
│   ├── PPR/
│   └── ATR/
├── third_party/
└── docs/
```

## Current TAS Release

The paper defines three TAS benchmarks in the current release:
- `TAS-S`
- `TAS-BL`
- `TAS-BR`

The current implementation uses the following launcher keys:
- `CAS` for `TAS-S`
- `FAS_L` for `TAS-BL`
- `FAS_R` for `TAS-BR`

Each method directory under `tasks/TAS/` contains:
- `configs/`: IMPACT TAS configurations
- `scripts/`: training and evaluation entrypoints
- `README.md`: method-specific usage notes
- `UPSTREAM_DIFF.md`: repository-specific changes relative to the upstream implementation

Public TAS scripts use the following defaults:
- annotation root: `dataset/TAS/`
- output root: `outputs/`
- log root: `logs/`

Feature files are intentionally excluded from version control and should be prepared separately.

## Licensing

This repository uses a split license structure:
- [LICENSE](/hkfs/work/workspace/scratch/jv8319-HAS/IMPACT_open_source/LICENSE) covers repository-authored code, scripts, and configuration files, unless otherwise noted
- [LICENSE-DATA](/hkfs/work/workspace/scratch/jv8319-HAS/IMPACT_open_source/LICENSE-DATA) covers dataset assets in `dataset/` and repository-authored documentation, including this `README.md`, files under `docs/`, and Markdown documentation under `tasks/`, unless otherwise noted
- `third_party/` retains the bundled upstream licenses of the corresponding methods

If a file or subdirectory provides a more specific license notice, that notice takes precedence.

## Release Roadmap

Directories for `PSR` (Procedure Step Recognition), `ASR` (Assembly State Recognition), `PPR` (Procedural Phase Recognition), and `ATR` (Anomaly Type Recognition) are included to keep the repository layout stable. These tasks will be populated in subsequent releases.
