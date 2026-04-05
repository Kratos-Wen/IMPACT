# IMPACT

Official implementation of "IMPACT: A Dataset for Multi-Granularity Human Procedural Action Understanding in Industrial Assembly".

IMPACT provides the benchmark codebase for the IMPACT dataset and evaluation suite. The repository is organized first by task and then by method so that each benchmark setting exposes a consistent configuration and execution interface.

The current public release covers the following released benchmark tasks:
- `TAS`: Temporal Action Segmentation
- `ASR`: Assembly State Recognition
- `PSR`: Procedure Step Recognition
- `PPR`: Procedural Phase Recognition
- `ATR`: Anomaly Type Recognition
- `CV-TA`: Cross-View Temporal Alignment
- `CV-SM`: Cross-View Semantic Matching
- `AF-S`: Short-term Action Anticipation

`AF-L` will be released separately with its dedicated forecasting baselines.

Included in this release:
- task protocol assets under `dataset/`
- IMPACT-specific benchmark wrappers under `tasks/`
- method source snapshots under `third_party/`

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
│   ├── TAS/
│   ├── ASR/
│   ├── PSR/
│   ├── PPR/
│   ├── ATR/
│   ├── CV/
│   └── AF-S/
├── tasks/
│   ├── TAS/
│   ├── ASR/
│   ├── PSR/
│   ├── PPR/
│   ├── ATR/
│   ├── CV-TA/
│   ├── CV-SM/
│   └── AF-S/
├── third_party/
└── docs/
```

Runtime `logs/` and `outputs/` directories are not shipped in the repository tree. Public launcher scripts create them lazily when needed, and they remain excluded from version control.

## Task Coverage

`TAS`
- reference implementations: `LTContext`, `ASQuery`, `DiffAct`, `FACT`
- paper protocols: `TAS-S`, `TAS-BL`, `TAS-BR`
- launcher keys: `CAS`, `FAS_L`, `FAS_R`

`ASR`
- reference implementations: `MS-TCN++`, `VideoMAE v2+Head`, `Gemini 3.1 Pro`
- current public split assets: front-view `split1`
- benchmark wrappers expect an external feature directory or external raw videos, together with the released `dataset/ASR/` protocol assets

`PSR`
- indirect pipelines: `MS-TCN++ -> PSR`, `VideoMAE v2+Head -> PSR`
- direct pipelines: `STORM-PSR`, `Gemini 3.1 Pro`
- released assets include converted PSR labels and `procedure_info_IMPACT.json`

`PPR`
- reference implementations: `ASQuery`, `DiffAct`, `LTContext`, `FACT`
- paper protocols: `PPR-L`, `PPR-R`
- launcher keys: `PPR_L`, `PPR_R`

`ATR`
- reference implementations: `LTContext`, `FACT`
- paper protocols: `ATR-L`, `ATR-R`
- launcher keys: `ATR_L`, `ATR_R`
- `FACT` is released with the stable training entrypoint; the standalone checkpoint scorer is intentionally not exposed for ATR because the bundled helper does not support it

`CV-TA`
- reference implementation: `Cosine kNN`
- paper protocols: `local`, `global`, `exo2ego`
- public assets: synchronized CAS annotations and split bundles under `dataset/CV/`
- default feature lookup: `features/cv/{videomaev2,i3d,mvitv2}`

`CV-SM`
- reference implementations: `CV-SMR`, `CV-SMC`
- paper protocols: `CV-SMR`, `CV-SMC`
- public assets: synchronized CAS annotations and split bundles under `dataset/CV/`
- default feature lookup: `features/cv/{videomaev2,i3d,mvitv2}`

`AF-S`
- reference implementations: `AVT`, `ScalAnt`, `Qwen3VL-8B`
- paper protocol: `AF-S`
- current public split assets: `split1`
- supervised baselines expect external feature roots; `Qwen3VL-8B` expects raw videos
- `AF-L` is not included in this release

## Licensing

This repository uses a split license structure:
- [LICENSE](LICENSE) covers repository-authored code, scripts, and configuration files, unless otherwise noted
- [LICENSE-DATA](LICENSE-DATA) covers dataset assets in `dataset/` and repository-authored documentation, including this `README.md`, files under `docs/`, and Markdown documentation under `tasks/`, unless otherwise noted
- `third_party/` retains the bundled upstream licenses of the corresponding methods

If a file or subdirectory provides a more specific license notice, that notice takes precedence.
