# IMPACT

Official repository for "IMPACT: A Dataset for Multi-Granularity Human Procedural Action Understanding in Industrial Assembly".

<p align="center">
  <a href="https://kratos-wen.github.io/IMPACT/"><strong>Project Page</strong></a>
  ·
  <a href="https://drive.google.com/drive/folders/1P7vBnxSVH9g_lQc5n0c0WA47QAGEhE_U?usp=sharing"><strong>Data Release</strong></a>
  ·
  <a href="https://github.com/Kratos-Wen/IMPACT/tree/main/tasks"><strong>Benchmark Tasks</strong></a>
  ·
  <a href="https://github.com/Kratos-Wen/IMPACT/tree/main/dataset"><strong>Protocol Assets</strong></a>
</p>

> [!IMPORTANT]
> Start with the [IMPACT Project Page](https://kratos-wen.github.io/IMPACT/) for the dataset overview, visual examples, benchmark coverage, and release status.
>
> Dataset release bundles are available via [Google Drive](https://drive.google.com/drive/folders/1P7vBnxSVH9g_lQc5n0c0WA47QAGEhE_U?usp=sharing), including the full annotation bundle, multi-view RGB and depth archives, ego audio, released feature packs, and a quick-start sample subset. A Hugging Face mirror will be added in a later update.

This repository hosts the benchmark codebase, protocol assets, method snapshots, and website source for IMPACT. The release is organized first by task and then by method so that each benchmark setting exposes a consistent configuration and execution interface.

The paper benchmark is organized around the following task families:
- Temporal Understanding: `TAS`
- Cross-View Understanding: `CV-TA`, `CV-SM`
- Action Forecasting: `AF-S`, `AF-L`
- State & Reasoning: `PSR`, `ASR`, `PPR`, `ATR`

The current code release provides runnable benchmark wrappers for `TAS`, `CV-TA`, `CV-SM`, `AF-S`, `PSR`, `ASR`, `PPR`, and `ATR`. `AF-L` is reserved in the repository structure and documented, but its runnable baselines are not released yet.

This release includes:
- task protocol assets under `dataset/`
- IMPACT-specific benchmark wrappers under `tasks/`
- method source snapshots under `third_party/`
- project-site source under `website/`
- release bundles for annotations, RGB media, depth streams, ego audio, feature packs, and a quick-start sample subset via Google Drive

## Data Release Layout

The public Google Drive release is organized into the following bundles. For the bundle-level layout and naming conventions, see [docs/DATA_RELEASE.md](docs/DATA_RELEASE.md).

| Bundle | Approx. size | Contents |
| --- | ---: | --- |
| `annotations_v1.zip` | `5.4M` | official annotation release bundle |
| `videos/videos_{ego,front,left,right,top}.zip` | `132G` | five-view RGB video archives |
| `depth/depth_{front,left,right,top}.zip` | `46G` | exocentric depth archives |
| `audio/audio_ego.zip` | `200M` | egocentric audio bundle |
| `features/features_{I3D,MViTv2,VideoMAEv2}.zip` | `71G` | released backbone feature bundles |
| `sample/` | `3.5G` | quick-start subset with 3 executions, multi-view media, features, and task annotations |

## Repository Layout

```text
IMPACT/
├── .github/
├── dataset/
│   ├── TAS/
│   ├── CV/
│   ├── AF-S/
│   ├── AF-L/
│   ├── PSR/
│   ├── ASR/
│   ├── PPR/
│   └── ATR/
├── tasks/
│   ├── TAS/
│   ├── CV-TA/
│   ├── CV-SM/
│   ├── AF-S/
│   ├── AF-L/
│   ├── PSR/
│   ├── ASR/
│   ├── PPR/
│   └── ATR/
├── third_party/
├── website/
└── docs/
```

Runtime `logs/` and `outputs/` directories are not shipped in the repository tree. Public launcher scripts create them lazily when needed, and they remain excluded from version control.

## Task Coverage

`TAS`
- reference implementations: `LTContext`, `ASQuery`, `DiffAct`, `FACT`
- paper protocols: `TAS-S` and `TAS-B` (`TAS-BL`, `TAS-BR`)
- public wrappers accept `TAS-S`, `TAS-BL`, and `TAS-BR`

`CV-TA`
- reference implementation: `Cosine kNN`
- paper protocols: `local`, `global`, `exo2ego`
- public assets: synchronized step-level (`TAS-S`) annotations and split bundles under `dataset/CV/`
- default feature lookup: `features/cv/{videomaev2,i3d,mvitv2}`

`CV-SM`
- reference implementations: `CV-SMR`, `CV-SMC`
- paper protocols: `CV-SMR`, `CV-SMC`
- public assets: synchronized step-level (`TAS-S`) annotations and split bundles under `dataset/CV/`
- default feature lookup: `features/cv/{videomaev2,i3d,mvitv2}`

`AF-S`
- reference implementations: `AVT`, `ScalAnt`, `Qwen3VL-8B`
- paper protocol: `AF-S`
- current public split assets: `split1`
- supervised baselines use released feature packs; `Qwen3VL-8B` uses the released raw-video bundle

`AF-L`
- paper baselines: `ScalAnt`, `AntGPT`, `PALM`, `Qwen3VL-8B`
- repository status: placeholder task directory and documentation are included; runnable baselines are pending release

`PSR`
- indirect pipelines: `MS-TCN++ -> PSR`, `VideoMAE v2+Head -> PSR`
- direct pipelines: `STORM-PSR`, `Gemini 3.1 Pro`
- released assets include converted PSR labels and `procedure_info_IMPACT.json`

`ASR`
- reference implementations: `MS-TCN++`, `VideoMAE v2+Head`, `Gemini 3.1 Pro`
- current public split assets: front-view `split1`
- benchmark wrappers use the released `dataset/ASR/` protocol assets together with feature packs or raw-video bundles distributed through the release

`PPR`
- reference implementations: `ASQuery`, `DiffAct`, `LTContext`, `FACT`
- paper protocols: `PPR-L`, `PPR-R`
- launcher keys: `PPR_L`, `PPR_R`

`ATR`
- reference implementations: `LTContext`, `FACT`
- paper protocols: `ATR-L`, `ATR-R`
- launcher keys: `ATR_L`, `ATR_R`
- `FACT` is released with the stable training entrypoint; the standalone checkpoint scorer is intentionally not exposed for ATR because the bundled helper does not support it

## Licensing

This repository uses a split license structure:
- [LICENSE](LICENSE) covers repository-authored code, scripts, and configuration files, unless otherwise noted
- [LICENSE-DATA](LICENSE-DATA) covers dataset assets in `dataset/` and repository-authored documentation, including this `README.md`, files under `docs/`, and Markdown documentation under `tasks/`, unless otherwise noted
- `third_party/` uses per-directory provenance and license notices: bundled upstream licenses are retained where available, and IMPACT-maintained method snapshots document their external references in the corresponding `README.md`

If a file or subdirectory provides a more specific license notice, that notice takes precedence.

## Project Site

The project-site source lives under `website/`. The public landing page is published at `https://kratos-wen.github.io/IMPACT/`.
