# Long-Horizon Forecasting

This directory documents the IMPACT `AF-L` benchmark and its reported baseline setup.

## Task Definition

- `AF-L` operates on step-level segments derived from `TAS-S`.
- The standard protocol observes `M=2` coarse steps and forecasts the next `Z=5` steps.
- Reported baselines use Split 1 and the four exocentric views: `front`, `left`, `right`, and `top`.
- Evaluation follows the paper protocol with `K=5` sampled futures and reports `AUED` and `ED@z`.

## Status

- The current public repository documents the `AF-L` benchmark and baseline design.

## Baseline Families

| Family | Methods | Input | Notes |
| --- | --- | --- | --- |
| Feature-based segment models | `ScalAnt`, `PALM` | mean-pooled segment features | Reported with `I3D` and `VideoMAEv2` features |
| LLM-based sequence models | `AntGPT-LLM`, `PALM-LLM` | observed step labels as text | Used to study oracle-label and two-stage forecasting settings |
| Zero-shot VLM baseline | `Qwen3VL-8B` | raw video covering the observed steps | No task-specific training |

## Included Documentation

- [BASELINES.md](./BASELINES.md): detailed baseline descriptions, protocol notes, and hyperparameter summary.

## Notes

- Feature-based baselines use segment representations obtained by mean-pooling frame features over `TAS-S` step boundaries.
- The paper analyzes both oracle-label and predicted-label settings for LLM-based forecasting to quantify the recognition bottleneck.
- This directory is kept in the public repository so the released task layout matches the paper taxonomy while preserving accurate release scope.
