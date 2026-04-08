# AF-L Baseline Notes

This note summarizes the `AF-L` baselines reported for IMPACT and the evaluation protocol used in the paper appendix.

## Evaluation Protocol

- Task: long-horizon forecasting over step-level segments from `TAS-S`
- Observed context: `M=2` coarse procedural steps
- Forecast horizon: `Z=5` future steps
- Candidate futures used for evaluation: best of `K=5`
- Reported split: `Split 1`
- Reported views: `front`, `left`, `right`, `top`
- Metrics: `AUED` and `ED@z`

## Segment Features

- Feature-based baselines operate on segment-level representations produced by mean-pooling frame features inside each `TAS-S` step boundary.
- Two feature backbones are reported:
  - `I3D` (`1024-D`)
  - `VideoMAEv2` (`1408-D`)

Pooling rule:

```text
x_bar = (1 / (f_end - f_start + 1)) * sum_{t=f_start}^{f_end} x_t
```

Mean-pooling is used as the default segment aggregation strategy.

## Baseline Summary

| Method | Family | Input | Backbone / Model | Key Setup |
| --- | --- | --- | --- | --- |
| `ScalAnt` | feature-based | segment features | Mamba encoder + cross-attention decoder | `Z=5` learned future queries, no `CLAM` memory |
| `PALM` | feature-based | segment features | forecasting encoder-decoder | independent heads for each future step |
| `AntGPT-LLM` | LLM-based | observed step labels | `Llama2-7B` + `LoRA` | fine-tuned sequence completion |
| `PALM-LLM` | LLM-based | observed step labels | `Llama2-7B` in-context learning | `MMR` exemplar selection |
| `Qwen3VL-8B` | zero-shot VLM | raw video | `Qwen3VL-8B` | predicts future step ids directly from video |

## Feature-Based Baselines

### ScalAnt

- Adapted to segment-level inputs.
- Uses a 4-layer Mamba encoder with `d_state=16` and `d_conv=4`.
- Uses a 4-layer query decoder with 8 attention heads and FFN dimension `2048`.
- Decodes with `Z=5` learnable queries, one for each forecast step.
- Maps each decoded query to the 26-class step vocabulary with a shared linear head.
- Disables `CLAM` memory because the observed sequence contains only two segments.
- Training setup:
  - optimizer: `AdamW`
  - learning rate: `3e-4`
  - weight decay: `1e-4`
  - scheduler: cosine
  - batch size: `32`
  - max epochs: `50`
  - early stopping on validation `AUED` with patience `10`

### PALM

- Uses the forecasting encoder-decoder design adapted from the Ego4D LTA codebase.
- Projects segment features to a `512-D` hidden space.
- Aggregates the observed segments by concatenation.
- Predicts `Z=5` future steps with independent linear classification heads.
- Uses equal-weight per-step cross-entropy losses.
- Does not use observed step labels as input.
- Training setup:
  - optimizer: `AdamW`
  - learning rate: `5e-4`
  - batch size: `32`
  - max epochs: `50`
  - early stopping on validation `AUED`

## LLM-Based Baselines

### AntGPT-LLM

- Fine-tunes `Llama2-7B` with `LoRA`.
- Reported LoRA setup:
  - rank `r=8`
  - `alpha=16`
  - dropout `0.05`
  - targets: `q_proj`, `v_proj`
- Uses a minimal sequence-completion format:

```text
step_A, step_B => step_C, step_D, step_E, step_F, step_G
```

- Trained with the Hugging Face `Trainer`.
- Reported training setup:
  - learning rate: `5e-4`
  - batch size: `4`
  - epochs: `10`
  - warmup steps: `100`
  - mixed precision: `bf16`
- Inference uses `K=5` outputs:
  - one greedy decoding
  - four sampled decodings with temperature `0.7` and `top-k=50`

### PALM-LLM

- Uses `Llama2-7B` without fine-tuning in an in-context learning setup.
- Retrieves candidate exemplars with `all-mpnet-base-v2` embeddings and `FAISS` cosine similarity.
- Selects `k=4` examples with Maximum Marginal Relevance using `lambda=0.5`.
- Uses the PALM-style prompt pattern:

```text
You are going to complete an action sequence. Given the past steps below, predict the next 5 steps.

<Example>
Past steps: step_A, step_B
Future steps: step_C, step_D, step_E, step_F, step_G

<Problem>
Past steps: step_X, step_Y
Future steps:
```

- Generation uses sampling with `top-k=50` and `top-p=0.5`.
- The reported IMPACT adaptation omits caption inputs and operates on symbolic step sequences only.

### Qwen3VL-8B

- Zero-shot vision-language baseline.
- Consumes the raw video clip covering the observed two-step context.
- Generates `Z=5` future step ids without task-specific training.
- Serves as a direct video-to-sequence lower bound alongside feature-based and label-based models.

## Recognition Bottleneck Notes

- The appendix also analyzes two-stage forecasting where LLM baselines receive predicted step labels instead of oracle labels.
- A simple segment-level recognition model is used to estimate that bottleneck.
- Reported findings show that replacing oracle labels with predicted labels substantially degrades downstream forecasting quality, which explains why two-stage LLM pipelines do not yet outperform the visual-only baselines in end-to-end settings.

## Hyperparameter Snapshot

| Setting | `ScalAnt` | `PALM` | `AntGPT-LLM` | `PALM-LLM` | `Qwen3VL-8B` |
| --- | --- | --- | --- | --- | --- |
| Backbone / core model | Mamba | Linear encoder-decoder | `Llama2-7B` | `Llama2-7B` | `Qwen3VL-8B` |
| Fine-tuning | yes | yes | `LoRA` | no | no |
| Input modality | features | features | labels | labels | raw video |
| Hidden dim | `512` | `512` | `4096` | `4096` | `-` |
| LR | `3e-4` | `5e-4` | `5e-4` | `-` | `-` |
| Batch size | `32` | `32` | `4` | `-` | `1` |
| Epochs | `50` | `50` | `10` | `-` | `-` |
| Exemplars | `-` | `-` | `-` | `4 (MMR)` | `-` |

## Release Scope

- This repository currently releases the `AF-L` task documentation only.
- Runnable `AF-L` wrappers and training code are not included in the present code drop.
- The documentation is provided so the repository matches the paper taxonomy and records the reported baseline protocol faithfully.
