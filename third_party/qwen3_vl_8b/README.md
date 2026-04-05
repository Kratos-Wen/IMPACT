# Qwen3VL-8B

This directory contains the IMPACT-maintained zero-shot evaluation snapshot used by the released `AF-S` wrapper for `Qwen3-VL-8B-Instruct`.

Reference model:
- technical report: `Qwen3-VL Technical Report` (2025)
- model card: `https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct`
- model card license: `Apache-2.0`

Snapshot status:
- this directory does not bundle the upstream model weights
- the included file is the IMPACT evaluation harness that calls the public `Qwen3-VL-8B-Instruct` model through `transformers`
- runnable benchmark entrypoints live under `tasks/AF-S/qwen3_vl_8b/`

Licensing:
- repository-authored files in this snapshot follow the repository root `LICENSE`
- the referenced model weights and configuration remain subject to the upstream model-card terms
