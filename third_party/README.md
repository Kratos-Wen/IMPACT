# Third-Party Implementations

This directory contains the method code used by the IMPACT benchmark release.

The contents are intentionally split into two categories:
- bundled upstream snapshots: original external repositories copied into the IMPACT release with their upstream documentation and licenses where available
- IMPACT-maintained method snapshots: method-specific source trees prepared for the IMPACT benchmark, together with explicit references to the corresponding external papers or model cards

## Provenance Index

| Directory | Snapshot type | External reference | License status in this release |
| --- | --- | --- | --- |
| `asquery` | bundled upstream snapshot | `ASQuery: A Query-Based Model for Action Segmentation` (ICME 2024), upstream repo: `https://github.com/zlngan/ASQuery` | no `LICENSE` file is bundled in the upstream snapshot; see `third_party/asquery/README.md` |
| `diffact` | bundled upstream snapshot | `Diffusion Action Segmentation` (ICCV 2023) | `third_party/diffact/LICENSE` |
| `fact` | bundled upstream snapshot | `FACT: Frame-Action Cross-Attention Temporal Modeling for Efficient Action Segmentation` (CVPR 2024), upstream repo: `https://github.com/ZijiaLewisLu/CVPR2024-FACT` | `third_party/fact/LICENSE` |
| `ltcontext` | bundled upstream snapshot | `How Much Temporal Long-Term Context Is Needed for Action Segmentation?` (ICCV 2023) | `third_party/ltcontext/LICENSE` |
| `storm_psr` | bundled upstream snapshot | `Learning to Recognize Correctly Completed Procedure Steps in Egocentric Assembly Videos through Spatio-Temporal Modeling` (CVIU 2025), upstream repo: `https://github.com/shaohsuanhung/STORM-PSR` | `third_party/storm_psr/LICENSE` |
| `ms_tcn2` | IMPACT-maintained method snapshot | `MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation` (TPAMI 2020), reference repo: `https://github.com/sj-li/MS-TCN2` | `third_party/ms_tcn2/LICENSE` |
| `videomae_v2_head` | IMPACT-maintained method snapshot | `VideoMAE V2: Scaling Video Masked Autoencoders With Dual Masking` (CVPR 2023), reference repo: `https://github.com/OpenGVLab/VideoMAEv2` | `third_party/videomae_v2_head/LICENSE` |
| `avt` | IMPACT-maintained method snapshot | `Anticipative Video Transformer` (ICCV 2021), reference repo: `https://github.com/facebookresearch/AVT` | repository-level `LICENSE` applies to repository-authored files in this snapshot; adapted components retain source attribution in file headers |
| `scalant` | IMPACT-maintained method snapshot | `Scalable Video Action Anticipation with Cross Linear Attentive Memory` (WACV 2026) | repository-level `LICENSE` applies to repository-authored files in this snapshot; adapted components retain source attribution in file headers |
| `qwen3_vl_8b` | IMPACT-maintained evaluation snapshot | `Qwen3-VL Technical Report` (2025), model card: `https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct` | repository-level `LICENSE` applies to the evaluator code; model weights are not bundled |

## Task Mapping

- `ltcontext`, `diffact`, `asquery`, and `fact` back the released `TAS`, `PPR`, and `ATR` wrappers
- `ms_tcn2` backs the released `ASR` and indirect `PSR` wrappers for `MS-TCN++`
- `videomae_v2_head` backs the released `ASR` and indirect `PSR` wrappers for `VideoMAE v2+Head`
- `storm_psr` backs the released direct `PSR` temporal-stream wrapper
- `avt`, `scalant`, and `qwen3_vl_8b` back the released `AF-S` wrappers

Repository-authored benchmark wrappers live under `tasks/`.

For the TAS methods, repository-specific changes relative to upstream are summarized in the method-local `UPSTREAM_DIFF.md` files under `tasks/TAS/`.
