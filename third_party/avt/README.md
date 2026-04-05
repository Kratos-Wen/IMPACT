# AVT

This directory contains the IMPACT-maintained source snapshot used by the released `AF-S` wrapper for `AVT`.

Reference method:
- paper: `Anticipative Video Transformer` (ICCV 2021)
- reference repository: `https://github.com/facebookresearch/AVT`
- reference repository license: `Apache-2.0`

Snapshot status:
- this is not a verbatim mirror of the upstream `AVT` repository
- the core model definition in `models/avt.py` is adapted from the official implementation
- the data pipeline, training entrypoints, and IMPACT-specific evaluation logic in this directory are part of the IMPACT benchmark release
- runnable benchmark entrypoints live under `tasks/AF-S/avt/`

Licensing:
- repository-authored files in this snapshot follow the repository root `LICENSE`
- adapted components retain source attribution in file-level comments
