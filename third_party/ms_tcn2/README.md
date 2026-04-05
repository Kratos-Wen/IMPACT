# MS-TCN++

This directory contains the IMPACT-maintained method snapshot used by the released `ASR` and indirect `PSR` wrappers for `MS-TCN++`.

Reference method:
- paper: `MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation` (TPAMI 2020)
- reference repository: `https://github.com/sj-li/MS-TCN2`

Snapshot status:
- this is not a verbatim mirror of the original `MS-TCN2` repository
- the snapshot keeps the MS-TCN++ training and evaluation structure while adding the IMPACT-specific state-recognition and procedure-graph utilities required by `ASR` and `PSR`
- runnable benchmark entrypoints live under `tasks/ASR/ms_tcn2/` and `tasks/PSR/ms_tcn2/`

Licensing:
- the bundled license for this snapshot is provided in `LICENSE`
