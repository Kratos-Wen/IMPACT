# VideoMAE v2+Head

This directory contains the IMPACT-maintained method snapshot used by the released `ASR` and indirect `PSR` wrappers for `VideoMAE v2+Head`.

Reference method:
- paper: `VideoMAE V2: Scaling Video Masked Autoencoders With Dual Masking` (CVPR 2023)
- reference repository: `https://github.com/OpenGVLab/VideoMAEv2`

Snapshot status:
- this is not the upstream `VideoMAEv2` repository
- the snapshot contains the IMPACT benchmark head and training logic built on frozen `VideoMAE V2` features
- runnable benchmark entrypoints live under `tasks/ASR/videomae_v2_head/` and `tasks/PSR/videomae_v2_head/`

Licensing:
- the bundled license for this snapshot is provided in `LICENSE`
