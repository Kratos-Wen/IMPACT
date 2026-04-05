# FACT

This directory bundles the upstream `FACT` repository snapshot used by the IMPACT benchmark release.

Reference method:
- paper: `FACT: Frame-Action Cross-Attention Temporal Modeling for Efficient Action Segmentation` (CVPR 2024)
- upstream repository: `https://github.com/ZijiaLewisLu/CVPR2024-FACT`

Snapshot layout:
- the bundled upstream source tree lives under `third_party/fact/src/`
- the upstream `README.md` is kept at `third_party/fact/src/README.md`
- runnable IMPACT benchmark entrypoints live under `tasks/TAS/fact/`, `tasks/PPR/fact/`, and `tasks/ATR/fact/`

Licensing:
- the bundled upstream license is provided in `third_party/fact/LICENSE`
