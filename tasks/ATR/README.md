# Anomaly Type Recognition

This directory contains the IMPACT Anomaly Type Recognition benchmark release.

Included reference implementations:
- `ltcontext`
- `fact`

Benchmarks defined in the paper:
- `ATR-L`
- `ATR-R`

Launcher mapping used in this repository:
- `ATR_L` for `ATR-L`
- `ATR_R` for `ATR-R`

Release note:
- `ltcontext` exposes train and standalone checkpoint evaluation entrypoints
- `fact` exposes the stable training entrypoint; the bundled standalone checkpoint helper is not released for ATR because it explicitly excludes ATR scoring
