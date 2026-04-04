# Procedure Step Recognition

This directory contains the IMPACT Procedure Step Recognition benchmark release.

Included reference implementations:
- `ms_tcn2`
- `videomae_v2_head`
- `storm_psr`

Release structure:
- `ms_tcn2` and `videomae_v2_head` expose the indirect `ASR -> procedure graph -> PSR` pipelines used in the paper
- `storm_psr` exposes the direct temporal-stream benchmark wrapper

Released protocol assets:
- front-view ASR annotations and split files in `dataset/ASR/`
- front-view PSR labels and procedure metadata in `dataset/PSR/`
