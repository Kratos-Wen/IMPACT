# AF-S Assets

This directory stores the released protocol assets for the IMPACT short-term action anticipation benchmark (`AF-S`).

Contents:
- `Annotation/Front`, `Left`, `Right`, `Top`: exocentric anticipation annotations
- `Annotation/ego_valid`: released egocentric subset used by the public `AF-S` wrappers
- `Annotation/splits`: official bundle files; the current release provides `split1`

Expected external inputs:
- extracted features for the supervised baselines under a user-managed root such as `features/af_s/{vmae,i3d}/...`
- raw videos for the `Qwen3VL-8B` zero-shot baseline

Not included:
- raw videos
- extracted features
- pretrained checkpoints
- model predictions

Dataset assets in this directory are covered by [LICENSE-DATA](../../LICENSE-DATA), unless otherwise noted.
