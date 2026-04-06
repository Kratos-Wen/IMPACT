# AF-S Assets

This directory stores the released protocol assets for the IMPACT short-term action anticipation benchmark (`AF-S`).

Contents:
- `Annotation/Front`, `Left`, `Right`, `Top`: exocentric anticipation annotations
- `Annotation/ego_valid`: released egocentric subset used by the public `AF-S` wrappers
- `Annotation/splits`: official bundle files; the current release provides `split1`

Expected external inputs:
- extracted features for the supervised baselines under a user-managed root such as `features/af_s/{vmae,i3d}/...`
- raw videos for the `Qwen3VL-8B` zero-shot baseline

Stored outside this directory:
- raw videos are distributed through the Google Drive release bundle
- extracted features are distributed through the Google Drive release bundle
- pretrained checkpoints are distributed through the Google Drive release bundle

This subdirectory stores only the lightweight protocol assets required by the benchmark wrappers.

Dataset assets in this directory are covered by [LICENSE-DATA](../../LICENSE-DATA), unless otherwise noted.
