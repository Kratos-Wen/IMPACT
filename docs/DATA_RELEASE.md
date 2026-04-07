# Data Release Layout

This document summarizes the current Google Drive release layout for IMPACT.

## Public Bundles

| Path | Approx. size | Contents |
| --- | ---: | --- |
| `annotations_v1.zip` | `5.4M` | official annotation release bundle |
| `videos/videos_ego.zip` | part of `132G` | egocentric RGB videos |
| `videos/videos_front.zip` | part of `132G` | front-view RGB videos |
| `videos/videos_left.zip` | part of `132G` | left-view RGB videos |
| `videos/videos_right.zip` | part of `132G` | right-view RGB videos |
| `videos/videos_top.zip` | part of `132G` | top-view RGB videos |
| `depth/depth_front.zip` | part of `46G` | front-view depth streams |
| `depth/depth_left.zip` | part of `46G` | left-view depth streams |
| `depth/depth_right.zip` | part of `46G` | right-view depth streams |
| `depth/depth_top.zip` | part of `46G` | top-view depth streams |
| `audio/audio_ego.zip` | `200M` | egocentric audio bundle |
| `features/features_I3D.zip` | part of `71G` | released I3D feature bundle |
| `features/features_MViTv2.zip` | part of `71G` | released MViTv2 feature bundle |
| `features/features_VideoMAEv2.zip` | part of `71G` | released VideoMAEv2 feature bundle |
| `sample/` | `3.5G` | quick-start subset for download verification, reviewer inspection, and lightweight debugging |

## Sample Subset

The sample subset currently contains:
- 3 executions
- 15 RGB video files
- 12 depth files
- 3 ego-audio files
- 45 feature files
- task-specific annotation folders covering `TAS-S`, `TAS-B`, `ASR`, `PSR`, `PPR`, and `ATR`

Selected sample executions:
- `AL07EJ17_Disassembly_A_001`
- `AL07EJ17_Reassembly_A_002`
- `ER07AD15_Disassembly_A_001`

Sample layout:

```text
sample/
├── annotations/
│   ├── ASR/
│   ├── ATR/
│   ├── PPR/
│   ├── PSR/
│   ├── TAS-B/
│   └── TAS-S/
├── audio/ego/
├── depth/{front,left,right,top}/
├── features/{I3D,MViTv2,VideoMAEv2}/
├── metadata/
└── videos/{ego,front,left,right,top}/
```

## Notes

- The repository tree stores lightweight protocol assets under `dataset/`; the larger Google Drive bundles are distributed separately.
- The description above reflects the current public Google Drive release layout.
- A public Hugging Face mirror is available at `https://huggingface.co/datasets/KratosWen/IMPACT`.
- The Hugging Face mirror follows the public IMPACT release layout.
