# STORM-PSR on IMPACT

This directory provides the IMPACT PSR benchmark wrapper for `STORM-PSR`.

Current public protocol:
- front-view `split1`
- external front-view embedding directory required

## Train the Temporal Stream

```bash
bash tasks/PSR/storm_psr/scripts/train_split.sh 1 /path/to/IMPACT_front/features 0 impact_storm_split1
```

## Run Inference

```bash
bash tasks/PSR/storm_psr/scripts/test_split.sh 1 impact_storm_split1 best_model test /path/to/IMPACT_front/features 0
```

## Evaluate the Temporal Stream

```bash
bash tasks/PSR/storm_psr/scripts/eval_temporal_stream.sh impact_storm_split1 best_model test /path/to/videos
```

Notes:
- the temporal-stream evaluator requires a `VIDEO_DIR` argument because the upstream evaluator constructs that path unconditionally, even when qualitative videos are not requested
- the released wrapper uses `dataset/PSR/labels_front_only_v1/` and `dataset/ASR/splits_front_only_v1/` by default
- `dataset/PSR/labels_front_only_v1/procedure_info_IMPACT.json` is the released procedure metadata used by the evaluator

Implementation provenance:
- source snapshot: `third_party/storm_psr/`
