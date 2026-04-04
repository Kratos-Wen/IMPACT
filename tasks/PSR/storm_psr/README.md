# STORM-PSR for PSR

This directory provides the released direct `STORM-PSR` pipeline for IMPACT `PSR`.

## Path Convention

- PSR labels: `dataset/PSR/labels_front_only_v1`
- ASR split assets: `dataset/ASR/splits_front_only_v1`
- default checkpoints: `outputs/psr/storm_psr/checkpoints`
- default run logs: `logs/psr/storm_psr/runs`
- source snapshot: `third_party/storm_psr`

## Common Arguments

- `SPLIT_ID`: released split id, currently front-view `split1`
- `FEATURE_DIR`: required external front-view embedding directory
- `GPU`: CUDA device used by training or inference
- `RUN_NAME`: run identifier shared across training, inference, and evaluation
- `CHECKPOINT_NAME`: checkpoint name, usually `best_model`
- `SUBSET`: evaluation subset such as `test`
- `VIDEO_DIR`: video root required by the upstream evaluator
- `LABEL_DIR`, `SPLIT_DIR`, `LOG_ROOT`, `CKPT_ROOT`, `CFG_PATH`, `PROCEDURE_INFO`: optional runtime overrides

## Scripts

- `scripts/train_split.sh`: trains the temporal stream
- `scripts/test_split.sh`: runs temporal-stream inference for a saved run
- `scripts/eval_temporal_stream.sh`: computes PSR metrics for an inferred run

## Examples

```bash
bash tasks/PSR/storm_psr/scripts/train_split.sh 1 /path/to/features_front 0 impact_storm_split1
```

```bash
bash tasks/PSR/storm_psr/scripts/test_split.sh 1 impact_storm_split1 best_model test /path/to/features_front 0
```

```bash
bash tasks/PSR/storm_psr/scripts/eval_temporal_stream.sh impact_storm_split1 best_model test /path/to/videos
```

## Notes

- `VIDEO_DIR` is still required because the upstream evaluator constructs that path unconditionally, even when qualitative videos are not requested.
