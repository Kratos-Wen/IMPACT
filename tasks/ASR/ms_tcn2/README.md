# MS-TCN++ for ASR

This directory provides the released IMPACT `ASR` wrapper for `MS-TCN++`.

## Path Convention

- dataset root: `dataset/ASR`
- split assets: `dataset/ASR/splits_front_only_v1`
- default logs: `logs/asr/ms_tcn2`
- source snapshot: `third_party/ms_tcn2`

## Common Arguments

- `SPLIT_ID`: released split id, currently front-view `split1`
- `FEATURE_DIR`: required external feature directory
- `GPU`: CUDA device used by training or evaluation
- `NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`: optional training hyperparameters
- `ANNOTATION_DIR`: override for `dataset/ASR/annotations`
- `SPLIT_DIR`: override for `dataset/ASR/splits_front_only_v1`
- `CHECKPOINT_NAME`: checkpoint name passed to the upstream evaluator
- `LOG_BASE`: optional log directory override

## Scripts

- `scripts/train_split.sh`: trains the state recognition model on the requested split
- `scripts/eval_checkpoint.sh`: evaluates a checkpoint on the released test bundle

## Examples

```bash
bash tasks/ASR/ms_tcn2/scripts/train_split.sh 1 /path/to/features_i3d_front 0
```

```bash
bash tasks/ASR/ms_tcn2/scripts/eval_checkpoint.sh 1 /path/to/features_i3d_front epoch-100.model 0
```

## Notes

- The public release covers the front-view protocol and expects external features.
- `CHECKPOINT_NAME` is resolved by the upstream code under the corresponding `ms_tcn2` model directory.
