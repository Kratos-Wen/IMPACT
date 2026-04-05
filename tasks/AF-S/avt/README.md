# AVT for AF-S

This directory provides the released IMPACT `AF-S` wrapper for `AVT`.

## Path Convention

- dataset annotations: `dataset/AF-S/Annotation`
- default logs: `logs/af_s/avt`
- default outputs: `outputs/af_s/avt`
- source snapshot: `third_party/avt`
- default model config: `third_party/avt/configs/avt.json`
- IMPACT-specific changes: `tasks/AF-S/UPSTREAM_DIFF.md`

## Common Arguments

- `FEATURE_TYPE`: `vmae` or `i3d`
- `VIEW`: `front`, `left`, `right`, `top`, `ego`, `all`, or `ego-exclude`
- `FEATURE_DIR`: external feature root
- `GPU_LIST`: comma-separated GPU ids for training
- `GPU`: CUDA device used for checkpoint evaluation
- `SPLIT_NAME`: official split id; default `split1`
- `RUN_TAG`: run identifier appended to logs and outputs
- `ANNOTATION_DIR`: optional dataset override
- `CKPT_PATH`: checkpoint file or directory accepted by `test_lightning.py`

## Scripts

- `scripts/train_split.sh`: trains `AVT` on one official `AF-S` split
- `scripts/eval_checkpoint.sh`: evaluates one `AVT` checkpoint on one official `AF-S` split

## Examples

```bash
bash tasks/AF-S/avt/scripts/train_split.sh vmae ego-exclude /path/to/features 0 split1 afs_avt_vmae
```

```bash
bash tasks/AF-S/avt/scripts/eval_checkpoint.sh vmae ego-exclude /path/to/features /path/to/best.ckpt 0 split1 afs_avt_eval
```

## Notes

- The wrapper uses the released `AVT` source snapshot in `third_party/avt`.
- Standard output is redirected to `logs/af_s/avt/`; checkpoints and metrics are written under `outputs/af_s/avt/`.
