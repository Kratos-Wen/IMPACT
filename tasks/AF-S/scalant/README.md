# ScalAnt for AF-S

This directory provides the released IMPACT `AF-S` wrapper for `ScalAnt`.

## Path Convention

- dataset annotations: `dataset/AF-S/Annotation`
- default logs: `logs/af_s/scalant`
- default outputs: `outputs/af_s/scalant`
- source snapshot: `third_party/scalant`
- default model config: `third_party/scalant/configs/sca.json`
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

- `scripts/train_split.sh`: trains `ScalAnt` on one official `AF-S` split
- `scripts/eval_checkpoint.sh`: evaluates one `ScalAnt` checkpoint on one official `AF-S` split

## Examples

```bash
bash tasks/AF-S/scalant/scripts/train_split.sh i3d all /path/to/features 0 split1 afs_scalant_i3d
```

```bash
bash tasks/AF-S/scalant/scripts/eval_checkpoint.sh i3d all /path/to/features /path/to/best.ckpt 0 split1 afs_scalant_eval
```

## Notes

- The public wrapper exposes the paper method name `ScalAnt`; the shared source snapshot keeps the internal model key `sca`.
- Standard output is redirected to `logs/af_s/scalant/`; checkpoints and metrics are written under `outputs/af_s/scalant/`.
