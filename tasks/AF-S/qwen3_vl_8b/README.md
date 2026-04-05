# Qwen3VL-8B for AF-S

This directory provides the released IMPACT `AF-S` wrapper for `Qwen3VL-8B`.

## Path Convention

- dataset annotations: `dataset/AF-S/Annotation`
- default logs: `logs/af_s/qwen3_vl_8b`
- default outputs: `outputs/af_s/qwen3_vl_8b`
- source snapshot: `third_party/qwen3_vl_8b`
- IMPACT-specific changes: `tasks/AF-S/UPSTREAM_DIFF.md`

## Common Arguments

- `VIDEO_ROOT`: raw video root
- `VIEW`: `front`, `left`, `right`, `top`, `ego`, `all`, or `ego-exclude`
- `SPLIT`: `train`, `val`, or `test`; benchmark evaluation uses `test`
- `GPU_LIST`: comma-separated GPU ids for `torchrun`
- `SPLIT_NAME`: official split id; default `split1`
- `RUN_TAG`: run identifier appended to logs and outputs
- `ANNOTATION_DIR`: optional dataset override

## Scripts

- `scripts/run_eval.sh`: runs zero-shot `Qwen3VL-8B` evaluation on one official `AF-S` split

## Examples

```bash
bash tasks/AF-S/qwen3_vl_8b/scripts/run_eval.sh /path/to/videos ego test 0 afs_qwen3vl_eval split1
```

## Notes

- The wrapper launches `test_qwen_anticipation.py` with `torchrun`; a single GPU is sufficient when `GPU_LIST` contains one id.
- The shared source snapshot expects `transformers`, `qwen-vl-utils`, and the corresponding video decoding dependencies to be installed in the runtime environment.
