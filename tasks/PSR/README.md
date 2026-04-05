# Procedure Step Recognition

This directory contains the IMPACT Procedure Step Recognition benchmark release.

## Path Convention

- Run all commands from the repository root.
- Repository-managed ASR roots: `dataset/ASR/annotations/` and `dataset/ASR/splits_front_only_v1/`
- Repository-managed PSR root: `dataset/PSR/labels_front_only_v1/`
- Repository-managed runtime roots: `logs/psr/` and `outputs/psr/`
- User-managed inputs: `FEATURE_DIR` and, for `storm_psr` evaluation, `VIDEO_DIR`

## Release Structure

- `MS-TCN++ -> PSR` and `VideoMAE v2+Head -> PSR` expose the indirect `ASR -> procedure graph -> PSR` pipelines used in the paper.
- `STORM-PSR` exposes the direct temporal-stream benchmark wrapper.

## Common Arguments

- `SPLIT_ID`: split index such as `1`
- `FEATURE_DIR`: path to extracted front-view features
- `CHECKPOINT_NAME`: upstream checkpoint file name such as `epoch-100.model` or `best_model`
- `RUN_NAME`: released `storm_psr` run identifier
- `VIDEO_DIR`: video directory used only by `storm_psr` temporal-stream evaluation

## Scripts

| Method | Script | Purpose | Usage |
| --- | --- | --- | --- |
| `MS-TCN++ -> PSR` | `tasks/PSR/ms_tcn2/scripts/train_state_model.sh` | train the ASR state model | `bash tasks/PSR/ms_tcn2/scripts/train_state_model.sh <SPLIT_ID> <FEATURE_DIR> [GPU] [NUM_EPOCHS] [BATCH_SIZE] [LR] [ANNOTATION_DIR] [SPLIT_DIR] [LOG_ROOT]` |
| `MS-TCN++ -> PSR` | `tasks/PSR/ms_tcn2/scripts/learn_graph.sh` | learn the procedure graph from the training bundle | `bash tasks/PSR/ms_tcn2/scripts/learn_graph.sh <SPLIT_ID> [OUTPUT_PATH] [ALIAS_MAP] [ANNOTATION_DIR] [SPLIT_DIR] [LOG_ROOT]` |
| `MS-TCN++ -> PSR` | `tasks/PSR/ms_tcn2/scripts/eval_psr.sh` | evaluate PSR on the test bundle | `bash tasks/PSR/ms_tcn2/scripts/eval_psr.sh <SPLIT_ID> <FEATURE_DIR> <CHECKPOINT_NAME> [GPU] [GRAPH_PATH] [ANNOTATION_DIR] [SPLIT_DIR] [ALIAS_MAP] [LOG_ROOT]` |
| `VideoMAE v2+Head -> PSR` | `tasks/PSR/videomae_v2_head/scripts/train_state_model.sh` | train the ASR state model | `bash tasks/PSR/videomae_v2_head/scripts/train_state_model.sh <SPLIT_ID> <FEATURE_DIR> [GPU] [NUM_EPOCHS] [BATCH_SIZE] [LR] [ANNOTATION_DIR] [SPLIT_DIR] [LOG_ROOT]` |
| `VideoMAE v2+Head -> PSR` | `tasks/PSR/videomae_v2_head/scripts/learn_graph.sh` | learn the procedure graph from the training bundle | `bash tasks/PSR/videomae_v2_head/scripts/learn_graph.sh <SPLIT_ID> [OUTPUT_PATH] [ALIAS_MAP] [ANNOTATION_DIR] [SPLIT_DIR] [LOG_ROOT]` |
| `VideoMAE v2+Head -> PSR` | `tasks/PSR/videomae_v2_head/scripts/eval_psr.sh` | evaluate PSR on the test bundle | `bash tasks/PSR/videomae_v2_head/scripts/eval_psr.sh <SPLIT_ID> <FEATURE_DIR> <CHECKPOINT_NAME> [GPU] [GRAPH_PATH] [ANNOTATION_DIR] [SPLIT_DIR] [ALIAS_MAP] [LOG_ROOT]` |
| `STORM-PSR` | `tasks/PSR/storm_psr/scripts/train_split.sh` | train the temporal stream | `bash tasks/PSR/storm_psr/scripts/train_split.sh <SPLIT_ID> <FEATURE_DIR> [GPU] [RUN_NAME] [LABEL_DIR] [SPLIT_DIR] [LOG_ROOT] [CKPT_ROOT] [CFG_PATH]` |
| `STORM-PSR` | `tasks/PSR/storm_psr/scripts/test_split.sh` | run temporal-stream inference | `bash tasks/PSR/storm_psr/scripts/test_split.sh <SPLIT_ID> <RUN_NAME> [CHECKPOINT_NAME] [SUBSET] <FEATURE_DIR> [GPU] [LABEL_DIR] [SPLIT_DIR] [LOG_ROOT]` |
| `STORM-PSR` | `tasks/PSR/storm_psr/scripts/eval_temporal_stream.sh` | evaluate the temporal stream outputs | `bash tasks/PSR/storm_psr/scripts/eval_temporal_stream.sh <RUN_NAME> [CHECKPOINT_NAME] [SUBSET] <VIDEO_DIR> [LABEL_DIR] [LOG_ROOT] [PROCEDURE_INFO]` |

## Examples

```bash
bash tasks/PSR/ms_tcn2/scripts/train_state_model.sh 1 /path/to/IMPACT_i3d_front/features 0
bash tasks/PSR/ms_tcn2/scripts/learn_graph.sh 1
bash tasks/PSR/ms_tcn2/scripts/eval_psr.sh 1 /path/to/IMPACT_i3d_front/features epoch-100.model 0

bash tasks/PSR/videomae_v2_head/scripts/train_state_model.sh 1 /path/to/IMPACT_front/features 0
bash tasks/PSR/videomae_v2_head/scripts/learn_graph.sh 1
bash tasks/PSR/videomae_v2_head/scripts/eval_psr.sh 1 /path/to/IMPACT_front/features epoch-100.model 0

bash tasks/PSR/storm_psr/scripts/train_split.sh 1 /path/to/IMPACT_front/features 0 impact_storm_split1
bash tasks/PSR/storm_psr/scripts/test_split.sh 1 impact_storm_split1 best_model test /path/to/IMPACT_front/features 0
bash tasks/PSR/storm_psr/scripts/eval_temporal_stream.sh impact_storm_split1 best_model test /path/to/videos
```

## Notes

- The released baseline order follows the non-VLM methods described in Sec. 5.1 of the paper.
- The indirect pipelines should be run in order: train state model, learn graph, then evaluate PSR.
- `STORM-PSR` evaluation requires `VIDEO_DIR` because the upstream evaluator builds that path unconditionally.
- `dataset/PSR/labels_front_only_v1/procedure_info_IMPACT.json` is the released PSR procedure metadata.
