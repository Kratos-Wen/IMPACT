# VideoMAE v2+Head for PSR

This directory provides the released indirect `VideoMAE v2+Head -> PSR` pipeline for IMPACT.

## Path Convention

- ASR annotations: `dataset/ASR/annotations`
- ASR split assets: `dataset/ASR/splits_front_only_v1`
- alias map: `tasks/PSR/videomae_v2_head/configs/component_alias.json`
- default procedure graph: `outputs/psr/videomae_v2_head/procedure_graph_split1.json`
- default logs: `logs/psr/videomae_v2_head`
- source snapshot: `third_party/asr_psr_experiment`

## Common Arguments

- `SPLIT_ID`: released split id, currently front-view `split1`
- `FEATURE_DIR`: required external feature directory
- `GPU`: CUDA device used by state-model training or evaluation
- `NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`: optional training hyperparameters
- `CHECKPOINT_NAME`: checkpoint name passed to the upstream evaluator
- `GRAPH_PATH`: output or input procedure graph path
- `ALIAS_MAP`: component alias file used by graph learning and PSR evaluation
- `ANNOTATION_DIR`, `SPLIT_DIR`, `LOG_BASE`: optional runtime directory overrides

## Scripts

- `scripts/train_state_model.sh`: trains the front-view state recognition model
- `scripts/learn_graph.sh`: learns a procedure graph from the training bundle
- `scripts/eval_psr.sh`: evaluates PSR on the test bundle with the learned graph

## Examples

```bash
bash tasks/PSR/videomae_v2_head/scripts/train_state_model.sh 1 /path/to/features_videomaev2_front 0
```

```bash
bash tasks/PSR/videomae_v2_head/scripts/learn_graph.sh 1
```

```bash
bash tasks/PSR/videomae_v2_head/scripts/eval_psr.sh 1 /path/to/features_videomaev2_front epoch-100.model 0
```

## Notes

- Run the pipeline in order: train the state model, learn the procedure graph, then evaluate PSR.
