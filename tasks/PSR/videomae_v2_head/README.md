# VideoMAE v2+Head to PSR on IMPACT

This directory provides the released indirect `VideoMAE v2+Head -> PSR` pipeline used for IMPACT PSR.

Pipeline stages:
- train the ASR state predictor
- learn the procedure graph from the training bundle
- evaluate PSR on the test bundle with the learned graph

## Train the State Model

```bash
bash tasks/PSR/videomae_v2_head/scripts/train_state_model.sh 1 /path/to/IMPACT_front/features 0
```

## Learn the Procedure Graph

```bash
bash tasks/PSR/videomae_v2_head/scripts/learn_graph.sh 1
```

## Evaluate PSR

```bash
bash tasks/PSR/videomae_v2_head/scripts/eval_psr.sh 1 /path/to/IMPACT_front/features epoch-100.model 0
```

Defaults:
- ASR annotations: `dataset/ASR/annotations/`
- ASR split assets: `dataset/ASR/splits_front_only_v1/`
- alias map: `tasks/PSR/videomae_v2_head/configs/component_alias.json`
- learned graph: `outputs/psr/videomae_v2_head/procedure_graph_split1.json`

Implementation provenance:
- source snapshot: `third_party/asr_psr_experiment/`
