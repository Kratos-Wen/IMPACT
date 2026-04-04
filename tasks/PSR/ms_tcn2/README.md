# MS-TCN++ to PSR on IMPACT

This directory provides the released indirect `MS-TCN++ -> PSR` pipeline used for IMPACT PSR.

Pipeline stages:
- train the ASR state predictor
- learn the procedure graph from the training bundle
- evaluate PSR on the test bundle with the learned graph

## Train the State Model

```bash
bash tasks/PSR/ms_tcn2/scripts/train_state_model.sh 1 /path/to/IMPACT_i3d_front/features 0
```

## Learn the Procedure Graph

```bash
bash tasks/PSR/ms_tcn2/scripts/learn_graph.sh 1
```

## Evaluate PSR

```bash
bash tasks/PSR/ms_tcn2/scripts/eval_psr.sh 1 /path/to/IMPACT_i3d_front/features epoch-100.model 0
```

Defaults:
- ASR annotations: `dataset/ASR/annotations/`
- ASR split assets: `dataset/ASR/splits_front_only_v1/`
- alias map: `tasks/PSR/ms_tcn2/configs/component_alias.json`
- learned graph: `outputs/psr/ms_tcn2/procedure_graph_split1.json`

Implementation provenance:
- source snapshot: `third_party/ASR-PSR-Experiment/`
