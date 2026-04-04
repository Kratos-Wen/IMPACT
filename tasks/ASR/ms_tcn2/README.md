# MS-TCN++ on IMPACT ASR

This directory provides the IMPACT ASR benchmark wrapper for `MS-TCN++`.

Current public protocol:
- front-view `split1`
- external feature directory required

## Train

```bash
bash tasks/ASR/ms_tcn2/scripts/train_split.sh 1 /path/to/IMPACT_i3d_front/features 0
```

## Evaluate a Checkpoint

```bash
bash tasks/ASR/ms_tcn2/scripts/eval_checkpoint.sh 1 /path/to/IMPACT_i3d_front/features epoch-100.model 0
```

Notes:
- the evaluation script expects a checkpoint name inside `third_party/asr_psr_experiment/models/ms_tcn2/split_1/`
- the released wrapper uses `dataset/ASR/annotations/` and `dataset/ASR/splits_front_only_v1/` by default
- runtime artifacts created by the upstream code remain excluded from version control

Implementation provenance:
- source snapshot: `third_party/asr_psr_experiment/`
