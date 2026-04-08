## Temporal Stream model
## Installation 

```
$ git clone [Redacted, url of this repo in the public github]
$ cd STORM-PSR/temporal_stream
$ conda create -n storm-psr python=3.9 -y
$ conda activate storm-psr
$ pip install -r storm-psr/temporal_stream/requirements.txt
```

## Usage
The general training procedures of the temporal stream model is:
1. Pre-training the spatial encoder using the upstream `pretrained_spatial/` pipeline. That directory is not bundled in this IMPACT release snapshot.
2. Extracting embeddings from the pretrained spatial encoder using the corresponding upstream extraction script. That script is also not bundled in this snapshot.
3. Use the extracted embeddings as dataset. Train the temporal encoder using [scripts](./train_spatial_temporal/).
4. End-to-end fine-tuning the temporal stream using `train_spatial_temporal/fine_tuning.py`. 


### Job script
To automated the training pipeline, the original upstream repository provides `job_script/` helpers. That directory is not bundled in this IMPACT release snapshot. The training pipeline is invoked in the upstream README as:
```
sh job_script/interactive/train_test_eval_ft_pipeline.sh
```
