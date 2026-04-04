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
1. Pre-training the spatial encoder using [scripts](./pretrained_spatial/)
2. Extract embeddings from the pretrained spatial encoder using [scripts](./pretrained_spatial/compute_embedding_from_dataset.py). 
3. Use the extracted embeddings as dataset. Train the temporal encoder using [scripts](./train_spatial_temporal/).
4. End-to-end fine-tuning the temporal stream using [scripts](./train_spatial_temporal/fine_tuning.py). 


### Job script
To automated the training pipeline, we provided the shell script to automated the training and testing process. Please refer to [STORM-PSR/temporal_stream/job_script](./job_script/). Indicating the path of the config file used to train the model by calling:
```
sh job_script/interactive/train_test_eval_ft_pipeline.sh
```