## Pre-train the Spatail encoder in Weakly-supervised Contrastive Learning Setting
## Installation 

```
$ git clone [To be add, url of this repo in the public github]
$ cd STROM-PSR/temporal_stream/train_sptial_temporal
$ conda create -n storm-psr python=3.9 -y
$ conda activate storm-psr
$ pip install -r storm-psr/temporal_stream/requirements.txt
```

## Usage
In this folder, there are two stages to train the model. First, train the temporal encoder. Second, end-to-end train the sptai and temporal encoder. 
### Train the temporal encoder
The temporal encoder learn the temporal context of the frame-level embeddings in the videos. So to train the tempoarl encoder, we need the embeddings of the video dataset, this can be generate by the pre-trained spatial encoder that trained in the `STORM-PSR/temporal_stream/pretrained_spaital` [folder](../pretrained_spatial). The following command can be used to train the spatial model (default hyperparameters can be found in our paper):
```
python train.py --data_dir DATA_DIR\
                --psr_label_path PSR_LABEL_PATH\
                --log_path LOG_PATH\
                --ckpt_dir CKPT_DIR\
                --run_name RUN_NAME\
                --config CONFIG\
                --dtype DTYPE\
                [--resume RESUME]\
                [--parallel]\
                [--job_file_mode]\
                [--baseline]\
                [--skip_factor SKIP_FACTOR]\
                [--lr LR] \
                [--scheduler SCHEDULER]\
                [--T_0 T_0]\
                [--lr_gamma LR_GAMMA]\
                [--lr_step LR_STEP]\
                [--warmup WARMUP]\
                [--weight_decay WEIGHT_DECAY]\
                [--epochs EPOCHS] \
                [--batch_size BATCH_SIZE]\
                [--workers WORKERS]\
                [--warmup_rate WARMUP_RATE]\
                [--sampling_strategy SAMPLING_STRATEGY]\
                [--n_iter N_ITER]\
                [--exe_mode EXE_MODE]\
                [--tmp_pretrained TMP_PRETRAINED]
```
### End-to-end fine-tuning the temproal stream model
We can also end-to-end trained the spatial encoder and temporal encoder. Here, we train the model using the procedural assembly video dataset. The following command can be used to end-to-end train the temporal stream.
```
python fine_tuning.py --data_dir DATA_DIR\
                      --psr_label_path PSR_LABEL_PATH\
                      --log_path LOG_PATH\
                      --run_name RUN_NAME\
                      --config CONFIG\
                      --dtype DTYPE\
                      [--resume RESUME]\
                      [--parallel]\
                      [--job_file_mode]\
                      [--baseline]\
                      [--sanity_check]\
                      [--pretrained_weight PRETRAINED_WEIGHT]\
                      [--spatial_pretrained_weight SPATIAL_PRETRAINED_WEIGHT]\
                      [--lr LR]\
                      [--scheduler SCHEDULER]\
                      [--T_0 T_0]\
                      [--lr_gamma LR_GAMMA]\
                      [--lr_step LR_STEP]\
                      [--warmup WARMUP]\
                      [--weight_decay WEIGHT_DECAY]
                      [--epochs EPOCHS]\
                      [--batch_size BATCH_SIZE]\
                      [--workers WORKERS]\
                      [--warmup_rate WARMUP_RATE]\
                      [--sampling_strategy SAMPLING_STRATEGY]\
                      [--skip_factor SKIP_FACTOR]
```
### Run inference 
When inference, the whole video would be split into consecutive video clips with fixed number (N) of frames. That is, the first forwarded video clip is from t=0 ~ N, the second is t=1 ~ N+1. The model would predict the action that completed in the input video clip. (It is many-to-one recurrent model.)  

The following command can be used to end-to-end train the temporal stream.
```
python test.py --run_name RUN_NAME\
                --checkpoint CHECKPOINT\
                --split SPLIT\
                --psr_label_path PSR_LABEL_PATH \
                --data_dir DATA_DIR\
                --csv_dir CSV_DIR\
                --log_path LOG_PATH\
                --dtype DTYPE\
                [--spatial_pretrained_weight SPATIAL_PRETRAINED_WEIGHT]\
                [--asd_label ASD_LABEL]\
                [--baseline]\
                [--skip_factor SKIP_FACTOR]
```
We can run the test on:
* Temporal encoder only: by specificiying `--dtype = embedding` and give the path to embedding dataset of the video: `--csv_dir = {path to embeddings}`.
* The whole temporal stream:  by specificiying `--dtype = video` and give the path to video dataset of the video: `--csv_dir = {path to video dataset}`.

### Job script
To automated the training pipeline, we provided the shell script to automated the training and testing process. Please refer to [STORM-PSR/temporal_stream/train_spatial_temporal/job_script/interactive](./job_script/interactive/). Indicating the path of the config file used to train the model by calling
```
sh job_script/interactive/train_temporal_enc_industreal.sh
sh job_script/interactive/fine_tune_temporal_Stream_industreal.sh
```