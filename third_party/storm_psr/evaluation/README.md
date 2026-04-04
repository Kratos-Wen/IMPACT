# Evaluate Procedure Step Recognition task

## Installation 

```
$ git clone [redacted, url of this repo in the public github]
$ cd STORM-PSR/PSR_evaluation
$ conda create -n storm-psr python=3.9 -y
$ conda activate storm-psr
$ pip install -r requirements.txt
$ pip install git+https://github.com/infoscout/weighted-levenshtein.git#egg=weighted_levenshtein
```
> The weighted-levenshtein gives an error when installing requirements.txt file, therefore it is installed manually from their [GitHub repo](https://github.com/infoscout/weighted-levenshtein).

## Usage
The STORM-PSR model consist of two stream model, object detection stream and temporal stream. For different models, there are different script to evaluate the model due to different output of the model. The predictions are provided at [prediction folder](./predictions/). Therefore, one can run the PSR code on CPU, since there are no new predictions from computationally expensive operations.

To evaluate PSR task on specific task, the `procedure_info.json` of the dataset is need (see [here](./utils/procedure_info_IndustReal.json)). To evaluate the STORM-PSR stream, you need predictions from both object detection stream and temporal stream. We provided the predictions in the paper. 

The following command can be used to evalute models (default hyperparameters can be found in our paper):
```
python evaluate_STORM_PSR.py --run_path_temporal_stream RUN_PATH_TEMPORAL_STREAM\
                           --run_path_obj_stream RUN_PATH_OBJ_STREAM\
                           --rec_path REC_PATH --psr_label_path PSR_LABEL_PATH\
                           --video_dir VIDEO_DIR\
                           --procedure_info PROCEDURE_INFO\
                           --split SPLIT\
                           --checkpoint_tmp CHECKPOINT_TMP\
                           [--conf_threshold CONF_THRESHOLD]\
                           [--cum_conf_threshold CUM_CONF_THRESHOLD]\
                           [--cum_decay CUM_DECAY]\
                           [--soft_voting_w SOFT_VOTING_W] \
                           [--num_dig_psr NUM_DIG_PSR]\
                           [--temporal_win TEMPORAL_WIN] \
                           [--create_video]\
                           [--create_PSR_plot] \
                           [--FPS FPS]\
                           [--width WIDTH]\
                           [--height HEIGHT]\
                           [--bbox]

```

