#%%
"""
Please use this to extract embeddings for the second stage training. 
"""
import argparse
import torch
import torch.utils.data
import numpy as np
import time
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_metric_learning.distances import LpDistance, CosineSimilarity
from pytorch_metric_learning.utils.inference import CustomKNN
import yaml
from argparse import Namespace
from model import ContrastiveModel
from PIL import Image
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
from utils import  get_recording_list
import multiprocessing
from video_dataset_action_label import testtime_dataset
import torchvision.transforms.functional as f
import torchvision
import time
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_image_multi(idx, image_path, size=(224, 224)):
    img = Image.open(image_path).convert("RGB")

    img = img.resize(size)

    return idx, torchvision.transforms.functional.pil_to_tensor(img).float() / 255.0


def set_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, help='Path to the run directory, e.g. ./runs/run_name')
    parser.add_argument("--checkpoint", type=str, default=None, help='Name of the checkpoint to be tested')
    parser.add_argument("--config",type=str,default='configs/STORM_F65-dim128.yaml',help='configuration of the mdoel')
    parser.add_argument("--loss",type=str,default='SupCon',help='type of loss used to train model. Either "ce" or "SupCon"')
    parser.add_argument("--data_path",type=str,default=None,help='location of the dataset')
    parser.add_argument("--psr_label_path",type=str,default=None,help='location of the psr labels')
    parser.add_argument("--log_path",type=str,default=None,help='save path of the output embedding')
    parser.add_argument("--num_workers",type=int,default=None,help='Number of workers for datalodaer')
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()

    return args

if __name__ == "__main__":
    #-- Directory setting
    args = set_options()
    run_name = args.run_name
    ckpt_name = args.checkpoint
    save_dir = Path(args.log_path) / run_name / 'embeddings'/ckpt_name
    save_dir.mkdir(parents=True,exist_ok=True)
    weights_dir = Path(args.log_path) / run_name / 'checkpoints' / ckpt_name
    #-- Load model and only use the spatial enc. 
    with open(args.config) as f:
        cfg = yaml.load(f,Loader=yaml.FullLoader)

    cfg = Namespace(**cfg)
    spatial_args = Namespace(**cfg.spatial_args)
    if args.loss == 'ce':
        spatial_enc = ContrastiveModel(spatial_args, weights_dir=None,classifier=False)
    elif args.loss =='SupCon':
        spatial_enc = ContrastiveModel(spatial_args, weights_dir=None,classifier=False)
    else:
        raise NotImplementedError(f"The type of loss is not implemented. Expect 'SupCon' or  'ce' loss.")
    

    spatial_enc.load_weights_encoder_from_ckpt(weights_dir)
    
    # spatial_enc.load_weights_encoder(weights_dir)
    spatial_enc.use_projection_head(False)  # Use spatial feature extractor only 
    spatial_enc.eval()
    spatial_enc.to(DEVICE)
    
    #TODO: Need further implementation for more datasets
    #-- preprocessing
    if 'industreal' in str(args.data_path).lower():
        means = [0.608, 0.545, 0.520]
        stds = [0.172, 0.197, 0.188] # Normalization for IndustReal 
        print("Run dataset:\t IndustReal\n")
        print("Mean: [0.608, 0.545, 0.520], std: [0.172, 0.197, 0.188].")
    elif 'meccano' in str(args.data_path).lower():
        means = [0.4144,0.4014,0.3777]
        stds = [0.2312,0.2458,0.2684] # Normalization for MECCANO
        print("Run dataset:\t MECCANO\n")
        print("Mean: [0.4144,0.4014,0.3777], std: [0.2312,0.2458,0.2684].")
    else:
        raise NotImplementedError(f"Currently only support industreal and meccano, but get {args.data_path}")
    
    preprocess = transforms.Compose([transforms.Normalize(mean=means, std=stds),])
    
    #-- glob get recordings list for train / val / test set. 
    train_recording = get_recording_list(Path(args.data_path), train=True)
    val_recording = get_recording_list(Path(args.data_path),val = True)
    test_recording = get_recording_list(Path(args.data_path),test = True)

    if 'industreal' in str(args.data_path).lower():
        train_recording = [rec / 'rgb' for rec in train_recording]
        val_recording   = [rec / 'rgb' for rec in val_recording]
        test_recording  = [rec / 'rgb' for rec in test_recording]
    
    recordings = [train_recording,val_recording,test_recording]
    split = ['train','val','test']

    #-- For each recording read frames
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for set in split:
        dataset = testtime_dataset(Path(args.data_path),set,preprocess=preprocess)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
        progress = tqdm(enumerate(loader),total=len(loader))
        #
        preds = np.zeros((len(dataset.df[dataset.df['FileName'] == dataset.prev_vid_name]), spatial_enc.embed_dim))
        prev_vid_name = dataset.prev_vid_name
        total_len = len(dataset.df[dataset.df['FileName'] == dataset.df.loc[0]['FileName']])
        frame_ID = 0
        #
        for i, img in progress:
            curr_vid_name = dataset.df.loc[i]['FileName']
            
            with torch.no_grad(): 
                embedding = spatial_enc(img.to(DEVICE))
            preds[frame_ID, :] = embedding.detach().cpu().numpy()
            progress.set_description(f"Current video:{dataset.df.loc[i]['FileName']}, Frame {dataset.df.loc[i]['FrameID'][:-4]} / {total_len}")
            frame_ID += 1

            if frame_ID == total_len:
                save_df_dir = save_dir/ set / curr_vid_name
                save_df_dir.mkdir(parents=True, exist_ok=True)
                df = pd.DataFrame(preds)
                df.to_csv(save_df_dir /'embeddings.csv', header=False, index=False, float_format='%.5f')
                print(f"Saved video embeddings to {save_df_dir}")
                if (dataset.df.loc[i]['FileName'] == list(dataset.df['FileName'])[-1]):
                    #- Check whether is the last file
                    break
                else:
                    total_len = len(dataset.df[dataset.df['FileName'] == dataset.df.loc[i+1]['FileName']])
                    preds = np.zeros((len(dataset.df[dataset.df['FileName'] == dataset.df.loc[i+1]['FileName']]), spatial_enc.embed_dim))
                    frame_ID = 0
