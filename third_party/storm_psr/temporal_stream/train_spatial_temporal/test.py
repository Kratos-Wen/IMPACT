# %%
import numpy as np
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
import platform
import yaml
from model import VTN_tmp_only, No_temporal_encoder, STORM, STORM_MLP
from torchvision import transforms
from video_dataset_action_label import (
    EmbeddingFrameDataset_PSR,
    EmbeddingRecord_PSR,
    EmbeddinglistToTensor,
    ImglistToTensor,
    ImpactASREmbeddingDataset,
    VideoRecord,
    get_image,
    infer_impact_split_dir,
    is_impact_embedding_dataset,
    load_impact_component_names,
)
from utils import load_yaml, load_embedding_df, load_embedding_and_label_df
from tqdm import tqdm
from einops.layers.torch import Rearrange

from natsort import natsorted
import os


DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
#-- Global variable Setting 
if platform.system() == "Windows":
    base_dir = Path(
        r"\your_path")
    data_dir = Path(
        r"\your_data_path")
    ckpt_dir = data_dir
    psr_path = Path(
        r"\your_label_path")
else:
    base_dir = Path("your/run_path")
    data_dir = Path("your/data_path")
    ckpt_dir = Path("your/checkpoint_path")
    psr_path = Path("your/label_path")


def set_options():
    parser = argparse.ArgumentParser()
    # About run name
    parser.add_argument("--run_name", required=True, type=str,
                        help='Path to the run directory, e.g. ./runs/run_name')
    parser.add_argument("--checkpoint", required=False, type=str, default='best_model',
                        help="Name of the checkpoint (just name) to be tested or evaluated")

    parser.add_argument("--split", required=False, type=str, default='test',
                        help='Name of the model of taht run to be test or evaluated')

    parser.add_argument("--spatial_pretrained_weight",type=str, default=None,
                        help="Pre-trained the spatial enc. then fine-tuned the spatial enc.")
    # About data
    parser.add_argument("--data_dir",required=False,type=str,default=None,
                        help="Read images/videos when running the test on the whole temporal stream.")
    parser.add_argument("--csv_dir", required=False, type=str,default=None, 
                        help='Read embeddings from csv file (extracted from spatial encoder)\
                            when running the test on only temporal encoder.')
    parser.add_argument("--asd_label", required=False, type=str,
                        default='./')
    parser.add_argument("--psr_label_path",required=True, type=str,default=None,
                        help='Directory of the psr label.')
    parser.add_argument("--split_dir", required=False, type=str, default=None,
                        help='Optional split directory. Required for bundle-based datasets such as IMPACT ASR.')
    parser.add_argument("--impact_split", required=False, type=str, default='split1',
                        help='Bundle split name for IMPACT ASR, e.g. split1.')
    parser.add_argument("--log_path",type=str,default=None,
                        help="save path to the data log.")
    parser.add_argument("--dtype", required=False, type=str, default='embedding',
                        help='Specific the data type of the dataset (only have embeddings and video)')

    parser.add_argument("--baseline", default=False, action="store_true",
                        help="when runing the MLP baseline.")

    parser.add_argument("--skip_factor", required=False, type=int, default=0,
                        help="skip factor, in order to have larger temporal receptive field.")

    args, _ = parser.parse_known_args()
    return args


def setup_path(args):
    run_path = Path(args.log_path)/ args.run_name
    if not run_path.exists():
        raise ValueError(
            f"The run {run_path} you are trying to test, does not exist!")
    save_dir = run_path / "test_result" / test_args.checkpoint/ test_args.split
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Save the hyperparameter settings & model config for future to review the result and config.
    with open(save_dir / 'test_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        print(f"Saved run parameters to {run_path / 'args.txt'}")

    with open(run_path / 'model_args.yaml', 'r') as cfg:
        config = yaml.safe_load(cfg)

    with open(save_dir / 'test_model_args.yaml', 'w') as f:
        # json.dump(config, f, indent=2)
        yaml.dump(config, f)
        print(f"Saved run parameters to {run_path / 'test_model_args.yaml'}")

    print(f"PyTorch Version {torch.__version__}")
    print(f"Device:{DEVICE}")
    print(f"type:{torch.cuda.get_device_name()}")
    print("\n" * 5, "-" * 79, "\n", "-" * 79)
    print("Args for this run: \n", args)
    return run_path,  save_dir


def load_images_from_list(img_path_list: list,  resize = None, existing_frames = None, preprocess = None):
    """ Used during testing
    Read the images from the img_list and processing the image to tensor
    """
    if resize == None:
        resize = (224,224)

    clipped_imgs = list()
    if existing_frames is None:
        # print(f"Load images from {img_path_list[0]} to {img_path_list[-1]}")
        for img_path in img_path_list:
            img = get_image(img_path,size = resize)
            clipped_imgs.append(img)
            # clipped_imgs = torch.stack((clipped_imgs,img))
        if preprocess is not None:
            clipped_imgs = preprocess(clipped_imgs)

    else:
        # print(f"Load images from {img_path_list[0]} to {img_path_list[-1]}")
        clipped_imgs = existing_frames[1:]
        # Only need to load one images. 
        img = get_image(img_path_list[-1],size = resize)
        # Preprocess the image alone, do not preprocess clipped_imgs again.
        if preprocess is not None:
            img = preprocess(img).unsqueeze(0)

        # clipped_imgs.append(img)
        clipped_imgs = torch.cat((clipped_imgs,img))
        
    return clipped_imgs


if __name__ == "__main__":
    # -- Setup save path, data path, load dataset, load model
    test_args = set_options()
    run_path, save_dir = setup_path(test_args)
    model_weight_path = run_path / "checkpoints" / \
        f"{test_args.checkpoint}.pth"
    cfg = load_yaml(run_path / 'model_args.yaml')
    is_impact_asr = is_impact_embedding_dataset(test_args.data_dir, test_args.psr_label_path, test_args.split_dir)

    # --- Set up dataset specific information
    if 'industreal' in str(test_args.psr_label_path).lower():
        print("Train on IndustReal dataset...")
        NUM_COMPONENT = 11
        means = (0.608, 0.545, 0.520)
        stds = (0.172, 0.197, 0.188)
        state_list = [ i for i in range(NUM_COMPONENT)]

    elif 'meccano' in str(test_args.psr_label_path).lower():
        print("Train on Meccano dataset...")
        NUM_COMPONENT = 17
        means = (0.4144,0.4014,0.3777)
        stds = (0.2312,0.2458,0.2684)
        state_list = [i for i in range(NUM_COMPONENT)]

    elif is_impact_asr:
        print("Test on IMPACT ASR dataset...")
        NUM_COMPONENT = len(load_impact_component_names(Path(test_args.psr_label_path)))
        means = None
        stds = None
        state_list = [i for i in range(NUM_COMPONENT)]
        
    else:
        raise NotImplementedError(
            f"Currently only support meccano, industreal, and IMPACT ASR, but get {test_args.data_dir}"
        )
    
    NUM_FRAMES = cfg.frames
    if test_args.dtype == 'embedding':
        #-- load model
        if test_args.baseline:
            model = No_temporal_encoder(NUM_FRAMES,cfg.num_classes)
            if platform.system() == "Windows":
                model.load_state_dict(torch.load(model_weight_path,map_location=torch.device('cpu'))['mlp_head'])

            else:
                model.load_state_dict(torch.load(model_weight_path)['mlp_head'])

        else:
            model = VTN_tmp_only(**vars(cfg))
            if platform.system() == "Windows":
                try:
                    model.temporal_enc.load_state_dict(torch.load(model_weight_path,map_location=torch.device('cpu'))['temporal_enc'])
                    model.mlp_head.load_state_dict(torch.load(model_weight_path,map_location=torch.device('cpu'))['mlp_head'])
                except:
                    print("Load legacy model pth....")
                    model.temporal_enc.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu'))['temporal_enc'])
                    model.mlp_head.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu'))['mlp_head'])

            else:
                try:
                    model.temporal_enc.load_state_dict(torch.load(model_weight_path)['temporal_enc'])
                    model.mlp_head.load_state_dict(torch.load(model_weight_path)['mlp_head'])
                except:
                    print("Load legacy model pth....")
                    print(f"Load model from {model_weight_path}")
                    model.load_state_dict(torch.load(model_weight_path))
                
        #-- Setup data preprocessing pipeline base on the data type
        preprocess = transforms.Compose([EmbeddinglistToTensor(),])
        if is_impact_asr:
            if test_args.data_dir is None:
                raise ValueError(f"Expect to give data_dir for IMPACT ASR embedding test, but got {test_args.data_dir}")
            split_dir = Path(test_args.split_dir) if test_args.split_dir is not None else infer_impact_split_dir(Path(test_args.data_dir))
            impact_testset = ImpactASREmbeddingDataset(
                test_args.data_dir,
                test_args.split,
                annotation_dir=test_args.psr_label_path,
                split_dir=split_dir,
                impact_split=test_args.impact_split,
                num_frame_per_seg=NUM_FRAMES,
                transform=preprocess,
                test_mode=True,
                sampling_strategy='uniform',
                skip_factor=test_args.skip_factor,
                exe_mode='no_error',
            )
            test_data_list = impact_testset.embedding_list
        else:
            if test_args.csv_dir is None:
                raise ValueError(f"Expect to give csv_dir, but got {test_args.csv_dir}")
            testset_df = load_embedding_and_label_df(
                Path(test_args.csv_dir) / test_args.split, None, statics=False, normalized=False)
            test_data_list = [EmbeddingRecord_PSR(testset_df[testset_df['filename'] == name],
                                                  root_datapath= Path(test_args.psr_label_path) / test_args.split,
                                                  name=name,
                                                  FRAMES_PER_SEGMENT=NUM_FRAMES,
                                                  sampling_strategy='uniform',
                                                  execution_mode='no_error',
                                                  skip_factor = test_args.skip_factor,
                                                  num_dig_psr = NUM_COMPONENT,
                                                  test_mode=True) for name in pd.unique(testset_df['filename'])]
        batch1 = Rearrange('f w -> 1 f w ', f=NUM_FRAMES)

    elif test_args.dtype == 'video':
        #-- load model
        if test_args.baseline:
            model = STORM_MLP(**vars(cfg),args=test_args)
            # Get spatial enc, if already fine-tuning the baseline model. 
            try:
                print(f"Load model from {model_weight_path}")
                model.spatial_enc.load_state_dict(torch.load(model_weight_path)['spatial_enc'])
            except:
                print(f"Debug mode, train tmp enc but test with video data type...")
                model.spatial_enc.load_state_dict(torch.load(f'{test_args.run_name}/checkpoints/best.pth'))
            model.spatial_enc.load_state_dict(torch.load(model_weight_path)['spatial_enc'])
            model.mlp_head.load_state_dict(torch.load(model_weight_path)['mlp_head'])
            print("load pretrained mlp_head weights...")
        else:
            model = STORM(**vars(cfg),args=test_args)

        if platform.system() == "Windows":
            model.spatial_enc.load_state_dict(
                torch.load(model_weight_path, map_location=torch.device('cpu'))['spatial_enc'])
            model.temporal_enc.load_state_dict(
                torch.load(model_weight_path, map_location=torch.device('cpu'))['temporal_enc'])
            model.mlp_head.load_state_dict(
                torch.load(model_weight_path, map_location=torch.device('cpu'))['mlp'])
            
        else:
            try:
                print(f"Load model from {model_weight_path}")
                model.spatial_enc.load_state_dict(torch.load(model_weight_path)['spatial_enc'])
            except:
                print(f"Debug mode, train tmp enc but test with video data type...")
                model.spatial_enc.load_state_dict(torch.load(f'{test_args.run_name}/checkpoints/best.pth'))
            print("load pretrained spatial weights...")
            model.temporal_enc.load_state_dict(torch.load(model_weight_path)['temporal_enc'])
            print("load pretrained temporal weights...")
            model.mlp_head.load_state_dict(torch.load(model_weight_path)['mlp_head'])
            print("load pretrained mlp_head weights...")
            
        #-- Setup data preprocessing pipeline base on the data type
        recording_name = list()
        preprocess = transforms.Compose([
            ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
            transforms.Normalize(mean=means, std=stds)])
        recording_name.append([Path(f.path)
                               for f in os.scandir(Path(test_args.data_dir) / test_args.split) if f.is_dir()])
        recording_name = [
            item for sublist in recording_name for item in sublist]
        print(f"{len(recording_name)} recordings to be test...")
        test_data_list = [VideoRecord(img_datapath = Path(test_args.data_dir) / test_args.split / video.name, 
                                      psr_datapath = Path(test_args.psr_label_path) / test_args.split,
                                      name=video.name, 
                                      num_dig_psr=NUM_COMPONENT, 
                                      frame_per_seg=NUM_FRAMES,  
                                      skip_factor=test_args.skip_factor,
                                      execution_mode='no_error', 
                                      test_mode=True, 
                                      img_size=cfg.img_size) for video in natsorted(recording_name)]
        batch1 = Rearrange('f c h w -> 1 f c h w', f=NUM_FRAMES)

    else:
        raise NotImplementedError(
            f'Expect args.dtype be video or embedding, but get {test_args.dtype}')

    model.eval()
    model.to(DEVICE)
    model.float()
    print("Testing...")
    # We evaluate the model video by video.
    for clip_idx, recording in enumerate(test_data_list):
        label_list = list()
        pred_list = list()
        data_list = list()
        conf_list = list()
        name_list = list()
        framenbr_list = list()
        emb_list = list()
        prev_frames = None
        if hasattr(recording, 'iter_test_clips'):
            clip_iterator = recording.iter_test_clips()
            total_clips = recording.num_test_clips
        else:
            clip_iterator = zip(
                recording.clipped_embeddings,
                recording.clipped_level_labels,
                recording.clipped_last_frameID
            )
            total_clips = len(recording.clipped_level_labels)
        progress = tqdm(
            enumerate(clip_iterator),
            total=total_clips,
            desc=f"Recording:{recording.name}"
        )

        for idx, (clips, clips_label, clips_frameID) in progress:
            if test_args.dtype == 'embedding':
                src = batch1(preprocess(clips))

            elif test_args.dtype == 'video':
                if test_args.skip_factor != 0 :
                    frames = load_images_from_list(clips, existing_frames = prev_frames, preprocess = preprocess)
                    prev_frames = frames
                    src = batch1(frames)
                else:
                    frames = load_images_from_list(clips, existing_frames = prev_frames,preprocess = preprocess)
                    prev_frames = frames
                    src = batch1(frames)
                

            if torch.is_tensor(clips_label):
                target = clips_label.float()
                clips_label_to_save = clips_label.cpu().tolist()
            else:
                target = torch.tensor(np.float32(clips_label))
                clips_label_to_save = np.asarray(clips_label, dtype=np.float32).tolist()

            if torch.cuda.is_available():
                src = torch.autograd.Variable(src).cuda()
                target = torch.autograd.Variable(target).cuda()

            with torch.no_grad():
                output = model(src) # embed: [Bxfxd]
                #-- Save embedding:
                output = torch.sigmoid(output)
                output_pred = output.round().to(dtype=torch.int32).data.cpu().numpy()

            conf_list.append(output.data.cpu().numpy().reshape(-1).tolist())
            label_list.append(clips_label_to_save)
            pred_list.append(output_pred.reshape(-1).tolist())
            name_list.append(recording.name)
            framenbr_list.append(clips_frameID)

        result_df = pd.DataFrame({"clip": name_list, "framenr": framenbr_list,
                                      "GT": label_list, "Pred": pred_list, "Conf": conf_list})
        result_df.to_csv(f'{save_dir}/{recording.name}_results_pred.csv')
