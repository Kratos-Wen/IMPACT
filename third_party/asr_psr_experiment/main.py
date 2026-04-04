#!/usr/bin/python2.7

import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import glob
import random
from impact_split_utils import (
    default_impact_annotation_dir,
    default_impact_feature_dir,
    load_bundle_feature_files,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="gtea")
parser.add_argument('--split', default='1')

parser.add_argument('--features_dim', default='1024', type=int)
parser.add_argument('--bz', default='6', type=int)
parser.add_argument('--lr', default='0.0005', type=float)


parser.add_argument('--num_f_maps', default='64', type=int)

# Need input
parser.add_argument('--num_epochs', default=10000, type=int)
parser.add_argument('--load', default=None, help='Path to checkpoint to load')
parser.add_argument('--num_layers_PG', default=11, type=int)
parser.add_argument('--num_layers_R', default=10, type=int)
parser.add_argument('--num_R', default=3, type=int)

# experiment = "ms_tcn2" or "linear"
parser.add_argument('--experiment', default="ms_tcn2", choices=["ms_tcn2", "videomae"])
parser.add_argument('--split_dir', default=None, help='Optional bundle split directory, e.g. ../data/IMPACT/splits_ASR_front_only_v1')
parser.add_argument('--impact_split', default='split1', help='Bundle suffix when --split_dir is used, e.g. split1')
parser.add_argument('--bundle_split', default='train', choices=['train', 'val', 'test'], help='Which bundle split to load when --split_dir is used')
parser.add_argument('--camera', default='front', choices=['front', 'left', 'right', 'top', 'ego'], help='Camera/view to train on when --split_dir is used')
parser.add_argument('--annotation_dir', default=None, help='Annotation directory. Defaults to ../data/IMPACT_ASR when --split_dir is used')
parser.add_argument('--feature_dir', default=None, help='Explicit feature directory for the selected experiment/camera')

args = parser.parse_args()

num_epochs = args.num_epochs
features_dim = args.features_dim
bz = args.bz
lr = args.lr

if args.experiment == "ms_tcn2":
    features_dim = 1024
elif args.experiment == "videomae":
    features_dim = 1408

num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R
num_f_maps = args.num_f_maps

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

features_path = "../data/features/"
gt_path = "../data/annotations/"

if args.split_dir:
    gt_path = args.annotation_dir or default_impact_annotation_dir("../data")
    features_path = args.feature_dir or default_impact_feature_dir(args.experiment, args.camera, "../data")
    selected_files = load_bundle_feature_files(
        args.split_dir,
        args.bundle_split,
        args.impact_split,
        features_path,
        camera=args.camera,
    )
    vid_list_file_trn = selected_files
    vid_list_file_tst = selected_files
    print(f"Using bundle split {args.bundle_split}.{args.impact_split} with {len(selected_files)} {args.camera} videos")
else:
    # vid_list_file = "../data/annotations/"+args.dataset+"/splits/train.split"+args.split+".bundle"
    # vid_list_file_tst = "../data/annotations/"+args.dataset+"/splits/test.split"+args.split+".bundle"
    vid_list_gt = glob.glob("../data/annotations/*")
    # get children names of each annotation file
    vid_list_file = []
    for f in [f.split('/')[-1].split('.')[0] for f in vid_list_gt]:
        d, t = f.split('_')[:2]  # remove the last part of the file name, which is the camera id
        if args.experiment == "ms_tcn2":
            files = glob.glob(f"../data/features/IMPACT_i3d*/*/*/*{d}_{t}*.npy")
        else:
            files = glob.glob(f"../data/features/IMPACT*/*/*{d}_{t}*.npy")
            files = [f for f in files if "i3d" not in f]
        vid_list_file += files

    # randomly split the videos into train and test sets
    random.shuffle(vid_list_file)
    split_point = int(0.8*len(vid_list_file))
    vid_list_file_trn = vid_list_file[:split_point]
    vid_list_file_tst = vid_list_file[split_point:]

model_dir = "./models/"+args.experiment+"/split_"+args.split
results_dir = "./results/"+args.experiment+"/split_"+args.split

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

num_classes = 17
trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, args.dataset, args.split, experiment=args.experiment)
if args.action == "train":
    if args.load:
        args.load = os.path.join("./models/", args.experiment, "split_" + args.split, args.load)
        trainer.model.load_state_dict(torch.load(args.load, map_location=device))
        print("Model loaded from", args.load)
    batch_gen = BatchGenerator(num_classes, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file_trn)
    trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

if args.action == "predict":
    trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, device, sample_rate)
