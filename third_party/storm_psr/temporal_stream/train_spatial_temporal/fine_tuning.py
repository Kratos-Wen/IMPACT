# %%
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import timm
from torchvision.transforms import ToTensor
import io
import pandas as pd
import math
import seaborn as sn
from sklearn.metrics import classification_report,  f1_score
from torchsummary import summary
import json
from video_dataset_action_label import ImglistToTensor,  VideoFrameDataset_PSR
from model import VTN, VTN_tmp_only, No_temporal_encoder, temp_enc_LSTM, STORM, STORM_MLP
from utils import load_yaml,  get_metrics, plot_trainlog_result
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.optim import AdamW, SGD, Adagrad
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import platform
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import argparse
import yaml
import datetime
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
import torch
import time
torch.backends.cudnn.benchmark = True

# --- Global Setting
DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
# Learning scheduler
LRS = [1, 0.1, 0.01]
STEPS = [1, 14, 25]

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
    
IndustReal_ACTION_LABEL = ['Base','Front chassis','Front chassis pin', 'Rear chassis', \
                'Short rear chassis', 'Front rear chassis pin', 'Rear rear chassis pin',\
                'Fron bracket','Front bracket screw','Front wheel assy', 'Rear wheel assy']

MECCANO_ACTION_LABEL = ['Left dampling fork','Right dampling fork','Left rear chassis','Right rear chassis',\
                'Left frame','Right frame','Left tail wings','right tail wings','Headlamp','Left handles','Right handles',\
                'Front wheel','Rear wheel','Swingarm','Fuel tank','Tail wings pin','Drive shaft']

# --- Setting gen to a constant value for future reproducability
gen = torch.Generator()
gen.manual_seed(1234)

# Parse arguments
def set_options():
   
    parser = argparse.ArgumentParser(description='Passing arguments to the training process')

    # -- Data path setting
    parser.add_argument("--data_dir", type=str, default='none',
                        help="Directory of the dataset")
    parser.add_argument("--psr_label_path",type=str,default=None,
                        help="Directory of the psr label")
    parser.add_argument("--log_path",type=str,default=None,
                        help="save path to the data log.")
    parser.add_argument("--run_name", type=str, default='default',
                        help="Name of the run to be tested or evaluated")
    parser.add_argument("--resume", type=int, default=0,
                        help='Resume training from')
    parser.add_argument("--config", type=str, default='configs/IndustReal/STORM_F65_dim128_ImgNet.yaml',
                        help="Config file")
    parser.add_argument("--dtype", type=str, default='video',
                        help='Specific the data type of the dataset (only have embeddings and video)')
    parser.add_argument("--parallel", default=False, action="store_true",
                        help="Enable the multiple GPU training")
    parser.add_argument("--job_file_mode", default=False, action="store_true",
                        help="If submit shell script to qsub, store true")
    parser.add_argument("--baseline", default=False, action="store_true",
                        help="If true, run the baseline (MLP).")
    parser.add_argument("--sanity_check", default=False, action="store_true",
                        help="If true, frozen weight and run some data to see the metrics.")
    parser.add_argument("--pretrained_weight",type=str, default=None,
                        help="Pre-trained the tmp enc. then fine-tuned the spatial enc.")
    parser.add_argument("--spatial_pretrained_weight",type=str, default=None,
                        help="Pre-trained the spatial enc. then fine-tuned the spatial enc.")
    
    # -- Hpyerparameters:
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--scheduler", type=str, default="cosine_restart",
                        help='Type of loss function to use. Implemented: stepLR, cosine_restart')
    parser.add_argument("--T_0", type=int, default=5,
                        help='T_0 value, indicating the number of epochs in first cycle of cos annealing warm restart')
    parser.add_argument("--lr_gamma", type=float, default=0.975,
                        help='Learning rate exponential decay factor. Gamma = 1 --> no decaying LR')
    parser.add_argument("--lr_step", type=int, default=5,
                        help='Learning rate step size.')
    parser.add_argument("--warmup", type=int, default=2,
                        help='Use warmup learning rate for this amount of epochs.')
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of the training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size of the training dataloader")
    parser.add_argument("--workers", type=int, default=8,
                        help="Num of worker for the dataloader")
    parser.add_argument("--warmup_rate", type=int, default=1e-3,
                        help="Use warmup learning rate for this amount of epochs")
    parser.add_argument("--sampling_strategy", type=str, default='bimodal',
                        help="Sampling method of the dataloader")
    parser.add_argument("--skip_factor", type=int, default=0,
                        help="skip factor, in order to have larger temporal receptive field.")

    return parser.parse_args()

def save_dict(log_path: Path, txt_name, dict_name):
    with open(log_path / txt_name, 'w') as file:
        file.write(json.dumps(dict_name))

def json_serializer(obj):
    if isinstance(obj,Path):
        return str(obj)

def save_model_state_dict(args,model,optimizer,scheduler,epoch,save_path,BEST:bool = False):
    if BEST:
        filename = 'best_model'
    else:
        filename = f'weights_{epoch}'

    if args.baseline:
        torch.save({'spatial_enc': model.spatial_enc.state_dict(),
                    'temporal_enc': model.temporal_enc.state_dict(),
                    'mlp_head'    : model.mlp_head.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'scheduler':scheduler.state_dict()}, 
                    f'{save_path}/{filename}.pth')
        
    elif args.parallel:
        torch.save({'spatial_enc': model.module.spatial_enc.state_dict(),
                    'temporal_enc': model.module.temporal_enc.state_dict(),
                    'mlp_head'    : model.module.mlp_head.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'scheduler':scheduler.state_dict()}, 
                    f'{save_path}/{filename}.pth')
        
    else:
        torch.save({'spatial_enc': model.spatial_enc.state_dict(),
                    'temporal_enc': model.temporal_enc.state_dict(),
                    'mlp_head': model.mlp_head.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'scheduler':scheduler.state_dict(),}, 
                    f'{save_path}/{filename}.pth')
        
    return

def setup_path(args):
    # --- Model paths and log file locations
    run_path = Path(args.log_path) / args.run_name
    if run_path.exists():
        if not args.job_file_mode:
            if args.resume != 0:
                ans = input(
                    f"Resuming training from latest checkpoint. This overwrites args.txt file. Continue? (y/*)")
            else:
                ans = input(f"Run name {args.run_name} already exists. "
                            f"Are you sure you want to overwrite this folder and tensorboard logs? (y/*)")
            if ans == "y":
                print(f"Continuing {args.run_name}")
            else:
                raise ValueError(f"Not overwriting {args.run_name}")

        else:
            if args.resume != 0:
                print(
                    f"Resuming training from latest checkpoint. This overwrites args.txt file.Continuing {args.run_name}")
            else:
                print(f"Duplicated run_name, rename it as {run_path.name}")

    modelsave_path = run_path / "checkpoints"
    log_path = run_path
    tb_dir = run_path / "tensorboard"
    save_ckpt_dir = ckpt_dir / args.run_name
    performance_report_dir = run_path / "classification_result"

    run_path.mkdir(parents=True, exist_ok=True)
    modelsave_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    save_ckpt_dir.mkdir(parents=True, exist_ok=True)
    performance_report_dir.mkdir(parents=True, exist_ok=True)

    # --- Save the hyperparameter settings & model config for future to review the result and config.
    with open(run_path / 'args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2, default=json_serializer)
        print(f"Saved run parameters to {run_path / 'args.txt'}")

    with open(args.config, 'r') as cfg:
        config = yaml.safe_load(cfg)

    #-- Store the model args, so the test script can load the modol parameter when inference.
    if args.resume == 0:
        with open(run_path / 'model_args.yaml', 'w') as f:
            yaml.dump(config, f)
            print(f"Saved run parameters to {run_path / 'model_args.txt'}")

    print(f"PyTorch Version {torch.__version__}")
    print(f"Start time: {datetime.datetime.now()}")
    print(f"Device:{DEVICE}")
    print(f"type:{torch.cuda.get_device_name()}")
    print("\n" * 5, "-" * 79, "\n", "-" * 79)
    print("Args for this run: \n", args)

    return run_path, modelsave_path, log_path, tb_dir, save_ckpt_dir, performance_report_dir

def batch_log_print(d, i, t, split):
    c = datetime.datetime.now().strftime('%H:%M:%S')
    str = f"{c} - {split} Iteration:{i} \t"
    for key in d:
        str += f" \t {key}: {d[key]:.5f}"
    str += f" \t {time.time() - t:.3f} seconds/iteration"
    print(str)

def classification_result(loader, model):
    y_pred = []  # save predction
    y_true = []  # save ground truth

    # iterate over data
    for inputs, labels in loader:
        inputs = torch.autograd.Variable(src).cuda()
        labels = torch.autograd.Variable(target).cuda()
        # Check the number of labels...

        with torch.no_grad():
            output = model(inputs)  # Feed Network
            # output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            # for PSR multi-label classification
            output = torch.sigmoid(output)
            output = output.round().data.cpu().numpy()

        y_pred.extend(output)  # save prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = [str(i) for i in range(11)]

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        target_names=ACTION_LABEL,
        zero_division=np.nan,
    )
    print(classification_report(
        y_true,
        y_pred,
        output_dict=False,
        target_names=ACTION_LABEL,
        zero_division=np.nan,
    ))
    return report

if __name__ == "__main__":
    # --  Before Training     
    # --- load args
    args = set_options()
    # --- Setup checkpoint path, data log
    if platform.system() == "Windows":
        print(f"Manually putting batch size to 2 and workers to 1 on CPU")
        args.batch_size = 1
        args.workers = 1

    #TODO: Need further implementation to train on more datasets
    # --- Set up dataset specific information
    if 'industreal' in str(args.psr_label_path).lower():
        print("Train on IndustReal dataset...")
        ACTION_LABEL  = IndustReal_ACTION_LABEL
        NUM_COMPONENT = 11
        means = (0.608, 0.545, 0.520)
        stds = (0.172, 0.197, 0.188)
        state_list = [ i for i in range(NUM_COMPONENT)]

    elif 'meccano' in str(args.psr_label_path).lower():
        print("Train on Meccano dataset...")
        ACTION_LABEL = MECCANO_ACTION_LABEL
        NUM_COMPONENT = 17
        means = (0.4144,0.4014,0.3777)
        stds = (0.2312,0.2458,0.2684)
        state_list = [i for i in range(NUM_COMPONENT)]
        
    else:
        raise NotImplementedError(f"Currently only support meccano and industreal, but get {args.data_dir}")
    
    # --- Model paths and log file locations
    run_path, modelsave_path, log_path, tb_dir, save_ckpt_dir, result_dir = setup_path(
        args)

    # --- Load model config
    cfg = load_yaml(args.config)
    NUM_FRAMES = cfg.frames
    # --- Load model, different data type would load different model
    if args.dtype == 'video':
        preprocess = transforms.Compose([
            ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.01, 2.0)),
            transforms.ColorJitter(
                brightness=0.1, saturation=0.7, contrast=0.1),
            transforms.Normalize(mean=means, std=stds),
        ])
        preprocess_sanity_check = transforms.Compose([
            ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
             transforms.Normalize(mean=means, std=stds),
        ])
        # model
        if args.baseline:
            model = STORM_MLP(**vars(cfg),args = args)
        else:
            model = STORM(**vars(cfg),args=args)

    else:
        raise NotImplementedError(
            f'Expecte video, but got {args.dtype}.')


   
    # --- Load the pre-trained temporal enc. / spatial enc.weight
    if (args.pretrained_weight != None) and (args.dtype == 'video') and (args.resume == 0):
        #- Load the temporal encoder from train.py given the directory in args.
        print("Load the pre-trained temporal enc. and   enc. to be fine-tuned..")
        
        #- change the old key name to new key name 
        if args.baseline:
            # Baseline only have mlp head
            mlp_state_dict = torch.load(f'{args.pretrained_weight}')['mlp_head']
            new_state_dict = dict()
            for new_key, key in zip(list(model.mlp_head.state_dict().keys()),list(mlp_state_dict.keys())):
                new_state_dict[new_key] = mlp_state_dict.pop(key)
            model.mlp_head.load_state_dict(new_state_dict)
            print(f"Done loading pre-trained mlp head from {args.pretrained_weight} for the Baseline model (MLP).")

        else:
            state_dict = torch.load(f'{args.pretrained_weight}')['temporal_enc']
            new_state_dict = dict()
            for new_key, key in zip(list(model.temporal_enc.state_dict().keys()),list(state_dict.keys())):
                new_state_dict[new_key] = state_dict.pop(key)
            model.temporal_enc.load_state_dict(new_state_dict)
            print(f"Done loading pre-trained temporal encder from {args.pretrained_weight}.")
            #
            mlp_state_dict = torch.load(f'{args.pretrained_weight}')['mlp_head']
            new_state_dict = dict()
            for new_key, key in zip(list(model.mlp_head.state_dict().keys()),list(mlp_state_dict.keys())):
                new_state_dict[new_key] = mlp_state_dict.pop(key)
            model.mlp_head.load_state_dict(new_state_dict)
            print(f"Done loading pre-trained mlp head from {args.pretrained_weight}.")

    if (args.pretrained_weight == None) and (args.dtype == 'video') and (args.resume == 0):
        print("No pretrained temporal enc. and mlp head, are going to end-to-end training the temproal enc from scratch.")

    # --- Resume weights
    if args.resume > 0:
        print("Resume the fine-tuning...")
        # Reset the best_val_loss for saving the checkpoints.
        best_val_loss = 0
        # Load model args and weight
        cfg = load_yaml(run_path / 'model_args.yaml')
        # TODO: if args.dtype = video, use other model to load.
        if args.dtype == 'embedding':
            print("Using data type: embedding..")
            model = VTN_tmp_only(**vars(cfg))
            model.temporal_enc.load_state_dict(torch.load(
                f'{save_ckpt_dir}/weights_{args.resume}.pth')['temporal_enc'])
            model.mlp_head.load_state_dict(torch.load(
                f'{save_ckpt_dir}/weights_{args.resume}.pth')['mlp_head'])


        elif args.dtype == 'video':
            print("Initializing model.....")
            if args.baseline:
                model.spatial_enc.load_state_dict(torch.load(
                    f'{save_ckpt_dir}/weights_{args.resume}.pth')['spatial_enc'])
                model.mlp_head.load_state_dict(torch.load(
                    f'{save_ckpt_dir}/weights_{args.resume}.pth')['mlp_head'])
            else:
                print(f"Load checkpoint at {args.resume} epoch...")
                #-- Since in the fine-tuning stage, the spatial encoder is no longer the original enc. that we used to train on  embedding
                model.spatial_enc.load_state_dict(torch.load(
                    f'{save_ckpt_dir}/weights_{args.resume}.pth')['spatial_enc'])
                model.temporal_enc.load_state_dict(torch.load(
                    f'{save_ckpt_dir}/weights_{args.resume}.pth')['temporal_enc'])
                model.mlp_head.load_state_dict(torch.load(
                    f'{save_ckpt_dir}/weights_{args.resume}.pth')['mlp_head'])

    

    # --- Use multiple GPU to trian the model
    if torch.cuda.is_available() and args.parallel:
        model = nn.DataParallel(model).cuda()
        # print(model)

    elif torch.cuda.is_available():
        model = model.to(DEVICE)

    else:
        pass

    # --- Load dataset
    if args.dtype == 'video':
        train_set = VideoFrameDataset_PSR(Path(args.data_dir), Path(args.psr_label_path), split='train',
                                          transform=preprocess, skip_factor=args.skip_factor, num_dig_psr= NUM_COMPONENT,
                                          num_frame_per_seg=NUM_FRAMES, img_size=cfg.img_size, test_mode=False, sampling_strategy=args.sampling_strategy)
        
        train_set_sanitiy_check = VideoFrameDataset_PSR(Path(args.data_dir), Path(args.psr_label_path), split='train',
                                          transform=preprocess_sanity_check, skip_factor=args.skip_factor, num_dig_psr= NUM_COMPONENT,
                                          num_frame_per_seg=NUM_FRAMES, img_size=cfg.img_size, test_mode=False, sampling_strategy=args.sampling_strategy)
        
        val_set = VideoFrameDataset_PSR(Path(args.data_dir), Path(args.psr_label_path), split='val',
                                        transform=preprocess, skip_factor=args.skip_factor, num_dig_psr= NUM_COMPONENT,
                                        num_frame_per_seg=NUM_FRAMES, img_size=cfg.img_size, test_mode=False, sampling_strategy=args.sampling_strategy)

    else:
        raise NotImplementedError(f"This script is only for training data tye for video.")

    # --- Split dataset
    # print("spliting...")
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, persistent_workers=True, pin_memory=True,generator=gen)
    train_loader_sanity_check = DataLoader(train_set_sanitiy_check, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, persistent_workers=True, pin_memory=True,generator=gen)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,num_workers=args.workers, persistent_workers=True, pin_memory=True,generator=gen)

    # --- Tensorboard
    # print("Tensorboard...")
    # tensorboard = SummaryWriter(tb_dir)

    # --- Optimizer, assign the sapatial encoder with smaller lr
    if (args.dtype == 'video') and (not cfg.spatial_frozen):
        if torch.cuda.is_available() and args.parallel:
            optimizer = SGD([{'params': model.module.spatial_enc.parameters(), 'lr': args.lr * 0.01},
                            {'params': model.module.temporal_enc.parameters(), 'lr': args.lr, 'momentum': 0.9, 'weight_decay': args.weight_decay}], lr=args.lr, momentum=0.9)
        else:
            optimizer = SGD([{'params': model.spatial_enc.parameters(), 'lr': args.lr * 0.01},
                            {'params': model.temporal_enc.parameters(), 'lr': args.lr, 'momentum': 0.9, 'weight_decay': args.weight_decay}], lr=args.lr, momentum=0.9)
    else:
        optimizer = SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.weight_decay)

     # --- Learning rate scheduler
    if args.scheduler == "cosine_restart":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0)
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=args.lr_step,
                           gamma=args.lr_gamma)
    else:
        raise ValueError(f"Scheduler {args.scheduler} not implemented..")

    if args.warmup > 0:
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, total_iters=args.warmup)
        scheduler = SequentialLR(
            optimizer, [warmup_scheduler, scheduler], milestones=[args.warmup])
        
    # --- Resume optimizer and scheduler
    if args.resume > 0:
        # Reset the best_val_loss for saving the checkpoints.
        if args.dtype == 'video':
            print("Resuming optimizer and scheduler model.....")
            print(f"Load checkpoint at {args.resume} epoch...")
            try:
                optimizer.load_state_dict(torch.load(
                    f'{save_ckpt_dir}/weights_{args.resume}.pth')['optimizer_state_dict'])
                scheduler.load_state_dict(torch.load(
                    f'{save_ckpt_dir}/weights_{args.resume}.pth')['scheduler'])
                print("Resume optimizer and scheduler.")
                
            except:
                print('No found ckpt for scheduler and optimizer. Use the new initialized opt. and scheduler.')
        else:
            raise NotImplementedError('This script is only for training spatial and temporal encoder.')
    # --- Loss
    loss_func = nn.BCELoss()
    softmax = nn.LogSoftmax(dim=1)
    best_val_acc = 0.0
    
    #-----------------------------------  Training   
    for epoch in range(max(args.resume+1, 1), args.epochs+1):
        # --- Train
        model.train()
        model.to(DEVICE)
        model.float()
        progress = tqdm(enumerate(train_loader), total=len(
            train_loader), desc=f"Epoch: {epoch}, loss: 0.000")
        train_loss_list = []
        train_acc_list = []
        multi_label_total = 0
        if cfg.spatial_frozen:
            lr_spatial_enc = 0
            lr_temporal_enc = scheduler.get_last_lr()[0]
        else:
            lr_spatial_enc = scheduler.get_last_lr()[0]
            lr_temporal_enc = scheduler.get_last_lr()[1]
        for i, (src, target) in progress:
            if torch.cuda.is_available():
                src = torch.autograd.Variable(src).cuda()
                target = torch.autograd.Variable(target).cuda()
            optimizer.zero_grad()
            # Forward + backprop + optimize
            output = model(src)
            output = torch.sigmoid(output)
            loss = loss_func(output, target)
            output_pred = output.round().data.cpu().numpy()

            # Cosine scheduler
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Show the F1 score
            train_loss = loss.item()
            # train_acc  = torch.sum(torch.argmax(output, dim=1) == target).cpu().detach().item() / args.batch_size
            train_acc = f1_score(target.data.cpu().numpy(),output_pred, labels=state_list, average=None, zero_division=0.0)
            result_train_acc = np.mean([score for score in train_acc if math.isnan(score) != True])
            # print(f'GT:{output_pred}')
            # print(f'Pred:{target.data.cpu().numpy(),}')
            progress.set_description(f"Epoch: {epoch}, train loss: {train_loss:.6f}, macro F1 score:{result_train_acc:.6f}")
            train_loss_list.append(train_loss)
            train_acc_list.append(result_train_acc)

            # Summary per iter on TB
            # tensorboard.add_scalar(
            #     'train_loss', train_loss, epoch * len(train_loader) + i)
            # tensorboard.add_scalar(
            #     'train_f1', result_train_acc * 100, epoch * len(train_loader) + i)
            # tensorboard.add_scalar('lr', lr, epoch * len(train_loader) + i)

        # # Summary per epoch on TB
        # avg_train_loss = sum(avg_train_loss_list)/ len(avg_train_loss_list)
        # avg_train_acc = sum(avg_train_acc_list) / len(avg_train_acc_list)
        # tensorboard.add_scalar('train_loss',avg_train_loss, epoch * len(train_loader) + i)
        # tensorboard.add_scalar('train_f1',avg_train_acc * 100, epoch * len(train_loader) + i)
        # tensorboard.add_scalar('lr', lr, epoch * len(train_loader) + i)

        # --- Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        val_loss_list = []
        val_acc_list = []
        y_pred = []
        y_true = []
        progress = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch:{epoch}, validating")
        # for src, target in progress:
        for i, (src, target) in progress:
            if torch.cuda.is_available():
                src = torch.autograd.Variable(src).cuda()
                target = torch.autograd.Variable(target).cuda()

            with torch.no_grad():
                output = model(src)
                output = torch.sigmoid(output)
                loss = loss_func(output, target)
                output_pred = output.round().data.cpu().numpy()
                val_loss = loss.item()
                val_loss_list.append(val_loss)
                y_pred.extend(output_pred)  # save prediction
                y_true.extend(target.data.cpu().numpy())  # save ground truth
                val_acc = f1_score(target.data.cpu().numpy(),output_pred, labels=state_list, average=None, zero_division=0.0)
                val_acc = np.mean([score for score in val_acc if math.isnan(score) != True])
                val_acc_list.append(val_acc)
                


            progress.set_description(f"Epoch: {epoch}, val loss: {val_loss:.6f}, macro F1 score:{val_acc:.6f}")
            # --- Summary per iter
            # tensorboard.add_scalar('val_loss', val_loss,
            #                        epoch * len(val_loader) + i)
            # tensorboard.add_scalar(
            #     'val_acc', val_acc*100, epoch * len(val_loader) + i)
        print(classification_report(y_true, y_pred,target_names=ACTION_LABEL, output_dict=False, zero_division=np.nan))
        # --- Creating dict to save training progress
        dict_to_save = {
            "train_loss": np.mean(train_loss_list),
            "train_accuracy": np.mean(train_acc_list),
            "validation_loss": np.mean(val_loss_list),
            "validation_accuracy": np.mean(val_acc_list),
            "lr_spatial": lr_spatial_enc,
            "lr_temporal": lr_temporal_enc
        }
        c = datetime.datetime.now().strftime('%H:%M:%S')
        log_str = f"{c} - Epoch: {epoch}\tTrain loss: {np.mean(train_loss_list):.6f}\tTrain acc (%): {np.mean(train_acc_list)*100:.4f}\t" \
            f"Val loss: {np.mean(val_loss_list):.6f}\tVal acc (%): {np.mean(val_acc_list)*100:.4f}\t" \
            f"LR (spatial): {lr_spatial_enc:.6f}\t LR (temporal):{lr_temporal_enc:.6f}\n"
        
        #--- Summary per epoch
        # tensorboard.add_scalar('val_loss', sum(avg_val_loss) / len(avg_val_loss), epoch)
        # tensorboard.add_scalar('val_acc', sum(avg_val_acc) / len(avg_val_acc) * 100, epoch)

        # --- Save spatial enc and tmp_enc separately.
        save_model_state_dict(args,model,optimizer,scheduler,epoch, save_ckpt_dir, BEST = False)

        # --- Save the best model depend on the highest val accsas
        # Not going to save 'opt' and 'scheduler' in the achived checkpoint
        if ((epoch == 1) or (epoch == args.resume+1)):
            best_val_loss = np.mean(val_loss_list)

        elif (np.mean(val_loss_list)) < best_val_loss:
            print(f'Update best model. at {epoch}')
            save_model_state_dict(args,model,optimizer,scheduler,epoch, modelsave_path, BEST = True)
            best_val_loss = np.mean(val_loss_list)
            best_val_acc = np.mean(val_acc_list)

        elif epoch == args.epochs:
            print(f'Save the final resulted model.')
            save_model_state_dict(args,model,optimizer,scheduler,epoch, modelsave_path, BEST = False)
        
        else:
            pass
        # print("Checkpoint Saved")
        progressfilename = 'train_progress_epoch' + str(epoch) + '.txt'
        save_dict(log_path, progressfilename, dict_to_save)
        f = open(log_path / 'train_log.txt', 'a')
        f.write(log_str)
        f.close()
        print(log_str)

        #
        # tensorboard.add_figure("Train Confusion Matrix",createConfusionMatrix(train_loader),epoch)
        # tensorboard.add_figure("Val Confusion Matrix",createConfusionMatrix(val_loader),epoch)
        # tensorboard.add_figure("Val Confusion Matirx", createMutliLabelConfusionMatrix(val_loader),epoch)
        # report = classification_result(val_loader, model)
        report = classification_report(
            y_true,
            y_pred,
            output_dict=True, target_names=ACTION_LABEL,
            zero_division=np.nan)
        val_file = 'result_on_val_' + str(epoch) + '.json'
        with open(result_dir / val_file, 'w') as outfile:
            json.dump(report, outfile)

        plot_trainlog_result(epoch, run_path, run_path)
    # tensorboard.close()
    # Run the testset
    avg_test_loss = []
    avg_test_acc = []
    #
    print("Run test on the testset")
    if args.dtype == 'video':
        test_set = VideoFrameDataset_PSR(Path(args.data_dir), Path(args.psr_label_path), split='test', transform=preprocess, skip_factor=args.skip_factor, num_dig_psr=NUM_COMPONENT,
                                         num_frame_per_seg=NUM_FRAMES, sampling_strategy=args.sampling_strategy)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,num_workers=args.workers, persistent_workers=True, pin_memory=True,generator=gen)
    report = classification_result(test_loader, model)
    with open(result_dir / 'result_on_test.json', 'w') as outfile:
        json.dump(report, outfile)

    print("Training Complete.")
