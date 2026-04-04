from sklearn.metrics import classification_report
import io
import pandas as pd
import math
import seaborn as sn
from sklearn.metrics import classification_report, f1_score
from torchsummary import summary
import json
from video_dataset_action_label import (
    EmbeddingFrameDataset_PSR,
    EmbeddinglistToTensor,
    ImglistToTensor,
    ImpactASREmbeddingDataset,
    VideoFrameDataset_PSR,
    VideoRecord,
    infer_impact_split_dir,
    is_impact_embedding_dataset,
    load_impact_component_names,
)
from model import VTN, VTN_tmp_only, No_temporal_encoder, temp_enc_LSTM
from utils import load_yaml, GradualWarmupScheduler, plot_trainlog_result
import utils as ut
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim import AdamW, SGD, Adagrad
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
import platform
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm, trange
import numpy as np
import torch.nn as nn
import argparse
import yaml
import os
import datetime
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
import torch
import time
torch.backends.cudnn.benchmark = True

# --- Global training Setting
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Learning scheduler
LRS = [1, 0.1, 0.01]
STEPS = [1, 14, 25]

IndustReal_ACTION_LABEL = ['Base','Front chassis','Front chassis pin', 'Rear chassis', \
                'Short rear chassis', 'Front rear chassis pin', 'Rear rear chassis pin',\
                'Fron bracket','Front bracket screw','Front wheel assy', 'Rear wheel assy']

MECCANO_ACTION_LABEL = ['Left dampling fork','Right dampling fork','Left rear chassis','Right rear chassis',\
                'Left frame','Right frame','Left tail wings','right tail wings','Headlamp','Left handles','Right handles',\
                'Front wheel','Rear wheel','Swingarm','Fuel tank','Tail wings pin','Drive shaft']

#-- For reproducibility
gen = torch.Generator()
gen.manual_seed(1234)

# Parse arguments
def set_options():

    parser = argparse.ArgumentParser(description='Passing arguments to the training process')
    # -- Data path setting
    parser.add_argument("--data_dir", type=str, default='none',
                        help="Directory of the embedding dataset")
    parser.add_argument("--psr_label_path",type=str,default=None,
                        help="Directory of the psr label")
    parser.add_argument("--split_dir", type=str, default=None,
                        help="Optional split directory. Required for datasets that use bundle-based splits, such as IMPACT ASR.")
    parser.add_argument("--impact_split", type=str, default="split1",
                        help="Bundle split name for IMPACT ASR, e.g. split1.")
    parser.add_argument("--log_path",type=str,default=None,
                        help="save path to the data log.")
    parser.add_argument("--ckpt_dir",type=str,default="check_point_path",help="Dir. to save the checkpoint model to at each epoch.")    

    # -- training setting
    parser.add_argument("--run_name", type=str, default='default',
                        help="Name of the run to be tested or evaluated")
    parser.add_argument("--resume", type=int, default=0,
                        help='Resume training from')
    parser.add_argument("--config", type=str, default='configs/full-vtn-temporal-vit.yaml',
                        help="Config file")
    parser.add_argument("--dtype", type=str, default='embedding',
                        help='Specific the data type of the dataset (only have embeddings and video)')
    parser.add_argument("--parallel", default=False, action="store_true",
                        help="Enable the multiple GPU training")
    parser.add_argument("--job_file_mode", default=False, action="store_true",
                        help="If submit shell script to qsub, store true")
    parser.add_argument("--baseline", default=False, action="store_true",
                        help="If true, run the baseline (MLP).")
    parser.add_argument("--skip_factor", type=int, default=0,
                        help="skip factor, in order to have larger temporal receptive field.")
    # -- Hpyerparameters
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--scheduler", type=str, default="cosine_restart",
                        help='Type of loss function to use. Implemented: stepLR, cosine_restart')
    parser.add_argument("--T_0", type=int, default=5,
                        help='T_0 value, indicating the number of epochs in first cycle of cos annealing warm restart')
    parser.add_argument("--lr_gamma", type=float, default=0.975,
                        help='Learning rate exponential decay factor. Gamma = 1 --> no decaying LR')
    parser.add_argument("--lr_step", type=int, default=5,
                        help='Learning rate step size.')
    parser.add_argument("--warmup", type=int, default=3,
                        help='Use warmup learning rate for this amount of epochs..')
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of the training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size of the training dataloader")
    parser.add_argument("--workers", type=int, default=16,
                        help="Num of worker for the dataloader")
    parser.add_argument("--warmup_rate", type=int, default=1e-3,
                        help="Use warmup learning rate for this amount of epochs")
    parser.add_argument("--sampling_strategy", type=str, default='bimodal',
                        help="Sampling method of the dataloader")
    
    parser.add_argument("--n_iter", type=int, default=160000,
                        help="Number of data to sampled in train/ test / val dataloader in an epoch.")
    #
    parser.add_argument("--exe_mode",  type=str, default='no_error',
                        help="Executive mode, either 'no_error' or 'error'. 'no_error' means only sampling videos clips that without error label. ")
    parser.add_argument("--tmp_pretrained",  type=str, default=None,
                        help="Path fo pre-trained temporal encoder,\
                        can use the temporal encoder trained on one dataset as pre-trained model to train on ther other dataset.")

    return parser.parse_args()


def save_dict(log_path: Path, txt_name, dict_name):
    with open(log_path / txt_name, 'w') as file:
        file.write(json.dumps(dict_name))


def setup_path(args):
    # --- Model paths and log file locations
    run_path = Path(args.log_path) / args.run_name
    if run_path.exists():
        if not args.job_file_mode:
            if args.resume != 0:
                ans = input(f"Resuming training from latest checkpoint. This overwrites args.txt file. Continue? (y/*)")
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
                print(f"Will overwrite duplicated run_name:{args.run_name}.")

    modelsave_path = run_path / "checkpoints"
    log_path = run_path
    tb_dir = run_path / "tensorboard"
    save_ckpt_dir = Path(args.ckpt_dir) / args.run_name
    performance_report_dir = run_path / "classification_result"

    run_path.mkdir(parents=True, exist_ok=True)
    modelsave_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    save_ckpt_dir.mkdir(parents=True, exist_ok=True)
    performance_report_dir.mkdir(parents=True, exist_ok=True)

    #-- Save the hyperparameter settings & model config for future to review the result and config.
    with open(run_path / 'args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        print(f"Saved run parameters to {run_path / 'args.txt'}")

    with open(args.config, 'r') as cfg:
        config = yaml.safe_load(cfg)

    #-- Store the model args, so the test script can load the modol parameter when inference.
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


def classification_result(loader, model, ACTION_LABEL):
    y_pred = []  # save predction
    y_true = []  # save ground truth

    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            output = model(inputs)
            # for PSR/ASR multi-label classification
            output = torch.sigmoid(output)
            output = output.round().data.cpu().numpy()

            y_pred.extend(output)  # save prediction
            y_true.extend(labels.data.cpu().numpy())  # save ground truth

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

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

def save_model_state_dict(args,model,optimizer,scheduler,epoch,save_path,BEST:bool = False):
    if BEST:
        filename = 'best_model'
    else:
        filename = f'weights_{epoch}'

    if args.baseline:
        torch.save({'mlp_head': model.state_dict(),
                    'optimizar_state_dict':optimizer.state_dict(),
                    'scheduler':scheduler}, 
                    f'{save_path}/{filename}.pth')
        
    elif args.parallel:
        torch.save({'temporal_enc': model.module.temporal_enc.state_dict(),
                    'mlp_head': model.module.mlp_head.state_dict(),
                    'optimizar_state_dict':optimizer.state_dict(),
                    'scheduler':scheduler}, 
                    f'{save_path}/{filename}.pth')
    else:
        torch.save({'temporal_enc': model.temporal_enc.state_dict(),
                    'mlp_head': model.mlp_head.state_dict(),
                    'optimizar_state_dict':optimizer.state_dict(),
                    'scheduler':scheduler},
                    f'{save_path}/{filename}.pth')

    return

if __name__ == "__main__":
    #-- Before the training
    args = set_options()

    #--- Setting for trial run at Windows OS
    if platform.system() == "Windows":
        print(f"Manually putting batch size to 2 and workers to 1 on CPU")
        args.batch_size = 1
        args.workers = 1

    is_impact_asr = is_impact_embedding_dataset(args.data_dir, args.psr_label_path, args.split_dir)

    #TODO: Need further implmentation for other datasets
    # --- Set up dataset specific information
    if 'industreal' in str(args.psr_label_path).lower():
        print("Train on IndustReal dataset...")
        ACTION_LABEL  = IndustReal_ACTION_LABEL
        NUM_COMPONENT = 11
        state_list = [ i for i in range(NUM_COMPONENT)]

    elif 'meccano' in str(args.psr_label_path).lower():
        print("Train on Meccano dataset...")
        ACTION_LABEL = MECCANO_ACTION_LABEL
        NUM_COMPONENT = 17
        state_list = [i for i in range(NUM_COMPONENT)]

    elif is_impact_asr:
        print("Train on IMPACT ASR dataset...")
        ACTION_LABEL = load_impact_component_names(Path(args.psr_label_path))
        NUM_COMPONENT = len(ACTION_LABEL)
        state_list = [i for i in range(NUM_COMPONENT)]
        
    else:
        raise NotImplementedError(
            f"Currently only support meccano, industreal, and IMPACT ASR, but got {args.data_dir}"
        )
    

    # --- Model paths and log file locations
    print(f"Get data from {args.data_dir}")
    split_dir = None
    if not is_impact_asr:
        train_path = args.data_dir + '/train'
        val_path = args.data_dir + '/val'
        test_path = args.data_dir + '/test'
    else:
        split_dir = Path(args.split_dir) if args.split_dir is not None else infer_impact_split_dir(Path(args.data_dir))

    run_path, modelsave_path, log_path, tb_dir, save_ckpt_dir, result_dir = setup_path(args)

    # --- Load model config
    cfg = load_yaml(args.config)
    NUM_FRAMES = cfg.frames
    # --- Load model, different data type would load different model
    if args.dtype == 'embedding':
        preprocess = transforms.Compose([
            # To (FRAMES x [dim of embedding vector]) tensor
            EmbeddinglistToTensor(),
        ])
        # model = temp_enc_LSTM(**vars(cfg))
        if args.baseline:
            model = No_temporal_encoder(NUM_FRAMES,cfg.num_classes)
        else:
            model = VTN_tmp_only(**vars(cfg))
            if args.tmp_pretrained is not None:
                model.load_weights_encoder(Path(args.tmp_pretrained))

    else:
        raise NotImplementedError(f'Expecte onlt embedding, but got {args.dtype}.')

    #--- Use multiple GPU to trian the model
    if torch.cuda.is_available() and args.parallel:
        model = nn.DataParallel(model).cuda()

    #--- Resume weights
    if args.resume > 0:
        #-- Reset the best_val_loss for saving checkpoin
        best_val_loss = 0
        best_val_acc = 0
        #-- Load model args and weight
        cfg = load_yaml(run_path / 'model_args.yaml')
        if args.baseline:
            model = No_temporal_encoder(NUM_FRAMES,cfg.num_classes)
            try: # For legacy version
                model.load_state_dict(torch.load(f'{save_ckpt_dir}/weights_{args.resume}.pth'))
            except:
                model.mlp_head.load_state_dict(torch.load(
                f'{save_ckpt_dir}/weights_{args.resume}.pth')['mlp_head'])
        else:
            model = VTN_tmp_only(**vars(cfg))
            try: # For legacy version, I did not save model by their name before July.
                model.load_state_dict(torch.load(f'{save_ckpt_dir}/weights_{args.resume}.pth'))
            except:
                model.temporal_enc.load_state_dict(torch.load(
                f'{save_ckpt_dir}/weights_{args.resume}.pth')['temporal_enc'])
                model.mlp_head.load_state_dict(torch.load(
                f'{save_ckpt_dir}/weights_{args.resume}.pth')['mlp_head'])

    #-- Load dataset
    if args.dtype == 'embedding':
        if is_impact_asr:
            train_set = ImpactASREmbeddingDataset(
                args.data_dir,
                'train',
                annotation_dir=args.psr_label_path,
                split_dir=split_dir,
                impact_split=args.impact_split,
                transform=preprocess,
                num_frame_per_seg=NUM_FRAMES,
                skip_factor=args.skip_factor,
                sampling_strategy=args.sampling_strategy,
                n_iter=args.n_iter,
                exe_mode=args.exe_mode,
            )
            val_set = ImpactASREmbeddingDataset(
                args.data_dir,
                'val',
                annotation_dir=args.psr_label_path,
                split_dir=split_dir,
                impact_split=args.impact_split,
                transform=preprocess,
                num_frame_per_seg=NUM_FRAMES,
                skip_factor=args.skip_factor,
                sampling_strategy=args.sampling_strategy,
                n_iter=args.n_iter,
                exe_mode=args.exe_mode,
            )
        else:
            train_set = EmbeddingFrameDataset_PSR(train_path, 'train', psr_root_dir=Path(args.psr_label_path)/'train',
                                                  transform=preprocess, num_frame_per_seg=NUM_FRAMES,
                                                  skip_factor= args.skip_factor, num_dig_psr= NUM_COMPONENT,
                                                  sampling_strategy=args.sampling_strategy, load_mode='directories', load_df_label=None, n_iter=args.n_iter, exe_mode=args.exe_mode)

            val_set = EmbeddingFrameDataset_PSR(val_path, 'val', psr_root_dir=Path(args.psr_label_path)/'val',
                                                transform=preprocess, num_frame_per_seg=NUM_FRAMES,
                                                skip_factor=args.skip_factor, num_dig_psr= NUM_COMPONENT,
                                                sampling_strategy=args.sampling_strategy, load_mode='directories', load_df_label=None,n_iter=args.n_iter,exe_mode=args.exe_mode)

    else:
        raise NotImplementedError(f'Expecte onlt embedding, but got {args.dtype}.')

    # --- Build dataloader
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, persistent_workers=True, pin_memory=True, generator=gen)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            num_workers=args.workers, persistent_workers=True, pin_memory=True,generator=gen)
    # --- Tensorboard
    # print("Tensorboard...")
    # tensorboard = SummaryWriter(tb_dir)

    # --- Optimizer
    if (args.dtype == 'embedding'):
        optimizer = SGD(model.parameters(), momentum=0.9,
                        lr=args.lr, weight_decay=args.weight_decay)

    else:
        raise NotImplementedError(f'Expecte onlt embedding, but got {args.dtype}.')

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
        
    if args.resume > 0:
        # Reset the best_val_loss for saving the checkpoints.
        print("Loading opti and scheduler")
        print(f"Load checkpoint at {args.resume} epoch...")
        try:
            optimizer.load_state_dict(torch.load(
                f'{save_ckpt_dir}/weights_{args.resume}.pth')['optimizer_state_dict'])
            scheduler.load_state_dict(torch.load(
                f'{save_ckpt_dir}/weights_{args.resume}.pth')['scheduler'])
        except:
            print('No found ckpt for scheduler and optimizer. Use the new initialized opt. and scheduler.')

    # --- Loss
    loss_func = nn.BCELoss()
    softmax = nn.LogSoftmax(dim=1)
    best_val_acc = 0.0

    # --- Training process
    for epoch in range(max(args.resume+1, 1), args.epochs+1):
        # --- Train
        model.train()
        model.to(DEVICE)
        model.float()
        progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch: {epoch}, loss: 0.000")
        train_loss_list = []
        train_acc_list = []
        multi_label_total = 0
        lr = scheduler.get_last_lr()[0]
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
            loss.backward()
            optimizer.step()

            # Show the F1 score
            train_loss = loss.item()
            # train_acc  = torch.sum(torch.argmax(output, dim=1) == target).cpu().detach().item() / args.batch_size
            train_acc = f1_score(target.data.cpu().numpy(),output_pred, labels=state_list, average=None, zero_division=0.0)
            result_train_acc = np.mean([score for score in train_acc if math.isnan(score) != True])
            # print(f'GT:{output_pred}')
            # print(f'Pred:{target.data.cpu().numpy(),}')
            progress.set_description(f"Epoch: {epoch}, train loss: {train_loss:.6f}, macro F1 score:{result_train_acc:.6f}")
            train_loss_list.append(train_loss)
            train_acc_list.append(np.mean([score for score in train_acc if math.isnan(score) != True]))

            #-- Summary per iter if using the tensorboard
            # avg_train_loss = sum(avg_train_loss_list) / \
            #     len(avg_train_loss_list)
            # avg_train_acc = sum(avg_train_acc_list) / len(avg_train_acc_list)
            # tensorboard.add_scalar(
            #     'train_loss', train_loss, epoch * len(train_loader) + i)
            # tensorboard.add_scalar(
            #     'train_f1', result_train_acc * 100, epoch * len(train_loader) + i)
            # tensorboard.add_scalar('lr', lr, epoch * len(train_loader) + i)

        #-- Summary per epoch if using the tensorboard
        # avg_train_loss = sum(avg_train_loss_list)/ len(avg_train_loss_list)
        # avg_train_acc = sum(avg_train_acc_list) / len(avg_train_acc_list)
        # tensorboard.add_scalar('train_loss',avg_train_loss, epoch * len(train_loader) + i)
        # tensorboard.add_scalar('train_f1',avg_train_acc * 100, epoch * len(train_loader) + i)
        # tensorboard.add_scalar('lr', lr, epoch * len(train_loader) + i)

        #--- Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        y_pred = []
        y_true = []
        val_loss_list = []
        val_acc_list = []
        progress = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch:{epoch}, validating")
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
                y_true.extend(target.data.cpu().numpy())
                # val_acc = torch.sum(torch.argmax(output, dim=1) == target).cpu().detach().item() / args.batch_size
                # val_acc =  torch.sum(output.round() == target).cpu().detach().item() / args.batch_size
                val_acc = f1_score(target.data.cpu().numpy(),output_pred, labels=state_list, average=None, zero_division=0.0)
                val_acc_list.append(np.mean([score for score in val_acc if math.isnan(score) != True]))
                val_acc = np.mean([score for score in val_acc if math.isnan(score) != True])

            progress.set_description(f"Epoch: {epoch}, val loss: {val_loss:.6f}, macro F1 score:{val_acc:.6f}")
            # --- Summary per iter
            # tensorboard.add_scalar('val_loss', val_loss,epoch * len(val_loader) + i)
            # tensorboard.add_scalar('val_acc', val_acc*100, epoch * len(val_loader) + i)

        print(classification_report(y_true, y_pred, target_names = ACTION_LABEL,output_dict = False, zero_division = np.nan))
        # --- Creating dict to save training progress
        dict_to_save = {
            "train_loss": np.mean(train_loss_list),
            "train_accuracy": np.mean(train_acc_list),
            "validation_loss": np.mean(val_loss_list),
            "validation_accuracy": np.mean(val_acc_list),
            "lr": lr
        }
        c = datetime.datetime.now().strftime('%H:%M:%S')
        log_str = f"{c} - Epoch: {epoch}\tTrain loss: {np.mean(train_loss_list):.6f}\tTrain acc (%): {np.mean(train_acc_list)*100:.4f}\t" \
            f"Val loss: {np.mean(val_loss_list):.6f}\tVal acc (%): {np.mean(val_acc_list)*100:.4f}\t" \
            f"LR: {lr:.6f}\n"
        # #--- Summary per epoch
        # tensorboard.add_scalar('val_loss', sum(avg_val_loss) / len(avg_val_loss), epoch)
        # tensorboard.add_scalar('val_acc', sum(avg_val_acc) / len(avg_val_acc) * 100, epoch)

        # --- Save weights to checkpoint folder (for saving storage space, store the weights every 10 eps)
        if (epoch % 10 ==0):
            save_model_state_dict(args,model,optimizer,scheduler,epoch, save_ckpt_dir, BEST = False)

        # --- Save the best model depend on the highest val acc
        if epoch == 1:
            best_val_loss = np.mean(val_loss_list)
            best_val_acc  = np.mean(val_acc_list)
        elif (np.mean(val_acc_list)) > best_val_acc:
            print(f'Update best model. at {epoch}')
            save_model_state_dict(args,model,optimizer,scheduler,epoch, modelsave_path, BEST = True)
                
            best_val_loss = np.mean(val_loss_list)
            best_val_acc  = np.mean(val_acc_list)
            
        elif epoch == args.epochs:
            print(f'Save the final resulted model.')
            save_model_state_dict(args,model,optimizer,scheduler,epoch, modelsave_path, BEST = False)

        else:
            pass

        #-- Save the data log
        scheduler.step()
        progressfilename = 'train_progress_epoch' + str(epoch) + '.txt'
        save_dict(log_path, progressfilename, dict_to_save)
        f = open(log_path / 'train_log.txt', 'a')
        f.write(log_str)
        f.close()
        print(log_str)

        #-- Uncommend if using hte tensorboard
        # tensorboard.add_figure("Train Confusion Matrix",createConfusionMatrix(train_loader),epoch)
        # tensorboard.add_figure("Val Confusion Matrix",createConfusionMatrix(val_loader),epoch)
        # tensorboard.add_figure("Val Confusion Matirx", createMutliLabelConfusionMatrix(val_loader),epoch)
        report = report = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=np.nan)
        val_file = 'result_on_val_' + str(epoch) + '.json'
        with open(result_dir / val_file, 'w') as outfile:
            json.dump(report, outfile)

        plot_trainlog_result(epoch, run_path, run_path)

    # tensorboard.close()

    #-- Run the testset
    avg_test_loss = []
    avg_test_acc = []
    #
    print("Run test on the testset")
    if args.dtype == 'embedding':
        if is_impact_asr:
            test_set = ImpactASREmbeddingDataset(
                args.data_dir,
                'test',
                annotation_dir=args.psr_label_path,
                split_dir=split_dir,
                impact_split=args.impact_split,
                transform=preprocess,
                num_frame_per_seg=NUM_FRAMES,
                skip_factor=args.skip_factor,
                sampling_strategy=args.sampling_strategy,
                n_iter=args.n_iter,
                exe_mode=args.exe_mode,
            )
        else:
            test_set = EmbeddingFrameDataset_PSR(
                test_path,
                'test',
                psr_root_dir=Path(args.psr_label_path) / 'test',
                transform=preprocess,
                num_frame_per_seg=NUM_FRAMES,
                skip_factor=args.skip_factor,
                num_dig_psr=NUM_COMPONENT,
                sampling_strategy=args.sampling_strategy,
                load_mode='directories',
                load_df_label=None,
                n_iter=args.n_iter,
                exe_mode=args.exe_mode,
            )
    else:
        raise NotImplementedError(f'Only for embedding, but got {args.dtype}.')
    
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             num_workers=args.workers, persistent_workers=True, pin_memory=True)
    report = classification_result(test_loader, model, ACTION_LABEL)
    with open(result_dir / 'result_on_test.json', 'w') as outfile:
        json.dump(report, outfile)
    print("Training Complete.")
