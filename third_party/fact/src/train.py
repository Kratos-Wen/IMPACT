#!/usr/bin/python3

import numpy as np
import argparse
import os
import json
from torch import optim
import torch
import wandb

from .utils.dataset import DataLoader, create_dataset
from .utils.evaluate import Checkpoint
from .utils.atr import evaluate_atr_predictions
from .home import get_project_base
from .configs.utils import cfg2flatdict, setup_cfg
from .utils.train_tools import resume_ckpt, compute_null_weight, save_results
from .models.loss import MatchCriterion


def is_atr_task(cfg):
    return str(cfg.impact_task if "impact_task" in cfg else "").upper().startswith("ATR_")

def evaluate(global_step, net, loader, run, savedir, split_name):
    print(f"EVALUATING {split_name.upper()}" + "~"*10)

    os.makedirs(savedir, exist_ok=True)
    if is_atr_task(net.cfg):
        pred_dict = {}
        net.eval()
        with torch.no_grad():
            for batch_idx, (vnames, seq_list, train_label_list, eval_label_list) in enumerate(loader):
                seq_list = [s.cuda() for s in seq_list]
                train_label_list = [
                    {
                        "target": item["target"].cuda(),
                        "mask": item["mask"].cuda(),
                        "video_name": item.get("video_name", vnames[idx]),
                    }
                    for idx, item in enumerate(train_label_list)
                ]
                video_saves = net(seq_list, train_label_list)
                for vname, save_data in zip(vnames, video_saves):
                    pred_dict[vname] = save_data["prob"]

        impact_root = net.cfg.impact_root if ("impact_root" in net.cfg and net.cfg.impact_root) else \
            os.path.abspath(os.path.join(BASE, '..', 'data', 'IMPACT'))
        side = str(net.cfg.impact_task).upper().split('_')[-1]
        metrics = evaluate_atr_predictions(
            pred_dict,
            os.path.join(impact_root, f'atr_segments_{side}'),
            split_name,
            net.cfg.split,
            os.path.join(impact_root, 'mapping_ATR.txt'),
        )
        print(", ".join([f"{k}:{v:.1f}" for k, v in metrics.items() if isinstance(v, (int, float)) and v == v]) + '\n')
        if run is not None:
            run.log({f'{split_name}-metric/{k}': v for k, v in metrics.items()}, step=global_step + 1)
        net.train()
        return metrics

    ckpt = Checkpoint(
        global_step+1,
        bg_class=([] if net.cfg.eval_bg else loader.dataset.bg_class),
        index2label=loader.dataset.index2label,
    )
    net.eval()
    with torch.no_grad():
        for batch_idx, (vnames, seq_list, train_label_list, eval_label_list) in enumerate(loader):

            seq_list = [ s.cuda() for s in seq_list ]
            train_label_list = [ s.cuda() for s in train_label_list ]
            video_saves = net(seq_list, train_label_list)
            save_results(ckpt, vnames, eval_label_list, video_saves)

    net.train()
    ckpt.compute_metrics()

    log_dict = {}
    string = ""
    for k, v in ckpt.metrics.items():
        string += "%s:%.1f, " % (k, v)
        log_dict[f'{split_name}-metric/{k}'] = v
    print(string + '\n')
    if run is not None:
        run.log(log_dict, step=global_step+1)

    fname = "%d.gz" % (global_step+1) 
    ckpt.save(os.path.join(savedir, fname))

    return ckpt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", dest="cfg_file", nargs="*",
                            help="optional config file", default=[])
    parser.add_argument("--set", dest="set_cfgs",
            help="set config keys", default=None, nargs=argparse.REMAINDER,)

    args = parser.parse_args()
    BASE = get_project_base()

    ### initialize experiment #########################################################
    cfg = setup_cfg(args.cfg_file, args.set_cfgs)
    try:
        torch.cuda.set_device('cuda:%d'%cfg.aux.gpu)
    except Exception as e:
        print(e)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.aux.gpu)

    print('============')
    print(cfg)
    print('============')

    if cfg.aux.debug:
        seed = 1 
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    logdir = os.path.join(BASE, cfg.aux.logdir)
    ckptdir = os.path.join(logdir, 'ckpts')
    savedir = os.path.join(logdir, 'saves')
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    print('Saving log at', logdir)

    try:
        run = wandb.init(
                    project=cfg.aux.wandb_project, entity=cfg.aux.wandb_user,
                    dir=cfg.aux.logdir,
                    group=cfg.aux.exp, resume="allow",
                    config=cfg2flatdict(cfg),
                    reinit=True, save_code=False,
                    mode="offline" if cfg.aux.debug else "online",
                    )
    except Exception as e:
        print("WARNING: Failed to initialize wandb.")
        run = None

    argSaveFile = os.path.join(logdir, 'args.json')
    with open(argSaveFile, 'w') as f:
        json.dump(cfg, f, indent=True)

    ### load dataset #########################################################
    dataset, val_dataset, test_dataset = create_dataset(cfg)
    if not cfg.aux.debug:
        trainloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    else:
        trainloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    print('Train dataset', dataset)
    print('Val dataset  ', val_dataset)
    print('Test dataset ', test_dataset)

    ### create network #########################################################
    if cfg.dataset == 'epic':
        from .models.blocks_SepVerbNoun import FACT
        net = FACT(cfg, dataset.input_dimension, 98, 301)
    else:
        from .models.blocks import FACT
        net = FACT(cfg, dataset.input_dimension, dataset.nclasses)

    if (not is_atr_task(cfg)) and cfg.Loss.nullw == -1:
        compute_null_weight(cfg, dataset)
    if not is_atr_task(cfg):
        net.mcriterion = MatchCriterion(cfg, dataset.nclasses, dataset.bg_class)

    global_step, ckpt_file = resume_ckpt(cfg, logdir)
    if ckpt_file is not None:
        ckpt = torch.load(ckpt_file, map_location="cpu")
        if 'frame_pe.pe' in ckpt: del ckpt['frame_pe.pe']
        if 'action_pe.pe' in ckpt: del ckpt['action_pe.pe']
        net.load_state_dict(ckpt, strict=False)
    net.cuda()

    print(net)

    if cfg.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(),
                            lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(),
                            lr=cfg.lr, weight_decay=cfg.weight_decay)

    ### start training #########################################################
    start_epoch = global_step // len(trainloader)
    ckpt = None if is_atr_task(cfg) else Checkpoint(
        -1,
        bg_class=([] if net.cfg.eval_bg else dataset.bg_class),
        eval_edit=False,
        index2label=dataset.index2label,
    )
    best_ckpt, best_metric = None, 0
    best_model_path = None
    val_savedir = os.path.join(savedir, 'val')
    test_savedir = os.path.join(savedir, 'test')
    os.makedirs(val_savedir, exist_ok=True)
    os.makedirs(test_savedir, exist_ok=True)

    print(f'Start Training from Epoch {start_epoch}...')
    for eidx in range(start_epoch, cfg.epoch):

        for batch_idx, (vnames, seq_list, train_label_list, eval_label_list) in enumerate(trainloader):

            seq_list = [ s.cuda() for s in seq_list ]
            if is_atr_task(cfg):
                train_label_list = [
                    {
                        "target": item["target"].cuda(),
                        "mask": item["mask"].cuda(),
                        "video_name": item.get("video_name", vnames[idx]),
                    }
                    for idx, item in enumerate(train_label_list)
                ]
            else:
                train_label_list = [ s.cuda() for s in train_label_list ]

            optimizer.zero_grad()
            loss, video_saves = net(seq_list, train_label_list, compute_loss=True)
            loss.backward()

            if cfg.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip_grad_norm)
            optimizer.step()

            if not is_atr_task(cfg):
                save_results(ckpt, vnames, eval_label_list, video_saves)

            # print some progress information
            if (global_step+1) % cfg.aux.print_every == 0:

                log_dict = {}
                string = "Iter%d, " % (global_step+1)
                _L = len(string)
                if is_atr_task(cfg):
                    loss_val = float(loss.item())
                    log_dict["train-loss/loss"] = loss_val
                    string += f"loss:{loss_val:.3f}, "
                else:
                    ckpt.compute_metrics()
                    ckpt.average_losses()
                    for k, v in ckpt.loss.items():
                        log_dict[f"train-loss/{k}"] = v
                        string += f"{k}:{v:.1f}, "
                print(string)

                if not is_atr_task(cfg):
                    string = " " * _L 
                    for k, v in ckpt.metrics.items():
                        string += "%s:%.3f, " % (k, v)
                        log_dict['train-metric/'+k] = v
                    print(string)

                if run is not None:
                    run.log(log_dict, step=global_step+1)

                if not is_atr_task(cfg):
                    ckpt = Checkpoint(
                        -1,
                        bg_class=(dataset.bg_class if cfg.eval_bg else []),
                        eval_edit=False,
                        index2label=dataset.index2label,
                    )

            # test and save model every x iterations
            if global_step != 0 and (global_step+1) % cfg.aux.eval_every == 0:
                val_ckpt = evaluate(global_step, net, valloader, run, val_savedir, "val")
                network_file = ckptdir + '/network.iter-' + str(global_step+1) + '.net'
                net.save_model(network_file)
                if is_atr_task(cfg):
                    selection_metric = val_ckpt.get('ATR_mAP_present', 0.0)
                else:
                    selection_metric = val_ckpt.metrics.get('Phase_MacroF1', val_ckpt.metrics['F1@0.50'])
                if selection_metric >= best_metric:
                    best_ckpt = val_ckpt
                    best_metric = selection_metric
                    best_model_path = network_file

            global_step += 1

        if cfg.lr_decay > 0 and (eidx + 1) % cfg.lr_decay == 0:
            for g in optimizer.param_groups:
                g['lr'] = cfg.lr * 0.1
            print('------------------------------------Update Learning rate--------------------------------')

    if best_model_path is not None and os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} (val metric={best_metric:.2f})")
        ckpt = torch.load(best_model_path, map_location="cpu")
        if 'frame_pe.pe' in ckpt:
            del ckpt['frame_pe.pe']
        if 'action_pe.pe' in ckpt:
            del ckpt['action_pe.pe']
        net.load_state_dict(ckpt, strict=False)
    else:
        print("No validation checkpoint selected; using last model for test.")

    evaluate(global_step, net, testloader, run, test_savedir, "test")
    if best_ckpt is not None and not is_atr_task(cfg):
        print(f'Best Checkpoint: {best_ckpt.iteration}')
        best_ckpt.eval_edit = True
        best_ckpt.compute_metrics()
        best_ckpt.save(os.path.join(logdir, 'best_ckpt.gz'))
    elif best_ckpt is not None:
        print('Best ATR validation metric: {:.2f}'.format(best_metric))
    else:
        print('No validation checkpoint selected; skipping best_ckpt.gz save.')
    if run is not None:
        run.finish()

    # create a file to mark this experiment has completed
    finish_proof_fname = os.path.join(logdir, "FINISH_PROOF")
    open(finish_proof_fname, "w").close()
