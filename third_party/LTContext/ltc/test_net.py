import os
from os.path import join, splitext
from tqdm import tqdm
from yacs.config import CfgNode

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import pandas as pd


import ltc.utils.checkpoint as cu
import ltc.utils.misc as misc
from ltc.dataset import loader
from ltc.model import model_builder
from ltc.utils.metrics import calculate_metrics, load_label_names
from ltc.utils.atr import evaluate_atr_predictions


import ltc.utils.logging as logging

logger = logging.get_logger(__name__)


def _is_atr_task(cfg: CfgNode) -> bool:
    return str(cfg.MODEL.LOSS_FUNC).lower() == "masked_bce"


@torch.no_grad()
def eval_model(
        val_loader: DataLoader,
        model: nn.Module,
        device: torch.device.type,
        cfg: CfgNode):
    """
    Evaluate the model on the val set.
    :param val_loader: data loader to provide validation data.
    :param model: model to evaluate the performance.
    :param device: device to use (cuda or cpu)
    :param cfg:
    :return:
    """
    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    logger.info(f"Testing the trained model.")

    cfg_filename = splitext(cfg.CONFIG_FILE.split("/")[-1])[0]
    save_path = join(cfg.TEST.SAVE_RESULT_PATH,
                     cfg.TEST.DATASET,
                     cfg_filename)
    logger.info(save_path)
    os.makedirs(save_path, exist_ok=True)

    test_metrics = {"video_name": []}
    ignored_class_idx = cfg.TEST.IGNORED_CLASSES + [cfg.MODEL.PAD_IGNORE_IDX]
    logger.info(f"Ignored class idxs: {ignored_class_idx}")
    mapping_path = cfg.DATA.MAPPING_FILE if os.path.isabs(cfg.DATA.MAPPING_FILE) else join(cfg.DATA.PATH_TO_DATA_DIR, cfg.DATA.MAPPING_FILE)
    label_names = load_label_names(mapping_path)
    is_atr_task = _is_atr_task(cfg)
    atr_predictions = {}

    for batch_dict in tqdm(val_loader, total=len(val_loader)):
        misc.move_to_device(batch_dict, device)
        logits = model(batch_dict['features'], batch_dict['masks'])
        if is_atr_task:
            probs = misc.prepare_probabilities(logits).cpu().numpy()
            for idx in range(probs.shape[0]):
                video_name = batch_dict['video_name'][0][idx]
                valid_len = int(batch_dict['masks'][idx, 0].sum().item())
                atr_predictions[video_name] = probs[idx, :valid_len]
                if cfg.TEST.SAVE_PREDICTIONS:
                    base_path = join(save_path, video_name)
                    os.makedirs(base_path, exist_ok=True)
                    np.save(join(base_path, "prob.npy"), atr_predictions[video_name])
        else:
            mb_size = batch_dict["targets"].shape[0]
            assert mb_size == 1, "Validation batch size should be one."

            prediction = misc.prepare_prediction(logits)

            target = batch_dict['targets'].cpu()
            prediction = prediction.cpu()
            video_name = batch_dict['video_name'][0][0]

            video_metrics = calculate_metrics(target,
                                              prediction,
                                              ignored_class_idx,
                                              label_names)

            test_metrics['video_name'].append(video_name)
            for name, score in video_metrics.items():
                if name != 'Edit':
                    score = score * 100
                if name not in test_metrics:
                    test_metrics[name] = []
                test_metrics[name].append(score)

            if cfg.TEST.SAVE_PREDICTIONS:
                base_path = join(save_path, video_name)
                os.makedirs(base_path, exist_ok=True)
                np.save(join(base_path, "pred.npy"), prediction[0].long().numpy())
                np.save(join(base_path, "gt.npy"), target[0].long().numpy())

    if is_atr_task:
        segments_dir = cfg.DATA.ATR_SEGMENTS_DIR if os.path.isabs(cfg.DATA.ATR_SEGMENTS_DIR) else join(
            cfg.DATA.PATH_TO_DATA_DIR,
            cfg.DATA.ATR_SEGMENTS_DIR,
        )
        mean_metrics = evaluate_atr_predictions(
            atr_predictions,
            segments_dir,
            cfg.TRAIN.EVAL_SPLIT,
            cfg.DATA.CV_SPLIT_NUM,
            mapping_path,
        )
        pd.DataFrame([mean_metrics]).round(5).to_csv(join(save_path, "testing_metrics.csv"), index=False)
        logger.info("Testing metric:")
        logging.log_json_stats(mean_metrics, precision=1)
    else:
        test_res_df = pd.DataFrame(test_metrics)
        test_res_df.round(5).to_csv(join(save_path, "testing_metrics.csv"))
        metric_columns = [c for c in test_res_df.columns if c != 'video_name']
        mean_metrics = test_res_df[metric_columns].mean()
        logger.info("Testing metric:")
        logging.log_json_stats(mean_metrics, precision=1)


def test(cfg: CfgNode):
    """
    Train an action segmentation model for many epochs on train set and validate it on val set
    :param cfg: config file. Details can be found in ltc/config/defaults.py
    :return:
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    logging.setup_logging(cfg.OUTPUT_DIR, cfg.EXPR_NUM)

    model = model_builder.build_model(cfg)
    logger.info(f"Number of params: {misc.params_to_string(misc.params_count(model))}")

    if cfg.NUM_GPUS > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Transfer the model to device(s)
    model = model.to(device)

    checkpoint_path = cfg.TEST.CHECKPOINT_PATH
    cu.load_model(
        checkpoint_path,
        model=model,
        num_gpus=cfg.NUM_GPUS,
    )
    test_loader = loader.construct_loader(cfg, cfg.TRAIN.EVAL_SPLIT)
    eval_model(test_loader, model, device, cfg)
