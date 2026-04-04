import os
import sys
import argparse
from os.path import join, split, splitext
from yacs.config import CfgNode

import ltc.utils.checkpoint as cu
from ltc.config.defaults import get_cfg, _assert_and_infer_cfg
import ltc.utils.misc as misc
from ltc.train_net import train
from ltc.test_net import test


def _count_mapping_classes(mapping_file: str) -> int:
    with open(mapping_file, "r") as f:
        return sum(1 for line in f if line.strip())


def _get_impact_task_settings(label_mode: str):
    settings = {
        "CAS": {
            "task_type": "segmentation",
            "splits_dir": "splits_CAS",
            "gt_dir": "groundTruth_CAS",
            "mapping_file": "mapping_CAS.txt",
        },
        "FAS_L": {
            "task_type": "segmentation",
            "splits_dir": "splits_FAS_L",
            "gt_dir": "groundTruth_FAS_L",
            "mapping_file": "mapping_FAS.txt",
        },
        "FAS_R": {
            "task_type": "segmentation",
            "splits_dir": "splits_FAS_R",
            "gt_dir": "groundTruth_FAS_R",
            "mapping_file": "mapping_FAS.txt",
        },
        "PPR_L": {
            "task_type": "segmentation",
            "splits_dir": "splits_PPR_L",
            "gt_dir": "groundTruth_PPR_L",
            "mapping_file": "mapping_PPR.txt",
        },
        "PPR_R": {
            "task_type": "segmentation",
            "splits_dir": "splits_PPR_R",
            "gt_dir": "groundTruth_PPR_R",
            "mapping_file": "mapping_PPR.txt",
        },
        "ATR_L": {
            "task_type": "atr",
            "splits_dir": "splits_FAS_L",
            "gt_dir": "groundTruth_ATR_L",
            "mask_dir": "mask_ATR_L",
            "atr_segments_dir": "atr_segments_L",
            "mapping_file": "mapping_ATR.txt",
        },
        "ATR_R": {
            "task_type": "atr",
            "splits_dir": "splits_FAS_R",
            "gt_dir": "groundTruth_ATR_R",
            "mask_dir": "mask_ATR_R",
            "atr_segments_dir": "atr_segments_R",
            "mapping_file": "mapping_ATR.txt",
        },
    }
    if label_mode not in settings:
        raise ValueError(f"Invalid --impact-label-mode={label_mode}.")
    return settings[label_mode]


def _apply_impact_overrides(cfg: CfgNode, args):
    if str(cfg.TRAIN.DATASET).lower() != "impact":
        return

    impact_root = args.impact_root if args.impact_root else cfg.DATA.PATH_TO_DATA_DIR
    impact_root = os.path.abspath(os.path.expanduser(impact_root))

    label_mode = args.impact_label_mode if args.impact_label_mode else "CAS"
    label_mode = str(label_mode).upper()
    task_settings = _get_impact_task_settings(label_mode)

    feature_type = args.impact_feature_type if args.impact_feature_type else cfg.DATA.FEATURE_TYPE
    feature_type = str(feature_type).lower()
    if feature_type == "auto":
        feature_type = "i3d" if int(cfg.MODEL.INPUT_DIM) == 1024 else "videomaev2"
    if feature_type not in ["i3d", "videomaev2"]:
        raise ValueError(f"Invalid --impact-feature-type={feature_type}.")

    if args.impact_split > 0:
        cfg.DATA.CV_SPLIT_NUM = int(args.impact_split)

    cfg.DATA.PATH_TO_DATA_DIR = impact_root
    cfg.DATA.TASK_TYPE = task_settings["task_type"]
    cfg.DATA.SPLITS_DIR = task_settings["splits_dir"]
    cfg.DATA.GROUND_TRUTH_DIR = task_settings["gt_dir"]
    cfg.DATA.MASK_DIR = task_settings.get("mask_dir", "")
    cfg.DATA.ATR_SEGMENTS_DIR = task_settings.get("atr_segments_dir", "")
    cfg.DATA.MAPPING_FILE = task_settings["mapping_file"]
    cfg.DATA.FEATURES_DIR = "features_i3d" if feature_type == "i3d" else "features"
    cfg.DATA.FEATURE_TYPE = feature_type
    cfg.DATA.SKIP_MISSING_FEATURES = True
    cfg.MODEL.INPUT_DIM = 1024 if feature_type == "i3d" else 1408
    if cfg.DATA.TASK_TYPE == "atr":
        cfg.MODEL.LOSS_FUNC = "masked_bce"

    mapping_path = cfg.DATA.MAPPING_FILE
    if not os.path.isabs(mapping_path):
        mapping_path = join(impact_root, mapping_path)
    cfg.MODEL.NUM_CLASSES = _count_mapping_classes(mapping_path)

    print("[IMPACT] root=", impact_root)
    print("[IMPACT] label_mode=", label_mode, "feature_type=", feature_type)
    print("[IMPACT] split=", cfg.DATA.CV_SPLIT_NUM)
    print("[IMPACT] task_type=", cfg.DATA.TASK_TYPE)
    print("[IMPACT] splits_dir=", cfg.DATA.SPLITS_DIR)
    print("[IMPACT] gt_dir=", cfg.DATA.GROUND_TRUTH_DIR)
    if cfg.DATA.MASK_DIR:
        print("[IMPACT] mask_dir=", cfg.DATA.MASK_DIR)
    if cfg.DATA.ATR_SEGMENTS_DIR:
        print("[IMPACT] atr_segments_dir=", cfg.DATA.ATR_SEGMENTS_DIR)
    print("[IMPACT] features_dir=", cfg.DATA.FEATURES_DIR)
    print("[IMPACT] mapping_file=", cfg.DATA.MAPPING_FILE)
    print("[IMPACT] input_dim=", cfg.MODEL.INPUT_DIM, "num_classes=", cfg.MODEL.NUM_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Provide the path to config and options. "
                    "See ltc/config/defaults.py for all options"
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/Assembly101/LTContext.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See ltc/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--impact-root",
        dest="impact_root",
        help="Override IMPACT data root path.",
        default="",
        type=str,
    )
    parser.add_argument(
        "--impact-label-mode",
        dest="impact_label_mode",
        help="IMPACT label mode.",
        default="",
        choices=["CAS", "FAS_L", "FAS_R", "PPR_L", "PPR_R", "ATR_L", "ATR_R"],
        type=str,
    )
    parser.add_argument(
        "--impact-feature-type",
        dest="impact_feature_type",
        help="IMPACT feature type.",
        default="",
        choices=["i3d", "videomaev2"],
        type=str,
    )
    parser.add_argument(
        "--impact-split",
        dest="impact_split",
        help="Override IMPACT split id.",
        default=-1,
        type=int,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """

    :param args: arguments including `cfg_file`, and `opts`
    :return:
        config file
    """

    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    _apply_impact_overrides(cfg, args)

    # Inherit parameters from args.
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "resume_expr_num"):
        cfg.RESUME_EXPR_NUM = args.resume_expr_num
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    cfg.CONFIG_FILE = args.cfg_file
    cfg_file_name = splitext(split(args.cfg_file)[1])[0]
    cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, cfg_file_name)

    return _assert_and_infer_cfg(cfg)


def prep_output_paths(cfg: CfgNode):
    """
    Preparing the path for tensorboard summary, config log and checkpoints
    :param cfg:
    :return:
    """
    if cfg.TRAIN.ENABLE:
        summary_path = misc.check_path(join(cfg.OUTPUT_DIR, "summary"))
        cfg.EXPR_NUM = misc.find_latest_experiment(join(cfg.OUTPUT_DIR, "summary")) + 1
        if cfg.TRAIN.AUTO_RESUME and cfg.TRAIN.RESUME_EXPR_NUM > 0:
            cfg.EXPR_NUM = cfg.TRAIN.RESUME_EXPR_NUM
        cfg.SUMMARY_PATH = misc.check_path(join(summary_path, "{}".format(cfg.EXPR_NUM)))
        cfg.CONFIG_LOG_PATH = misc.check_path(
            join(cfg.OUTPUT_DIR, "config", "{}".format(cfg.EXPR_NUM))
        )
        # Create the checkpoint dir.
        cu.make_checkpoint_dir(cfg.OUTPUT_DIR, cfg.EXPR_NUM)
    if cfg.TEST.ENABLE:
        os.makedirs(cfg.TEST.SAVE_RESULT_PATH, exist_ok=True)


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    prep_output_paths(cfg)
    if cfg.TRAIN.ENABLE:
        train(cfg=cfg)

    if cfg.TEST.ENABLE:
        test(cfg)


if __name__ == "__main__":
    main()
