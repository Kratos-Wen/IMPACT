import argparse
import json
import os

import torch
from tqdm import tqdm

from .configs.utils import setup_cfg
from .utils.dataset import DataLoader, create_dataset
from .utils.evaluate import Checkpoint
from .utils.train_tools import save_results


def is_atr_task(cfg):
    return str(cfg.impact_task if "impact_task" in cfg else "").upper().startswith("ATR_")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one FACT checkpoint on the test split.")
    parser.add_argument("--cfg", required=True, help="Path to yaml config.")
    parser.add_argument("--ckpt", required=True, help="Path to network.iter-*.net weight file.")
    parser.add_argument("--split", required=True, help="Split name such as split1.")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index inside current visibility.")
    parser.add_argument("--impact-root", default=None, help="Override IMPACT root path.")
    parser.add_argument("--impact-task", default=None, help="Optional override such as CAS/FAS_L/FAS_R.")
    parser.add_argument("--impact-feature-type", default=None, help="Optional override such as i3d/videomaev2.")
    parser.add_argument("--save-gz", default=None, help="Optional output .gz path for full test checkpoint.")
    parser.add_argument("--save-json", default=None, help="Optional output .json path for metrics only.")
    return parser.parse_args()


def main():
    args = parse_args()

    set_cfgs = ["aux.gpu", str(args.gpu), "split", args.split]
    if args.impact_root:
        set_cfgs.extend(["impact_root", args.impact_root])
    if args.impact_task:
        set_cfgs.extend(["impact_task", args.impact_task])
    if args.impact_feature_type:
        set_cfgs.extend(["impact_feature_type", args.impact_feature_type])

    cfg = setup_cfg([args.cfg], set_cfgs)

    if is_atr_task(cfg):
        raise NotImplementedError(
            "ATR evaluation is not implemented in this helper. "
            "Use it for CAS/FAS/PPR checkpoints."
        )

    try:
        torch.cuda.set_device(f"cuda:{cfg.aux.gpu}")
    except Exception:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.aux.gpu)

    dataset, _, test_dataset = create_dataset(cfg)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    from .models.blocks import FACT

    model = FACT(cfg, dataset.input_dimension, dataset.nclasses)
    weights = torch.load(args.ckpt, map_location="cpu")
    if "frame_pe.pe" in weights:
        del weights["frame_pe.pe"]
    if "action_pe.pe" in weights:
        del weights["action_pe.pe"]
    model.load_state_dict(weights, strict=False)
    model.eval().cuda()

    ckpt = Checkpoint(
        -1,
        bg_class=([] if cfg.eval_bg else dataset.bg_class),
        index2label=dataset.index2label,
    )

    with torch.no_grad():
        for vnames, seq_list, train_label_list, eval_label_list in tqdm(loader):
            seq_list = [s.cuda() for s in seq_list]
            train_label_list = [s.cuda() for s in train_label_list]
            video_saves = model(seq_list, train_label_list)
            save_results(ckpt, vnames, eval_label_list, video_saves)

    ckpt.compute_metrics()

    print(json.dumps(ckpt.metrics, indent=2, sort_keys=True))

    if args.save_gz:
        os.makedirs(os.path.dirname(args.save_gz), exist_ok=True)
        ckpt.save(args.save_gz)
        print(f"Saved checkpoint results to {args.save_gz}")

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(ckpt.metrics, f, indent=2, sort_keys=True)
        print(f"Saved metrics json to {args.save_json}")


if __name__ == "__main__":
    main()
