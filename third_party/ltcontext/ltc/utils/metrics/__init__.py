from typing import List, Dict
import numpy as np
from torch import Tensor

from .base import Metric
from .segmentation import MoFAccuracyMetric
from .fully_supervised import F1Score, Edit


def _normalize_label_name(label_name: str) -> str:
    return str(label_name).strip().lower()


def infer_phase_class_ids(class_names):
    if not class_names:
        return None
    name_to_idx = {_normalize_label_name(name): idx for idx, name in enumerate(class_names)}
    required = ["normal", "anomaly", "recovery"]
    if not all(name in name_to_idx for name in required):
        return None
    return {name: name_to_idx[name] for name in required}


def load_label_names(mapping_path: str) -> List[str]:
    labels = []
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, label = line.split(maxsplit=1)
            labels.append(label)
    return labels


def _compute_phase_metrics(target: Tensor, pred: Tensor, class_names) -> Dict:
    phase_ids = infer_phase_class_ids(class_names)
    if phase_ids is None:
        return {}

    gt_arr = target.numpy().reshape(-1)
    pred_arr = pred.numpy().reshape(-1)
    eps = 1e-8
    per_class_f1 = {}
    for phase_name, class_id in phase_ids.items():
        tp = np.sum((gt_arr == class_id) & (pred_arr == class_id))
        fp = np.sum((gt_arr != class_id) & (pred_arr == class_id))
        fn = np.sum((gt_arr == class_id) & (pred_arr != class_id))
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        per_class_f1[phase_name] = 2.0 * precision * recall / (precision + recall + eps)

    return {
        "Phase_Acc": float((gt_arr == pred_arr).mean()),
        "Phase_MacroF1": float(np.mean(list(per_class_f1.values()))),
        "Phase_F1_Normal": float(per_class_f1["normal"]),
        "Phase_F1_Anomaly": float(per_class_f1["anomaly"]),
        "Phase_F1_Recovery": float(per_class_f1["recovery"]),
    }


def calculate_metrics(target: Tensor, pred: Tensor, ignored_class_ids: List[int], class_names=None) -> Dict:
    """
    Calculates the action segmentation metrics (MoF, Edit, F1@{10, 25, 50}) for a video
    :param target: a Tensor of shape [batch_size, sequence_len]
    :param pred: a Tensor of shape [batch_size, sequence_len]
    :param ignored_class_ids: a list of class ids to ignore during calculation
    :return:
      a dict of metrics values
    """
    assert target.shape[0] == 1, "Batch size should be one for validation due to limitation of metric functions."

    result_dict = {}
    mof_func = MoFAccuracyMetric(ignore_ids=ignored_class_ids)
    edit_func = Edit(ignore_ids=ignored_class_ids)
    f1_func = F1Score(ignore_ids=ignored_class_ids)

    mof_func.add(target, pred)
    edit_func.add(target, pred)
    f1_func.add(target, pred)

    result_dict['MoF'] = mof_func.summary()
    result_dict['Edit'] = edit_func.summary()
    f1_dict = f1_func.summary()
    result_dict.update(f1_dict)
    result_dict.update(_compute_phase_metrics(target.cpu(), pred.cpu(), class_names))
    return result_dict


def get_metric(name):
    return {
        "MoF": MoFAccuracyMetric,
        "F1": F1Score,
        "Edit": Edit
    }[name]
