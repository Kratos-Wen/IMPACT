import json
import os
from typing import Dict, List

import numpy as np


def load_label_names(mapping_path: str) -> List[str]:
    labels = []
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, label = line.split(maxsplit=1)
            labels.append(label.strip().lower())
    return labels


def average_precision_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    positives = int(y_true.sum())
    if positives == 0:
        return np.nan

    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    precision = tp / np.maximum(tp + fp, 1)
    return float(precision[y_true == 1].sum() / positives)


def _metric_name(label_name: str) -> str:
    label_name = str(label_name).strip().lower()
    return label_name.replace("-", "_").replace(" ", "_")


def evaluate_atr_predictions(
    predictions_by_video: Dict[str, np.ndarray],
    segments_dir: str,
    split_name: str,
    split_num: int,
    mapping_path: str,
) -> Dict[str, float]:
    manifest_path = os.path.join(segments_dir, f"{split_name}.split{split_num}.jsonl")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"ATR manifest not found: {manifest_path}")

    label_names = load_label_names(mapping_path)
    pooled_scores = []
    pooled_targets = []
    used_videos = set()
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            labels = np.asarray(record["labels"], dtype=np.uint8)
            if labels.sum() == 0:
                continue
            video_id = record["video_id"]
            if video_id not in predictions_by_video:
                continue
            probs = predictions_by_video[video_id]
            if probs.ndim != 2:
                raise ValueError(f"Expected probs[T, C], got {probs.shape} for {video_id}")
            start = max(0, int(record["start_frame"]))
            end = min(len(probs) - 1, int(record["end_frame"]))
            if end < start:
                continue
            pooled_scores.append(probs[start:end + 1].mean(axis=0))
            pooled_targets.append(labels)
            used_videos.add(video_id)

    metrics = {
        "ATR_NumSegments": 0.0,
        "ATR_NumVideos": 0.0,
        "ATR_mAP_present": 0.0,
        "ATR_mAP_all": 0.0,
    }
    if not pooled_targets:
        return metrics

    y_true = np.asarray(pooled_targets, dtype=np.uint8)
    y_score = np.asarray(pooled_scores, dtype=np.float32)
    metrics["ATR_NumSegments"] = float(len(y_true))
    metrics["ATR_NumVideos"] = float(len(used_videos))

    aps_present = []
    aps_all = []
    for idx, label_name in enumerate(label_names):
        ap = average_precision_score_binary(y_true[:, idx], y_score[:, idx])
        metric_key = f"ATR_AP_{_metric_name(label_name)}"
        metrics[metric_key] = float(ap * 100.0) if not np.isnan(ap) else float("nan")
        if not np.isnan(ap):
            aps_present.append(ap * 100.0)
            aps_all.append(ap * 100.0)

    if aps_present:
        metrics["ATR_mAP_present"] = float(np.mean(aps_present))
        metrics["ATR_mAP_all"] = float(np.mean(aps_all))

    return metrics
