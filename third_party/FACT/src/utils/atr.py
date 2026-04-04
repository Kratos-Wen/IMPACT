import json
import os
from collections import OrderedDict

import numpy as np


def load_label_names(mapping_file):
    labels = []
    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, label = line.split(maxsplit=1)
            labels.append(label.strip().lower())
    return labels


def average_precision_binary(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    positives = int(y_true.sum())
    if positives == 0:
        return np.nan

    order = np.argsort(-y_score, kind='mergesort')
    y_true = y_true[order]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    precision = tp / np.maximum(tp + fp, 1)
    return float(precision[y_true == 1].sum() / positives)


def _metric_name(label_name):
    label_name = str(label_name).strip().lower()
    return label_name.replace('-', '_').replace(' ', '_')


def evaluate_atr_predictions(predictions_by_video, atr_segments_dir, split_name, split_id, mapping_file):
    labels = load_label_names(mapping_file)
    manifest_path = os.path.join(atr_segments_dir, '{}.{}.jsonl'.format(split_name, split_id))
    if not os.path.isfile(manifest_path):
        manifest_path = os.path.join(atr_segments_dir, '{}.split{}.jsonl'.format(split_name, split_id.replace('split', '')))
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError('ATR manifest not found for {} {} in {}'.format(split_name, split_id, atr_segments_dir))

    pooled_targets = []
    pooled_scores = []
    used_videos = set()
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            target = np.asarray(record['labels'], dtype=np.uint8)
            if target.sum() == 0:
                continue
            video_id = record['video_id']
            if video_id not in predictions_by_video:
                continue
            probs = predictions_by_video[video_id]
            start = max(0, int(record['start_frame']))
            end = min(len(probs) - 1, int(record['end_frame']))
            if end < start:
                continue
            pooled_targets.append(target)
            pooled_scores.append(probs[start:end + 1].mean(axis=0))
            used_videos.add(video_id)

    metrics = OrderedDict({
        'ATR_NumSegments': 0.0,
        'ATR_NumVideos': 0.0,
        'ATR_mAP_present': 0.0,
        'ATR_mAP_all': 0.0,
    })
    if not pooled_targets:
        return metrics

    y_true = np.asarray(pooled_targets, dtype=np.uint8)
    y_score = np.asarray(pooled_scores, dtype=np.float32)
    metrics['ATR_NumSegments'] = float(len(y_true))
    metrics['ATR_NumVideos'] = float(len(used_videos))

    aps = []
    for idx, label_name in enumerate(labels):
        ap = average_precision_binary(y_true[:, idx], y_score[:, idx])
        metrics['ATR_AP_{}'.format(_metric_name(label_name))] = float(ap * 100.0) if not np.isnan(ap) else float('nan')
        if not np.isnan(ap):
            aps.append(ap * 100.0)

    if aps:
        metrics['ATR_mAP_present'] = float(np.mean(aps))
        metrics['ATR_mAP_all'] = float(np.mean(aps))

    return metrics
