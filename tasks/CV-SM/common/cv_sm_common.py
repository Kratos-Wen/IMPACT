#!/usr/bin/env python3
"""Shared utilities for CV-SM evaluation on IMPACT CAS."""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
CV_TA_DIR = THIS_DIR.parent.parent / "CV-TA" / "cosine_knn"
if str(CV_TA_DIR) not in sys.path:
    sys.path.insert(0, str(CV_TA_DIR))

from cv_ta_retrieval import (
    DEFAULT_ANNOTATION_ROOT,
    DEFAULT_SPLIT_DIR,
    KNOWN_VIEWS,
    NULL_LABEL,
    REPO_ROOT,
    SegmentRecord,
    attach_feature_paths,
    counter_to_sorted_dict,
    load_metadata_from_annotations,
    load_rows_from_json,
    parse_optional_bool,
    parse_view_list,
    resolve_split_bundle,
    trial_id_from_name,
)

DEFAULT_FEATURE_ROOTS = {
    "videomaev2": REPO_ROOT / "features" / "cv" / "videomaev2",
    "i3d": REPO_ROOT / "features" / "cv" / "i3d",
    "mvitv2": REPO_ROOT / "features" / "cv" / "mvitv2",
}

FEATURE_DIM_HINTS = {
    "i3d": 1024,
    "videomaev2": 1408,
    "mvitv2": 768,
}


def resolve_feature_root(feature_type: str, feature_root: Optional[Path]) -> Path:
    if feature_root is not None:
        return feature_root
    default_root = DEFAULT_FEATURE_ROOTS.get(feature_type)
    if default_root is None or not default_root.exists():
        raise FileNotFoundError(
            f"No default feature directory found for feature_type='{feature_type}'. "
            "Pass --feature-root explicitly or place features under features/cv/."
        )
    return default_root


def load_semantic_metadata_from_file(metadata_path: Path, split: str) -> List[SegmentRecord]:
    if metadata_path.suffix.lower() == ".csv":
        with metadata_path.open("r", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif metadata_path.suffix.lower() == ".json":
        rows = load_rows_from_json(metadata_path)
    else:
        raise ValueError("Metadata must be .csv or .json.")

    records: List[SegmentRecord] = []
    for index, row in enumerate(rows):
        row_split = row.get("split")
        if split.lower() != "all" and row_split not in (split, None, ""):
            continue

        view_id = str(row["view_id"])
        coarse_label = str(
            row.get("coarse_label")
            or row.get("label_coarse")
            or row.get("label")
        )
        start_frame = int(row["start_frame"])
        end_frame = int(row["end_frame"])
        trial_id = str(row["trial_id"])
        video_id = str(
            row.get("video_id")
            or row.get("feature_key")
            or f"{trial_id}_{view_id}"
        )
        segment_id = str(row.get("segment_id", index))
        segment_uid = f"{video_id}:{segment_id}"
        video_num_frames = int(
            row.get("video_num_frames")
            or row.get("num_frames")
            or (end_frame + 1)
        )
        norm_start = start_frame / max(video_num_frames, 1)
        norm_end = end_frame / max(video_num_frames, 1)

        records.append(
            SegmentRecord(
                split=row_split or split,
                subject_id=row.get("subject_id") or None,
                trial_id=trial_id,
                execution_id=row.get("execution_id") or trial_id,
                view_id=view_id,
                video_id=video_id,
                segment_id=segment_id,
                coarse_label=coarse_label,
                start_frame=start_frame,
                end_frame=end_frame,
                visible=parse_optional_bool(row.get("visible")),
                feature_key=row.get("feature_key") or None,
                feature_path=Path(row["feature_path"]) if row.get("feature_path") else None,
                occurrence_id=None,
                video_num_frames=video_num_frames,
                norm_start=norm_start,
                norm_end=norm_end,
                norm_center=0.5 * (norm_start + norm_end),
                is_null=(coarse_label == NULL_LABEL),
                source_path=metadata_path,
                segment_uid=segment_uid,
            )
        )
    return records


def load_split_records(
    metadata: Optional[Path],
    annotation_root: Path,
    split: str,
    split_dir: Path,
    split_index: int,
    split_bundle: Optional[Path],
) -> Tuple[List[SegmentRecord], Optional[Path], Optional[Set[str]]]:
    split_bundle_path, split_members = resolve_split_bundle(
        split=split,
        split_dir=split_dir,
        split_index=split_index,
        split_bundle=split_bundle,
    )
    if metadata is not None:
        records = load_semantic_metadata_from_file(metadata, split=split)
        return records, split_bundle_path, split_members

    split_trial_ids = (
        {trial_id_from_name(item) for item in split_members}
        if split_members is not None
        else None
    )
    records = load_metadata_from_annotations(
        annotation_root=annotation_root,
        split=split,
        split_trial_ids=split_trial_ids,
    )
    return records, split_bundle_path, split_members


def load_embeddings_for_views(
    records: Sequence[SegmentRecord],
    selected_views: Set[str],
    feature_type: str,
    feature_root: Optional[Path],
    pooling: str,
) -> Tuple[Path, Dict[str, np.ndarray], Dict[str, int], Dict[str, int]]:
    resolved_root = resolve_feature_root(feature_type=feature_type, feature_root=feature_root)
    feature_records = [record for record in records if record.view_id in selected_views]
    feature_attach_stats = attach_feature_paths(feature_records, feature_root=resolved_root)
    embeddings, feature_stats = compute_segment_embeddings_robust(
        feature_records,
        pooling=pooling,
        feature_type=feature_type,
    )
    return resolved_root, embeddings, feature_attach_stats, feature_stats


def to_time_major_feature(
    feature: np.ndarray,
    record: SegmentRecord,
    feature_type: str,
) -> np.ndarray:
    if feature.ndim == 1:
        return feature.reshape(1, -1).astype(np.float32, copy=False)
    if feature.ndim != 2:
        raise ValueError(f"Expected 1D or 2D feature array, got shape {feature.shape}.")

    feat_dim_hint = FEATURE_DIM_HINTS.get(feature_type)
    candidates = []
    for axis in (0, 1):
        time_len = feature.shape[axis]
        feat_dim = feature.shape[1 - axis]
        score = abs(time_len - record.video_num_frames)
        if feat_dim_hint is not None and feat_dim != feat_dim_hint:
            score += abs(feat_dim - feat_dim_hint)
        candidates.append((score, axis))

    _, time_axis = min(candidates, key=lambda item: item[0])
    time_major = feature if time_axis == 0 else feature.T
    return time_major.astype(np.float32, copy=False)


def pool_segment_feature_robust(
    feature: np.ndarray,
    record: SegmentRecord,
    pooling: str,
    feature_type: str,
) -> np.ndarray:
    if feature.ndim == 1:
        return feature.astype(np.float32, copy=False)

    time_major = to_time_major_feature(feature, record=record, feature_type=feature_type)
    if pooling == "none":
        raise ValueError(
            "Pooling='none' only works when each segment already has a fixed 1D feature."
        )

    time_steps = time_major.shape[0]
    if time_steps <= 0:
        raise ValueError(f"Invalid empty time axis for feature shape {feature.shape}.")

    start = int(np.floor(record.start_frame * time_steps / max(record.video_num_frames, 1)))
    end = int(np.ceil((record.end_frame + 1) * time_steps / max(record.video_num_frames, 1))) - 1
    start = max(0, min(start, time_steps - 1))
    end = max(0, min(end, time_steps - 1))
    if end < start:
        end = start

    segment_slice = time_major[start : end + 1]
    if segment_slice.size == 0:
        raise ValueError(
            f"Empty segment slice for mapped range [{start}, {end}] from frames "
            f"[{record.start_frame}, {record.end_frame}]."
        )
    return segment_slice.mean(axis=0).astype(np.float32, copy=False)


def compute_segment_embeddings_robust(
    records: Sequence[SegmentRecord],
    pooling: str,
    feature_type: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    records_by_feature: Dict[Path, List[SegmentRecord]] = {}
    for record in records:
        if record.is_null or record.feature_path is None:
            continue
        records_by_feature.setdefault(record.feature_path, []).append(record)

    embeddings: Dict[str, np.ndarray] = {}
    stats = {"videos_loaded": 0, "segments_embedded": 0}
    for feature_path, path_records in records_by_feature.items():
        feature = np.load(feature_path, allow_pickle=True)
        stats["videos_loaded"] += 1
        for record in path_records:
            vector = pool_segment_feature_robust(
                feature=feature,
                record=record,
                pooling=pooling,
                feature_type=feature_type,
            )
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                continue
            embeddings[record.segment_uid] = (vector / norm).astype(np.float32, copy=False)
            stats["segments_embedded"] += 1
    return embeddings, stats


def filter_embedded_records(
    records: Sequence[SegmentRecord],
    embeddings: Dict[str, np.ndarray],
    views: Sequence[str],
) -> List[SegmentRecord]:
    view_set = set(views)
    return [
        record
        for record in records
        if (record.view_id in view_set)
        and (not record.is_null)
        and (record.segment_uid in embeddings)
    ]


def build_label_vocab(records: Sequence[SegmentRecord]) -> List[str]:
    return sorted({record.coarse_label for record in records if not record.is_null})


def average_precision_from_hits(hits: np.ndarray) -> Optional[float]:
    if hits.size == 0:
        return None
    total_positives = float(hits.sum())
    if total_positives <= 0:
        return None
    cumulative_hits = np.cumsum(hits)
    precision = cumulative_hits / (np.arange(hits.size) + 1)
    return float((precision * hits).sum() / total_positives)


def macro_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> float:
    f1_scores: List[float] = []
    for class_index in range(num_classes):
        true_mask = y_true == class_index
        pred_mask = y_pred == class_index
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum((~true_mask) & pred_mask))
        fn = int(np.sum(true_mask & (~pred_mask)))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def top1_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
