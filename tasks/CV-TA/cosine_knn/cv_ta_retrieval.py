#!/usr/bin/env python3
"""Cross-View Temporal Alignment (CV-TA) for IMPACT CAS.

This script evaluates cross-view segment retrieval with cosine similarity only.
It supports three protocols:

1. local:
   Query: one non-null segment from a source view.
   Gallery: all non-null segments from a target view within the same trial.
   Positive:
     - metadata mode: same occurrence_id in the target view.
     - CAS annotation mode, exo-exo pair: same coarse label over the same
       synchronized frame interval in the target view.
     - CAS annotation mode, pair involving ego: same occurrence recovered by
       non-null label-sequence alignment within the trial.

2. global:
   Query: one non-null segment from a source view.
   Gallery: all non-null segments from a target view in the selected split.
   Positive: defined the same way as local, but searched in the full split.

3. exo2ego:
   Same as cross-view retrieval, but intended for exocentric -> ego retrieval.
   If the synchronized ego interval is labeled "null", or sequence alignment has
   no ego counterpart, the query is excluded from the retrieval metrics.
   Coverage is reported as the fraction of source queries with a valid ego match.

The script can read an external metadata CSV/JSON, or construct metadata
 directly from CAS annotation JSON files without creating a separate metadata
file on disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANNOTATION_ROOT = REPO_ROOT / "dataset" / "CV" / "annotations_CAS"
DEFAULT_SPLIT_DIR = REPO_ROOT / "dataset" / "CV" / "splits_CAS"
DEFAULT_FEATURE_ROOTS = {
    "videomaev2": REPO_ROOT / "features" / "cv" / "videomaev2",
    "i3d": REPO_ROOT / "features" / "cv" / "i3d",
    "mvitv2": REPO_ROOT / "features" / "cv" / "mvitv2",
}
KNOWN_VIEWS = ("ego", "front", "left", "right", "top")
NULL_LABEL = "null"
OCCURRENCE_TIME_TOLERANCE = 0.01


@dataclass
class SegmentRecord:
    split: Optional[str]
    subject_id: Optional[str]
    trial_id: str
    execution_id: Optional[str]
    view_id: str
    video_id: str
    segment_id: str
    coarse_label: str
    start_frame: int
    end_frame: int
    visible: Optional[bool]
    feature_key: Optional[str]
    feature_path: Optional[Path]
    occurrence_id: Optional[str]
    video_num_frames: int
    norm_start: float
    norm_end: float
    norm_center: float
    is_null: bool
    source_path: Optional[Path]
    segment_uid: str


@dataclass
class OccurrenceCluster:
    trial_id: str
    coarse_label: str
    members: List[SegmentRecord] = field(default_factory=list)
    member_views: Set[str] = field(default_factory=set)

    def add(self, record: SegmentRecord) -> None:
        self.members.append(record)
        self.member_views.add(record.view_id)

    @property
    def norm_start(self) -> float:
        return statistics.median(member.norm_start for member in self.members)

    @property
    def norm_end(self) -> float:
        return statistics.median(member.norm_end for member in self.members)

    @property
    def norm_center(self) -> float:
        return 0.5 * (self.norm_start + self.norm_end)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Cross-View Temporal Alignment (CV-TA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Optional CSV/JSON segment metadata file. If omitted, metadata is "
            "built directly from CAS annotation JSONs."
        ),
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=DEFAULT_ANNOTATION_ROOT,
        help="CAS annotation root used when --metadata is omitted.",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=None,
        help="Directory containing precomputed features. Defaults from --feature-type.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split name to evaluate. Use 'all' to ignore split filtering.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help="Directory that stores train/val/test split bundles.",
    )
    parser.add_argument(
        "--split-index",
        type=int,
        default=1,
        help="Which split bundle index to use when multiple split bundles exist.",
    )
    parser.add_argument(
        "--split-bundle",
        type=Path,
        default=None,
        help="Explicit split bundle file. Overrides --split-dir and --split-index.",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        required=True,
        choices=("local", "global", "exo2ego"),
        help="Retrieval protocol.",
    )
    parser.add_argument(
        "--source-views",
        type=str,
        required=True,
        help="Comma-separated source views, e.g. front,left,right,top.",
    )
    parser.add_argument(
        "--target-views",
        type=str,
        required=True,
        help="Comma-separated target views, e.g. ego or front,left.",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        required=True,
        choices=("i3d", "videomaev2", "mvitv2"),
        help="Backbone feature family.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=("mean", "none"),
        help=(
            "Pooling for segment features. Use 'mean' for per-frame/per-clip "
            "video features, 'none' when the loaded feature is already 1D."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path to the output metrics JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the final JSON summary print to stdout.",
    )
    return parser.parse_args()


def parse_view_list(raw: str) -> List[str]:
    views: List[str] = []
    for part in raw.replace(" ", ",").split(","):
        view = part.strip()
        if not view:
            continue
        if view not in KNOWN_VIEWS:
            raise ValueError(f"Unknown view '{view}'. Known views: {KNOWN_VIEWS}")
        views.append(view)
    if not views:
        raise ValueError("At least one view must be provided.")
    return views


def resolve_feature_root(feature_type: str, feature_root: Optional[Path]) -> Path:
    if feature_root is not None:
        return feature_root
    return DEFAULT_FEATURE_ROOTS[feature_type]


def resolve_split_bundle(
    split: str,
    split_dir: Path,
    split_index: int,
    split_bundle: Optional[Path],
) -> Tuple[Optional[Path], Optional[Set[str]]]:
    if split.lower() == "all":
        return None, None

    if split_bundle is not None:
        bundle_path = split_bundle
    else:
        bundle_path = split_dir / f"{split}.split{split_index}.bundle"
        if not bundle_path.exists():
            matches = sorted(split_dir.glob(f"{split}.split*.bundle"))
            if len(matches) == 1:
                bundle_path = matches[0]
            elif not matches:
                raise FileNotFoundError(
                    f"Could not find a bundle for split='{split}' under {split_dir}."
                )
            else:
                raise FileNotFoundError(
                    f"Multiple bundles found for split='{split}'. "
                    "Pass --split-index or --split-bundle explicitly."
                )

    members = {
        line.strip()
        for line in bundle_path.read_text().splitlines()
        if line.strip()
    }
    return bundle_path, members


def trial_id_from_name(name: str) -> str:
    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot infer trial_id from '{name}'.")
    return "_".join(parts[:2])


def unique_preserve_order(values: Iterable[Optional[str]]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for value in values:
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def counter_to_sorted_dict(counter: Counter) -> Dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def build_trial_subject_map(annotation_root: Path) -> Dict[str, str]:
    trial_to_subject: Dict[str, str] = {}
    for view in ("front", "left", "right", "top"):
        for path in sorted((annotation_root / view).glob("*/*.json")):
            subject_id = path.parent.name
            trial_id = trial_id_from_name(path.stem)
            existing = trial_to_subject.get(trial_id)
            if existing is not None and existing != subject_id:
                raise ValueError(
                    f"Conflicting subject ids for trial '{trial_id}': "
                    f"'{existing}' vs '{subject_id}'."
                )
            trial_to_subject[trial_id] = subject_id
    return trial_to_subject


def iter_annotation_files(annotation_root: Path) -> Iterable[Path]:
    for view in KNOWN_VIEWS:
        view_root = annotation_root / view
        if view == "ego":
            yield from sorted(view_root.glob("*.json"))
        else:
            yield from sorted(view_root.glob("*/*.json"))


def record_from_annotation(
    path: Path,
    payload: Dict[str, Any],
    subject_id: Optional[str],
    split: Optional[str],
) -> List[SegmentRecord]:
    video_id = payload["video_id"]
    trial_id = trial_id_from_name(video_id)
    view_id = payload["view"]
    total_frames = int(payload["view_end"]) - int(payload["view_start"]) + 1
    execution_id = trial_id

    records: List[SegmentRecord] = []
    for segment_index, segment in enumerate(payload["segments"]):
        # CAS JSON segments already represent coarse action segments, so each
        # annotated segment becomes one retrieval unit.
        coarse_label = str(segment["label"])
        start_frame = int(segment["f_start"])
        end_frame = int(segment["f_end"])
        norm_start = start_frame / total_frames
        norm_end = end_frame / total_frames
        is_null = coarse_label == NULL_LABEL
        visible = None if is_null else True
        # CAS segment["id"] is often a class id, not a unique segment-instance id.
        segment_id = str(segment_index)
        segment_uid = f"{video_id}:{segment_id}"

        records.append(
            SegmentRecord(
                split=split,
                subject_id=subject_id,
                trial_id=trial_id,
                execution_id=execution_id,
                view_id=view_id,
                video_id=video_id,
                segment_id=segment_id,
                coarse_label=coarse_label,
                start_frame=start_frame,
                end_frame=end_frame,
                visible=visible,
                feature_key=None,
                feature_path=None,
                occurrence_id=None,
                video_num_frames=total_frames,
                norm_start=norm_start,
                norm_end=norm_end,
                norm_center=0.5 * (norm_start + norm_end),
                is_null=is_null,
                source_path=path,
                segment_uid=segment_uid,
            )
        )
    return records


def load_metadata_from_annotations(
    annotation_root: Path,
    split: str,
    split_trial_ids: Optional[Set[str]],
) -> List[SegmentRecord]:
    trial_to_subject = build_trial_subject_map(annotation_root)
    all_records: List[SegmentRecord] = []

    for path in iter_annotation_files(annotation_root):
        payload = json.loads(path.read_text())
        video_id = payload["video_id"]
        trial_id = trial_id_from_name(video_id)
        if split_trial_ids is not None and trial_id not in split_trial_ids:
            continue
        subject_id = (
            path.parent.name if payload["view"] != "ego" else trial_to_subject.get(trial_id)
        )
        records = record_from_annotation(path, payload, subject_id=subject_id, split=split)
        all_records.extend(records)
    return all_records


def load_rows_from_json(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("rows", "segments", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported JSON metadata structure in {path}.")


def parse_optional_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value '{value}'.")


def load_metadata_from_file(metadata_path: Path, split: str) -> List[SegmentRecord]:
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
        coarse_label = str(row["coarse_label"])
        start_frame = int(row["start_frame"])
        end_frame = int(row["end_frame"])
        video_id = str(
            row.get("video_id")
            or row.get("feature_key")
            or f"{row['trial_id']}_{view_id}"
        )
        segment_id = str(row.get("segment_id", index))
        segment_uid = f"{video_id}:{segment_id}"

        occurrence_id = row.get("occurrence_id") or None
        if occurrence_id is None:
            raise ValueError(
                "Metadata file is missing occurrence_id. Provide occurrence_id in "
                "metadata, or omit --metadata and let the script build metadata "
                "directly from CAS annotations."
            )

        video_num_frames = int(row.get("video_num_frames") or (end_frame + 1))
        norm_start = start_frame / max(video_num_frames, 1)
        norm_end = end_frame / max(video_num_frames, 1)

        records.append(
            SegmentRecord(
                split=row_split or split,
                subject_id=row.get("subject_id") or None,
                trial_id=str(row["trial_id"]),
                execution_id=row.get("execution_id") or None,
                view_id=view_id,
                video_id=video_id,
                segment_id=segment_id,
                coarse_label=coarse_label,
                start_frame=start_frame,
                end_frame=end_frame,
                visible=parse_optional_bool(row.get("visible")),
                feature_key=row.get("feature_key") or None,
                feature_path=Path(row["feature_path"]) if row.get("feature_path") else None,
                occurrence_id=occurrence_id,
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


def candidate_feature_stems(record: SegmentRecord) -> List[str]:
    annotation_stem = record.source_path.stem if record.source_path is not None else None
    candidates = [
        record.feature_key,
        record.video_id,
        annotation_stem,
        record.video_id[:-8] if record.video_id.endswith("_clipped") else None,
        annotation_stem[:-8] if annotation_stem and annotation_stem.endswith("_clipped") else None,
        record.video_id.replace("_ego_sync_clipped", "_ego_sync")
        if record.video_id.endswith("_ego_sync_clipped")
        else None,
        record.video_id.replace("_ego_sync", "_ego")
        if record.video_id.endswith("_ego_sync")
        else None,
        annotation_stem.replace("_ego", "_ego_sync")
        if annotation_stem and annotation_stem.endswith("_ego")
        else None,
    ]
    return unique_preserve_order(candidates)


def index_feature_files(feature_root: Path) -> Dict[str, Path]:
    feature_index: Dict[str, Path] = {}
    for path in sorted(feature_root.glob("*.npy")):
        feature_index[path.stem] = path
    if not feature_index:
        raise FileNotFoundError(f"No .npy files found under {feature_root}.")
    return feature_index


def attach_feature_paths(
    records: Sequence[SegmentRecord],
    feature_root: Path,
) -> Dict[str, int]:
    feature_index = index_feature_files(feature_root)
    missing = 0
    resolved = 0

    for record in records:
        if record.is_null:
            continue

        if record.feature_path is not None:
            if not record.feature_path.is_absolute():
                record.feature_path = feature_root / record.feature_path
            if not record.feature_path.exists():
                missing += 1
                record.feature_path = None
            else:
                resolved += 1
            continue

        matched_path = None
        for stem in candidate_feature_stems(record):
            matched_path = feature_index.get(stem)
            if matched_path is not None:
                break

        if matched_path is None:
            missing += 1
        else:
            resolved += 1
            record.feature_path = matched_path

    return {"resolved": resolved, "missing": missing}


def cluster_matches(record: SegmentRecord, cluster: OccurrenceCluster) -> bool:
    if record.view_id in cluster.member_views:
        return False
    return (
        record.norm_end + OCCURRENCE_TIME_TOLERANCE >= cluster.norm_start
        and record.norm_start - OCCURRENCE_TIME_TOLERANCE <= cluster.norm_end
    )


def build_occurrence_index(
    records: Sequence[SegmentRecord],
) -> Dict[str, OccurrenceCluster]:
    trial_label_groups: Dict[Tuple[str, str], List[SegmentRecord]] = defaultdict(list)
    for record in records:
        if record.is_null:
            continue
        trial_label_groups[(record.trial_id, record.coarse_label)].append(record)

    occurrences: Dict[str, OccurrenceCluster] = {}
    for (trial_id, coarse_label), label_records in trial_label_groups.items():
        label_records.sort(key=lambda item: (item.norm_center, item.norm_start, item.view_id))
        clusters: List[OccurrenceCluster] = []
        for record in label_records:
            if clusters and cluster_matches(record, clusters[-1]):
                clusters[-1].add(record)
            else:
                cluster = OccurrenceCluster(trial_id=trial_id, coarse_label=coarse_label)
                cluster.add(record)
                clusters.append(cluster)

        for cluster_index, cluster in enumerate(clusters):
            occurrence_id = f"{trial_id}:{coarse_label}:{cluster_index:03d}"
            for record in cluster.members:
                record.occurrence_id = occurrence_id
            occurrences[occurrence_id] = cluster
    return occurrences


def ensure_occurrence_ids(records: Sequence[SegmentRecord]) -> Dict[str, OccurrenceCluster]:
    missing_occurrences = any(
        (not record.is_null) and (record.occurrence_id is None)
        for record in records
    )
    if missing_occurrences:
        return build_occurrence_index(records)

    occurrences: Dict[str, OccurrenceCluster] = {}
    grouped: Dict[str, List[SegmentRecord]] = defaultdict(list)
    for record in records:
        if record.is_null or record.occurrence_id is None:
            continue
        grouped[record.occurrence_id].append(record)

    for occurrence_id, members in grouped.items():
        cluster = OccurrenceCluster(
            trial_id=members[0].trial_id,
            coarse_label=members[0].coarse_label,
        )
        for member in members:
            cluster.add(member)
        occurrences[occurrence_id] = cluster
    return occurrences


def as_time_major(feature: np.ndarray, expected_frames: int) -> np.ndarray:
    if feature.ndim == 1:
        return feature.reshape(1, -1).astype(np.float32, copy=False)
    if feature.ndim != 2:
        raise ValueError(f"Expected 1D or 2D feature array, got shape {feature.shape}.")
    if feature.shape[0] == expected_frames:
        return feature.astype(np.float32, copy=False)
    if feature.shape[1] == expected_frames:
        return feature.T.astype(np.float32, copy=False)

    diff0 = abs(feature.shape[0] - expected_frames)
    diff1 = abs(feature.shape[1] - expected_frames)
    time_major = feature if diff0 <= diff1 else feature.T
    return time_major.astype(np.float32, copy=False)


def pool_segment_feature(
    feature: np.ndarray,
    record: SegmentRecord,
    pooling: str,
) -> np.ndarray:
    if feature.ndim == 1:
        return feature.astype(np.float32, copy=False)

    time_major = as_time_major(feature, expected_frames=record.video_num_frames)
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


def compute_segment_features(
    records: Sequence[SegmentRecord],
    pooling: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    records_by_feature: Dict[Path, List[SegmentRecord]] = defaultdict(list)
    for record in records:
        if record.is_null or record.feature_path is None:
            continue
        records_by_feature[record.feature_path].append(record)

    embeddings: Dict[str, np.ndarray] = {}
    stats = {"videos_loaded": 0, "segments_embedded": 0}
    for feature_path, path_records in records_by_feature.items():
        feature = np.load(feature_path, allow_pickle=True)
        stats["videos_loaded"] += 1
        for record in path_records:
            vector = pool_segment_feature(feature, record, pooling=pooling)
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                continue
            embeddings[record.segment_uid] = (vector / norm).astype(np.float32, copy=False)
            stats["segments_embedded"] += 1
    return embeddings, stats


def group_by(records: Sequence[SegmentRecord], key_fn) -> Dict[Any, List[SegmentRecord]]:
    grouped: Dict[Any, List[SegmentRecord]] = defaultdict(list)
    for record in records:
        grouped[key_fn(record)].append(record)
    return grouped


def temporal_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    return min(end_a, end_b) - max(start_a, start_b)


def frame_overlap(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> int:
    return min(end_a, end_b) - max(start_a, start_b) + 1


def find_overlapping_segment(
    segments: Sequence[SegmentRecord],
    occurrence: OccurrenceCluster,
) -> Optional[SegmentRecord]:
    best: Optional[SegmentRecord] = None
    best_overlap = -1.0
    for record in segments:
        overlap = temporal_overlap(
            record.norm_start,
            record.norm_end,
            occurrence.norm_start,
            occurrence.norm_end,
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best = record
        if record.norm_start <= occurrence.norm_center <= record.norm_end:
            return record
    return best


def resolve_positive_by_synchronized_frames(
    query_record: SegmentRecord,
    target_segments: Sequence[SegmentRecord],
) -> Tuple[Optional[SegmentRecord], str]:
    best_same_label: Optional[SegmentRecord] = None
    best_same_label_overlap = 0
    covering_target: Optional[SegmentRecord] = None
    covering_overlap = 0
    midpoint = 0.5 * (query_record.start_frame + query_record.end_frame)

    for target_record in target_segments:
        overlap = frame_overlap(
            query_record.start_frame,
            query_record.end_frame,
            target_record.start_frame,
            target_record.end_frame,
        )
        if overlap <= 0:
            continue
        if (
            target_record.start_frame <= midpoint <= target_record.end_frame
            and overlap > covering_overlap
        ):
            covering_target = target_record
            covering_overlap = overlap
        if (
            (not target_record.is_null)
            and target_record.coarse_label == query_record.coarse_label
            and overlap > best_same_label_overlap
        ):
            best_same_label = target_record
            best_same_label_overlap = overlap

    if covering_target is not None:
        if covering_target.is_null:
            return None, 'invisible'
        if covering_target.coarse_label == query_record.coarse_label:
            return covering_target, 'positive'

    if best_same_label is not None:
        return best_same_label, 'positive'
    return None, 'missing'


def sort_segments_for_sequence(records: Sequence[SegmentRecord]) -> List[SegmentRecord]:
    return sorted(
        [record for record in records if not record.is_null],
        key=lambda record: (record.start_frame, record.end_frame, record.segment_id),
    )


def lcs_align_segments(
    source_segments: Sequence[SegmentRecord],
    target_segments: Sequence[SegmentRecord],
) -> List[Tuple[int, int]]:
    n = len(source_segments)
    m = len(target_segments)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if source_segments[i].coarse_label == target_segments[j].coarse_label:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    aligned_pairs: List[Tuple[int, int]] = []
    i = 0
    j = 0
    while i < n and j < m:
        if (
            source_segments[i].coarse_label == target_segments[j].coarse_label
            and dp[i][j] == 1 + dp[i + 1][j + 1]
        ):
            aligned_pairs.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return aligned_pairs


def build_trial_sequence_alignment(
    source_records: Sequence[SegmentRecord],
    target_records: Sequence[SegmentRecord],
) -> Dict[str, Dict[str, SegmentRecord]]:
    source_by_trial = group_by(
        [record for record in source_records if not record.is_null],
        lambda record: record.trial_id,
    )
    target_by_trial = group_by(
        [record for record in target_records if not record.is_null],
        lambda record: record.trial_id,
    )

    trial_alignment: Dict[str, Dict[str, SegmentRecord]] = {}
    for trial_id in sorted(set(source_by_trial) | set(target_by_trial)):
        source_sequence = sort_segments_for_sequence(source_by_trial.get(trial_id, []))
        target_sequence = sort_segments_for_sequence(target_by_trial.get(trial_id, []))
        matched_pairs = lcs_align_segments(source_sequence, target_sequence)
        trial_alignment[trial_id] = {
            source_sequence[source_index].segment_uid: target_sequence[target_index]
            for source_index, target_index in matched_pairs
        }
    return trial_alignment


def determine_positive_mode(
    metadata_present: bool,
    source_view: str,
    target_view: str,
) -> str:
    if metadata_present:
        return "occurrence"
    if source_view != "ego" and target_view != "ego":
        return "sync"
    return "sequence"


def compute_rank(similarities: np.ndarray, positive_index: int) -> int:
    positive_score = float(similarities[positive_index])
    return int(np.sum(similarities > positive_score)) + 1


def summarize_ranks(ranks: Sequence[int]) -> Dict[str, Optional[float]]:
    if not ranks:
        return {
            "recall@1": None,
            "recall@5": None,
            "median_rank": None,
        }
    rank_array = np.asarray(ranks)
    return {
        "recall@1": float(np.mean(rank_array <= 1)),
        "recall@5": float(np.mean(rank_array <= 5)),
        "median_rank": float(np.median(rank_array)),
    }


def evaluate_pair(
    protocol: str,
    source_view: str,
    target_view: str,
    source_records: Sequence[SegmentRecord],
    target_records: Sequence[SegmentRecord],
    all_trial_view_segments: Dict[Tuple[str, str], List[SegmentRecord]],
    occurrences: Dict[str, OccurrenceCluster],
    embeddings: Dict[str, np.ndarray],
    positive_mode: str,
) -> Dict[str, Any]:
    if source_view == target_view:
        return {
            "source_view": source_view,
            "target_view": target_view,
            "skipped_same_view_pair": True,
        }

    candidate_queries = [
        record
        for record in source_records
        if (not record.is_null) and record.segment_uid in embeddings
    ]

    target_non_null_records = [
        record
        for record in target_records
        if (not record.is_null) and record.segment_uid in embeddings
    ]
    target_by_trial = group_by(target_non_null_records, lambda record: record.trial_id)
    target_by_occurrence = {
        record.occurrence_id: record
        for record in target_non_null_records
        if record.occurrence_id is not None
    }
    trial_sequence_alignment = (
        build_trial_sequence_alignment(source_records, target_records)
        if positive_mode == "sequence"
        else {}
    )

    gallery_order: List[SegmentRecord] = []
    gallery_matrix: Optional[np.ndarray] = None
    if protocol == "global":
        gallery_order = list(target_non_null_records)
        if gallery_order:
            gallery_matrix = np.stack(
                [embeddings[record.segment_uid] for record in gallery_order], axis=0
            )

    ranks: List[int] = []
    skipped_invisible = 0
    skipped_no_positive = 0
    skipped_unaligned = 0
    skipped_empty_gallery = 0
    skipped_missing_query_occurrence = 0
    per_query_gallery_sizes: List[int] = []
    candidate_by_label: Counter = Counter(record.coarse_label for record in candidate_queries)
    evaluated_by_label: Counter = Counter()

    for query_record in candidate_queries:
        positive_record: Optional[SegmentRecord] = None
        if positive_mode == "occurrence":
            if query_record.occurrence_id is None:
                skipped_missing_query_occurrence += 1
                continue

            occurrence = occurrences[query_record.occurrence_id]
            positive_record = target_by_occurrence.get(query_record.occurrence_id)
            if positive_record is None:
                overlapping_target = find_overlapping_segment(
                    all_trial_view_segments.get((query_record.trial_id, target_view), []),
                    occurrence,
                )
                if protocol == "exo2ego" and overlapping_target is not None and overlapping_target.is_null:
                    skipped_invisible += 1
                else:
                    skipped_no_positive += 1
                continue
        elif positive_mode == "sync":
            positive_record, positive_status = resolve_positive_by_synchronized_frames(
                query_record,
                all_trial_view_segments.get((query_record.trial_id, target_view), []),
            )
            if positive_status != "positive":
                skipped_unaligned += 1
                if positive_status == "invisible":
                    skipped_invisible += 1
                else:
                    skipped_no_positive += 1
                continue
            if positive_record is None or positive_record.segment_uid not in embeddings:
                skipped_no_positive += 1
                continue
        elif positive_mode == "sequence":
            positive_record = trial_sequence_alignment.get(query_record.trial_id, {}).get(
                query_record.segment_uid
            )
            if positive_record is None:
                skipped_unaligned += 1
                if protocol == "exo2ego":
                    skipped_invisible += 1
                else:
                    skipped_no_positive += 1
                continue
            if positive_record.segment_uid not in embeddings:
                skipped_no_positive += 1
                continue
        else:
            raise ValueError(f"Unsupported positive_mode '{positive_mode}'.")

        if protocol == "global":
            if gallery_matrix is None:
                skipped_empty_gallery += 1
                continue
            gallery_order_local = gallery_order
            gallery_matrix_local = gallery_matrix
        else:
            gallery_order_local = target_by_trial.get(query_record.trial_id, [])
            if not gallery_order_local:
                skipped_empty_gallery += 1
                continue
            gallery_matrix_local = np.stack(
                [embeddings[record.segment_uid] for record in gallery_order_local], axis=0
            )

        positive_index = next(
            (
                idx
                for idx, gallery_record in enumerate(gallery_order_local)
                if gallery_record.segment_uid == positive_record.segment_uid
            ),
            None,
        )
        if positive_index is None:
            skipped_no_positive += 1
            continue

        query_vector = embeddings[query_record.segment_uid]
        similarities = gallery_matrix_local @ query_vector
        rank = compute_rank(similarities, positive_index=positive_index)
        ranks.append(rank)
        evaluated_by_label[query_record.coarse_label] += 1
        per_query_gallery_sizes.append(len(gallery_order_local))

    metrics = summarize_ranks(ranks)
    result: Dict[str, Any] = {
        "source_view": source_view,
        "target_view": target_view,
        "protocol": protocol,
        "positive_mode": positive_mode,
        "candidate_queries": len(candidate_queries),
        "evaluated_queries": len(ranks),
        "skipped_invisible": skipped_invisible,
        "skipped_no_positive": skipped_no_positive,
        "skipped_unaligned": skipped_unaligned,
        "skipped_empty_gallery": skipped_empty_gallery,
        "skipped_missing_query_occurrence": skipped_missing_query_occurrence,
        "average_gallery_size": (
            float(np.mean(per_query_gallery_sizes)) if per_query_gallery_sizes else None
        ),
        "candidate_queries_by_label": counter_to_sorted_dict(candidate_by_label),
        "evaluated_queries_by_label": counter_to_sorted_dict(evaluated_by_label),
    }
    result.update(metrics)

    if protocol == "exo2ego":
        result["coverage"] = (
            len(ranks) / len(candidate_queries) if candidate_queries else None
        )
    return result


def flatten_pair_metrics(pair_metrics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid_pairs = [
        metrics
        for metrics in pair_metrics
        if not metrics.get("skipped_same_view_pair", False)
    ]
    all_r1 = []
    all_r5 = []
    all_median = []
    total_candidate_queries = 0
    total_evaluated_queries = 0
    total_invisible = 0
    total_no_positive = 0
    total_empty_gallery = 0
    total_missing_occurrence = 0
    total_unaligned = 0
    candidate_by_label: Counter = Counter()
    evaluated_by_label: Counter = Counter()

    for metrics in valid_pairs:
        total_candidate_queries += int(metrics["candidate_queries"])
        total_evaluated_queries += int(metrics["evaluated_queries"])
        total_invisible += int(metrics["skipped_invisible"])
        total_no_positive += int(metrics["skipped_no_positive"])
        total_empty_gallery += int(metrics["skipped_empty_gallery"])
        total_missing_occurrence += int(metrics["skipped_missing_query_occurrence"])
        total_unaligned += int(metrics.get("skipped_unaligned", 0))
        candidate_by_label.update(metrics.get("candidate_queries_by_label", {}))
        evaluated_by_label.update(metrics.get("evaluated_queries_by_label", {}))

        if metrics["recall@1"] is not None:
            all_r1.append((metrics["recall@1"], metrics["evaluated_queries"]))
            all_r5.append((metrics["recall@5"], metrics["evaluated_queries"]))
        if metrics["median_rank"] is not None:
            all_median.append(metrics["median_rank"])

    def weighted_average(values: Sequence[Tuple[float, int]]) -> Optional[float]:
        if not values:
            return None
        numerator = sum(value * weight for value, weight in values)
        denominator = sum(weight for _, weight in values)
        return float(numerator / denominator) if denominator else None

    result = {
        "num_pairs": len(valid_pairs),
        "candidate_queries": total_candidate_queries,
        "evaluated_queries": total_evaluated_queries,
        "skipped_invisible": total_invisible,
        "skipped_no_positive": total_no_positive,
        "skipped_empty_gallery": total_empty_gallery,
        "skipped_missing_query_occurrence": total_missing_occurrence,
        "skipped_unaligned": total_unaligned,
        "recall@1": weighted_average(all_r1),
        "recall@5": weighted_average(all_r5),
        "median_rank": float(np.median(np.asarray(all_median))) if all_median else None,
    }

    if total_candidate_queries:
        result["coverage"] = total_evaluated_queries / total_candidate_queries
    else:
        result["coverage"] = None
    result["candidate_queries_by_label"] = counter_to_sorted_dict(candidate_by_label)
    result["evaluated_queries_by_label"] = counter_to_sorted_dict(evaluated_by_label)
    return result


def main() -> None:
    args = parse_args()
    source_views = parse_view_list(args.source_views)
    target_views = parse_view_list(args.target_views)
    feature_root = resolve_feature_root(args.feature_type, args.feature_root)
    split_bundle_path, split_members = resolve_split_bundle(
        split=args.split,
        split_dir=args.split_dir,
        split_index=args.split_index,
        split_bundle=args.split_bundle,
    )

    if args.protocol == "exo2ego":
        if any(view == "ego" for view in source_views):
            raise ValueError("exo2ego expects exocentric source views, not ego.")
        if target_views != ["ego"]:
            raise ValueError("exo2ego expects --target-views ego.")

    split_trial_ids = (
        {trial_id_from_name(item) for item in split_members}
        if (split_members is not None and args.metadata is None)
        else None
    )

    if args.metadata is not None:
        all_records = load_metadata_from_file(args.metadata, split=args.split)
    else:
        all_records = load_metadata_from_annotations(
            annotation_root=args.annotation_root,
            split=args.split,
            split_trial_ids=split_trial_ids,
        )

    metadata_present = args.metadata is not None
    config_positive_mode = "occurrence" if metadata_present else "pair_specific"
    occurrences = ensure_occurrence_ids(all_records) if metadata_present else {}

    selected_feature_views = set(source_views) | set(target_views)
    feature_records = [
        record for record in all_records if record.view_id in selected_feature_views
    ]
    feature_attach_stats = attach_feature_paths(feature_records, feature_root=feature_root)
    embeddings, feature_stats = compute_segment_features(feature_records, pooling=args.pooling)

    all_records_by_view = group_by(all_records, lambda record: record.view_id)
    sorted_records = sorted(
        all_records,
        key=lambda record: (record.trial_id, record.view_id, record.norm_start),
    )
    all_trial_view_segments = group_by(
        sorted_records,
        lambda record: (record.trial_id, record.view_id),
    )

    pair_metrics: List[Dict[str, Any]] = []
    for source_view in source_views:
        for target_view in target_views:
            pair_positive_mode = determine_positive_mode(
                metadata_present=metadata_present,
                source_view=source_view,
                target_view=target_view,
            )
            metrics = evaluate_pair(
                protocol=args.protocol,
                source_view=source_view,
                target_view=target_view,
                source_records=all_records_by_view.get(source_view, []),
                target_records=all_records_by_view.get(target_view, []),
                all_trial_view_segments=all_trial_view_segments,
                occurrences=occurrences,
                embeddings=embeddings,
                positive_mode=pair_positive_mode,
            )
            pair_metrics.append(metrics)

    summary = {
        "config": {
            "metadata": str(args.metadata) if args.metadata else None,
            "annotation_root": str(args.annotation_root) if args.metadata is None else None,
            "feature_root": str(feature_root),
            "feature_type": args.feature_type,
            "pooling": args.pooling,
            "protocol": args.protocol,
            "positive_mode": config_positive_mode,
            "annotation_pair_modes": (
                None
                if metadata_present
                else {"exo_exo": "sync", "ego_related": "sequence"}
            ),
            "split": args.split,
            "split_bundle": str(split_bundle_path) if split_bundle_path else None,
            "split_index": args.split_index,
            "split_filter_mode": "trial" if args.metadata is None else "row",
            "source_views": source_views,
            "target_views": target_views,
        },
        "dataset": {
            "records_total": len(all_records),
            "retrieval_records_non_null": sum(1 for record in all_records if not record.is_null),
            "occurrences_total": len(occurrences) if occurrences else None,
            "feature_paths_resolved": feature_attach_stats["resolved"],
            "feature_paths_missing": feature_attach_stats["missing"],
            "videos_loaded": feature_stats["videos_loaded"],
            "segments_embedded": feature_stats["segments_embedded"],
        },
        "pair_metrics": pair_metrics,
        "overall": flatten_pair_metrics(pair_metrics),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    if not args.quiet:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
