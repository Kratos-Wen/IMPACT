from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

HAND_TO_INDEX: Dict[str, int] = {"left": 0, "l": 0, "right": 1, "r": 1}
LEFT_HAND = 0
RIGHT_HAND = 1
SUPPORTED_VIEWS = ("front", "left", "right", "top", "ego")
SUPPORTED_VIEW_OPTIONS = (*SUPPORTED_VIEWS, "all", "ego-exclude")
SUPPORTED_FEATURE_TYPES = ("vmae", "i3d")
VIEW_TO_ANNOTATION_DIR = {
    "front": "Front",
    "left": "Left",
    "right": "Right",
    "top": "Top",
    "ego": "ego_valid",
}


@dataclass(frozen=True)
class _Segment:
    start_frame: int
    end_frame: int
    hand_idx: int
    action_label: int
    verb: int
    noun: int
    order: int


@dataclass(frozen=True)
class _SampleIndex:
    video_id: str
    t0: int
    t1: int
    hand_idx: int
    future_action: int
    future_verb: int
    future_noun: int


@dataclass
class _VideoRecord:
    features: np.ndarray
    feature_time_first: bool
    num_frames: int
    action_timeline: np.ndarray
    verb_timeline: np.ndarray
    noun_timeline: np.ndarray


class FeatureDataset(Dataset):
    """
    Frame-aligned feature dataset that yields one sample per hand segment.

    Returned layout:
    - features: [T, D]
    - past_{actions,verbs,nouns}: [T, 2], hand order = [left, right]
    - future_{actions,verbs,nouns}: scalar label for hand_of_interest only
    - hand_of_interest: scalar tensor, 0=left, 1=right

    Notes:
    - Expected layout:
      * annotations: `annotation_dir/{Front,Left,Right,Top,ego_valid}/*.json`
      * features (VMAE): `feature_dir/vmae/IMPACT_{front,left,right,top,ego}/features/*.npy`
      * features (I3D): `feature_dir/i3d/IMPACT_i3d_{front,left,right,top,ego}/features/*.npy`
    - `view` selects one camera view, all views, or all non-ego views (`ego-exclude`).
    - Source annotations/features are frame-aligned at source FPS (typically 30 FPS),
      but samples are built at target FPS (default 4 FPS).
    """

    def __init__(
        self,
        args: Any,
    ) -> None:
        annotation_dir = str(getattr(args, "annotation_dir", "data/Annotation"))
        feature_dir = str(getattr(args, "feature_dir", "data/Features"))
        feature_type = str(getattr(args, "feature_type", "vmae")).strip().lower()
        view = str(getattr(args, "view", "front")).strip().lower()
        tau_obs = float(getattr(args, "tau_obs", 5.0))
        tau_ant = float(getattr(args, "tau_ant", 1.0))
        tau_unit = str(getattr(args, "tau_unit", "frames"))
        strict_missing_features = bool(getattr(args, "strict_missing_features", False))
        target_fps = float(getattr(args, "target_fps", 4.0))

        if tau_obs < 0 or tau_ant < 0 or target_fps <= 0:
            raise ValueError("tau_obs and tau_ant must be non-negative, and target_fps must be positive.")

        tau_unit_normalized = tau_unit.strip().lower()
        if tau_unit_normalized not in {"frames", "seconds"}:
            raise ValueError("tau_unit must be 'frames' or 'seconds'.")
        if feature_type not in SUPPORTED_FEATURE_TYPES:
            raise ValueError(f"feature_type must be one of {SUPPORTED_FEATURE_TYPES}, got '{feature_type}'.")
        if view not in SUPPORTED_VIEW_OPTIONS:
            raise ValueError(f"view must be one of {SUPPORTED_VIEW_OPTIONS}, got '{view}'.")

        self.annotation_dir = Path(annotation_dir)
        self.feature_dir = Path(feature_dir)
        self.feature_type = feature_type
        self.view = view
        self.views = self._resolve_views(self.view)
        self.tau_obs = float(tau_obs)
        self.tau_ant = float(tau_ant)
        self.tau_unit = tau_unit_normalized
        self.strict_missing_features = strict_missing_features
        self.target_fps = target_fps

        if not self.annotation_dir.exists():
            raise FileNotFoundError(f"Annotation directory not found: {self.annotation_dir}")
        if not self.feature_dir.exists():
            raise FileNotFoundError(f"Feature directory not found: {self.feature_dir}")

        self._annotation_dirs = {
            view_name: self.annotation_dir / VIEW_TO_ANNOTATION_DIR[view_name] for view_name in SUPPORTED_VIEWS
        }
        self._feature_dirs = {
            view_name: self.feature_dir / self.feature_type / self._feature_subdir(view_name) / "features"
            for view_name in SUPPORTED_VIEWS
        }
        self._validate_view_dirs()

        self._video_records: Dict[str, _VideoRecord] = {}
        self._samples: List[_SampleIndex] = []
        self._build_index()

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index < 0 or index >= len(self._samples):
            raise IndexError(f"Sample index out of range: {index}")

        sample = self._samples[index]
        video_record = self._video_records[sample.video_id]

        if video_record.feature_time_first:
            feature_window = video_record.features[sample.t0 : sample.t1 + 1, :]
        else:
            feature_window = video_record.features[:, sample.t0 : sample.t1 + 1].T

        if feature_window.shape[0] == 0:
            raise RuntimeError(
                f"Encountered an empty feature window for sample index {index}: "
                f"video={sample.video_id}, t0={sample.t0}, t1={sample.t1}"
            )

        features = torch.from_numpy(np.ascontiguousarray(feature_window, dtype=np.float32))
        past_actions = torch.from_numpy(
            np.ascontiguousarray(video_record.action_timeline[sample.t0 : sample.t1 + 1], dtype=np.int64)
        )
        past_verbs = torch.from_numpy(
            np.ascontiguousarray(video_record.verb_timeline[sample.t0 : sample.t1 + 1], dtype=np.int64)
        )
        past_nouns = torch.from_numpy(
            np.ascontiguousarray(video_record.noun_timeline[sample.t0 : sample.t1 + 1], dtype=np.int64)
        )

        return {
            "features": features,
            "past_actions": past_actions,
            "past_verbs": past_verbs,
            "past_nouns": past_nouns,
            "future_actions": torch.tensor(sample.future_action, dtype=torch.long),
            "future_verbs": torch.tensor(sample.future_verb, dtype=torch.long),
            "future_nouns": torch.tensor(sample.future_noun, dtype=torch.long),
            "hand_of_interest": torch.tensor(sample.hand_idx, dtype=torch.long),
        }

    def _validate_view_dirs(self) -> None:
        missing_annotation: List[str] = []
        missing_feature: List[str] = []
        active_views: List[str] = []
        allow_missing_dirs = self.view in {"all", "ego-exclude"}
        for view_name in self.views:
            missing_items: List[str] = []
            if not self._annotation_dirs[view_name].is_dir():
                missing_annotation.append(str(self._annotation_dirs[view_name]))
                missing_items.append(str(self._annotation_dirs[view_name]))
            if not self._feature_dirs[view_name].is_dir():
                missing_feature.append(str(self._feature_dirs[view_name]))
                missing_items.append(str(self._feature_dirs[view_name]))

            if missing_items and allow_missing_dirs:
                warnings.warn(
                    f"Skipping view '{view_name}' because required directories are missing: {', '.join(missing_items)}",
                    stacklevel=2,
                )
                continue
            if not missing_items:
                active_views.append(view_name)

        if missing_annotation and not allow_missing_dirs:
            raise FileNotFoundError("Missing annotation view directories: " + ", ".join(missing_annotation))
        if missing_feature and not allow_missing_dirs:
            raise FileNotFoundError("Missing feature view directories: " + ", ".join(missing_feature))
        if not active_views:
            raise FileNotFoundError(f"No valid view directories found for view setting '{self.view}'.")

        self.views = tuple(active_views)

    @staticmethod
    def _resolve_views(view: str) -> tuple[str, ...]:
        if view == "all":
            return SUPPORTED_VIEWS
        if view == "ego-exclude":
            return tuple(view_name for view_name in SUPPORTED_VIEWS if view_name != "ego")
        return (view,)

    def _build_index(self) -> None:
        found_any_annotation = False
        for view_name in self.views:
            annotation_files = sorted(self._annotation_dirs[view_name].glob("*.json"))
            if not annotation_files:
                continue
            found_any_annotation = True

            for annotation_path in annotation_files:
                annotation = json.loads(annotation_path.read_text())
                video_id = str(annotation["video_id"])
                feature_path = self._feature_dirs[view_name] / f"{video_id}.npy"
                if not feature_path.exists():
                    message = (
                        f"Missing feature file for video_id={video_id} (view={view_name}). "
                        f"Expected path: {feature_path}"
                    )
                    if self.strict_missing_features:
                        raise FileNotFoundError(message)
                    warnings.warn(message, stacklevel=2)
                    continue

                features = np.load(feature_path, mmap_mode="r")
                if features.ndim != 2:
                    raise ValueError(f"Feature array for {video_id} must be 2D, got shape {features.shape}")

                meta_data = annotation.get("meta_data", {})
                fps = float(meta_data.get("fps", 30.0))
                num_frames_hint = (
                    int(meta_data.get("num_frames", 0)) if meta_data.get("num_frames") is not None else None
                )
                feature_time_first, num_frames = self._infer_feature_layout(
                    features=features, num_frames_hint=num_frames_hint
                )

                segments = self._normalize_segments(annotation.get("segments", []), num_frames=num_frames)
                if not segments:
                    continue

                action_timeline, verb_timeline, noun_timeline = self._build_hand_timelines(
                    segments=segments,
                    num_frames=num_frames,
                )

                frame_indices = self._build_resample_indices(num_frames=num_frames, source_fps=fps)
                features = self._resample_features(
                    features=features,
                    feature_time_first=feature_time_first,
                    frame_indices=frame_indices,
                )
                action_timeline = np.ascontiguousarray(action_timeline[frame_indices], dtype=np.int64)
                verb_timeline = np.ascontiguousarray(verb_timeline[frame_indices], dtype=np.int64)
                noun_timeline = np.ascontiguousarray(noun_timeline[frame_indices], dtype=np.int64)
                num_frames = int(frame_indices.shape[0])

                tau_ant_frames = self._tau_to_frames(self.tau_ant, source_fps=fps)
                tau_obs_frames = self._tau_to_frames(self.tau_obs, source_fps=fps)

                self._video_records[video_id] = _VideoRecord(
                    features=features,
                    feature_time_first=True,
                    num_frames=num_frames,
                    action_timeline=action_timeline,
                    verb_timeline=verb_timeline,
                    noun_timeline=noun_timeline,
                )

                left_segments = [segment for segment in segments if segment.hand_idx == LEFT_HAND]
                right_segments = [segment for segment in segments if segment.hand_idx == RIGHT_HAND]
                union_segments = left_segments + right_segments
                union_segments.sort(key=lambda s: (s.start_frame, s.hand_idx, s.end_frame, s.order))

                for segment in union_segments:
                    segment_start = self._frame_to_target_index(frame_idx=segment.start_frame, source_fps=fps)
                    t1 = segment_start - tau_ant_frames
                    if t1 <= 0:
                        continue

                    t0 = max(0, t1 - tau_obs_frames)
                    t1 = min(t1, num_frames - 1)
                    if t0 > t1:
                        continue

                    self._samples.append(
                        _SampleIndex(
                            video_id=video_id,
                            t0=t0,
                            t1=t1,
                            hand_idx=segment.hand_idx,
                            future_action=segment.action_label,
                            future_verb=segment.verb,
                            future_noun=segment.noun,
                        )
                    )

        if not found_any_annotation:
            view_list = ", ".join(self.views)
            warnings.warn(
                f"No annotation files found for views [{view_list}] under {self.annotation_dir}",
                stacklevel=2,
            )

    @staticmethod
    def _build_hand_timelines(
        segments: Sequence[_Segment],
        num_frames: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        action_timeline = np.full((num_frames, 2), 0, dtype=np.int64)
        verb_timeline = np.full((num_frames, 2), -1, dtype=np.int64)
        noun_timeline = np.full((num_frames, 2), -1, dtype=np.int64)

        for hand_idx in (LEFT_HAND, RIGHT_HAND):
            hand_segments = [segment for segment in segments if segment.hand_idx == hand_idx]
            for segment in hand_segments:
                action_timeline[segment.start_frame : segment.end_frame + 1, hand_idx] = segment.action_label
                verb_timeline[segment.start_frame : segment.end_frame + 1, hand_idx] = segment.verb
                noun_timeline[segment.start_frame : segment.end_frame + 1, hand_idx] = segment.noun

        return action_timeline, verb_timeline, noun_timeline

    def _build_resample_indices(self, num_frames: int, source_fps: float) -> np.ndarray:
        if source_fps <= 0:
            raise ValueError(f"Invalid source FPS: {source_fps}")
        if num_frames <= 0:
            raise ValueError(f"Invalid frame count: {num_frames}")

        target_num_frames = max(1, int(np.floor((num_frames - 1) * self.target_fps / source_fps)) + 1)
        source_indices = np.rint(
            np.arange(target_num_frames, dtype=np.float64) * (source_fps / self.target_fps)
        ).astype(np.int64)
        source_indices = np.clip(source_indices, 0, num_frames - 1)
        return np.unique(source_indices)

    @staticmethod
    def _resample_features(
        features: np.ndarray,
        feature_time_first: bool,
        frame_indices: np.ndarray,
    ) -> np.ndarray:
        if feature_time_first:
            sampled = features[frame_indices, :]
        else:
            sampled = features[:, frame_indices].T
        return np.ascontiguousarray(sampled, dtype=np.float32)

    def _frame_to_target_index(self, frame_idx: int, source_fps: float) -> int:
        return int(round(frame_idx * self.target_fps / source_fps))

    def _normalize_segments(self, segments: Sequence[Mapping[str, Any]], num_frames: int) -> List[_Segment]:
        normalized_segments: List[_Segment] = []
        for idx, segment in enumerate(segments):
            hand_idx = self._parse_hand(segment)
            start_frame = int(segment["start_frame"])
            end_frame = int(segment["end_frame"])

            if end_frame < 0 or start_frame >= num_frames:
                continue

            start_frame = max(0, start_frame)
            end_frame = min(num_frames - 1, end_frame)
            if end_frame < start_frame:
                continue

            normalized_segments.append(
                _Segment(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    hand_idx=hand_idx,
                    action_label=int(segment["action_label"]),
                    verb=int(segment["verb"]),
                    noun=int(segment["noun"]),
                    order=idx,
                )
            )
        return normalized_segments

    @staticmethod
    def _parse_hand(segment: Mapping[str, Any]) -> int:
        raw = segment.get("entity", segment.get("hand"))
        if raw is None:
            raise KeyError("Segment is missing both 'entity' and 'hand' fields.")

        hand_key = str(raw).strip().lower()
        if hand_key not in HAND_TO_INDEX:
            raise ValueError(f"Unsupported hand value: {raw}")
        return HAND_TO_INDEX[hand_key]

    @staticmethod
    def _infer_feature_layout(features: np.ndarray, num_frames_hint: Optional[int]) -> tuple[bool, int]:
        if num_frames_hint is not None:
            if features.shape[0] == num_frames_hint and features.shape[1] != num_frames_hint:
                return True, int(num_frames_hint)
            if features.shape[1] == num_frames_hint and features.shape[0] != num_frames_hint:
                return False, int(num_frames_hint)
            if features.shape[0] == num_frames_hint:
                return True, int(num_frames_hint)
            if features.shape[1] == num_frames_hint:
                return False, int(num_frames_hint)

        # Heuristic fallback for datasets where frame count metadata is missing:
        # time axis is usually the longer one for per-frame features.
        if features.shape[0] >= features.shape[1]:
            return True, int(features.shape[0])
        return False, int(features.shape[1])

    def _tau_to_frames(self, tau: float, source_fps: float) -> int:
        if self.tau_unit == "frames":
            return int(round((tau / source_fps) * self.target_fps))
        return int(round(tau * self.target_fps))

    def _feature_subdir(self, view_name: str) -> str:
        if self.feature_type == "i3d":
            return f"IMPACT_i3d_{view_name}"
        return f"IMPACT_{view_name}"


__all__ = ["FeatureDataset", "HAND_TO_INDEX", "LEFT_HAND", "RIGHT_HAND"]
