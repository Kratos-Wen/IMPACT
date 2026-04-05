from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from qwen_vl_utils import process_vision_info
from qwen_vl_utils.vision_process import SPATIAL_MERGE_SIZE
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


HAND_TO_INDEX: Dict[str, int] = {"left": 0, "l": 0, "right": 1, "r": 1}
INDEX_TO_HAND: Dict[int, str] = {0: "left", 1: "right"}
VIEW_TO_ANNOTATION_DIR: Dict[str, str] = {
    "front": "Front",
    "left": "Left",
    "right": "Right",
    "top": "Top",
    "ego": "ego_valid",
}
SUPPORTED_VIEWS: Sequence[str] = ("front", "left", "right", "top", "ego")


def _repo_root() -> Path:
    project_root = Path(__file__).resolve().parent
    if project_root.parent.name == "third_party":
        return project_root.parent.parent
    return project_root


def _default_annotation_dir() -> str:
    return str(_repo_root() / "dataset" / "AF-S" / "Annotation")


def _default_output_dir() -> str:
    return str(_repo_root() / "outputs" / "af_s" / "qwen3_vl_8b")


@dataclass(frozen=True)
class AnticipationSample:
    video_id: str
    view: str
    hand: str
    hand_idx: int
    obs_start_sec: float
    obs_end_sec: float
    future_action_id: int
    future_action_name: str
    source_annotation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3VL-8B on the IMPACT AF-S benchmark."
    )
    parser.add_argument("--annotation-dir", type=str, default=_default_annotation_dir())
    parser.add_argument("--split-dir", type=str, default=None)
    parser.add_argument("--split-name", type=str, default="split1")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--view", type=str, choices=[*SUPPORTED_VIEWS, "all", "ego-exclude"], default="front")
    parser.add_argument("--tau-obs", type=float, default=5.0, help="Observation duration in seconds.")
    parser.add_argument(
        "--tau-ant",
        type=float,
        default=1.0,
        help="Anticipation gap in seconds between observation end and future-action start.",
    )
    parser.add_argument("--target-fps", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-samples", action="store_true")
    parser.add_argument(
        "--score-records",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more existing JSONL record files to merge and evaluate without running inference. "
            "Later files override earlier ones for duplicate samples."
        ),
    )

    parser.add_argument("--video-root", type=str, default=None)
    parser.add_argument(
        "--video-template",
        type=str,
        default="",
        help=(
            "Optional path template relative to --video-root. "
            "Available fields: {video_id}, {view}, {view_dir}."
        ),
    )
    parser.add_argument(
        "--video-exts",
        type=str,
        default="mp4",
        help="Comma-separated extensions used when --video-template is not provided.",
    )
    parser.add_argument("--skip-missing-video", action="store_true")

    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--processor-name-or-path", type=str, default=None)
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")

    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--video-max-pixels", type=int, default=512)
    parser.add_argument("--video-min-pixels", type=int, default=128)
    parser.add_argument("--video-total-pixels", type=int, default=256000)
    parser.add_argument("--max-frames", type=int, default=128)

    parser.add_argument("--output-dir", type=str, default=_default_output_dir())
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--dist-backend", type=str, default=None, choices=["nccl", "gloo"])
    parser.add_argument("--dist-timeout-sec", type=int, default=3600)
    return parser.parse_args()


def resolve_views(view: str) -> tuple[str, ...]:
    if view == "all":
        return tuple(SUPPORTED_VIEWS)
    if view == "ego-exclude":
        return tuple(v for v in SUPPORTED_VIEWS if v != "ego")
    return (view,)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def get_dist_info() -> tuple[int, int, int]:
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)
    return rank, world_size, local_rank


def maybe_init_distributed(args: argparse.Namespace) -> tuple[int, int, int]:
    rank, world_size, local_rank = get_dist_info()
    if world_size <= 1:
        return rank, world_size, local_rank
    if not torch.distributed.is_available():
        raise RuntimeError("WORLD_SIZE>1 but torch.distributed is not available in this environment.")
    if not torch.distributed.is_initialized():
        backend = args.dist_backend
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        timeout = timedelta(seconds=max(1, int(args.dist_timeout_sec)))
        torch.distributed.init_process_group(backend=backend, timeout=timeout)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def load_split_video_ids(split_dir: Path, split: str, split_name: str) -> set[str]:
    split_path = split_dir / f"{split}.{split_name}.bundle"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split bundle not found: {split_path}")
    lines = split_path.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip()}


def _build_action_name_list(action_labels: Sequence[dict]) -> List[str]:
    if not action_labels:
        raise ValueError("No action_labels found in annotation file.")
    max_id = max(int(item["id"]) for item in action_labels)
    names = [f"unknown_{i}" for i in range(max_id + 1)]
    for item in action_labels:
        idx = int(item["id"])
        names[idx] = str(item["name"])
    return names


def load_action_vocab(annotation_dir: Path, views: Sequence[str]) -> List[str]:
    for view in views:
        ann_subdir = annotation_dir / VIEW_TO_ANNOTATION_DIR[view]
        if not ann_subdir.is_dir():
            continue
        for ann_path in sorted(ann_subdir.glob("*.json")):
            payload = json.loads(ann_path.read_text(encoding="utf-8"))
            labels = payload.get("action_labels", [])
            if labels:
                return _build_action_name_list(labels)
    raise RuntimeError("Failed to load action vocabulary from annotation JSON files.")


def _build_label_name_list(label_items: Sequence[dict], prefix: str) -> List[str]:
    if not label_items:
        return []
    max_id = max(int(item["id"]) for item in label_items)
    names = [f"{prefix}_{i}" for i in range(max_id + 1)]
    for item in label_items:
        idx = int(item["id"])
        names[idx] = str(item["name"])
    return names


def load_action_to_verb_noun_map(
    annotation_dir: Path,
    views: Sequence[str],
) -> tuple[Dict[int, tuple[int, int]], List[str], List[str]]:
    # Keep the function signature stable, but build this taxonomy globally so
    # verb/noun metrics are not view-dependent.
    _ = views
    action_to_verb_noun: Dict[int, tuple[int, int]] = {}
    verb_names: List[str] = []
    noun_names: List[str] = []
    declared_action_ids: set[int] = set()

    for view in SUPPORTED_VIEWS:
        ann_subdir = annotation_dir / VIEW_TO_ANNOTATION_DIR[view]
        if not ann_subdir.is_dir():
            continue
        for ann_path in sorted(ann_subdir.glob("*.json")):
            payload = json.loads(ann_path.read_text(encoding="utf-8"))

            current_verb_names = _build_label_name_list(payload.get("verbs", []), "verb")
            current_noun_names = _build_label_name_list(payload.get("nouns", []), "noun")
            if current_verb_names:
                if not verb_names:
                    verb_names = current_verb_names
                elif verb_names != current_verb_names:
                    raise ValueError(f"Inconsistent verb label taxonomy in {ann_path}.")
            if current_noun_names:
                if not noun_names:
                    noun_names = current_noun_names
                elif noun_names != current_noun_names:
                    raise ValueError(f"Inconsistent noun label taxonomy in {ann_path}.")

            for action_item in payload.get("action_labels", []):
                declared_action_ids.add(int(action_item["id"]))

            for seg in payload.get("segments", []):
                action_id = int(seg.get("action_label", -1))
                if action_id < 0:
                    continue
                pair = (int(seg.get("verb", -1)), int(seg.get("noun", -1)))
                existing = action_to_verb_noun.get(action_id)
                if existing is not None and existing != pair:
                    raise ValueError(
                        f"Inconsistent action->(verb,noun) mapping for action {action_id}: "
                        f"{existing} vs {pair} in {ann_path}."
                    )
                action_to_verb_noun[action_id] = pair

    if not action_to_verb_noun:
        raise RuntimeError("Failed to build action->(verb,noun) mapping from annotation files.")
    if declared_action_ids:
        missing_ids = sorted(action_id for action_id in declared_action_ids if action_id not in action_to_verb_noun)
        if missing_ids:
            preview = ",".join(str(item) for item in missing_ids[:10])
            suffix = "..." if len(missing_ids) > 10 else ""
            raise RuntimeError(
                "Global action->(verb,noun) mapping is incomplete; "
                f"missing action IDs: {preview}{suffix}."
            )
    return action_to_verb_noun, verb_names, noun_names


def _segment_to_sample(
    *,
    annotation_path: Path,
    payload: dict,
    view: str,
    segment: dict,
    action_names: Sequence[str],
    tau_obs: float,
    tau_ant: float,
    target_fps: float,
) -> Optional[AnticipationSample]:
    raw_hand = str(segment.get("entity", segment.get("hand", ""))).strip().lower()
    if raw_hand not in HAND_TO_INDEX:
        return None
    hand_idx = HAND_TO_INDEX[raw_hand]
    hand = INDEX_TO_HAND[hand_idx]

    start_frame = int(segment["start_frame"])
    fps = float(payload.get("meta_data", {}).get("fps", 30.0))
    if fps <= 0:
        return None

    tau_ant_target_frames = int(round(tau_ant * target_fps))
    segment_start_target_frame = int(round(start_frame * target_fps / fps))
    t1 = segment_start_target_frame - tau_ant_target_frames
    if t1 <= 0:
        return None

    event_start_sec = start_frame / fps
    obs_end_sec = max(0.0, event_start_sec - tau_ant)
    obs_start_sec = max(0.0, obs_end_sec - tau_obs)
    if obs_end_sec <= obs_start_sec:
        return None

    future_action_id = int(segment["action_label"])
    if future_action_id < 0 or future_action_id >= len(action_names):
        return None

    return AnticipationSample(
        video_id=str(payload["video_id"]),
        view=view,
        hand=hand,
        hand_idx=hand_idx,
        obs_start_sec=float(obs_start_sec),
        obs_end_sec=float(obs_end_sec),
        future_action_id=future_action_id,
        future_action_name=action_names[future_action_id],
        source_annotation=str(annotation_path),
    )


def _candidate_split_video_ids(video_id: str, annotation_path: Path) -> set[str]:
    candidates = {video_id, annotation_path.stem}
    if video_id.endswith("_clipped"):
        candidates.add(video_id[: -len("_clipped")])
    else:
        candidates.add(f"{video_id}_clipped")
    return {item for item in candidates if item}


def summarize_view_matches(
    *,
    annotation_dir: Path,
    split_video_ids: set[str],
    views: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for view in views:
        ann_subdir = annotation_dir / VIEW_TO_ANNOTATION_DIR[view]
        file_count = 0
        matched_file_count = 0
        for ann_path in sorted(ann_subdir.glob("*.json")):
            file_count += 1
            payload = json.loads(ann_path.read_text(encoding="utf-8"))
            video_id = str(payload.get("video_id", ""))
            candidate_video_ids = _candidate_split_video_ids(video_id=video_id, annotation_path=ann_path)
            if candidate_video_ids and not candidate_video_ids.isdisjoint(split_video_ids):
                matched_file_count += 1
        summary[view] = {
            "annotation_files": file_count,
            "matched_split_files": matched_file_count,
        }
    return summary


def build_samples(
    *,
    annotation_dir: Path,
    split_video_ids: set[str],
    views: Sequence[str],
    action_names: Sequence[str],
    tau_obs: float,
    tau_ant: float,
    target_fps: float,
) -> List[AnticipationSample]:
    samples: List[AnticipationSample] = []
    for view in views:
        ann_subdir = annotation_dir / VIEW_TO_ANNOTATION_DIR[view]
        if not ann_subdir.is_dir():
            continue

        for ann_path in sorted(ann_subdir.glob("*.json")):
            payload = json.loads(ann_path.read_text(encoding="utf-8"))
            video_id = str(payload.get("video_id", ""))
            candidate_video_ids = _candidate_split_video_ids(video_id=video_id, annotation_path=ann_path)
            if not candidate_video_ids or candidate_video_ids.isdisjoint(split_video_ids):
                continue

            for segment in payload.get("segments", []):
                sample = _segment_to_sample(
                    annotation_path=ann_path,
                    payload=payload,
                    view=view,
                    segment=segment,
                    action_names=action_names,
                    tau_obs=tau_obs,
                    tau_ant=tau_ant,
                    target_fps=target_fps,
                )
                if sample is not None:
                    samples.append(sample)
    return samples


def _format_key_float(value: object) -> str:
    return f"{float(value):.6f}"


def make_sample_key(
    *,
    video_id: object,
    view: object,
    hand: object,
    obs_start_sec: object,
    obs_end_sec: object,
    future_action_id: object,
) -> str:
    return "|".join(
        (
            str(video_id),
            str(view),
            str(hand).lower(),
            _format_key_float(obs_start_sec),
            _format_key_float(obs_end_sec),
            str(int(future_action_id)),
        )
    )


def sample_key_from_sample(sample: AnticipationSample) -> str:
    return make_sample_key(
        video_id=sample.video_id,
        view=sample.view,
        hand=sample.hand,
        obs_start_sec=sample.obs_start_sec,
        obs_end_sec=sample.obs_end_sec,
        future_action_id=sample.future_action_id,
    )


def sample_key_from_record(record: Mapping[str, object]) -> str:
    return make_sample_key(
        video_id=record["video_id"],
        view=record["view"],
        hand=record["hand"],
        obs_start_sec=record["obs_start_sec"],
        obs_end_sec=record["obs_end_sec"],
        future_action_id=record["future_action_id"],
    )


def load_record_files(record_paths: Sequence[Path]) -> tuple[List[dict], Dict[str, object]]:
    merged_by_key: Dict[str, dict] = {}
    stats: Dict[str, object] = {
        "num_input_files": len(record_paths),
        "num_loaded_records": 0,
        "num_unique_records": 0,
        "num_overwritten_records": 0,
        "per_file_counts": {},
    }

    for record_path in record_paths:
        if not record_path.is_file():
            raise FileNotFoundError(f"Record file not found: {record_path}")
        count = 0
        for line in record_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = sample_key_from_record(record)
            if key in merged_by_key:
                stats["num_overwritten_records"] = int(stats["num_overwritten_records"]) + 1
            merged_by_key[key] = record
            count += 1
        stats["per_file_counts"][str(record_path)] = count
        stats["num_loaded_records"] = int(stats["num_loaded_records"]) + count

    merged_records = list(merged_by_key.values())
    merged_records.sort(key=sample_key_from_record)
    stats["num_unique_records"] = len(merged_records)
    return merged_records, stats


def _candidate_video_paths(
    *,
    video_id: str,
    view: str,
    view_dir: str,
    video_root: Path,
    video_exts: Sequence[str],
    video_template: str,
) -> Iterable[Path]:
    if video_template:
        rendered = video_template.format(video_id=video_id, view=view, view_dir=view_dir)
        template_path = Path(rendered)
        yield template_path if template_path.is_absolute() else video_root / template_path
        return

    base_names: List[str] = [video_id]
    path_video_id = Path(video_id)
    if path_video_id.suffix:
        base_names.append(path_video_id.stem)

    ext_candidates: List[str] = []
    for ext in video_exts:
        ext_clean = ext.strip().lstrip(".")
        if ext_clean:
            ext_candidates.append(ext_clean)

    for base in base_names:
        if Path(base).suffix:
            yield video_root / base
            yield video_root / view / base
            yield video_root / view_dir / base
            yield video_root / "videos" / base
            yield video_root / "videos" / view / base
            yield video_root / "videos" / view_dir / base
            continue

        for ext in ext_candidates:
            filename = f"{base}.{ext}"
            yield video_root / filename
            yield video_root / view / filename
            yield video_root / view_dir / filename
            yield video_root / "videos" / filename
            yield video_root / "videos" / view / filename
            yield video_root / "videos" / view_dir / filename


def resolve_video_path(
    *,
    sample: AnticipationSample,
    video_root: Path,
    video_exts: Sequence[str],
    video_template: str,
) -> Optional[Path]:
    view_dir = VIEW_TO_ANNOTATION_DIR[sample.view]
    seen: set[Path] = set()
    for candidate in _candidate_video_paths(
        video_id=sample.video_id,
        view=sample.view,
        view_dir=view_dir,
        video_root=video_root,
        video_exts=video_exts,
        video_template=video_template,
    ):
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def build_action_catalog_text(action_names: Sequence[str]) -> str:
    return "\n".join(f"{i}: {name}" for i, name in enumerate(action_names))


def build_prompt(
    *,
    sample: AnticipationSample,
    action_catalog_text: str,
    num_classes: int,
) -> str:
    return (
        "You are given a 5-second video clip from a two-hand assembly task.\n"
        "The clip ends exactly 1 second before the next labeled action starts.\n"
        f"Predict the 5 most likely next action classes for the {sample.hand} hand.\n\n"
        "Choose exactly five unique class IDs from the full class list below, ordered from most likely to least likely.\n"
        "Probability ordering constraint: the first ID must have higher probability than each later ID.\n"
        f"Valid ID range: 0 to {num_classes - 1}.\n\n"
        "Action classes (id: name):\n"
        f"{action_catalog_text}\n\n"
        "Output format requirement: reply with only five integer IDs separated by commas."
    )


def _torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def run_qwen_generation(
    *,
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    video_path: Path,
    video_start: float,
    video_end: float,
    prompt: str,
    video_max_pixels: int,
    video_min_pixels: int,
    video_total_pixels: int,
    max_frames: int,
    max_new_tokens: int,
    temperature: float,
) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "video_start": float(video_start),
                    "video_end": float(video_end),
                    "min_pixels": video_min_pixels,
                    "max_pixels": video_max_pixels,
                    "total_pixels": video_total_pixels,
                    "max_frames": max_frames,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    texts = processor.apply_chat_template([conversation], tokenize=False, add_generation_prompt=True)
    _, video_inputs, video_kwargs = process_vision_info(
        [conversation],
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    videos, video_metadatas = zip(*video_inputs)
    video_tensor_batch = list(videos)[0]
    video_metadata_batch = list(video_metadatas)[0]

    inputs = processor(
        text=texts,
        videos=video_tensor_batch,
        video_metadata=video_metadata_batch,
        padding=True,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )
    if "pixel_values_videos" in inputs and inputs["pixel_values_videos"] is not None:
        inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(dtype=torch.bfloat16)

    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            inputs[key] = value.to(device=model.device)

    do_sample = temperature > 0.0
    generation_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = float(temperature)

    generated_ids = model.generate(**inputs, **generation_kwargs)
    trimmed_ids = [
        output_ids[input_ids.shape[0] :]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]
    text = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text.strip()


def parse_predicted_action_ids(
    response_text: str,
    action_names: Sequence[str],
    *,
    top_k: int = 5,
) -> List[int]:
    if top_k <= 0:
        return []

    parsed: List[int] = []
    seen: set[int] = set()
    for token in re.findall(r"-?\d+", response_text):
        idx = int(token)
        if 0 <= idx < len(action_names) and idx not in seen:
            parsed.append(idx)
            seen.add(idx)
            if len(parsed) >= top_k:
                return parsed

    if len(parsed) >= top_k:
        return parsed[:top_k]

    normalized_name_to_id: Dict[str, int] = {}
    for idx, name in enumerate(action_names):
        normalized = re.sub(r"\s+", "_", name.strip().lower().replace("-", "_"))
        normalized_name_to_id[normalized] = idx

    for chunk in re.split(r"[\n,;|]+", response_text):
        normalized_chunk = re.sub(r"\s+", "_", chunk.strip().lower().replace("-", "_"))
        if not normalized_chunk:
            continue
        idx = normalized_name_to_id.get(normalized_chunk)
        if idx is None or idx in seen:
            continue
        parsed.append(idx)
        seen.add(idx)
        if len(parsed) >= top_k:
            return parsed

    return parsed


def _map_action_topk_to_task_topk(
    action_topk: Sequence[int],
    action_to_verb_noun: Mapping[int, tuple[int, int]],
    *,
    task: str,
    top_k: int = 5,
) -> List[int]:
    task_index = 0 if task == "verb" else 1
    mapped: List[int] = []
    seen: set[int] = set()
    for action_id in action_topk:
        pair = action_to_verb_noun.get(int(action_id))
        if pair is None:
            continue
        label = int(pair[task_index])
        if label < 0 or label in seen:
            continue
        mapped.append(label)
        seen.add(label)
        if len(mapped) >= top_k:
            break
    return mapped


def _mean_recall_from_counts(hit_counts: Sequence[int], class_counts: Sequence[int]) -> float:
    observed: List[float] = []
    for hit, count in zip(hit_counts, class_counts):
        if count > 0:
            observed.append(float(hit) / float(count))
    return (sum(observed) / len(observed)) if observed else 0.0


def _evaluate_task_from_topk_records(
    *,
    records: Sequence[dict],
    num_classes: int,
    key_gt: str,
    key_pred_top5: str,
) -> dict:
    split_names = ("overall", "left", "right")
    stats: Dict[str, Dict[str, object]] = {
        split: {
            "class_counts": [0] * num_classes,
            "top1_hit_counts": [0] * num_classes,
            "top5_hit_counts": [0] * num_classes,
            "total_valid": 0,
            "parsed_total": 0,
            "strict_top1_correct": 0,
            "strict_top5_correct": 0,
            "parsed_top1_correct": 0,
            "parsed_top5_correct": 0,
        }
        for split in split_names
    }

    for rec in records:
        gt = int(rec.get(key_gt, -1))
        if gt < 0 or gt >= num_classes:
            continue
        hand = str(rec.get("hand", "")).lower()
        relevant_splits = ["overall"]
        if hand in {"left", "right"}:
            relevant_splits.append(hand)
        for split in relevant_splits:
            split_stats = stats[split]
            split_stats["total_valid"] = int(split_stats["total_valid"]) + 1
            class_counts = split_stats["class_counts"]
            class_counts[gt] += 1

        raw_top5 = rec.get(key_pred_top5, [])
        pred_top5: List[int] = []
        seen_pred: set[int] = set()
        if isinstance(raw_top5, list):
            for pred in raw_top5:
                pred_idx = int(pred)
                if 0 <= pred_idx < num_classes and pred_idx not in seen_pred:
                    pred_top5.append(pred_idx)
                    seen_pred.add(pred_idx)
                if len(pred_top5) >= 5:
                    break
        if not pred_top5:
            continue

        for split in relevant_splits:
            split_stats = stats[split]
            split_stats["parsed_total"] = int(split_stats["parsed_total"]) + 1
            if pred_top5[0] == gt:
                split_stats["strict_top1_correct"] = int(split_stats["strict_top1_correct"]) + 1
                split_stats["parsed_top1_correct"] = int(split_stats["parsed_top1_correct"]) + 1
                top1_hit_counts = split_stats["top1_hit_counts"]
                top1_hit_counts[gt] += 1
            if gt in pred_top5:
                split_stats["strict_top5_correct"] = int(split_stats["strict_top5_correct"]) + 1
                split_stats["parsed_top5_correct"] = int(split_stats["parsed_top5_correct"]) + 1
                top5_hit_counts = split_stats["top5_hit_counts"]
                top5_hit_counts[gt] += 1

    result: Dict[str, object] = {}
    for split in split_names:
        split_stats = stats[split]
        class_counts = split_stats["class_counts"]
        top1_hit_counts = split_stats["top1_hit_counts"]
        top5_hit_counts = split_stats["top5_hit_counts"]
        total_valid = int(split_stats["total_valid"])
        parsed_total = int(split_stats["parsed_total"])
        strict_top1_correct = int(split_stats["strict_top1_correct"])
        strict_top5_correct = int(split_stats["strict_top5_correct"])
        parsed_top1_correct = int(split_stats["parsed_top1_correct"])
        parsed_top5_correct = int(split_stats["parsed_top5_correct"])

        per_class_top5_recall: Dict[str, float] = {}
        observed_classes = 0
        for class_idx, count in enumerate(class_counts):
            if count <= 0:
                continue
            observed_classes += 1
            per_class_top5_recall[str(class_idx)] = float(top5_hit_counts[class_idx]) / float(count)

        result[f"{split}_num_valid_samples"] = total_valid
        result[f"{split}_strict_top1_accuracy"] = (strict_top1_correct / total_valid) if total_valid > 0 else 0.0
        result[f"{split}_strict_top5_accuracy"] = (strict_top5_correct / total_valid) if total_valid > 0 else 0.0
        result[f"{split}_parsed_top1_accuracy"] = (parsed_top1_correct / parsed_total) if parsed_total > 0 else 0.0
        result[f"{split}_parsed_top5_accuracy"] = (parsed_top5_correct / parsed_total) if parsed_total > 0 else 0.0
        result[f"{split}_parse_success_rate"] = (parsed_total / total_valid) if total_valid > 0 else 0.0
        result[f"{split}_mean_class_recall"] = _mean_recall_from_counts(top1_hit_counts, class_counts)
        result[f"{split}_mean_top5_recall"] = _mean_recall_from_counts(top5_hit_counts, class_counts)
        result[f"{split}_num_observed_classes"] = observed_classes
        result[f"{split}_per_class_top5_recall"] = per_class_top5_recall

    result["num_valid_samples"] = result["overall_num_valid_samples"]
    result["strict_top1_accuracy"] = result["overall_strict_top1_accuracy"]
    result["strict_top5_accuracy"] = result["overall_strict_top5_accuracy"]
    result["parsed_top1_accuracy"] = result["overall_parsed_top1_accuracy"]
    result["parsed_top5_accuracy"] = result["overall_parsed_top5_accuracy"]
    result["parse_success_rate"] = result["overall_parse_success_rate"]
    result["mean_class_recall"] = result["overall_mean_class_recall"]
    result["mean_top5_recall"] = result["overall_mean_top5_recall"]
    result["num_observed_classes"] = result["overall_num_observed_classes"]
    result["per_class_top5_recall"] = result["overall_per_class_top5_recall"]
    return result


def evaluate(
    *,
    records: Sequence[dict],
    num_classes: int,
    action_to_verb_noun: Mapping[int, tuple[int, int]],
    num_verbs: int,
    num_nouns: int,
) -> dict:
    normalized_records: List[dict] = []
    parse_failures = 0
    full_top5_parsed_total = 0
    for rec in records:
        raw_top5 = rec.get("pred_action_top5_ids")
        pred_action_top5: List[int] = []
        seen: set[int] = set()
        if isinstance(raw_top5, list):
            for pred in raw_top5:
                pred_idx = int(pred)
                if 0 <= pred_idx < num_classes and pred_idx not in seen:
                    pred_action_top5.append(pred_idx)
                    seen.add(pred_idx)
                if len(pred_action_top5) >= 5:
                    break
        elif rec.get("pred_action_id") is not None:
            pred_idx = int(rec["pred_action_id"])
            if 0 <= pred_idx < num_classes:
                pred_action_top5 = [pred_idx]

        if not pred_action_top5:
            parse_failures += 1
        if len(pred_action_top5) >= 5:
            full_top5_parsed_total += 1

        gt_action = int(rec["future_action_id"])
        gt_verb, gt_noun = action_to_verb_noun.get(gt_action, (-1, -1))
        normalized = dict(rec)
        normalized["pred_action_top5_ids"] = pred_action_top5
        normalized["future_verb_id"] = int(gt_verb)
        normalized["future_noun_id"] = int(gt_noun)
        normalized["pred_verb_top5_ids"] = _map_action_topk_to_task_topk(
            pred_action_top5,
            action_to_verb_noun,
            task="verb",
            top_k=5,
        )
        normalized["pred_noun_top5_ids"] = _map_action_topk_to_task_topk(
            pred_action_top5,
            action_to_verb_noun,
            task="noun",
            top_k=5,
        )
        normalized_records.append(normalized)

    action_metrics = _evaluate_task_from_topk_records(
        records=normalized_records,
        num_classes=num_classes,
        key_gt="future_action_id",
        key_pred_top5="pred_action_top5_ids",
    )
    verb_metrics = _evaluate_task_from_topk_records(
        records=normalized_records,
        num_classes=num_verbs,
        key_gt="future_verb_id",
        key_pred_top5="pred_verb_top5_ids",
    )
    noun_metrics = _evaluate_task_from_topk_records(
        records=normalized_records,
        num_classes=num_nouns,
        key_gt="future_noun_id",
        key_pred_top5="pred_noun_top5_ids",
    )

    metrics: Dict[str, object] = {
        "num_samples": len(records),
        "parse_failures": parse_failures,
        "full_top5_parse_rate": (full_top5_parsed_total / len(records)) if records else 0.0,
    }
    for prefix, task_metrics in (("action", action_metrics), ("verb", verb_metrics), ("noun", noun_metrics)):
        for key, value in task_metrics.items():
            metrics[f"{prefix}_{key}"] = value
    metrics["strict_top1_accuracy"] = action_metrics["strict_top1_accuracy"]
    metrics["strict_top5_accuracy"] = action_metrics["strict_top5_accuracy"]
    metrics["parsed_top1_accuracy"] = action_metrics["parsed_top1_accuracy"]
    metrics["parsed_top5_accuracy"] = action_metrics["parsed_top5_accuracy"]
    metrics["parse_success_rate"] = action_metrics["parse_success_rate"]
    metrics["mean_class_recall"] = action_metrics["mean_class_recall"]
    metrics["mean_top5_recall"] = action_metrics["mean_top5_recall"]
    metrics["num_observed_classes"] = action_metrics["num_observed_classes"]
    metrics["per_class_top5_recall"] = action_metrics["per_class_top5_recall"]
    return metrics


def resolve_run_tag(args: argparse.Namespace, rank: int, world_size: int) -> str:
    if args.run_tag:
        return str(args.run_tag)

    base = f"qwen3vl8b_{args.split}_{args.view}"
    timestamp = ""
    if world_size <= 1:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return f"{base}_{timestamp}"

    if rank == 0:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    object_list = [timestamp]
    torch.distributed.broadcast_object_list(object_list, src=0)
    timestamp = str(object_list[0])
    return f"{base}_{timestamp}_ws{world_size}"


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.score_records:
        rank, world_size, local_rank = 0, 1, 0
    else:
        rank, world_size, local_rank = maybe_init_distributed(args)
    try:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        annotation_dir = Path(args.annotation_dir).expanduser().resolve()
        split_dir = (
            Path(args.split_dir).expanduser().resolve()
            if args.split_dir is not None
            else (annotation_dir / "splits").resolve()
        )
        video_root = Path(args.video_root).expanduser().resolve() if args.video_root else None
        if not annotation_dir.is_dir():
            raise FileNotFoundError(f"Annotation dir not found: {annotation_dir}")
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split dir not found: {split_dir}")
        if args.score_records is None and video_root is None:
            raise ValueError("--video-root is required unless --score-records is used.")
        if video_root is not None and not video_root.is_dir():
            raise FileNotFoundError(f"Video root not found: {video_root}")

        views = resolve_views(args.view)
        split_video_ids = load_split_video_ids(split_dir=split_dir, split=args.split, split_name=args.split_name)
        action_names = load_action_vocab(annotation_dir=annotation_dir, views=views)
        action_to_verb_noun, verb_names, noun_names = load_action_to_verb_noun_map(
            annotation_dir=annotation_dir,
            views=views,
        )

        samples = build_samples(
            annotation_dir=annotation_dir,
            split_video_ids=split_video_ids,
            views=views,
            action_names=action_names,
            tau_obs=float(args.tau_obs),
            tau_ant=float(args.tau_ant),
            target_fps=float(args.target_fps),
        )
        if not samples:
            view_match_summary = summarize_view_matches(
                annotation_dir=annotation_dir,
                split_video_ids=split_video_ids,
                views=views,
            )
            raise RuntimeError(
                "No anticipation samples found for the selected split/view settings. "
                f"annotation_dir={annotation_dir}, split_dir={split_dir}, split={args.split}, "
                f"split_name={args.split_name}, view={args.view}, view_match_summary={view_match_summary}"
            )

        if args.shuffle_samples:
            random.Random(args.seed).shuffle(samples)
        if args.max_samples is not None:
            samples = samples[: max(0, int(args.max_samples))]
        if not samples:
            raise RuntimeError("No samples left after applying --max-samples.")

        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        run_tag = resolve_run_tag(args=args, rank=rank, world_size=world_size)
        records_path = output_dir / f"{run_tag}.jsonl"
        summary_path = output_dir / f"{run_tag}.summary.json"

        if args.score_records:
            input_record_paths = [Path(item).expanduser().resolve() for item in args.score_records]
            merged_input_records, load_stats = load_record_files(input_record_paths)
            expected_keys = {sample_key_from_sample(sample) for sample in samples}
            filtered_records = [
                record for record in merged_input_records if sample_key_from_record(record) in expected_keys
            ]
            found_keys = {sample_key_from_record(record) for record in filtered_records}
            missing_keys = expected_keys - found_keys
            if missing_keys:
                preview = ", ".join(sorted(missing_keys)[:5])
                suffix = " ..." if len(missing_keys) > 5 else ""
                raise RuntimeError(
                    f"Merged record files are missing {len(missing_keys)} required samples for view={args.view}: "
                    f"{preview}{suffix}"
                )

            metrics = evaluate(
                records=filtered_records,
                num_classes=len(action_names),
                action_to_verb_noun=action_to_verb_noun,
                num_verbs=max(1, len(verb_names)),
                num_nouns=max(1, len(noun_names)),
            )
            with records_path.open("w", encoding="utf-8") as f:
                for rec in filtered_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            summary = {
                "run_tag": run_tag,
                "args": vars(args),
                "world_size": world_size,
                "num_classes": len(action_names),
                "num_verbs": max(1, len(verb_names)),
                "num_nouns": max(1, len(noun_names)),
                "num_total_samples": len(samples),
                "num_processed_records": len(filtered_records),
                "num_extra_records_ignored": len(merged_input_records) - len(filtered_records),
                "metrics": metrics,
                "records_path": str(records_path),
                "score_record_paths": [str(path) for path in input_record_paths],
                "record_load_stats": load_stats,
            }
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

            print("\nScoring finished.")
            print(f"Records: {records_path}")
            print(f"Summary: {summary_path}")
            print(
                "Key metrics: "
                f"action_top1={metrics['action_strict_top1_accuracy']:.4f}, "
                f"action_top5={metrics['action_strict_top5_accuracy']:.4f}, "
                f"action_mean_top5_recall={metrics['action_mean_top5_recall']:.4f}, "
                f"verb_top1={metrics['verb_strict_top1_accuracy']:.4f}, "
                f"verb_top5={metrics['verb_strict_top5_accuracy']:.4f}, "
                f"verb_mean_top5_recall={metrics['verb_mean_top5_recall']:.4f}, "
                f"noun_top1={metrics['noun_strict_top1_accuracy']:.4f}, "
                f"noun_top5={metrics['noun_strict_top5_accuracy']:.4f}, "
                f"noun_mean_top5_recall={metrics['noun_mean_top5_recall']:.4f}"
            )
            return

        action_catalog_text = build_action_catalog_text(action_names)

        sharded_indices = list(range(rank, len(samples), world_size))
        shard_samples = [samples[idx] for idx in sharded_indices]

        processor_name = args.processor_name_or_path or args.model_name_or_path
        model_dtype = _torch_dtype(args.dtype)
        effective_device_map: object = args.device_map
        if world_size > 1 and str(args.device_map).lower() == "auto":
            if torch.cuda.is_available():
                effective_device_map = {"": local_rank}
            else:
                effective_device_map = {"": "cpu"}
        # In distributed runs, avoid simultaneous multi-rank downloads that can
        # leave incomplete snapshots in the local HF cache.
        if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            if rank == 0:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    args.model_name_or_path,
                    torch_dtype=model_dtype,
                    attn_implementation=args.attn_implementation,
                    device_map=effective_device_map,
                )
                processor = AutoProcessor.from_pretrained(processor_name, padding_side="left")
            torch.distributed.barrier()
            if rank != 0:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    args.model_name_or_path,
                    torch_dtype=model_dtype,
                    attn_implementation=args.attn_implementation,
                    device_map=effective_device_map,
                    local_files_only=True,
                )
                processor = AutoProcessor.from_pretrained(
                    processor_name,
                    padding_side="left",
                    local_files_only=True,
                )
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model_name_or_path,
                torch_dtype=model_dtype,
                attn_implementation=args.attn_implementation,
                device_map=effective_device_map,
            )
            processor = AutoProcessor.from_pretrained(processor_name, padding_side="left")

        image_factor = 16 * SPATIAL_MERGE_SIZE
        video_max_pixels = int(args.video_max_pixels) * image_factor * image_factor
        video_min_pixels = int(args.video_min_pixels) * image_factor * image_factor
        video_total_pixels = int(args.video_total_pixels) * image_factor * image_factor

        rank_records_path = output_dir / f"{run_tag}.rank{rank:03d}.jsonl"

        video_exts = [ext.strip() for ext in args.video_exts.split(",") if ext.strip()]
        results: List[dict] = []
        missing_video_count = 0
        generation_error_count = 0

        print(
            f"[rank {rank}/{world_size}] shard_size={len(shard_samples)} total_samples={len(samples)} "
            f"| split={args.split} view={args.view} | model={args.model_name_or_path}"
        )

        for local_index, (global_index, sample) in enumerate(zip(sharded_indices, shard_samples)):
            prompt = build_prompt(
                sample=sample,
                action_catalog_text=action_catalog_text,
                num_classes=len(action_names),
            )
            video_path = resolve_video_path(
                sample=sample,
                video_root=video_root,
                video_exts=video_exts,
                video_template=args.video_template,
            )

            record = asdict(sample)
            record["sample_key"] = sample_key_from_sample(sample)
            record["sample_index"] = int(global_index)
            record["rank"] = rank
            gt_verb_id, gt_noun_id = action_to_verb_noun.get(sample.future_action_id, (-1, -1))
            record["future_verb_id"] = int(gt_verb_id)
            record["future_noun_id"] = int(gt_noun_id)
            record["future_verb_name"] = (
                verb_names[gt_verb_id] if 0 <= gt_verb_id < len(verb_names) else None
            )
            record["future_noun_name"] = (
                noun_names[gt_noun_id] if 0 <= gt_noun_id < len(noun_names) else None
            )
            record["pred_action_id"] = None
            record["pred_action_name"] = None
            record["pred_action_top5_ids"] = []
            record["pred_action_top5_names"] = []
            record["pred_verb_top5_ids"] = []
            record["pred_noun_top5_ids"] = []
            record["response_text"] = None
            record["is_correct"] = False
            record["is_top5_correct"] = False
            record["video_path"] = None
            record["error"] = None

            if video_path is None:
                missing_video_count += 1
                record["error"] = "missing_video"
                results.append(record)
                if not args.skip_missing_video:
                    raise FileNotFoundError(
                        f"Could not resolve video file for sample {global_index}: "
                        f"video_id={sample.video_id}, view={sample.view}"
                    )
                continue

            record["video_path"] = str(video_path)

            try:
                response_text = run_qwen_generation(
                    model=model,
                    processor=processor,
                    video_path=video_path,
                    video_start=sample.obs_start_sec,
                    video_end=sample.obs_end_sec,
                    prompt=prompt,
                    video_max_pixels=video_max_pixels,
                    video_min_pixels=video_min_pixels,
                    video_total_pixels=video_total_pixels,
                    max_frames=int(args.max_frames),
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                )
                pred_action_top5_ids = parse_predicted_action_ids(response_text, action_names, top_k=5)
            except Exception as exc:  # noqa: BLE001
                generation_error_count += 1
                record["error"] = f"generation_error: {type(exc).__name__}: {exc}"
                results.append(record)
                continue

            record["response_text"] = response_text
            record["pred_action_top5_ids"] = pred_action_top5_ids
            record["pred_action_top5_names"] = [action_names[idx] for idx in pred_action_top5_ids]
            record["pred_verb_top5_ids"] = _map_action_topk_to_task_topk(
                pred_action_top5_ids,
                action_to_verb_noun,
                task="verb",
                top_k=5,
            )
            record["pred_noun_top5_ids"] = _map_action_topk_to_task_topk(
                pred_action_top5_ids,
                action_to_verb_noun,
                task="noun",
                top_k=5,
            )
            if pred_action_top5_ids:
                top1_id = pred_action_top5_ids[0]
                record["pred_action_id"] = top1_id
                record["pred_action_name"] = action_names[top1_id]
                record["is_correct"] = top1_id == sample.future_action_id
                record["is_top5_correct"] = sample.future_action_id in pred_action_top5_ids
            results.append(record)

            if (local_index + 1) % 20 == 0 or (local_index + 1) == len(shard_samples):
                print(
                    f"[rank {rank}] [{local_index + 1}/{len(shard_samples)}] "
                    f"missing_video={missing_video_count} generation_errors={generation_error_count}"
                )

        with rank_records_path.open("w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        local_metrics = evaluate(
            records=results,
            num_classes=len(action_names),
            action_to_verb_noun=action_to_verb_noun,
            num_verbs=max(1, len(verb_names)),
            num_nouns=max(1, len(noun_names)),
        )
        print(
            f"[rank {rank}] local_action_top1={local_metrics['action_strict_top1_accuracy']:.4f} "
            f"local_action_top5={local_metrics['action_strict_top5_accuracy']:.4f} "
            f"local_action_mean_top5_recall={local_metrics['action_mean_top5_recall']:.4f}"
        )

        if world_size > 1:
            torch.distributed.barrier()

        if rank != 0:
            return

        merged_results: List[dict] = []
        per_rank_records: Dict[str, int] = {}
        for shard_rank in range(world_size):
            shard_path = output_dir / f"{run_tag}.rank{shard_rank:03d}.jsonl"
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing shard records file: {shard_path}")
            shard_records = [
                json.loads(line)
                for line in shard_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            per_rank_records[str(shard_rank)] = len(shard_records)
            merged_results.extend(shard_records)

        merged_results.sort(key=lambda item: int(item.get("sample_index", -1)))
        with records_path.open("w", encoding="utf-8") as f:
            for rec in merged_results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        metrics = evaluate(
            records=merged_results,
            num_classes=len(action_names),
            action_to_verb_noun=action_to_verb_noun,
            num_verbs=max(1, len(verb_names)),
            num_nouns=max(1, len(noun_names)),
        )
        missing_video_total = sum(1 for rec in merged_results if rec.get("error") == "missing_video")
        generation_error_total = sum(
            1
            for rec in merged_results
            if isinstance(rec.get("error"), str) and rec["error"].startswith("generation_error:")
        )
        summary = {
            "run_tag": run_tag,
            "args": vars(args),
            "world_size": world_size,
            "num_classes": len(action_names),
            "num_verbs": max(1, len(verb_names)),
            "num_nouns": max(1, len(noun_names)),
            "num_total_samples": len(samples),
            "num_processed_records": len(merged_results),
            "per_rank_records": per_rank_records,
            "missing_video_count": missing_video_total,
            "generation_error_count": generation_error_total,
            "metrics": metrics,
            "records_path": str(records_path),
            "rank_record_paths": {
                str(shard_rank): str(output_dir / f"{run_tag}.rank{shard_rank:03d}.jsonl")
                for shard_rank in range(world_size)
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("\nRun finished.")
        print(f"Records: {records_path}")
        print(f"Summary: {summary_path}")
        print(
            "Key metrics: "
            f"action_top1={metrics['action_strict_top1_accuracy']:.4f}, "
            f"action_top5={metrics['action_strict_top5_accuracy']:.4f}, "
            f"action_mean_top5_recall={metrics['action_mean_top5_recall']:.4f}, "
            f"verb_top1={metrics['verb_strict_top1_accuracy']:.4f}, "
            f"verb_top5={metrics['verb_strict_top5_accuracy']:.4f}, "
            f"verb_mean_top5_recall={metrics['verb_mean_top5_recall']:.4f}, "
            f"noun_top1={metrics['noun_strict_top1_accuracy']:.4f}, "
            f"noun_top5={metrics['noun_strict_top5_accuracy']:.4f}, "
            f"noun_mean_top5_recall={metrics['noun_mean_top5_recall']:.4f}"
        )
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
