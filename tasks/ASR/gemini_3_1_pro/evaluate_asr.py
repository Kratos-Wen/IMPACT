#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER_ROOT = REPO_ROOT / "third_party" / "ms_tcn2"
DEFAULT_PRED_DIR = REPO_ROOT / "outputs" / "asr" / "gemini_3_1_pro" / "predictions"
DEFAULT_SPLIT_DIR = REPO_ROOT / "dataset" / "ASR" / "splits_front_only_v1"
DEFAULT_ANNOTATION_DIR = REPO_ROOT / "dataset" / "ASR" / "annotations"

if str(HELPER_ROOT) not in sys.path:
    sys.path.insert(0, str(HELPER_ROOT))

from impact_split_utils import (
    annotation_filename_from_video_stem,
    entry_to_camera_stem,
    read_bundle_entries,
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dense_gt_states(content, num_frames, num_components):
    states = np.zeros((num_components, num_frames), dtype=int)
    seq = content["state_sequence"]

    for i in range(len(seq) - 1):
        frame_start = max(0, min(int(seq[i]["frame"]), num_frames))
        frame_end = max(0, min(int(seq[i + 1]["frame"]), num_frames))
        if frame_end <= frame_start:
            continue
        state_vec = np.array(seq[i]["state"], dtype=int)
        states[:, frame_start:frame_end] = state_vec[:, None]

    if seq:
        last_start = max(0, min(int(seq[-1]["frame"]), num_frames))
        last_vec = np.array(seq[-1]["state"], dtype=int)
        states[:, last_start:num_frames] = last_vec[:, None]

    return states


def gemini_states_to_dense(prediction, num_frames, num_components, default_fps=30.0):
    fps = float(prediction.get("fps", default_fps) or default_fps)
    timeline = []
    for item in prediction.get("states_over_time", []):
        frame_idx = int(round(float(item.get("time_s", 0.0)) * fps))
        frame_idx = max(0, min(frame_idx, num_frames))
        state = np.array(item.get("state", []), dtype=int)
        if state.shape[0] != num_components:
            raise ValueError(
                f"states_over_time has {state.shape[0]} components, expected {num_components}"
            )
        if timeline and timeline[-1][0] == frame_idx:
            timeline[-1] = (frame_idx, state)
        else:
            timeline.append((frame_idx, state))

    if not timeline:
        raise ValueError("Prediction does not contain states_over_time")

    dense = np.zeros((num_components, num_frames), dtype=int)
    first_frame, first_state = timeline[0]
    dense[:, :first_frame] = first_state[:, None]

    for idx, (frame_idx, state) in enumerate(timeline):
        next_frame = num_frames
        if idx + 1 < len(timeline):
            next_frame = timeline[idx + 1][0]
        if next_frame <= frame_idx:
            continue
        dense[:, frame_idx:next_frame] = state[:, None]

    last_frame, last_state = timeline[-1]
    if last_frame < num_frames:
        dense[:, last_frame:num_frames] = last_state[:, None]
    return dense


def get_transitions(states):
    transitions = []
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            transitions.append((i, states[i]))
    return transitions


def compute_transition_scores(pred_states, gt_states, tolerance=30):
    gt_trans = get_transitions(gt_states)
    pred_trans = get_transitions(pred_states)

    hits_gt = 0
    used_pred = set()

    for gt_t, gt_l in gt_trans:
        for i, (pred_t, pred_l) in enumerate(pred_trans):
            if i in used_pred:
                continue
            if pred_l == gt_l and abs(pred_t - gt_t) <= tolerance:
                hits_gt += 1
                used_pred.add(i)
                break

    recall = hits_gt / len(gt_trans) if gt_trans else 0.0
    precision = hits_gt / len(pred_trans) if pred_trans else 0.0
    if recall + precision == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * recall * precision / (recall + precision)
    return precision, recall, f1


def iter_video_stems(split_dir, impact_split, bundle_split, camera, pred_dir):
    if split_dir:
        return [
            entry_to_camera_stem(entry, camera)
            for entry in read_bundle_entries(split_dir, bundle_split, impact_split)
        ]
    return sorted(path.name.replace("_pred_asd_psr.json", "") for path in Path(pred_dir).glob("*_pred_asd_psr.json"))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gemini states_over_time as 3-state ASR.")
    parser.add_argument("--pred_dir", default=str(DEFAULT_PRED_DIR))
    parser.add_argument("--split_dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--impact_split", default="split1")
    parser.add_argument("--bundle_split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--camera", default="front", choices=["front", "left", "right", "top", "ego"])
    parser.add_argument("--annotation_dir", default=str(DEFAULT_ANNOTATION_DIR))
    parser.add_argument("--video", default=None, help="Optional single video stem")
    parser.add_argument("--verbose_missing", action="store_true")
    args = parser.parse_args()

    pred_root = Path(args.pred_dir)
    annotation_root = Path(args.annotation_dir)
    if not pred_root.exists():
        raise FileNotFoundError(f"Missing prediction directory: {pred_root}")
    if not annotation_root.exists():
        raise FileNotFoundError(f"Missing annotation directory: {annotation_root}")

    video_stems = iter_video_stems(args.split_dir, args.impact_split, args.bundle_split, args.camera, pred_root)
    if args.video:
        video_stems = [stem for stem in video_stems if stem == args.video]
        if not video_stems:
            raise ValueError(f"Video {args.video} not found for bundle_split={args.bundle_split}")

    num_components = 17
    component_metrics = defaultdict(
        lambda: {"tp": defaultdict(int), "fp": defaultdict(int), "fn": defaultdict(int), "trans_f1": [], "final_acc": []}
    )
    totals = {"videos": 0, "missing_pred": 0, "missing_gt": 0}

    print(f"Evaluating Gemini ASR on {len(video_stems)} videos")
    print(f"Bundle split: {args.bundle_split}")

    for idx, video_stem in enumerate(video_stems, start=1):
        print(f"[{idx}/{len(video_stems)}] {video_stem}", flush=True)
        pred_path = pred_root / f"{video_stem}_pred_asd_psr.json"
        gt_path = annotation_root / annotation_filename_from_video_stem(video_stem)

        if not pred_path.exists():
            totals["missing_pred"] += 1
            if args.verbose_missing:
                print(f"Missing prediction: {pred_path}")
            continue
        if not gt_path.exists():
            totals["missing_gt"] += 1
            if args.verbose_missing:
                print(f"Missing GT: {gt_path}")
            continue

        pred_content = load_json(pred_path)
        gt_content = load_json(gt_path)
        num_frames = int(gt_content.get("frame_count") or gt_content["meta_data"]["num_frames"])
        gt_states_raw = build_dense_gt_states(gt_content, num_frames, num_components)
        pred_states_raw = gemini_states_to_dense(pred_content, num_frames, num_components)

        gt_states = gt_states_raw + 1
        pred_states = pred_states_raw + 1

        for c in range(num_components):
            pred_c = pred_states[c, :]
            gt_c = gt_states[c, :]

            for cls in [0, 1, 2]:
                p_mask = pred_c == cls
                g_mask = gt_c == cls
                component_metrics[c]["tp"][cls] += int(np.sum(p_mask & g_mask))
                component_metrics[c]["fp"][cls] += int(np.sum(p_mask & ~g_mask))
                component_metrics[c]["fn"][cls] += int(np.sum(~p_mask & g_mask))

            _, _, t_f1 = compute_transition_scores(pred_c, gt_c, tolerance=30)
            component_metrics[c]["trans_f1"].append(t_f1)
            component_metrics[c]["final_acc"].append(1 if pred_c[-1] == gt_c[-1] else 0)

        totals["videos"] += 1

    if totals["videos"] == 0:
        print("No videos were evaluated.")
        return

    print("\nASR Results:")
    print("--------------------------------------------------")
    print(f"{'Comp':<5} | {'Macro F1':<10} | {'Trans F1':<10} | {'Final Acc':<10}")
    print("--------------------------------------------------")

    avg_macro_f1 = []
    avg_trans_f1 = []
    avg_final_acc = []

    for c in range(num_components):
        f1s = []
        for cls in [0, 1, 2]:
            tp = component_metrics[c]["tp"][cls]
            fp = component_metrics[c]["fp"][cls]
            fn = component_metrics[c]["fn"][cls]
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            f1s.append(f1)

        macro_f1 = float(np.mean(f1s))
        trans_f1 = float(np.mean(component_metrics[c]["trans_f1"])) if component_metrics[c]["trans_f1"] else 0.0
        final_acc = float(np.mean(component_metrics[c]["final_acc"])) if component_metrics[c]["final_acc"] else 0.0

        avg_macro_f1.append(macro_f1)
        avg_trans_f1.append(trans_f1)
        avg_final_acc.append(final_acc)
        print(f"{c + 1:<5} | {macro_f1:<10.4f} | {trans_f1:<10.4f} | {final_acc:<10.4f}")

    print("--------------------------------------------------")
    print(
        f"{'AVG':<5} | {np.mean(avg_macro_f1):<10.4f} | "
        f"{np.mean(avg_trans_f1):<10.4f} | {np.mean(avg_final_acc):<10.4f}"
    )
    print("--------------------------------------------------")
    print(
        f"Evaluated videos: {totals['videos']} | Missing pred: {totals['missing_pred']} | Missing GT: {totals['missing_gt']}"
    )


if __name__ == "__main__":
    main()
