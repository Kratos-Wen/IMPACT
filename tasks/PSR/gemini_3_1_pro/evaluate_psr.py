#!/usr/bin/env python3
import argparse
import csv
import heapq
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRED_DIR = REPO_ROOT / "outputs" / "psr" / "gemini_3_1_pro" / "predictions"
DEFAULT_GT_DIR = REPO_ROOT / "dataset" / "PSR" / "labels_front_only_v1"
DEFAULT_PROCEDURE_GRAPH = Path(__file__).resolve().parent / "configs" / "procedure_graph.json"
DEFAULT_COMPONENT_ALIAS = Path(__file__).resolve().parent / "configs" / "component_alias.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_alias_map(path):
    if not path:
        return {}
    alias_map = load_json(path)
    if not isinstance(alias_map, dict):
        raise ValueError(f"Alias map must be a JSON object: {path}")
    return alias_map


def load_procedure_graph(path):
    graph = load_json(path)
    nodes = graph.get("nodes", [])
    prereq = graph.get("prereq", {})
    if not isinstance(nodes, list) or not nodes:
        raise ValueError(f"procedure_graph.json must contain a non-empty 'nodes' list: {path}")
    if not isinstance(prereq, dict):
        prereq = {}
    return nodes, prereq


def normalize_component_name(raw_name, alias_map):
    return alias_map.get(raw_name, raw_name)


def normalize_graph_step(raw_label, alias_map):
    if "__" not in raw_label:
        return None
    raw_component, event_type = raw_label.split("__", 1)
    component = normalize_component_name(raw_component, alias_map)
    if event_type == "install_ok":
        return f"{component}__install"
    if event_type in {"recover_ok", "remove_ok"}:
        return f"{component}__remove"
    return None


def build_proc_info(component_names, alias_map):
    labels = []
    for raw_name in component_names:
        component = normalize_component_name(raw_name, alias_map)
        labels.append(f"{component}__install")
        labels.append(f"{component}__remove")
    return [{"id": idx, "description": label} for idx, label in enumerate(labels)]


def collapse_prereq_graph(component_names, alias_map, original_nodes, original_prereq):
    valid_nodes = {entry["description"] for entry in build_proc_info(component_names, alias_map)}
    collapsed = {node: set() for node in valid_nodes}
    for dst in original_nodes:
        dst_norm = normalize_graph_step(dst, alias_map)
        if dst_norm is None or dst_norm not in valid_nodes:
            continue
        for src in original_prereq.get(dst, []):
            src_norm = normalize_graph_step(src, alias_map)
            if src_norm is None or src_norm not in valid_nodes or src_norm == dst_norm:
                continue
            collapsed[dst_norm].add(src_norm)
    return {key: sorted(value) for key, value in collapsed.items()}


def get_f1_score(fn_count, fp_count, tp_count):
    positives = tp_count + fn_count
    predicted_positives = tp_count + fp_count
    precision = tp_count / predicted_positives if predicted_positives else 1e-6
    recall = tp_count / positives if positives else 1e-6
    return 2 * (precision * recall) / (precision + recall + 1e-6)


def get_fn_fp_single_entry(gt_frame_n, pred_frame_n, conf_pred):
    sys_fp = False
    per_fn = False
    per_fp = False
    delay = None
    if conf_pred == 0:
        per_fn = True

    delta_frames = pred_frame_n - gt_frame_n
    if delta_frames < 0:
        if conf_pred == 0:
            per_fp = True
        else:
            sys_fp = True
    else:
        delay = delta_frames
    return sys_fp, per_fn, per_fp, delay


def match_indices(idxes_a, all_times_a, idxes_b, all_times_b):
    assert len(idxes_a) >= len(idxes_b)
    times_a = np.ones(len(all_times_a)) * 1e9
    for idx in idxes_a:
        times_a[idx] = all_times_a[idx]
    times_b = np.array([all_times_b[i] for i in idxes_b])
    matching_idxes = []
    for time_b in times_b:
        time_diff = np.abs(times_a - time_b)
        time_diff_pen = np.where(time_diff > 0, time_diff, np.inf)
        min_idx = np.argmin(time_diff_pen)
        matching_idxes.append(min_idx)
        times_a[min_idx] = 1e9
    return matching_idxes


def determine_step_metrics(gt, pred, proc_info):
    gt_obs_times = np.array([entry["frame"] for entry in gt], dtype=int)
    gt_order = np.array([int(entry["id"]) for entry in gt], dtype=int)
    pred_obs_times = np.array([entry["frame"] for entry in pred], dtype=int)
    pred_order = np.array([int(entry["id"]) for entry in pred], dtype=int)
    pred_confs = np.array([int(entry.get("conf", 1)) for entry in pred], dtype=int)

    sys_fns = 0
    sys_fps = 0
    per_fns = 0
    per_fps = 0
    delays = np.empty(len(gt_obs_times), dtype=float)
    delays[:] = np.nan

    for step_info in proc_info:
        idxes_gt = list(np.where(gt_order == step_info["id"])[0])
        idxes_pred = list(np.where(pred_order == step_info["id"])[0])
        calculate = True

        if len(idxes_gt) == len(idxes_pred) and len(idxes_pred) > 1:
            idxes_pred = match_indices(idxes_pred, pred_obs_times, idxes_gt, gt_obs_times)
        elif len(idxes_gt) == 0 and len(idxes_pred) > 0:
            sys_fps += len(idxes_pred)
            per_fps += len(idxes_pred)
            calculate = False
        elif len(idxes_gt) > 0 and len(idxes_pred) == 0:
            sys_fns += len(idxes_gt)
            per_fns += len(idxes_gt)
            calculate = False
        else:
            if len(idxes_gt) > len(idxes_pred):
                sys_fns += len(idxes_gt) - len(idxes_pred)
                per_fns += len(idxes_gt) - len(idxes_pred)
                idxes_gt = match_indices(idxes_gt, gt_obs_times, idxes_pred, pred_obs_times)
            elif len(idxes_pred) > len(idxes_gt):
                sys_fps += len(idxes_pred) - len(idxes_gt)
                per_fps += len(idxes_pred) - len(idxes_gt)
                idxes_pred = match_indices(idxes_pred, pred_obs_times, idxes_gt, gt_obs_times)

        if not calculate:
            continue

        for idx_gt, idx_pred in zip(idxes_gt, idxes_pred):
            gt_frame_n = gt_obs_times[idx_gt]
            pred_frame_n = pred_obs_times[idx_pred]
            conf_pred = pred_confs[idx_pred]
            sys_fp, per_fn, per_fp, delay = get_fn_fp_single_entry(gt_frame_n, pred_frame_n, conf_pred)
            if sys_fp:
                sys_fps += 1
            if per_fn:
                per_fns += 1
            if per_fp:
                per_fps += 1
            if delay is not None:
                delays[idx_gt] = delay

    sys_tps = len(pred_order) - sys_fps
    f1 = get_f1_score(sys_fns, sys_fps, sys_tps)
    valid_delays = delays[~np.isnan(delays)]
    avg_delay = float(np.mean(valid_delays)) if valid_delays.size else 0.0
    return {
        "perception_FPs": per_fps,
        "perception_FNs": per_fns,
        "system_FNs": sys_fns,
        "system_FPs": sys_fps,
        "system_TPs": sys_tps,
        "f1": f1,
        "avg_delay": avg_delay,
    }


def damerau_levenshtein_distance(seq_a, seq_b):
    len_a = len(seq_a)
    len_b = len(seq_b)
    if len_a == 0:
        return float(len_b)
    if len_b == 0:
        return float(len_a)

    dist = [[0.0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(len_a + 1):
        dist[i][0] = float(i)
    for j in range(len_b + 1):
        dist[0][j] = float(j)

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            substitution_cost = 0.0 if seq_a[i - 1] == seq_b[j - 1] else 2.0
            dist[i][j] = min(
                dist[i - 1][j] + 1.0,
                dist[i][j - 1] + 1.0,
                dist[i - 1][j - 1] + substitution_cost,
            )
            if i > 1 and j > 1 and seq_a[i - 1] == seq_b[j - 2] and seq_a[i - 2] == seq_b[j - 1]:
                dist[i][j] = min(dist[i][j], dist[i - 2][j - 2] + 1.0)
    return dist[len_a][len_b]


def build_occurrence_prereq_masks(gt_labels, prereq):
    n = len(gt_labels)
    predecessor_masks = [0] * n
    occurrences_by_label = defaultdict(list)

    for idx, label in enumerate(gt_labels):
        occurrences_by_label[label].append(idx)
        if len(occurrences_by_label[label]) > 1:
            predecessor_masks[idx] |= 1 << occurrences_by_label[label][-2]

    for dst_idx, label_b in enumerate(gt_labels):
        for label_a in prereq.get(label_b, []):
            for src_idx in occurrences_by_label.get(label_a, []):
                if src_idx != dst_idx:
                    predecessor_masks[dst_idx] |= 1 << src_idx

    return predecessor_masks


def closest_valid_topological_order_exact(gt_labels, pred_labels, prereq):
    n = len(gt_labels)
    m = len(pred_labels)

    if n == 0:
        return [], (0 if m == 0 else m)
    if n == 1:
        return list(gt_labels), damerau_levenshtein_distance(gt_labels, pred_labels)

    predecessor_masks = build_occurrence_prereq_masks(gt_labels, prereq)
    all_mask = (1 << n) - 1

    label_to_idx = {}
    for label in gt_labels:
        if label not in label_to_idx:
            label_to_idx[label] = len(label_to_idx)
    for label in pred_labels:
        if label not in label_to_idx:
            label_to_idx[label] = len(label_to_idx)

    num_label_types = len(label_to_idx)
    gt_label_ids = [label_to_idx[label] for label in gt_labels]
    pred_label_ids = [label_to_idx[label] for label in pred_labels]

    total_gt_counts = [0] * num_label_types
    for label_id in gt_label_ids:
        total_gt_counts[label_id] += 1

    pred_suffix_counts = np.zeros((m + 1, num_label_types), dtype=int)
    for j in range(m - 1, -1, -1):
        pred_suffix_counts[j] = pred_suffix_counts[j + 1]
        pred_suffix_counts[j, pred_label_ids[j]] += 1

    @lru_cache(maxsize=None)
    def available_nodes(mask):
        nodes = []
        for idx in range(n):
            bit = 1 << idx
            if mask & bit:
                continue
            if predecessor_masks[idx] & ~mask:
                continue
            nodes.append(idx)
        return tuple(nodes)

    @lru_cache(maxsize=None)
    def remaining_label_counts(mask):
        counts = list(total_gt_counts)
        tmp_mask = mask
        while tmp_mask:
            lsb = tmp_mask & -tmp_mask
            idx = lsb.bit_length() - 1
            counts[gt_label_ids[idx]] -= 1
            tmp_mask ^= lsb
        return tuple(counts)

    @lru_cache(maxsize=None)
    def heuristic(mask, j):
        rem_counts = remaining_label_counts(mask)
        pred_counts = pred_suffix_counts[j]
        return int(sum(abs(rem_counts[idx] - pred_counts[idx]) for idx in range(num_label_types)))

    def update_state(costs, parents, pq, cur_state, next_state, next_cost, emitted_nodes):
        best_cost = costs.get(next_state)
        if best_cost is not None and next_cost >= best_cost:
            return
        costs[next_state] = next_cost
        parents[next_state] = (cur_state, emitted_nodes)
        next_mask, next_j = next_state
        heapq.heappush(pq, (next_cost + heuristic(next_mask, next_j), next_cost, next_mask, next_j))

    start_state = (0, 0)
    costs = {start_state: 0}
    parents = {start_state: (None, ())}
    pq = [(heuristic(0, 0), 0, 0, 0)]
    goal_state = None

    while pq:
        _, cur_cost, mask, j = heapq.heappop(pq)
        cur_state = (mask, j)
        if cur_cost != costs.get(cur_state):
            continue
        if mask == all_mask and j == m:
            goal_state = cur_state
            break

        if j < m:
            update_state(costs, parents, pq, cur_state, (mask, j + 1), cur_cost + 1, ())

        avail = available_nodes(mask)
        for node_idx in avail:
            next_mask = mask | (1 << node_idx)
            update_state(costs, parents, pq, cur_state, (next_mask, j), cur_cost + 1, (node_idx,))

            if j < m:
                sub_cost = 0 if gt_labels[node_idx] == pred_labels[j] else 2
                update_state(costs, parents, pq, cur_state, (next_mask, j + 1), cur_cost + sub_cost, (node_idx,))

        if j + 1 < m:
            for first_idx in avail:
                if gt_labels[first_idx] != pred_labels[j + 1]:
                    continue
                mask_after_first = mask | (1 << first_idx)
                for second_idx in available_nodes(mask_after_first):
                    if gt_labels[second_idx] != pred_labels[j]:
                        continue
                    next_mask = mask_after_first | (1 << second_idx)
                    update_state(costs, parents, pq, cur_state, (next_mask, j + 2), cur_cost + 1, (first_idx, second_idx))

    if goal_state is None:
        return list(gt_labels), damerau_levenshtein_distance(gt_labels, pred_labels)

    emitted_reversed = []
    cur_state = goal_state
    while cur_state != start_state:
        prev_state, emitted_nodes = parents[cur_state]
        if emitted_nodes:
            emitted_reversed.extend(reversed(emitted_nodes))
        cur_state = prev_state
    emitted_reversed.reverse()
    best_order = [gt_labels[idx] for idx in emitted_reversed]
    return best_order, costs[goal_state]


def make_metric_entries(events, label_to_id):
    entries = []
    for event in events:
        event_id = label_to_id.get(event["label"])
        if event_id is None:
            continue
        entries.append(
            {
                "frame": event["frame"],
                "id": event_id,
                "description": event["label"],
                "conf": event.get("conf", 1),
            }
        )
    return entries


def list_split_videos(gt_root, bundle_split):
    if bundle_split == "all":
        splits = ["train", "val", "test"]
    else:
        splits = [bundle_split]

    pairs = []
    for split in splits:
        split_dir = gt_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing GT split directory: {split_dir}")
        for video_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            pairs.append((split, video_dir.name))
    return pairs


def load_gt_events(gt_csv_path, proc_info_by_id, component_names, alias_map):
    events = []
    skipped = 0
    with open(gt_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            frame = int(Path(row[0]).stem)
            step_id = int(row[1])
            step_info = proc_info_by_id[step_id]
            raw_component = component_names[step_info["state_idx"]]
            component = normalize_component_name(raw_component, alias_map)
            description = step_info["description"]
            if description.startswith("Install "):
                label = f"{component}__install"
            elif description.startswith("Remove "):
                label = f"{component}__remove"
            else:
                skipped += 1
                continue
            events.append({"frame": frame, "label": label, "conf": 1})
    return events, skipped


def load_pred_events(pred_path, alias_map, fps, time_source, min_conf):
    data = load_json(pred_path)
    seen = set()
    events = []
    skipped = 0
    for item in data.get("psr_over_time", []):
        detect_time_s = float(item.get("time_s", 0.0))
        for completed in item.get("completed", []):
            conf = float(completed.get("c", 1.0))
            if conf < min_conf:
                continue
            label = normalize_graph_step(completed.get("step", ""), alias_map)
            if label is None:
                skipped += 1
                continue
            if label in seen:
                continue
            seen.add(label)
            if time_source == "tau" and completed.get("tau_s") is not None:
                event_time_s = float(completed["tau_s"])
            else:
                event_time_s = detect_time_s
            frame = int(round(event_time_s * fps))
            events.append({"frame": frame, "label": label, "conf": 1})
    return data, events, skipped


def dedupe_first_occurrence(events):
    seen = set()
    deduped = []
    dropped = 0
    for event in events:
        label = event["label"]
        if label in seen:
            dropped += 1
            continue
        seen.add(label)
        deduped.append(event)
    return deduped, dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", default=str(DEFAULT_PRED_DIR))
    parser.add_argument("--gt_dir", default=str(DEFAULT_GT_DIR))
    parser.add_argument("--bundle_split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--video", default=None, help="Optional single video stem, e.g. 20250407_1420_color_front_clipped")
    parser.add_argument("--gt_filename", default="PSR_labels.csv", help="Use PSR_labels.csv for install/remove evaluation")
    parser.add_argument("--procedure_graph", default=str(DEFAULT_PROCEDURE_GRAPH))
    parser.add_argument("--component_alias", default=str(DEFAULT_COMPONENT_ALIAS))
    parser.add_argument("--pred_time_source", default="detect", choices=["detect", "tau"])
    parser.add_argument("--min_conf", default=0.0, type=float)
    parser.add_argument(
        "--keep_repeated_steps",
        action="store_true",
        help="Keep repeated normalized steps instead of evaluating first-completion semantics",
    )
    parser.add_argument("--verbose_missing", action="store_true")
    args = parser.parse_args()

    pred_root = Path(args.pred_dir)
    gt_root = Path(args.gt_dir)
    if not pred_root.exists():
        raise FileNotFoundError(f"Missing prediction directory: {pred_root}")
    if not gt_root.exists():
        raise FileNotFoundError(f"Missing GT directory: {gt_root}")

    alias_map = load_alias_map(args.component_alias)
    original_nodes, original_prereq = load_procedure_graph(args.procedure_graph)
    component_names = load_json(gt_root / "component_names.json")
    proc_info_raw = load_json(gt_root / "procedure_info_IMPACT.json")
    proc_info_by_id = {int(entry["id"]): entry for entry in proc_info_raw}

    proc_info = build_proc_info(component_names, alias_map)
    label_to_id = {entry["description"]: entry["id"] for entry in proc_info}
    prereq = collapse_prereq_graph(component_names, alias_map, original_nodes, original_prereq)

    pairs = list_split_videos(gt_root, args.bundle_split)
    if args.video:
        pairs = [pair for pair in pairs if pair[1] == args.video]
        if not pairs:
            raise ValueError(f"Video {args.video} not found under split={args.bundle_split}")
    psr_metrics = {"f1": [], "avg_delay_frames": [], "avg_delay_seconds": [], "pos_partial": []}
    psr_counts = {"system_TPs": 0, "system_FPs": 0, "system_FNs": 0}
    totals = {
        "videos": 0,
        "missing_pred": 0,
        "missing_gt": 0,
        "skipped_gt_rows": 0,
        "skipped_pred_steps": 0,
        "dropped_gt_duplicates": 0,
        "dropped_pred_duplicates": 0,
    }

    print(f"Evaluating Gemini PSR on {len(pairs)} videos from split={args.bundle_split}")
    print("Mapping: install_ok -> install, recover_ok/remove_ok -> remove")
    print(f"Prediction time source: {args.pred_time_source}")
    print(f"GT file: {args.gt_filename}")
    print(f"First-completion semantics: {not args.keep_repeated_steps}")

    for idx, (split_name, video_name) in enumerate(pairs, start=1):
        print(f"[{idx}/{len(pairs)}] {video_name}", flush=True)
        gt_csv_path = gt_root / split_name / video_name / args.gt_filename
        pred_path = pred_root / f"{video_name}_pred_asd_psr.json"

        if not gt_csv_path.exists():
            totals["missing_gt"] += 1
            if args.verbose_missing:
                print(f"Missing GT: {gt_csv_path}")
            continue
        if not pred_path.exists():
            totals["missing_pred"] += 1
            if args.verbose_missing:
                print(f"Missing prediction: {pred_path}")
            continue

        pred_data, pred_events, skipped_pred = load_pred_events(
            pred_path,
            alias_map,
            fps=float(load_json(pred_path).get("fps", 30.0) or 30.0),
            time_source=args.pred_time_source,
            min_conf=args.min_conf,
        )
        gt_events, skipped_gt = load_gt_events(gt_csv_path, proc_info_by_id, component_names, alias_map)

        dropped_gt_duplicates = 0
        dropped_pred_duplicates = 0
        if not args.keep_repeated_steps:
            gt_events, dropped_gt_duplicates = dedupe_first_occurrence(gt_events)
            pred_events, dropped_pred_duplicates = dedupe_first_occurrence(pred_events)

        fps = float(pred_data.get("fps", 30.0) or 30.0)
        gt_metric_entries = make_metric_entries(gt_events, label_to_id)
        pred_metric_entries = make_metric_entries(pred_events, label_to_id)
        step_metrics = determine_step_metrics(gt_metric_entries, pred_metric_entries, proc_info)

        gt_labels = [event["label"] for event in gt_events]
        pred_labels = [event["label"] for event in pred_events]
        best_valid_gt, best_pos_distance = closest_valid_topological_order_exact(gt_labels, pred_labels, prereq)
        if best_valid_gt:
            pos_partial = 1 - min((best_pos_distance / len(best_valid_gt)), 1)
        else:
            pos_partial = 1.0 if not pred_labels else 0.0

        psr_metrics["f1"].append(step_metrics["f1"])
        psr_metrics["avg_delay_frames"].append(step_metrics["avg_delay"])
        psr_metrics["avg_delay_seconds"].append(step_metrics["avg_delay"] / fps if fps > 0 else 0.0)
        psr_metrics["pos_partial"].append(pos_partial)
        psr_counts["system_TPs"] += step_metrics["system_TPs"]
        psr_counts["system_FPs"] += step_metrics["system_FPs"]
        psr_counts["system_FNs"] += step_metrics["system_FNs"]
        totals["videos"] += 1
        totals["skipped_gt_rows"] += skipped_gt
        totals["skipped_pred_steps"] += skipped_pred
        totals["dropped_gt_duplicates"] += dropped_gt_duplicates
        totals["dropped_pred_duplicates"] += dropped_pred_duplicates

    if totals["videos"] == 0:
        print("No videos were evaluated.")
        return

    tp = psr_counts["system_TPs"]
    fp = psr_counts["system_FPs"]
    fn_count = psr_counts["system_FNs"]
    global_f1 = get_f1_score(fn_count, fp, tp)

    print("\nPSR Results:")
    print("--------------------------------------------------")
    print(f"Step Completion F1 (mean):   {np.mean(psr_metrics['f1']):.4f}")
    print(f"Step Completion F1 (global): {global_f1:.4f}")
    print(f"Average Detection Delay:     {np.mean(psr_metrics['avg_delay_frames']):.2f} frames")
    print(f"Average Detection Delay:     {np.mean(psr_metrics['avg_delay_seconds']):.3f} s")
    print(f"Partial-Order POS:           {np.mean(psr_metrics['pos_partial']):.4f}")
    print(f"Counts: TP={tp} FP={fp} FN={fn_count}")
    print("--------------------------------------------------")
    print(
        "Evaluated videos: "
        f"{totals['videos']} | Missing pred: {totals['missing_pred']} | Missing GT: {totals['missing_gt']}"
    )
    print(
        "Skipped rows/steps after normalization: "
        f"GT={totals['skipped_gt_rows']} | Pred={totals['skipped_pred_steps']}"
    )
    print(
        "Dropped repeated normalized steps: "
        f"GT={totals['dropped_gt_duplicates']} | Pred={totals['dropped_pred_duplicates']}"
    )


if __name__ == "__main__":
    main()
