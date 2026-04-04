#!/usr/bin/python
import argparse
import glob
import heapq
import json
import os
import random
from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch

from model import MS_TCN2, Refinement
from impact_split_utils import (
    annotation_filename_from_video_stem,
    default_impact_annotation_dir,
    default_impact_feature_dir,
    load_bundle_feature_files,
)


DEFAULT_COMPONENT_NAMES = [
    "anti_vibration_handle",
    "gearbox_housing",
    "drive_shaft",
    "bevel_gear",
    "adapter_plate",
    "bearing_plate",
    "screw_lever",
    "screw_adaptor_topleft",
    "screw_adaptor_lowright",
    "bearing_screw_topleft",
    "bearing_screw_lowright",
    "nut_M4_bearing_plate_TL",
    "nut_M4_bearing_plate_BR",
    "spring",
    "lever",
    "washer",
    "nut_M6",
]


def get_transitions(states):
    transitions = []
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            transitions.append((i, states[i]))
    return transitions


def compute_transition_scores(pred_states, gt_states, tolerance=30):
    """
    Compute Precision, Recall, F1 for transitions.
    tolerance: frames.
    """
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
    if not path:
        return None, None, None, None
    graph = load_json(path)
    nodes = graph.get("nodes", [])
    prereq = graph.get("prereq", {})
    meta = graph.get("meta", {})
    if not isinstance(nodes, list) or not nodes:
        raise ValueError(f"procedure_graph.json must contain a non-empty 'nodes' list: {path}")
    if not isinstance(prereq, dict):
        prereq = {}
    event_types = meta.get("event_types", [])
    if not isinstance(event_types, list):
        event_types = []
    return nodes, set(nodes), prereq, set(event_types)


def canonicalize_component_names(content, num_components, alias_map):
    components = content.get("components")
    if isinstance(components, list) and len(components) >= num_components:
        names = []
        for idx in range(num_components):
            comp = components[idx]
            raw_name = comp.get("name") if isinstance(comp, dict) else None
            if not raw_name:
                raw_name = DEFAULT_COMPONENT_NAMES[idx] if idx < len(DEFAULT_COMPONENT_NAMES) else f"component_{idx}"
            names.append(alias_map.get(raw_name, raw_name))
        return names

    names = []
    for idx in range(num_components):
        raw_name = DEFAULT_COMPONENT_NAMES[idx] if idx < len(DEFAULT_COMPONENT_NAMES) else f"component_{idx}"
        names.append(alias_map.get(raw_name, raw_name))
    return names


def build_dense_gt_states(content, T, sample_rate, num_components):
    gt_states = np.zeros((num_components, T), dtype=int)
    seq = content["state_sequence"]

    for i in range(len(seq) - 1):
        frame_start = int(seq[i]["frame"] / sample_rate)
        frame_end = int(seq[i + 1]["frame"] / sample_rate)
        frame_start = max(0, min(frame_start, T))
        frame_end = max(0, min(frame_end, T))
        if frame_end <= frame_start:
            continue
        state_vec = np.array(seq[i]["state"], dtype=int)
        gt_states[:, frame_start:frame_end] = state_vec[:, None]

    if seq:
        last_start = int(seq[-1]["frame"] / sample_rate)
        last_start = max(0, min(last_start, T))
        last_vec = np.array(seq[-1]["state"], dtype=int)
        gt_states[:, last_start:T] = last_vec[:, None]

    return gt_states


def transition_to_event(prev_s, curr_s, comp_name):
    if prev_s == curr_s:
        return None
    if prev_s == -1 and curr_s == 0:
        return None
    if prev_s == 0 and curr_s == 1:
        return f"{comp_name}__install_ok"
    if prev_s == 0 and curr_s == -1:
        return f"{comp_name}__install_bad"
    if prev_s == 1 and curr_s == 0:
        return f"{comp_name}__remove_ok"
    if prev_s == -1 and curr_s == 1:
        return f"{comp_name}__recover_ok"
    return None


def extract_event_sequence_from_dense(states_raw, component_names, allowed_nodes=None, include_event_types=None):
    events = []
    T = states_raw.shape[1]
    for t in range(1, T):
        prev_vec = states_raw[:, t - 1]
        curr_vec = states_raw[:, t]
        diffs = np.nonzero(prev_vec != curr_vec)[0]
        if diffs.size == 0:
            continue
        for k in diffs.tolist():
            event_label = transition_to_event(int(prev_vec[k]), int(curr_vec[k]), component_names[k])
            if event_label is None:
                continue
            event_type = event_label.split("__", 1)[1]
            if include_event_types and event_type not in include_event_types:
                continue
            if allowed_nodes is not None and event_label not in allowed_nodes:
                continue
            events.append({"frame": t, "label": event_label, "conf": 1})
    return events


def make_proc_info(node_labels):
    return [{"id": idx, "description": label} for idx, label in enumerate(node_labels)]


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
    metrics = {
        "perception_FPs": per_fps,
        "perception_FNs": per_fns,
        "system_FNs": sys_fns,
        "system_FPs": sys_fps,
        "system_TPs": sys_tps,
        "f1": f1,
        "avg_delay": avg_delay,
    }
    return metrics


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


def procedure_order_similarity(gt_order, pred_order):
    if not gt_order:
        return (1.0, 0.0) if not pred_order else (0.0, float(len(pred_order)))
    distance = damerau_levenshtein_distance(gt_order, pred_order)
    score = 1 - min((distance / len(gt_order)), 1)
    return score, distance


def label_count_lower_bound(seq_a, seq_b):
    counts = defaultdict(int)
    for label in seq_a:
        counts[label] += 1
    for label in seq_b:
        counts[label] -= 1
    return int(sum(abs(delta) for delta in counts.values()))


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
    """
    Find the exact valid topological ordering whose Damerau-Levenshtein distance
    to the predicted step sequence is minimal.

    The search space is the occurrence-level DAG induced by:
    1. prerequisite constraints from procedure_graph["prereq"], and
    2. a same-label chain to preserve duplicate-step occurrence order.

    We solve this exactly as a shortest-path search over states
    (placed_gt_occurrences_mask, consumed_pred_prefix_len), using the same
    operation costs as the POS distance:
    delete=1, insert=1, substitute=2, adjacent-transpose=1.

    Important optimization:
    POS clips the final distance by len(gt_labels), i.e. scores are identical
    for all distances >= len(gt_labels). We therefore run an exact bounded
    search for costs < len(gt_labels). If no path below that bound exists, we
    can return distance=len(gt_labels) immediately and the POS score remains
    exact.
    """
    n = len(gt_labels)
    m = len(pred_labels)

    if n == 0:
        return [], (0 if m == 0 else m)
    if n == 1:
        return list(gt_labels), damerau_levenshtein_distance(gt_labels, pred_labels)

    # POS clips any distance >= n to score 0, so a lower bound already >= n
    # lets us return the exact clipped score without exploring the DAG.
    if abs(n - m) >= n:
        return list(gt_labels), n
    if label_count_lower_bound(gt_labels, pred_labels) >= n:
        return list(gt_labels), n

    predecessor_masks = build_occurrence_prereq_masks(gt_labels, prereq)
    all_mask = (1 << n) - 1
    distance_cap = n

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
        next_mask, next_j = next_state
        lower_bound = next_cost + heuristic(next_mask, next_j)
        if lower_bound >= distance_cap:
            return
        best_cost = costs.get(next_state)
        if best_cost is not None and next_cost >= best_cost:
            return
        costs[next_state] = next_cost
        parents[next_state] = (cur_state, emitted_nodes)
        heapq.heappush(pq, (lower_bound, next_cost, next_mask, next_j))

    start_state = (0, 0)
    costs = {start_state: 0}
    parents = {start_state: (None, ())}
    start_lower_bound = heuristic(0, 0)
    if start_lower_bound >= distance_cap:
        return list(gt_labels), n
    pq = [(start_lower_bound, 0, 0, 0)]
    goal_state = None

    while pq:
        est_total, cur_cost, mask, j = heapq.heappop(pq)
        cur_state = (mask, j)
        if cur_cost != costs.get(cur_state):
            continue
        if est_total >= distance_cap:
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
        return list(gt_labels), n

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gtea")
    parser.add_argument("--split", default="1")
    parser.add_argument("--features_dim", default=1024, type=int)
    parser.add_argument("--num_f_maps", default=64, type=int)
    parser.add_argument("--num_layers_PG", default=11, type=int)
    parser.add_argument("--num_layers_R", default=10, type=int)
    parser.add_argument("--num_R", default=3, type=int)
    parser.add_argument("--experiment", default="ms_tcn2", choices=["ms_tcn2", "videomae"])
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--procedure_graph", default=None, help="Optional procedure_graph.json for rule-based PSR evaluation")
    parser.add_argument("--component_alias", default=None, help="Optional component_alias.json for canonical component naming")
    parser.add_argument("--split_dir", default=None, help="Optional bundle split directory, e.g. ../data/IMPACT/splits_ASR_front_only_v1")
    parser.add_argument("--impact_split", default="split1", help="Bundle suffix when --split_dir is used, e.g. split1")
    parser.add_argument("--bundle_split", default="test", choices=["train", "val", "test"], help="Which bundle split to evaluate when --split_dir is used")
    parser.add_argument("--camera", default="front", choices=["front", "left", "right", "top", "ego"], help="Camera/view to evaluate when --split_dir is used")
    parser.add_argument("--annotation_dir", default=None, help="Annotation directory. Defaults to ../data/IMPACT_ASR when --split_dir is used")
    parser.add_argument("--feature_dir", default=None, help="Explicit feature directory for the selected experiment/camera")
    args = parser.parse_args()

    seed = 1538574472
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.experiment == "ms_tcn2":
        features_dim = 1024
    elif args.experiment == "videomae":
        features_dim = 1408
    else:
        features_dim = args.features_dim

    if args.split_dir:
        gt_path = args.annotation_dir or default_impact_annotation_dir("../data")
        feature_dir = args.feature_dir or default_impact_feature_dir(args.experiment, args.camera, "../data")
        vid_list_file_tst = load_bundle_feature_files(
            args.split_dir,
            args.bundle_split,
            args.impact_split,
            feature_dir,
            camera=args.camera,
        )
        print(f"Using bundle split {args.bundle_split}.{args.impact_split} with {len(vid_list_file_tst)} {args.camera} videos")
    else:
        vid_list_gt = glob.glob("../data/annotations/*")
        vid_list_file = []
        for f in [f.split("/")[-1].split(".")[0] for f in vid_list_gt]:
            d, t = f.split("_")[:2]
            if args.experiment == "ms_tcn2":
                files = glob.glob(f"../data/features/IMPACT_i3d*/*/*/*{d}_{t}*.npy")
            else:
                files = glob.glob(f"../data/features/IMPACT*/*/*{d}_{t}*.npy")
                files = [f for f in files if "i3d" not in f]
            vid_list_file += files

        random.shuffle(vid_list_file)
        split_point = int(0.8 * len(vid_list_file))
        vid_list_file_tst = vid_list_file[split_point:]
        gt_path = "../data/annotations/"

    num_components = 17
    num_classes = num_components * 3

    if args.experiment == "ms_tcn2":
        model = MS_TCN2(args.num_layers_PG, args.num_layers_R, args.num_R, args.num_f_maps, features_dim, num_classes)
    else:
        model = Refinement(args.num_layers_R, args.num_f_maps, features_dim, num_classes)

    if args.checkpoint:
        model_path = "./models/" + args.experiment + "/split_" + args.split + "/" + args.checkpoint
        print(f"Loading {args.checkpoint}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model_dir = "./models/" + args.experiment + "/split_" + args.split
        models = glob.glob(model_dir + "/epoch-*.model")
        if not models:
            print(f"No models found in {model_dir}")
            return
        models.sort(key=lambda x: int(x.split("epoch-")[-1].split(".model")[0]))
        latest_model = models[-1]
        print(f"Loading {latest_model}")
        model.load_state_dict(torch.load(latest_model, map_location=device))

    model.to(device)
    model.eval()

    component_metrics = defaultdict(
        lambda: {"tp": defaultdict(int), "fp": defaultdict(int), "fn": defaultdict(int), "trans_f1": [], "final_acc": []}
    )

    sample_rate = 1
    if args.dataset == "50salads":
        sample_rate = 2

    graph_nodes, allowed_nodes, prereq, include_event_types = load_procedure_graph(args.procedure_graph)
    alias_map = load_alias_map(args.component_alias)
    psr_enabled = allowed_nodes is not None
    if psr_enabled:
        label_to_id = {label: idx for idx, label in enumerate(graph_nodes)}
        proc_info = make_proc_info(graph_nodes)
        psr_metrics = {"f1": [], "avg_delay_frames": [], "avg_delay_seconds": [], "pos_partial": []}
        psr_counts = {"system_TPs": 0, "system_FPs": 0, "system_FNs": 0}

    print(f"Evaluating on {len(vid_list_file_tst)} videos...")

    for vid in vid_list_file_tst:
        features = np.load(vid)
        if "i3d" in vid:
            features = features.transpose()

        features = features[:, ::sample_rate]

        input_x = torch.tensor(features, dtype=torch.float)
        input_x.unsqueeze_(0)
        input_x = input_x.to(device)

        with torch.no_grad():
            outputs = model(input_x)
            prediction = outputs[-1]
            prediction = prediction.view(1, num_components, 3, -1)
            predicted_states = torch.argmax(prediction, dim=2).squeeze(0).cpu().numpy()

        file_name = vid.split("/")[-1].split(".")[0]
        gt_filename = os.path.join(gt_path, annotation_filename_from_video_stem(file_name))

        if not os.path.exists(gt_filename):
            print(f"Warning: GT file {gt_filename} not found.")
            continue

        content = load_json(gt_filename)
        T = features.shape[1]
        gt_states_raw = build_dense_gt_states(content, T, sample_rate, num_components)
        gt_states = gt_states_raw + 1

        for c in range(num_components):
            pred_c = predicted_states[c, :]
            gt_c = gt_states[c, :]

            for cls in [0, 1, 2]:
                p_mask = pred_c == cls
                g_mask = gt_c == cls

                tp = np.sum(p_mask & g_mask)
                fp = np.sum(p_mask & ~g_mask)
                fn = np.sum(~p_mask & g_mask)

                component_metrics[c]["tp"][cls] += tp
                component_metrics[c]["fp"][cls] += fp
                component_metrics[c]["fn"][cls] += fn

            _, _, t_f1 = compute_transition_scores(pred_c, gt_c, tolerance=30)
            component_metrics[c]["trans_f1"].append(t_f1)

            if len(gt_c) > 0:
                final_correct = 1 if pred_c[-1] == gt_c[-1] else 0
                component_metrics[c]["final_acc"].append(final_correct)

        if psr_enabled:
            component_names = canonicalize_component_names(content, num_components, alias_map)
            pred_states_raw = predicted_states - 1
            gt_events = extract_event_sequence_from_dense(
                gt_states_raw,
                component_names,
                allowed_nodes=allowed_nodes,
                include_event_types=include_event_types,
            )
            pred_events = extract_event_sequence_from_dense(
                pred_states_raw,
                component_names,
                allowed_nodes=allowed_nodes,
                include_event_types=include_event_types,
            )

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

            fps = float(content.get("meta_data", {}).get("fps", 0) or content.get("fps", 0) or 0)
            delay_seconds = 0.0
            if fps > 0:
                delay_seconds = step_metrics["avg_delay"] * sample_rate / fps

            psr_metrics["f1"].append(step_metrics["f1"])
            psr_metrics["avg_delay_frames"].append(step_metrics["avg_delay"])
            psr_metrics["avg_delay_seconds"].append(delay_seconds)
            psr_metrics["pos_partial"].append(pos_partial)
            psr_counts["system_TPs"] += step_metrics["system_TPs"]
            psr_counts["system_FPs"] += step_metrics["system_FPs"]
            psr_counts["system_FNs"] += step_metrics["system_FNs"]

    print("\nEvaluation Results:")
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

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1s.append(f1)

        macro_f1 = np.mean(f1s)
        trans_f1 = np.mean(component_metrics[c]["trans_f1"]) if component_metrics[c]["trans_f1"] else 0.0
        final_acc = np.mean(component_metrics[c]["final_acc"]) if component_metrics[c]["final_acc"] else 0.0

        print(f"{c + 1:<5} | {macro_f1:.4f}     | {trans_f1:.4f}     | {final_acc:.4f}")

        avg_macro_f1.append(macro_f1)
        avg_trans_f1.append(trans_f1)
        avg_final_acc.append(final_acc)

    print("--------------------------------------------------")
    print(f"{'AVG':<5} | {np.mean(avg_macro_f1):.4f}     | {np.mean(avg_trans_f1):.4f}     | {np.mean(avg_final_acc):.4f}")
    print("--------------------------------------------------")

    if psr_enabled and psr_metrics["f1"]:
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


if __name__ == "__main__":
    main()
