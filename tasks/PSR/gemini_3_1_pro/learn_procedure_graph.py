#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learn a robust procedure graph (DAG prerequisites) from ASR annotation JSON files.

We use FRACTION thresholds (percent of videos) to be paper-justifiable and scalable:
- min_cooccur_frac: A and B must co-occur in at least this fraction of videos
- min_support_frac: A must occur before B in at least this fraction of videos
- min_confidence: P(A before B | A,B) must exceed this threshold

Default for N≈80:
- min_cooccur_frac=0.25 => 20 videos
- min_support_frac=0.20 => 16 videos
- min_confidence=0.90
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import numpy as np


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def expand_state_sequence(state_sequence: List[dict], T: int, K: int) -> np.ndarray:
    dense = np.zeros((T, K), dtype=np.int8)
    frames = [x["frame"] for x in state_sequence]
    states = [np.array(x["state"], dtype=np.int8) for x in state_sequence]
    for i, f in enumerate(frames):
        end = frames[i+1] if i+1 < len(frames) else T
        dense[f:end] = states[i]
    return dense


def transition_to_event(prev_s: int, curr_s: int, comp_name: str) -> str | None:
    if prev_s == curr_s:
        return None
    # ignore undo_bad (-1 -> 0) by default
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


def extract_event_sequence(asr_json: dict,
                           alias_map: Dict[str, str] | None,
                           include_event_types: Set[str]) -> List[Tuple[int, str]]:
    T = int(asr_json.get("frame_count") or asr_json["meta_data"]["num_frames"])
    comps = asr_json["components"]
    K = len(comps)
    dense = expand_state_sequence(asr_json["state_sequence"], T=T, K=K)

    events: List[Tuple[int, str]] = []
    for t in range(1, T):
        prev_vec = dense[t-1]
        curr_vec = dense[t]
        diffs = np.nonzero(prev_vec != curr_vec)[0]
        if diffs.size == 0:
            continue
        for k in diffs.tolist():
            raw = comps[k]["name"]
            name = alias_map.get(raw, raw) if alias_map else raw
            ev = transition_to_event(int(prev_vec[k]), int(curr_vec[k]), name)
            if ev is None:
                continue
            etype = ev.split("__", 1)[1]
            if etype not in include_event_types:
                continue
            events.append((t, ev))
    events.sort(key=lambda x: x[0])
    return events


def learn_pairwise_constraints(seqs: List[List[Tuple[int, str]]],
                               min_cooccur: int,
                               min_support: int,
                               min_confidence: float):
    before = defaultdict(int)
    cooc = defaultdict(int)
    nodes: Set[str] = set()

    for seq in seqs:
        if not seq:
            continue
        first_time: Dict[str, int] = {}
        for t, ev in seq:
            nodes.add(ev)
            if ev not in first_time:
                first_time[ev] = t
        evs = list(first_time.keys())
        for i in range(len(evs)):
            for j in range(len(evs)):
                if i == j:
                    continue
                A, B = evs[i], evs[j]
                cooc[(A, B)] += 1
                if first_time[A] < first_time[B]:
                    before[(A, B)] += 1

    kept = {}
    for (A, B), co in cooc.items():
        if co < min_cooccur:
            continue
        sup = before.get((A, B), 0)
        if sup < min_support:
            continue
        conf = sup / float(co)
        if conf >= min_confidence:
            kept[(A, B)] = {"support": sup, "cooccur": co, "confidence": conf}
    return kept, nodes


def resolve_conflicts_and_make_dag(kept: Dict[Tuple[str, str], dict], nodes: Set[str]):
    # resolve A->B and B->A conflicts
    cand = {}
    visited = set()
    for (A, B), info in kept.items():
        if (A, B) in visited:
            continue
        rev = kept.get((B, A))
        if rev is not None:
            if abs(info["confidence"] - rev["confidence"]) < 1e-12:
                visited.add((A, B)); visited.add((B, A))
                continue
            if info["confidence"] > rev["confidence"]:
                cand[(A, B)] = info["confidence"]
            else:
                cand[(B, A)] = rev["confidence"]
            visited.add((A, B)); visited.add((B, A))
        else:
            cand[(A, B)] = info["confidence"]
            visited.add((A, B))

    edges = list(cand.keys())

    def topo_sort(edges_list):
        indeg = {n: 0 for n in nodes}
        adj = {n: [] for n in nodes}
        for a, b in edges_list:
            adj.setdefault(a, []).append(b)
            indeg[b] = indeg.get(b, 0) + 1
            indeg.setdefault(a, indeg.get(a, 0))
        q = [n for n, d in indeg.items() if d == 0]
        out = []
        while q:
            n = q.pop()
            out.append(n)
            for m in adj.get(n, []):
                indeg[m] -= 1
                if indeg[m] == 0:
                    q.append(m)
        return out if len(out) == len(indeg) else None

    # break cycles by removing weakest edges
    while True:
        if topo_sort(edges) is not None:
            break
        weakest = min(edges, key=lambda e: cand[e])
        edges.remove(weakest)

    return edges, cand


def edges_to_prereq(edges: List[Tuple[str, str]], nodes: Set[str]):
    prereq = {n: [] for n in nodes}
    for a, b in edges:
        prereq[b].append(a)
    for k in prereq:
        prereq[k] = sorted(prereq[k])
    return prereq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr_dir", type=str, required=True, help="Folder containing ASR JSON files (*.json).")
    ap.add_argument("--out", type=str, required=True, help="Output procedure_graph.json.")
    ap.add_argument("--alias_map", type=str, default=None, help="Optional component_alias.json.")

    ap.add_argument("--min_cooccur_frac", type=float, default=0.25,
                    help="Min fraction of videos where A and B co-occur (default 0.25).")
    ap.add_argument("--min_support_frac", type=float, default=0.20,
                    help="Min fraction of videos supporting A before B (default 0.20).")
    ap.add_argument("--min_confidence", type=float, default=0.90,
                    help="Min confidence P(A before B | A,B) (default 0.90).")

    ap.add_argument("--event_types", type=str, default="install_ok,remove_ok,recover_ok",
                    help="Comma-separated event types to include (default: install_ok,remove_ok,recover_ok).")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.asr_dir, "*.json")))
    if not paths:
        raise FileNotFoundError(f"No ASR JSON files found under: {args.asr_dir}")

    n_videos = len(paths)
    min_cooccur = max(3, int(np.ceil(args.min_cooccur_frac * n_videos)))
    min_support = max(3, int(np.ceil(args.min_support_frac * n_videos)))
    include_types = set([x.strip() for x in args.event_types.split(",") if x.strip()])
    alias_map = load_json(args.alias_map) if args.alias_map else None

    seqs = []
    for p in paths:
        asr = load_json(p)
        seqs.append(extract_event_sequence(asr, alias_map, include_types))

    kept, nodes = learn_pairwise_constraints(seqs, min_cooccur, min_support, args.min_confidence)
    edges, edge_conf = resolve_conflicts_and_make_dag(kept, nodes)
    prereq = edges_to_prereq(edges, nodes)

    graph = {
        "meta": {
            "num_videos": n_videos,
            "min_cooccur_frac": args.min_cooccur_frac,
            "min_support_frac": args.min_support_frac,
            "min_cooccur": min_cooccur,
            "min_support": min_support,
            "min_confidence": args.min_confidence,
            "event_types": sorted(list(include_types)),
            "alias_map_used": bool(alias_map),
            "note": "Edges are robust prerequisites mined from data. Missing edge => flexible ordering."
        },
        "nodes": sorted(list(nodes)),
        "edges": [[a, b] for a, b in edges],
        "prereq": prereq
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote: {args.out}")
    print(f"     videos={n_videos}, nodes={len(graph['nodes'])}, edges={len(graph['edges'])}")
    print(f"     thresholds: min_cooccur={min_cooccur} ({args.min_cooccur_frac:.0%}), "
          f"min_support={min_support} ({args.min_support_frac:.0%}), "
          f"min_confidence={args.min_confidence:.2f}")


if __name__ == "__main__":
    main()
