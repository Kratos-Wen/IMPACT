#!/usr/bin/env python3
"""Run the released Gemini 3.1 Pro prompting pipeline for IMPACT.

The prompt jointly predicts assembly states and completed procedure steps.
"""


import os
import io
import glob
import json
import math
import argparse
from typing import List, Dict, Any, Tuple, Set

from PIL import Image
from moviepy import VideoFileClip
from google import genai
from google.genai import types


# -------------------------
# Sampling (STRICT policy)
# -------------------------
def round_t(t: float, nd: int = 3) -> float:
    return float(f"{t:.{nd}f}")


def build_sample_times(
    key_times_s: List[float],
    duration_s: float,
    key_half_window_s: float = 0.5,
    key_fps: float = 10.0,
    base_fps: float = 1.0,
) -> Tuple[List[float], Set[float]]:
    """
    STRICT:
    - key window [t-0.5, t+0.5]: 10 fps
    - elsewhere: 1 fps
    Return:
      all_times_sorted
      key_dense_set
    """
    key_step = 1.0 / key_fps
    base_step = 1.0 / base_fps

    key_dense = set()
    for t in key_times_s:
        s = max(0.0, t - key_half_window_s)
        e = min(duration_s, t + key_half_window_s)
        cur = s
        while cur <= e + 1e-9:
            key_dense.add(round_t(cur, 3))
            cur += key_step

    base_set = set()
    cur = 0.0
    while cur <= duration_s + 1e-9:
        base_set.add(round_t(cur, 3))
        cur += base_step

    all_times = sorted(base_set.union(key_dense))
    return all_times, key_dense


# -------------------------
# Frames -> JPEG
# -------------------------
def frame_to_jpeg_bytes(frame_rgb, quality: int = 85) -> bytes:
    img = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


# -------------------------
# JSON extraction
# -------------------------
def extract_first_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model output.")
    depth = 0
    end = None
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError("Unclosed JSON object in model output.")
    return json.loads(text[start:end])


# -------------------------
# Graph helpers (procedure_graph.json)
# -------------------------
def load_procedure_graph(path: str) -> Tuple[Set[str], List[List[str]], Dict[str, List[str]]]:
    """
    returns:
      allowed_nodes_set
      edges: list of [a,b] meaning a -> b precedence constraint
      prereq: dict b -> [a1,a2,...] (prerequisites)
    """
    with open(path, "r", encoding="utf-8") as f:
        g = json.load(f)
    nodes = g.get("nodes", [])
    edges = g.get("edges", [])
    prereq = g.get("prereq", {})
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("procedure_graph.json must contain non-empty list field 'nodes'")
    if not isinstance(edges, list):
        edges = []
    if not isinstance(prereq, dict):
        prereq = {}
    return set(nodes), edges, prereq


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def find_asr_for_video(video_path: str, asr_dir: str, asr_suffix: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    candidates = [
        os.path.join(asr_dir, f"{stem}{asr_suffix}.json"),
        os.path.join(asr_dir, f"{stem}.json"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    fuzzy = sorted(glob.glob(os.path.join(asr_dir, f"{stem}*.json")))
    if len(fuzzy) == 1:
        return fuzzy[0]
    return ""


# -------------------------
# Validation (for evaluation reliability)
# -------------------------
def validate_state_vector(vec: Any, n: int) -> Tuple[bool, str]:
    if not isinstance(vec, list) or len(vec) != n:
        return False, f"state must be a list of length {n}"
    for x in vec:
        if x not in (-1, 0, 1):
            return False, "state entries must be in {-1,0,1}"
    return True, ""


def validate_psr_completed_list(
    completed: Any,
    allowed_nodes: Set[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings = []
    out = []
    if completed is None:
        return out, warnings
    if not isinstance(completed, list):
        return out, ["completed must be a list"]

    for i, item in enumerate(completed):
        if not isinstance(item, dict):
            warnings.append(f"completed[{i}] not an object")
            continue
        step = item.get("step")
        tau = item.get("tau_s")
        c = item.get("c")
        if not isinstance(step, str):
            warnings.append(f"completed[{i}].step missing/invalid")
            continue
        if step not in allowed_nodes:
            warnings.append(f"completed[{i}].step not in procedure_graph.nodes: {step}")
        try:
            tau = float(tau)
        except Exception:
            warnings.append(f"completed[{i}].tau_s missing/invalid")
            tau = None
        try:
            c = float(c)
        except Exception:
            warnings.append(f"completed[{i}].c missing/invalid")
            c = None
        if c is not None and not (0.0 <= c <= 1.0):
            warnings.append(f"completed[{i}].c out of range [0,1]: {c}")
        out.append({"step": step, "tau_s": tau, "c": c})
    return out, warnings


def check_partial_order(
    completed: List[Dict[str, Any]],
    prereq: Dict[str, List[str]],
    tol_s: float = 0.25,
) -> List[str]:
    """
    Verify prerequisite constraints under partial order.
    - For each step b in completed, all prereq[b] must appear earlier with tau <= tau_b + tol_s
    """
    warnings = []
    pos = {x["step"]: i for i, x in enumerate(completed) if isinstance(x.get("step"), str)}
    tau_map = {x["step"]: x.get("tau_s") for x in completed if isinstance(x.get("step"), str)}

    for b, pres in prereq.items():
        if b not in pos:
            continue
        tau_b = tau_map.get(b)
        if tau_b is None:
            continue
        for a in pres:
            if a not in pos:
                warnings.append(f"partial-order violation: missing prerequisite {a} before {b}")
                continue
            # order in list
            if pos[a] > pos[b]:
                warnings.append(f"partial-order violation: {a} appears after {b}")
            # time consistency
            tau_a = tau_map.get(a)
            if tau_a is not None and tau_a > tau_b + tol_s:
                warnings.append(f"partial-order time violation: tau({a})={tau_a:.3f} > tau({b})={tau_b:.3f} (+{tol_s}s)")
    return warnings


def in_sampled_time(t: float, sampled: Set[float], tol: float = 0.051) -> bool:
    for s in sampled:
        if abs(s - t) <= tol:
            return True
    return False


# -------------------------
# Prompt (simple + evaluable)
# -------------------------
def build_prompt(
    video_id: str,
    fps: float,
    workflow: str,
    components: List[Dict[str, Any]],
    key_times_s: List[float],
    allowed_nodes: List[str],
    edges: List[List[str]],
    alias_map: Dict[str, str],
) -> str:
    comp_order = "\n".join([f"{c['id']}: {c['name']}" for c in components])
    nodes_text = "\n".join([f"- {n}" for n in allowed_nodes])
    # edges are precedence constraints (partial order)
    edges_text = "\n".join([f"- {a} -> {b}" for a, b in edges[:200]])  # safe cap

    alias_text = "\n".join([f"- {k} -> {v}" for k, v in sorted(alias_map.items())])

    return f"""
You will receive a sequence of frames from an industrial (dis-)assembly video.
Each frame is preceded by a header:
  [FRAME-KEY] t=12.30s   or   [FRAME-BASE] t=45.00s

Meta:
- video_id: {video_id}
- fps: {fps}
- workflow: {workflow}

Component order for the state vector (length = {len(components)}):
(Meaning: 1=installed correctly, 0=not installed, -1=installed wrongly)
{comp_order}

Key query times (seconds). You MUST output one state vector and one PSR completed-step list for EACH time:
{[round_t(t, 3) for t in key_times_s]}

Alias map (do NOT invent new aliases):
{alias_text}

PSR step vocabulary (ALLOWED_NODES). Every step label MUST be exactly one of these strings:
{nodes_text}

Partial-order prerequisite constraints (a -> b means a must be completed before b; missing edges imply flexible ordering):
{edges_text}

TASK:
A) ASD/ASR-style state tracking:
For each key query time t, output the component installation state vector AFTER time t.
State values MUST be in {{-1,0,1}}.

B) Procedure Step Recognition (completion-centric, IndustReal-style):
At each key query time t, output the ORDERED list of steps that have been correctly completed so far:
  E_t = [(label, completion_time, confidence), ...]
- label must be in ALLOWED_NODES
- list must be ordered by completion_time ascending
- completion_time <= t
- must respect the partial-order constraints (a -> b)

Confidence calibration rules (for field c):
- c MUST reflect visual certainty, not a default value.
- Do NOT assign 1.0 by default.
- Use two decimals (e.g., 0.73, 0.91).
- 0.95-1.00: transition is directly visible and supported by >=2 nearby frames, with no contradiction.
- 0.80-0.94: strong evidence, but not fully explicit in all nearby frames.
- 0.60-0.79: plausible with partial/indirect evidence.
- 0.40-0.59: weak evidence; use sparingly.
- If evidence is ambiguous or only a single weak frame supports it, keep c <= 0.90.
- 1.00 is only allowed for exceptionally clear transitions with explicit before/after visual evidence.

OUTPUT (STRICT JSON ONLY, no markdown, no extra text):
{{
  "video_id": "{video_id}",
  "fps": {fps},
  "workflow": "{workflow}",
  "states_over_time": [
    {{
      "time_s": number,            // one entry per key query time
      "state": [-1/0/1 x {len(components)}]
    }}
  ],
  "psr_over_time": [
    {{
      "time_s": number,            // one entry per key query time
      "completed": [
        {{
          "step": string,          // in ALLOWED_NODES
          "tau_s": number,         // completion time (<= time_s)
          "c": number              // confidence 0..1
        }}
      ],
      "evidence_times_s": [number] // timestamps taken from frame headers (>=2 if possible)
    }}
  ]
}}

Important:
- Do NOT output any step not in ALLOWED_NODES.
- Do NOT output markdown.
- Use timestamps shown in headers for evidence_times_s.
""".strip()


# -------------------------
# One-video inference
# -------------------------
def run_one_video(
    video_path: str,
    asr_json_path: str,
    procedure_graph_path: str,
    component_alias_path: str,
    model: str,
    out_json_path: str,
    out_raw_path: str,
    jpeg_quality: int,
    temperature: float,
    max_frames: int,
    base_fps: float,
    api_key: str,
) -> Dict[str, Any]:
    if max_frames < 0:
        raise ValueError("--max_frames must be >= 0 (0 means no frame cap)")
    if base_fps <= 0:
        raise ValueError("--base_fps must be > 0")

    # Load ASR
    with open(asr_json_path, "r", encoding="utf-8") as f:
        asr = json.load(f)

    # fps and video_id
    # (your ASR has both top-level fps/video_id and meta_data; we prioritize meta_data if present)
    fps = float(asr.get("meta_data", {}).get("fps", asr.get("fps")))
    workflow = str(asr.get("meta_data", {}).get("workflow", asr.get("workflow", "unknown")))
    video_id = str(asr.get("meta_data", {}).get("video_id", asr.get("video_id")))

    # components (deterministic order by id)
    components = asr.get("components", [])
    components = sorted([{"id": int(c["id"]), "name": str(c["name"])} for c in components], key=lambda x: x["id"])
    n_comp = len(components)

    # key times from ASR state_changes frames
    state_changes = asr.get("state_changes", [])
    key_times_s = sorted({float(c["frame"]) / fps for c in state_changes if "frame" in c})

    # duration
    with VideoFileClip(video_path) as vtmp:
        duration_s = float(vtmp.duration) if vtmp.duration else None
    if duration_s is None:
        raise RuntimeError("Cannot read video duration.")

    # load procedure graph nodes + edges + prereq
    allowed_nodes_set, edges, prereq = load_procedure_graph(procedure_graph_path)
    allowed_nodes_list = sorted(list(allowed_nodes_set))

    # alias map (passed into prompt; useful if model wants to reason about components vs steps)
    with open(component_alias_path, "r", encoding="utf-8") as f:
        alias_map = json.load(f)
    if not isinstance(alias_map, dict):
        alias_map = {}

    # sampling
    all_times, key_dense_set = build_sample_times(
        key_times_s=key_times_s,
        duration_s=duration_s,
        key_half_window_s=0.5,
        key_fps=10.0,
        base_fps=base_fps,
    )

    # cap frames only when requested: keep all KEY frames, subsample BASE if needed
    if max_frames > 0 and len(all_times) > max_frames:
        key_times = sorted(key_dense_set)
        base_times = sorted(set(all_times) - key_dense_set)
        remain = max_frames - len(key_times)
        if remain <= 0:
            step = max(1, math.ceil(len(key_times) / max_frames))
            all_times = key_times[::step][:max_frames]
        else:
            step = max(1, math.ceil(len(base_times) / remain)) if base_times else 1
            base_pick = base_times[::step][:remain]
            all_times = sorted(set(key_times).union(base_pick))

    sampled_time_set = set(round_t(t, 3) for t in all_times)

    # build prompt
    prompt = build_prompt(
        video_id=video_id,
        fps=fps,
        workflow=workflow,
        components=components,
        key_times_s=key_times_s,
        allowed_nodes=allowed_nodes_list,
        edges=edges,
        alias_map=alias_map,
    )

    # build parts: prompt + per-frame header + image
    parts: List[types.Part] = [types.Part(text=prompt)]

    with VideoFileClip(video_path) as video:
        for t in all_times:
            t3 = round_t(t, 3)
            tag = "KEY" if t3 in key_dense_set else "BASE"
            parts.append(types.Part(text=f"[FRAME-{tag}] t={t3:.2f}s"))
            # avoid decoder edge case when t == duration
            t_safe = min(t, max(0.0, duration_s - 1e-3))
            frame = video.get_frame(t_safe)
            jpg = frame_to_jpeg_bytes(frame, quality=jpeg_quality)
            parts.append(types.Part(inline_data=types.Blob(data=jpg, mime_type="image/jpeg")))

    # call Gemini
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=types.Content(parts=parts),
        config=types.GenerateContentConfig(temperature=temperature),
    )

    raw = resp.text or ""
    ensure_parent_dir(out_raw_path)
    with open(out_raw_path, "w", encoding="utf-8") as f:
        f.write(raw)

    pred = extract_first_json(raw)

    # -------------------------
    # Validation warnings (help your evaluation pipeline)
    # -------------------------
    warnings = []
    all_conf_values: List[float] = []

    # states_over_time must be one per key time
    sot = pred.get("states_over_time", [])
    if not isinstance(sot, list) or len(sot) != len(key_times_s):
        warnings.append(f"states_over_time length {len(sot) if isinstance(sot, list) else 'N/A'} != key_times {len(key_times_s)}")
    else:
        for i, item in enumerate(sot):
            if not isinstance(item, dict):
                warnings.append(f"states_over_time[{i}] not an object")
                continue
            vec = item.get("state")
            ok, msg = validate_state_vector(vec, n_comp)
            if not ok:
                warnings.append(f"states_over_time[{i}]: {msg}")

    # psr_over_time must be one per key time
    pot = pred.get("psr_over_time", [])
    if not isinstance(pot, list) or len(pot) != len(key_times_s):
        warnings.append(f"psr_over_time length {len(pot) if isinstance(pot, list) else 'N/A'} != key_times {len(key_times_s)}")
    else:
        for i, item in enumerate(pot):
            if not isinstance(item, dict):
                warnings.append(f"psr_over_time[{i}] not an object")
                continue
            t_query = item.get("time_s")
            try:
                t_query = float(t_query)
            except Exception:
                warnings.append(f"psr_over_time[{i}].time_s invalid")
                t_query = None

            completed, w = validate_psr_completed_list(item.get("completed", []), allowed_nodes_set)
            warnings.extend([f"psr_over_time[{i}]: {x}" for x in w])
            conf_vals = [float(x["c"]) for x in completed if x.get("c") is not None]
            all_conf_values.extend(conf_vals)
            if conf_vals and all(abs(v - 1.0) <= 1e-9 for v in conf_vals):
                warnings.append(f"psr_over_time[{i}]: all confidence values are 1.0 (possible uncalibrated confidence)")

            # enforce tau_s <= time_s
            if t_query is not None:
                for j, citem in enumerate(completed):
                    tau = citem.get("tau_s")
                    if tau is not None and tau > t_query + 1e-6:
                        warnings.append(f"psr_over_time[{i}].completed[{j}]: tau_s {tau:.3f} > time_s {t_query:.3f}")

            # check order by tau_s ascending
            taus = [x.get("tau_s") for x in completed if x.get("tau_s") is not None]
            if taus and any(taus[k] > taus[k+1] for k in range(len(taus)-1)):
                warnings.append(f"psr_over_time[{i}]: completed list not ordered by tau_s ascending")

            # check partial order constraints using prereq dict
            warnings.extend([f"psr_over_time[{i}]: {x}" for x in check_partial_order(completed, prereq)])

            # evidence times should come from sampled headers
            ev = item.get("evidence_times_s", [])
            if isinstance(ev, list) and ev:
                bad = 0
                for tt in ev:
                    try:
                        tt = float(tt)
                    except Exception:
                        bad += 1
                        continue
                    if not in_sampled_time(tt, sampled_time_set):
                        bad += 1
                if bad:
                    warnings.append(f"psr_over_time[{i}]: {bad} evidence_times_s not in sampled timestamps")

    if all_conf_values:
        one_ratio = sum(1 for v in all_conf_values if abs(v - 1.0) <= 1e-9) / float(len(all_conf_values))
        uniq_count = len(set(round(v, 4) for v in all_conf_values))
        if one_ratio >= 0.8:
            warnings.append(f"confidence calibration warning: {one_ratio:.0%} of confidence values are 1.0")
        if uniq_count == 1:
            warnings.append("confidence calibration warning: all confidence values are identical")

    pred["_validation_warnings"] = warnings

    ensure_parent_dir(out_json_path)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(pred, f, ensure_ascii=False, indent=2)

    print("=== DONE ===")
    print(f"video_id={video_id} | fps={fps} | duration={duration_s:.2f}s")
    print(f"key_times={len(key_times_s)} | sampled_frames={len(all_times)}")
    print(f"raw: {out_raw_path}")
    print(f"json: {out_json_path}")

    if warnings:
        print("\n=== WARNINGS (check before evaluation) ===")
        for w in warnings[:80]:
            print("-", w)
        if len(warnings) > 80:
            print(f"... ({len(warnings)-80} more)")
    else:
        print("\nNo validation warnings. Output is evaluation-ready.")


    return {
        "video_id": video_id,
        "video_path": video_path,
        "asr_json_path": asr_json_path,
        "out_json_path": out_json_path,
        "out_raw_path": out_raw_path,
        "key_times": len(key_times_s),
        "sampled_frames": len(all_times),
        "warnings": len(warnings),
    }


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    # single-video mode
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--asr_json", type=str, default=None)

    # batch mode
    ap.add_argument("--video_dir", type=str, default=None, help="Folder containing videos to evaluate.")
    ap.add_argument("--asr_dir", type=str, default=None, help="Folder containing ASR JSON files.")
    ap.add_argument("--out_dir", type=str, default="pred_batch", help="Output folder for batch mode.")
    ap.add_argument("--video_glob", type=str, default="*.mp4", help="Filename pattern in --video_dir subtree (default: *.mp4).")
    ap.add_argument("--asr_suffix", type=str, default="_asr", help="ASR filename suffix after video stem (default: _asr).")

    ap.add_argument("--procedure_graph", type=str, required=True)
    ap.add_argument("--component_alias", type=str, required=True)
    ap.add_argument("--model", type=str, default="gemini-3.1-pro-preview")
    ap.add_argument("--out_json", type=str, default="pred_asd_psr.json")
    ap.add_argument("--out_raw", type=str, default="pred_raw.txt")
    ap.add_argument("--jpeg_quality", type=int, default=85)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_frames", type=int, default=0,
                    help="Max sampled frames; 0 means no cap (default: 0).")
    ap.add_argument("--base_fps", type=float, default=0.5,
                    help="Sampling fps outside key-transition windows (default: 0.5).")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Please set GEMINI_API_KEY env var.")

    single_mode = bool(args.video or args.asr_json)
    batch_mode = bool(args.video_dir or args.asr_dir)
    if single_mode and batch_mode:
        raise ValueError("Use either single-video mode (--video + --asr_json) OR batch mode (--video_dir + --asr_dir), not both.")

    if single_mode:
        if not args.video or not args.asr_json:
            raise ValueError("Single-video mode requires both --video and --asr_json.")
        run_one_video(
            video_path=args.video,
            asr_json_path=args.asr_json,
            procedure_graph_path=args.procedure_graph,
            component_alias_path=args.component_alias,
            model=args.model,
            out_json_path=args.out_json,
            out_raw_path=args.out_raw,
            jpeg_quality=args.jpeg_quality,
            temperature=args.temperature,
            max_frames=args.max_frames,
            base_fps=args.base_fps,
            api_key=api_key,
        )
        return

    if not batch_mode:
        raise ValueError("Provide either --video + --asr_json (single mode) or --video_dir + --asr_dir (batch mode).")
    if not args.video_dir or not args.asr_dir:
        raise ValueError("Batch mode requires both --video_dir and --asr_dir.")

    # recursive scan so all mp4 under subfolders are included
    pattern = os.path.join(args.video_dir, "**", args.video_glob)
    video_paths = sorted([p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)])
    if not video_paths:
        raise FileNotFoundError(f"No videos found: {pattern}")

    os.makedirs(args.out_dir, exist_ok=True)
    ok = 0
    failed = 0
    skipped = 0
    for i, vp in enumerate(video_paths, 1):
        stem = os.path.splitext(os.path.basename(vp))[0]
        asr_path = find_asr_for_video(vp, args.asr_dir, args.asr_suffix)
        if not asr_path:
            print(f"[SKIP {i}/{len(video_paths)}] {stem}: ASR json not found in {args.asr_dir}")
            skipped += 1
            continue

        out_json = os.path.join(args.out_dir, f"{stem}_pred_asd_psr.json")
        out_raw = os.path.join(args.out_dir, f"{stem}_pred_raw.txt")
        print(f"\n[RUN {i}/{len(video_paths)}] {stem}")
        try:
            run_one_video(
                video_path=vp,
                asr_json_path=asr_path,
                procedure_graph_path=args.procedure_graph,
                component_alias_path=args.component_alias,
                model=args.model,
                out_json_path=out_json,
                out_raw_path=out_raw,
                jpeg_quality=args.jpeg_quality,
                temperature=args.temperature,
                max_frames=args.max_frames,
                base_fps=args.base_fps,
                api_key=api_key,
            )
            ok += 1
        except Exception as e:
            print(f"[FAIL {i}/{len(video_paths)}] {stem}: {e}")
            failed += 1

    print("\n=== BATCH SUMMARY ===")
    print(f"videos_total={len(video_paths)}")
    print(f"ok={ok} | failed={failed} | skipped={skipped}")
    print(f"out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
