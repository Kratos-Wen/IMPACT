import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert IMPACT ASR annotations into STORM-PSR label files."
    )
    parser.add_argument(
        "--annotation_dir",
        type=Path,
        required=True,
        help="Directory containing IMPACT ASR JSON annotations.",
    )
    parser.add_argument(
        "--split_dir",
        type=Path,
        required=True,
        help="Directory containing bundle split files, e.g. train.split1.bundle.",
    )
    parser.add_argument(
        "--impact_split",
        type=str,
        default="split1",
        help="Split bundle suffix, e.g. split1.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for STORM-style labels.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Dataset splits to export.",
    )
    return parser.parse_args()


def frame_name(frame_idx: int) -> str:
    return f"{int(frame_idx)}.jpg"


def read_bundle(split_dir: Path, split: str, impact_split: str) -> List[str]:
    bundle_path = split_dir / f"{split}.{impact_split}.bundle"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing bundle file: {bundle_path}")
    return [line.strip() for line in bundle_path.read_text().splitlines() if line.strip()]


def resolve_annotation_path(annotation_dir: Path, entry: str) -> Path:
    candidates = [annotation_dir / f"{entry}_asr.json"]
    fallback_entry = re.sub(r"_(left|right|top)_clipped$", "_front_clipped", entry)
    fallback_entry = re.sub(r"_ego_sync$", "_front_clipped", fallback_entry)
    if fallback_entry != entry:
        candidates.append(annotation_dir / f"{fallback_entry}_asr.json")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find IMPACT ASR annotation for {entry}. Tried: {candidates}"
    )


def normalize_no_error(state: Sequence[int]) -> List[int]:
    return [0 if int(value) == -1 else int(value) for value in state]


def build_procedure_info(component_names: Sequence[str]) -> List[dict]:
    procedure_info = []
    for idx, component_name in enumerate(component_names):
        procedure_info.append({
            "id": idx * 3 + 0,
            "description": f"Install {component_name}",
            "install": True,
            "state_idx": idx,
            "expected_in_assy": False,
            "expected_before_subgoal": False,
            "expected_in_main": False,
        })
        procedure_info.append({
            "id": idx * 3 + 1,
            "description": f"Incorrectly installed {component_name}",
            "install": True,
            "state_idx": idx,
            "expected_in_assy": False,
            "expected_before_subgoal": False,
            "expected_in_main": False,
        })
        procedure_info.append({
            "id": idx * 3 + 2,
            "description": f"Remove {component_name}",
            "install": False,
            "state_idx": idx,
            "expected_in_assy": False,
            "expected_before_subgoal": False,
            "expected_in_main": False,
        })
    return procedure_info


def transition_to_action_id(prev_state: int, curr_state: int, state_idx: int):
    if prev_state == curr_state:
        return None
    if prev_state == -1 and curr_state == 0:
        return None
    if prev_state == -1 and curr_state == 1:
        return state_idx * 3 + 0
    if prev_state == 0 and curr_state == -1:
        return state_idx * 3 + 1
    if prev_state == 0 and curr_state == 1:
        return state_idx * 3 + 0
    if prev_state == 1 and curr_state == -1:
        return state_idx * 3 + 1
    if prev_state == 1 and curr_state == 0:
        return state_idx * 3 + 2
    raise ValueError(
        f"Unsupported state transition for component {state_idx}: {prev_state} -> {curr_state}"
    )


def states_to_actions(state_sequence: Sequence[dict], procedure_info: Sequence[dict], replace_errors_with_zero: bool) -> List[List[object]]:
    if not state_sequence:
        return []

    def map_state(entry_state: Sequence[int]) -> List[int]:
        if replace_errors_with_zero:
            return normalize_no_error(entry_state)
        return [int(value) for value in entry_state]

    ordered_states = sorted(state_sequence, key=lambda item: int(item["frame"]))
    prev_state = map_state(ordered_states[0]["state"])
    rows: List[List[object]] = []

    for item in ordered_states[1:]:
        frame_idx = int(item["frame"])
        curr_state = map_state(item["state"])
        for state_idx, (prev_value, curr_value) in enumerate(zip(prev_state, curr_state)):
            action_id = transition_to_action_id(prev_value, curr_value, state_idx)
            if action_id is None:
                continue
            rows.append([
                frame_name(frame_idx),
                action_id,
                procedure_info[action_id]["description"],
            ])
        prev_state = curr_state

    return rows


def ensure_initial_state(annotation: dict) -> List[dict]:
    state_sequence = sorted(annotation["state_sequence"], key=lambda item: int(item["frame"]))
    if state_sequence and int(state_sequence[0]["frame"]) == 0:
        return state_sequence

    initial_state = annotation.get("initial_state_vector")
    if initial_state is None:
        initial_state = annotation.get("meta_data", {}).get("initial_state_vector")
    if initial_state is None:
        raise ValueError("Annotation has no frame-0 state and no initial_state_vector to recover it.")

    return [{"frame": 0, "state": initial_state}] + state_sequence


def write_csv(rows: Iterable[Sequence[object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerows(rows)


def export_record(annotation_path: Path, out_dir: Path, procedure_info: Sequence[dict]) -> dict:
    annotation = json.loads(annotation_path.read_text())
    state_sequence = ensure_initial_state(annotation)

    raw_rows = []
    for item in state_sequence:
        raw_rows.append([frame_name(int(item["frame"]))] + [int(value) for value in item["state"]])

    psr_rows = states_to_actions(state_sequence, procedure_info, replace_errors_with_zero=True)
    psr_error_rows = states_to_actions(state_sequence, procedure_info, replace_errors_with_zero=False)

    write_csv(raw_rows, out_dir / "PSR_labels_raw.csv")
    write_csv(psr_rows, out_dir / "PSR_labels.csv")
    write_csv(psr_error_rows, out_dir / "PSR_labels_with_errors.csv")

    return {
        "num_states": len(raw_rows),
        "num_events_no_error": len(psr_rows),
        "num_events_with_errors": len(psr_error_rows),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reference_annotation = None
    for split in args.splits:
        for entry in read_bundle(args.split_dir, split, args.impact_split):
            reference_annotation = resolve_annotation_path(args.annotation_dir, entry)
            break
        if reference_annotation is not None:
            break
    if reference_annotation is None:
        raise FileNotFoundError("No bundle entries found to infer IMPACT component names.")

    annotation = json.loads(reference_annotation.read_text())
    component_names = [component["name"] for component in annotation["components"]]
    procedure_info = build_procedure_info(component_names)

    (args.out_dir / "procedure_info_IMPACT.json").write_text(
        json.dumps(procedure_info, indent=2)
    )
    (args.out_dir / "component_names.json").write_text(
        json.dumps(component_names, indent=2)
    )

    summary = {
        "annotation_dir": str(args.annotation_dir),
        "split_dir": str(args.split_dir),
        "impact_split": args.impact_split,
        "splits": {},
    }

    for split in args.splits:
        split_counts = {
            "records": 0,
            "states": 0,
            "events_no_error": 0,
            "events_with_errors": 0,
        }
        entries = read_bundle(args.split_dir, split, args.impact_split)
        for entry in entries:
            annotation_path = resolve_annotation_path(args.annotation_dir, entry)
            record_out_dir = args.out_dir / split / entry
            record_stats = export_record(annotation_path, record_out_dir, procedure_info)
            split_counts["records"] += 1
            split_counts["states"] += record_stats["num_states"]
            split_counts["events_no_error"] += record_stats["num_events_no_error"]
            split_counts["events_with_errors"] += record_stats["num_events_with_errors"]
        summary["splits"][split] = split_counts

    (args.out_dir / f"conversion_summary_{args.impact_split}.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
