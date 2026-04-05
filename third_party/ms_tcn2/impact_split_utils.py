import re
from pathlib import Path


FEATURE_SUBDIRS = {
    "ms_tcn2": {
        "front": "IMPACT_i3d_front/features",
        "left": "IMPACT_i3d_left/features",
        "right": "IMPACT_i3d_right/features",
        "top": "IMPACT_i3d_top/features",
        "ego": "IMPACT_i3d_ego/features",
    },
    "videomae": {
        "front": "IMPACT_front/features",
        "left": "IMPACT_left/features",
        "right": "IMPACT_right/features",
        "top": "IMPACT_top/features",
        "ego": "IMPACT_ego/features",
    },
}


def read_bundle_entries(split_dir, bundle_split, impact_split):
    bundle_path = Path(split_dir) / f"{bundle_split}.{impact_split}.bundle"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing bundle file: {bundle_path}")
    return [line.strip() for line in bundle_path.read_text().splitlines() if line.strip()]


def default_impact_annotation_dir(base_data_dir="../data"):
    return str(Path(base_data_dir) / "IMPACT_ASR")


def default_impact_feature_dir(experiment, camera="front", base_data_dir="../data"):
    try:
        rel_path = FEATURE_SUBDIRS[experiment][camera]
    except KeyError as exc:
        raise ValueError(f"Unsupported experiment/camera combination: {experiment}, {camera}") from exc
    return str(Path(base_data_dir) / rel_path)


def entry_to_camera_stem(entry, camera):
    if camera == "front":
        return entry
    if camera in {"left", "right", "top"}:
        return re.sub(r"_front_clipped$", f"_{camera}_clipped", entry)
    if camera == "ego":
        return re.sub(r"_front_clipped$", "_ego_sync", entry)
    raise ValueError(f"Unsupported camera: {camera}")


def annotation_filename_from_video_stem(video_stem):
    front_stem = re.sub(r"_(left|right|top)_clipped$", "_front_clipped", video_stem)
    front_stem = re.sub(r"_ego_sync$", "_front_clipped", front_stem)
    return f"{front_stem}_asr.json"


def load_bundle_feature_files(split_dir, bundle_split, impact_split, feature_dir, camera="front"):
    feature_root = Path(feature_dir)
    files = []
    missing = []
    for entry in read_bundle_entries(split_dir, bundle_split, impact_split):
        feature_stem = entry_to_camera_stem(entry, camera)
        candidate = feature_root / f"{feature_stem}.npy"
        if candidate.exists():
            files.append(str(candidate))
            continue
        matches = sorted(feature_root.rglob(f"{feature_stem}.npy"))
        if len(matches) == 1:
            files.append(str(matches[0]))
            continue
        if len(matches) > 1:
            raise ValueError(
                f"Found multiple feature files for {feature_stem} under {feature_root}: "
                + ", ".join(str(match) for match in matches[:5])
            )
        else:
            missing.append(str(candidate))
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} feature files in {feature_root}. First missing paths:\n{preview}"
        )
    return files
