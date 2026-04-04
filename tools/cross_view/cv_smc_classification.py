#!/usr/bin/env python3
"""Cross-View Semantic Matching via Classification (CV-SMC) for IMPACT CAS.

This script trains a linear classifier on frozen segment features.

Input unit:
- one coarse action segment.

Label space:
- coarse action labels from CAS annotations
- or verb / noun / verb-noun labels derived from the coarse label string.

Training protocol:
- backbone features stay frozen.
- only a linear layer is trained with cross-entropy loss.
- all selected train views are pooled into one training set.
- all selected test views are pooled into one test set.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cv_sm_common import (
    DEFAULT_ANNOTATION_ROOT,
    DEFAULT_SPLIT_DIR,
    KNOWN_VIEWS,
    counter_to_sorted_dict,
    filter_embedded_records,
    load_embeddings_for_views,
    load_split_records,
    macro_f1_score,
    parse_view_list,
    set_random_seed,
    top1_accuracy,
)

LABEL_MODE_CHOICES = ("coarse", "verb", "noun", "verb_noun")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Cross-View Semantic Matching via Classification (CV-SMC).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=DEFAULT_ANNOTATION_ROOT,
        help="CAS annotation root used when --metadata is omitted.",
    )
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--split-index", type=int, default=1)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--train-split-bundle", type=Path, default=None)
    parser.add_argument("--test-split-bundle", type=Path, default=None)
    parser.add_argument(
        "--train-views",
        type=str,
        default=",".join(KNOWN_VIEWS),
        help="Comma-separated train views.",
    )
    parser.add_argument(
        "--test-views",
        type=str,
        default=",".join(KNOWN_VIEWS),
        help="Comma-separated test views.",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        required=True,
        choices=("i3d", "videomaev2", "mvitv2"),
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=("mean", "none"),
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default="coarse",
        choices=LABEL_MODE_CHOICES,
        help=(
            "Classification target space. 'verb_noun' derives labels by "
            "splitting each CAS coarse label into verb=first token and "
            "noun=remaining tokens."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    parser.set_defaults(class_weighting=True)
    parser.add_argument(
        "--no-class-weighting",
        dest="class_weighting",
        action="store_false",
        help="Disable inverse-frequency class weights.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def split_coarse_label(coarse_label: str) -> Tuple[str, str]:
    parts = coarse_label.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"Cannot derive verb/noun from coarse label {coarse_label!r}. Expected at least one underscore."
        )
    return parts[0], "_".join(parts[1:])


def get_record_label(record: Any, label_mode: str) -> str:
    coarse_label = record.coarse_label
    if label_mode == "coarse":
        return coarse_label
    verb, noun = split_coarse_label(coarse_label)
    if label_mode == "verb":
        return verb
    if label_mode == "noun":
        return noun
    if label_mode == "verb_noun":
        return f"{verb}::{noun}"
    raise ValueError(f"Unsupported label mode: {label_mode}")


def build_label_vocab_from_records(records: List[Any], label_mode: str) -> List[str]:
    return sorted({get_record_label(record, label_mode) for record in records})


def build_feature_matrix(
    records: List[Any],
    embeddings: Dict[str, np.ndarray],
    label_to_index: Dict[str, int],
    label_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    features = np.stack([embeddings[record.segment_uid] for record in records], axis=0).astype(np.float32)
    labels = np.asarray(
        [label_to_index[get_record_label(record, label_mode)] for record in records],
        dtype=np.int64,
    )
    return features, labels


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    weights = np.zeros_like(counts)
    nonzero = counts > 0
    weights[nonzero] = counts.sum() / counts[nonzero]
    if np.any(nonzero):
        weights[nonzero] /= weights[nonzero].mean()
    weights[~nonzero] = 0.0
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_classifier(
    model: nn.Module,
    records: List[Any],
    embeddings: Dict[str, np.ndarray],
    label_to_index: Dict[str, int],
    label_mode: str,
    device: torch.device,
) -> Dict[str, Any]:
    if not records:
        return {
            "num_segments": 0,
            "top1_accuracy": None,
            "macro_f1": None,
        }

    features, labels = build_feature_matrix(records, embeddings, label_to_index, label_mode)
    with torch.no_grad():
        logits = model(torch.from_numpy(features).to(device))
    predictions = torch.argmax(logits, dim=1).cpu().numpy()

    return {
        "num_segments": int(labels.shape[0]),
        "top1_accuracy": top1_accuracy(labels, predictions),
        "macro_f1": macro_f1_score(labels, predictions, num_classes=len(label_to_index)),
        "label_counts": counter_to_sorted_dict(Counter(get_record_label(record, label_mode) for record in records)),
    }


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    train_views = parse_view_list(args.train_views)
    test_views = parse_view_list(args.test_views)
    device = resolve_device(args.device)

    train_records, train_bundle_path, _ = load_split_records(
        metadata=args.metadata,
        annotation_root=args.annotation_root,
        split=args.train_split,
        split_dir=args.split_dir,
        split_index=args.split_index,
        split_bundle=args.train_split_bundle,
    )
    test_records, test_bundle_path, _ = load_split_records(
        metadata=args.metadata,
        annotation_root=args.annotation_root,
        split=args.test_split,
        split_dir=args.split_dir,
        split_index=args.split_index,
        split_bundle=args.test_split_bundle,
    )

    combined_by_uid = {record.segment_uid: record for record in train_records}
    combined_by_uid.update({record.segment_uid: record for record in test_records})
    all_records = list(combined_by_uid.values())

    selected_views = set(train_views) | set(test_views)
    feature_root, embeddings, feature_attach_stats, feature_stats = load_embeddings_for_views(
        records=all_records,
        selected_views=selected_views,
        feature_type=args.feature_type,
        feature_root=args.feature_root,
        pooling=args.pooling,
    )

    train_segments = filter_embedded_records(train_records, embeddings, train_views)
    test_segments = filter_embedded_records(test_records, embeddings, test_views)
    label_vocab = build_label_vocab_from_records(train_segments + test_segments, args.label_mode)
    label_to_index = {label: index for index, label in enumerate(label_vocab)}

    if not train_segments:
        raise ValueError("No train segments available after view/filter/feature matching.")
    if not test_segments:
        raise ValueError("No test segments available after view/filter/feature matching.")

    train_features, y_train = build_feature_matrix(train_segments, embeddings, label_to_index, args.label_mode)
    input_dim = int(train_features.shape[1])
    num_classes = len(label_vocab)

    model = nn.Linear(input_dim, num_classes).to(device)
    class_weights = compute_class_weights(y_train, num_classes).to(device) if args.class_weighting else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_features), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    final_train_loss = None
    for _ in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_count = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            this_batch_size = int(batch_labels.shape[0])
            running_loss += float(loss.item()) * this_batch_size
            running_count += this_batch_size
        final_train_loss = running_loss / max(running_count, 1)

    model.eval()
    train_eval = evaluate_classifier(model, train_segments, embeddings, label_to_index, args.label_mode, device)
    test_eval = evaluate_classifier(model, test_segments, embeddings, label_to_index, args.label_mode, device)
    per_view_metrics: List[Dict[str, Any]] = []
    for view in test_views:
        per_view_metrics.append(
            {
                "view_id": view,
                **evaluate_classifier(
                    model,
                    [record for record in test_segments if record.view_id == view],
                    embeddings,
                    label_to_index,
                    args.label_mode,
                    device,
                ),
            }
        )

    train_label_set = {get_record_label(record, args.label_mode) for record in train_segments}
    test_label_set = {get_record_label(record, args.label_mode) for record in test_segments}
    train_only_labels = sorted(train_label_set - test_label_set)
    test_only_labels = sorted(test_label_set - train_label_set)

    summary = {
        "config": {
            "metadata": str(args.metadata) if args.metadata else None,
            "annotation_root": str(args.annotation_root) if args.metadata is None else None,
            "feature_root": str(feature_root),
            "feature_type": args.feature_type,
            "pooling": args.pooling,
            "label_mode": args.label_mode,
            "train_split": args.train_split,
            "test_split": args.test_split,
            "train_split_bundle": str(train_bundle_path) if train_bundle_path else None,
            "test_split_bundle": str(test_bundle_path) if test_bundle_path else None,
            "split_index": args.split_index,
            "train_views": train_views,
            "test_views": test_views,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": str(device),
            "class_weighting": args.class_weighting,
        },
        "dataset": {
            "labels_total": len(label_vocab),
            "label_vocab": label_vocab,
            "train_segments": len(train_segments),
            "test_segments": len(test_segments),
            "feature_paths_resolved": feature_attach_stats["resolved"],
            "feature_paths_missing": feature_attach_stats["missing"],
            "videos_loaded": feature_stats["videos_loaded"],
            "segments_embedded": feature_stats["segments_embedded"],
            "train_only_labels": train_only_labels,
            "test_only_labels": test_only_labels,
        },
        "train": {
            "final_train_loss": final_train_loss,
            "top1_accuracy": train_eval["top1_accuracy"],
            "macro_f1": train_eval["macro_f1"],
            "label_counts": train_eval["label_counts"],
        },
        "test": {
            "top1_accuracy": test_eval["top1_accuracy"],
            "macro_f1": test_eval["macro_f1"],
            "label_counts": test_eval["label_counts"],
        },
        "per_view_metrics": per_view_metrics,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    if not args.quiet:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
