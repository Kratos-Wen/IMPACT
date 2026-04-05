#!/usr/bin/env python3
"""Cross-View Semantic Matching via Retrieval (CV-SMR) for IMPACT CAS.

This task evaluates semantic invariance across views.

Query unit:
- one coarse action segment from a source view.

Gallery construction:
- all non-null segments from the selected gallery views in the same split,
  excluding segments from the same trial as the query,
  and excluding the query's own view so retrieval is strictly cross-view.

Positive / negative definition:
- positive: gallery segment with the same coarse action label as the query.
- negative: gallery segment with a different coarse action label.

Multiple positives may exist for one query, so the script reports Recall@1,
Recall@5, and mAP.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
COMMON_DIR = THIS_DIR.parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from cv_sm_common import (
    DEFAULT_ANNOTATION_ROOT,
    DEFAULT_SPLIT_DIR,
    KNOWN_VIEWS,
    average_precision_from_hits,
    build_label_vocab,
    counter_to_sorted_dict,
    filter_embedded_records,
    load_embeddings_for_views,
    load_split_records,
    parse_view_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Cross-View Semantic Matching via Retrieval (CV-SMR).",
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
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--split-index", type=int, default=1)
    parser.add_argument("--split-bundle", type=Path, default=None)
    parser.add_argument(
        "--query-views",
        type=str,
        default=",".join(KNOWN_VIEWS),
        help="Comma-separated query views.",
    )
    parser.add_argument(
        "--gallery-views",
        type=str,
        default=",".join(KNOWN_VIEWS),
        help="Comma-separated gallery views.",
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
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def summarize_query_metrics(hit1: List[float], hit5: List[float], aps: List[float]) -> Dict[str, Optional[float]]:
    if not aps:
        return {
            "recall@1": None,
            "recall@5": None,
            "mAP": None,
        }
    return {
        "recall@1": float(np.mean(np.asarray(hit1, dtype=np.float32))),
        "recall@5": float(np.mean(np.asarray(hit5, dtype=np.float32))),
        "mAP": float(np.mean(np.asarray(aps, dtype=np.float32))),
    }


def main() -> None:
    args = parse_args()
    query_views = parse_view_list(args.query_views)
    gallery_views = parse_view_list(args.gallery_views)

    records, split_bundle_path, _ = load_split_records(
        metadata=args.metadata,
        annotation_root=args.annotation_root,
        split=args.split,
        split_dir=args.split_dir,
        split_index=args.split_index,
        split_bundle=args.split_bundle,
    )

    selected_views = set(query_views) | set(gallery_views)
    feature_root, embeddings, feature_attach_stats, feature_stats = load_embeddings_for_views(
        records=records,
        selected_views=selected_views,
        feature_type=args.feature_type,
        feature_root=args.feature_root,
        pooling=args.pooling,
    )

    query_records = filter_embedded_records(records, embeddings, query_views)
    gallery_records = filter_embedded_records(records, embeddings, gallery_views)
    label_vocab = build_label_vocab(records)

    if gallery_records:
        gallery_matrix = np.stack(
            [embeddings[record.segment_uid] for record in gallery_records],
            axis=0,
        )
        gallery_trials = np.asarray([record.trial_id for record in gallery_records], dtype=object)
        gallery_views_arr = np.asarray([record.view_id for record in gallery_records], dtype=object)
        gallery_labels = np.asarray([record.coarse_label for record in gallery_records], dtype=object)
    else:
        gallery_matrix = None
        gallery_trials = np.asarray([], dtype=object)
        gallery_views_arr = np.asarray([], dtype=object)
        gallery_labels = np.asarray([], dtype=object)

    overall_hit1: List[float] = []
    overall_hit5: List[float] = []
    overall_aps: List[float] = []
    overall_gallery_sizes: List[int] = []
    overall_candidate_by_label: Counter = Counter()
    overall_evaluated_by_label: Counter = Counter()
    overall_candidate_queries = 0
    overall_evaluated_queries = 0
    overall_skipped_empty_gallery = 0
    overall_skipped_no_positive = 0

    per_query_view_metrics: List[Dict[str, Any]] = []
    for query_view in query_views:
        view_queries = [record for record in query_records if record.view_id == query_view]
        candidate_by_label: Counter = Counter(record.coarse_label for record in view_queries)
        evaluated_by_label: Counter = Counter()
        view_hit1: List[float] = []
        view_hit5: List[float] = []
        view_aps: List[float] = []
        view_gallery_sizes: List[int] = []
        skipped_empty_gallery = 0
        skipped_no_positive = 0

        for query_record in view_queries:
            if gallery_matrix is None:
                skipped_empty_gallery += 1
                continue

            valid_mask = (gallery_trials != query_record.trial_id) & (gallery_views_arr != query_record.view_id)
            gallery_size = int(np.sum(valid_mask))
            if gallery_size == 0:
                skipped_empty_gallery += 1
                continue

            positive_mask = valid_mask & (gallery_labels == query_record.coarse_label)
            if not np.any(positive_mask):
                skipped_no_positive += 1
                continue

            scores = gallery_matrix @ embeddings[query_record.segment_uid]
            valid_indices = np.flatnonzero(valid_mask)
            ranked_indices = valid_indices[np.argsort(-scores[valid_indices], kind="stable")]
            hits = (gallery_labels[ranked_indices] == query_record.coarse_label).astype(np.float32)
            ap = average_precision_from_hits(hits)
            if ap is None:
                skipped_no_positive += 1
                continue

            hit1 = float(np.any(hits[:1]))
            hit5 = float(np.any(hits[:5]))
            view_hit1.append(hit1)
            view_hit5.append(hit5)
            view_aps.append(ap)
            view_gallery_sizes.append(gallery_size)
            evaluated_by_label[query_record.coarse_label] += 1

        metrics = summarize_query_metrics(view_hit1, view_hit5, view_aps)
        view_summary: Dict[str, Any] = {
            "query_view": query_view,
            "candidate_queries": len(view_queries),
            "evaluated_queries": len(view_aps),
            "skipped_empty_gallery": skipped_empty_gallery,
            "skipped_no_positive": skipped_no_positive,
            "average_gallery_size": (
                float(np.mean(np.asarray(view_gallery_sizes, dtype=np.float32)))
                if view_gallery_sizes
                else None
            ),
            "candidate_queries_by_label": counter_to_sorted_dict(candidate_by_label),
            "evaluated_queries_by_label": counter_to_sorted_dict(evaluated_by_label),
        }
        view_summary.update(metrics)
        per_query_view_metrics.append(view_summary)

        overall_hit1.extend(view_hit1)
        overall_hit5.extend(view_hit5)
        overall_aps.extend(view_aps)
        overall_gallery_sizes.extend(view_gallery_sizes)
        overall_candidate_by_label.update(candidate_by_label)
        overall_evaluated_by_label.update(evaluated_by_label)
        overall_candidate_queries += len(view_queries)
        overall_evaluated_queries += len(view_aps)
        overall_skipped_empty_gallery += skipped_empty_gallery
        overall_skipped_no_positive += skipped_no_positive

    overall = {
        "candidate_queries": overall_candidate_queries,
        "evaluated_queries": overall_evaluated_queries,
        "skipped_empty_gallery": overall_skipped_empty_gallery,
        "skipped_no_positive": overall_skipped_no_positive,
        "average_gallery_size": (
            float(np.mean(np.asarray(overall_gallery_sizes, dtype=np.float32)))
            if overall_gallery_sizes
            else None
        ),
        "candidate_queries_by_label": counter_to_sorted_dict(overall_candidate_by_label),
        "evaluated_queries_by_label": counter_to_sorted_dict(overall_evaluated_by_label),
    }
    overall.update(summarize_query_metrics(overall_hit1, overall_hit5, overall_aps))

    summary = {
        "config": {
            "metadata": str(args.metadata) if args.metadata else None,
            "annotation_root": str(args.annotation_root) if args.metadata is None else None,
            "feature_root": str(feature_root),
            "feature_type": args.feature_type,
            "pooling": args.pooling,
            "split": args.split,
            "split_bundle": str(split_bundle_path) if split_bundle_path else None,
            "split_index": args.split_index,
            "query_views": query_views,
            "gallery_views": gallery_views,
            "gallery_rule": "other views only, exclude same trial",
            "positive_definition": "same coarse_label",
            "negative_definition": "different coarse_label",
        },
        "dataset": {
            "records_total": len(records),
            "records_non_null": sum(1 for record in records if not record.is_null),
            "labels_total": len(label_vocab),
            "feature_paths_resolved": feature_attach_stats["resolved"],
            "feature_paths_missing": feature_attach_stats["missing"],
            "videos_loaded": feature_stats["videos_loaded"],
            "segments_embedded": feature_stats["segments_embedded"],
        },
        "per_query_view_metrics": per_query_view_metrics,
        "overall": overall,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    if not args.quiet:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
