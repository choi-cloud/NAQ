"""exp052: Descriptive Cora-Full statistics using the original NAQ class split."""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from data import load_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Cora_full")
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def distribution_stats(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    ordered = np.sort(counts)
    shares = counts / total
    n = len(counts)
    return {
        "num_classes": int(n),
        "num_nodes": int(total),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "median": float(np.median(counts)),
        "std": float(counts.std(ddof=0)),
        "cv": float(counts.std(ddof=0) / counts.mean()),
        "max_min_ratio": float(counts.max() / counts.min()),
        "p10": float(np.percentile(counts, 10)),
        "p25": float(np.percentile(counts, 25)),
        "p75": float(np.percentile(counts, 75)),
        "p90": float(np.percentile(counts, 90)),
        "gini": float((2 * np.dot(np.arange(1, n + 1), ordered) / (n * total)) - (n + 1) / n),
        "normalized_entropy": float(-np.sum(shares * np.log(shares)) / math.log(n)) if n > 1 else 1.0,
        "top_10pct_classes_node_share": float(np.sort(shares)[-max(1, math.ceil(n * 0.1)) :].sum()),
        "bottom_50pct_classes_node_share": float(np.sort(shares)[: max(1, math.floor(n * 0.5))].sum()),
    }


def main():
    args = parse_args()
    if args.dataset != "Cora_full":
        raise ValueError("exp052 is specifically for the Cora_full dataset")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Original NAQ data loader: class order is first appearance in labels,
    # followed by Python random.shuffle and a 25/20/25 class split.
    data = load_data(args)
    labels = data.labels.detach().cpu().numpy().reshape(-1)
    edges = data.edge_index.detach().cpu().numpy()
    features = data.features.detach().cpu()

    split_ids = {
        "train": [int(x) for x in data.class_list_train],
        "valid": [int(x) for x in data.class_list_valid],
        "test": [int(x) for x in data.class_list_test],
    }
    all_classes = sorted(int(x) for x in np.unique(labels))
    class_to_split = {c: split for split, ids in split_ids.items() for c in ids}
    if set(class_to_split) != set(all_classes):
        raise RuntimeError("Class split is not a complete partition of observed classes")

    counts = Counter(int(x) for x in labels)
    node_degrees = np.bincount(edges[0], minlength=len(labels)).astype(np.int64)
    per_class = []
    for cls in all_classes:
        ids = np.flatnonzero(labels == cls)
        degrees = node_degrees[ids]
        per_class.append(
            {
                "class_id": cls,
                "split": class_to_split[cls],
                "num_nodes": int(len(ids)),
                "fraction_of_all_nodes": float(len(ids) / len(labels)),
                "degree_mean": float(degrees.mean()),
                "degree_median": float(np.median(degrees)),
                "isolated_node_count": int((degrees == 0).sum()),
            }
        )

    splits = {}
    for name, ids in split_ids.items():
        per_counts = [counts[c] for c in ids]
        splits[name] = {
            "class_ids_original_order": ids,
            "class_ids_sorted": sorted(ids),
            "class_counts": {str(c): int(counts[c]) for c in sorted(ids)},
            "node_fraction_of_graph": float(sum(per_counts) / len(labels)),
            "imbalance": distribution_stats(per_counts),
            "classes_smallest_first": sorted(ids, key=lambda c: (counts[c], c)),
            "classes_largest_first": sorted(ids, key=lambda c: (-counts[c], c)),
        }

    # Match against all observed nodes, not only classes named 'train'.
    raw_train_ids = np.asarray(data.id_train, dtype=np.int64)
    unique_edges = np.unique(np.sort(edges.T, axis=1), axis=0)
    source = edges[0].astype(np.int64)
    target = edges[1].astype(np.int64)
    cross_split_edges = int(
        np.count_nonzero(np.asarray([class_to_split[int(labels[a])] != class_to_split[int(labels[b])] for a, b in zip(source, target)], dtype=bool))
    )

    report = {
        "experiment": "exp052",
        "dataset": args.dataset,
        "seed": args.seed,
        "source": "NAQ original data.load_data(Cora_full), no model training",
        "split_protocol": {
            "method": "first-appearance class order, Python random.shuffle, 25/20/25",
            "counts": {name: len(ids) for name, ids in split_ids.items()},
            "important_note": "For unsupervised NAQ training, data.id_train contains ALL nodes across train/valid/test classes. These splits designate downstream labeled tasks, not node access restrictions.",
        },
        "graph": {
            "num_nodes": int(len(labels)),
            "num_classes": len(all_classes),
            "num_features": int(features.shape[1]),
            "edge_index_entries_directed": int(edges.shape[1]),
            "undirected_edge_pairs_including_self_loops": int(len(unique_edges)),
            "self_loop_entries": int(np.count_nonzero(source == target)),
            "mean_edge_index_degree": float(node_degrees.mean()),
            "median_edge_index_degree": float(np.median(node_degrees)),
            "isolated_nodes_in_edge_index": int(np.count_nonzero(node_degrees == 0)),
            "cross_downstream_split_edge_entries": cross_split_edges,
            "cross_downstream_split_edge_fraction": float(cross_split_edges / edges.shape[1]) if edges.shape[1] else 0.0,
            "feature_nonzero_fraction": float(torch.count_nonzero(features).item() / features.numel()),
            "feature_row_l2_norm_mean": float(torch.linalg.vector_norm(features.float(), dim=1).mean().item()),
        },
        "all_classes": {
            "imbalance": distribution_stats([counts[c] for c in all_classes]),
            "class_counts": {str(c): int(counts[c]) for c in all_classes},
        },
        "splits": splits,
        "per_class": per_class,
        "unsupervised_train_pool": {
            "id_train_nodes": int(len(raw_train_ids)),
            "unique_id_train_nodes": int(len(np.unique(raw_train_ids))),
            "covers_all_graph_nodes": bool(np.array_equal(np.sort(raw_train_ids), np.arange(len(labels)))),
            "nodes_by_downstream_split": {name: int(np.isin(labels[raw_train_ids], ids).sum()) for name, ids in split_ids.items()},
        },
    }
    out = Path(__file__).resolve().with_suffix(".json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    print(f"[exp052] nodes={len(labels)} classes={len(all_classes)} edges={edges.shape[1]}")
    for name, stats in splits.items():
        s = stats["imbalance"]
        print(
            f"[{name}] classes={s['num_classes']} nodes={s['num_nodes']} "
            f"min={s['min']} median={s['median']:.1f} max={s['max']} "
            f"max/min={s['max_min_ratio']:.2f} gini={s['gini']:.3f} cv={s['cv']:.3f}"
        )
    print(f"[exp052] JSON saved: {out}")


if __name__ == "__main__":
    main()
