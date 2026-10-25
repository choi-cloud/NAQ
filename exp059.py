"""exp059: GT homophily of 1-hop edges within X / A_tilde X clusters.

Diagnostic only; no model training and no GT used for clustering.
Run from the NAQ repository root where data.py is available.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import MiniBatchKMeans

from data import load_data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Cora_full")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--k", type=int, nargs="+", default=[70])
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--max_iter", type=int, default=200)
    p.add_argument("--n_init", type=int, default=3)
    return p.parse_args()


def calculate_stats(mask, same_gt, total_edges):
    count = int(mask.sum())
    positives = int(np.count_nonzero(mask & same_gt))
    all_positive = int(same_gt.sum())
    precision = positives / count if count else None
    baseline = all_positive / total_edges if total_edges else None
    return {
        "selected_edges": count,
        "selected_fraction_of_edges": count / total_edges if total_edges else None,
        "gt_same_edges": positives,
        "gt_different_edges": count - positives,
        "gt_same_rate": precision,
        "gt_same_rate_pct": precision * 100 if precision is not None else None,
        "lift_over_edge_baseline": precision / baseline if precision is not None and baseline else None,
        "recall_of_gt_same_edges": positives / all_positive if all_positive else None,
    }


def within_cluster_pairs(labels, y):
    # Exact GT pair purity for unordered node pairs in the same cluster,
    # regardless of whether an edge connects them. No O(N^2) enumeration.
    n = len(labels)
    selected, same = 0, 0
    for cl in np.unique(labels):
        ys = y[labels == cl]
        size = len(ys)
        selected += size * (size - 1) // 2
        _, counts = np.unique(ys, return_counts=True)
        same += int(np.sum(counts * (counts - 1) // 2))
    all_counts = np.unique(y, return_counts=True)[1]
    all_same = int(np.sum(all_counts * (all_counts - 1) // 2))
    all_pairs = n * (n - 1) // 2
    rate = same / selected if selected else None
    base = all_same / all_pairs if all_pairs else None
    return {
        "same_cluster_pairs": int(selected),
        "same_cluster_and_gt_pairs": same,
        "gt_same_rate": rate,
        "random_pair_gt_same_rate": base,
        "lift_over_random_pair": rate / base if rate is not None and base else None,
    }


def main():
    args = parse_args()
    if args.dataset != "Cora_full":
        raise ValueError("This experiment is configured for Cora_full")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    data = load_data(args)
    y = data.labels.detach().cpu().numpy().reshape(-1)
    x = data.features.detach().cpu().numpy().astype(np.float32, copy=False)
    edges = data.edge_index.detach().cpu().numpy()
    n = len(y)

    # Unique, undirected, non-self edges; avoids counting both i->j and j->i.
    u = np.minimum(edges[0], edges[1]).astype(np.int64)
    v = np.maximum(edges[0], edges[1]).astype(np.int64)
    valid = u != v
    edge_pairs = np.unique(np.stack([u[valid], v[valid]], axis=1), axis=0)
    src, dst = edge_pairs[:, 0], edge_pairs[:, 1]
    same_gt = y[src] == y[dst]
    total_edges = len(src)

    # Use exactly NAQ's row-normalized adjacency with self loops: data.adj.
    adj = data.adj.coalesce().cpu()
    ij = adj.indices().numpy()
    values = adj.values().numpy()
    a_tilde = sp.coo_matrix((values, (ij[0], ij[1])), shape=(n, n)).tocsr()
    ax = np.asarray(a_tilde @ x, dtype=np.float32)

    report = {
        "experiment": "exp059",
        "status": "completed",
        "args": vars(args),
        "definitions": {
            "X": "Original Cora_full node features; MiniBatchKMeans on raw X",
            "A_tilde_X": "NAQ data.adj (row-normalized adjacency with self-loops) multiplied by raw X; MiniBatchKMeans",
            "edge": "An undirected unique 1-hop graph edge, excluding self loops and duplicates",
            "precision": "P(GT labels equal | chosen pair has 1-hop edge AND satisfies cluster condition)",
            "GT_use": "Only post-clustering evaluation; no GT labels used in KMeans",
            "joint": "Both X-based and A_tilde_X-based clusters agree on co-membership; cluster IDs need not align",
        },
        "graph": {
            "num_nodes": n,
            "num_gt_classes": int(np.unique(y).size),
            "num_unique_undirected_edges": total_edges,
            "edge_gt_same_rate": float(same_gt.mean()),
            "edge_gt_same_rate_pct": float(same_gt.mean() * 100),
        },
        "results": {},
    }

    for k in args.k:
        if k < 2 or k > n:
            raise ValueError(f"Invalid K: {k}")
        methods = {}
        for name, feats in [("X", x), ("A_tilde_X", ax)]:
            model = MiniBatchKMeans(n_clusters=k, random_state=args.seed, batch_size=args.batch_size, max_iter=args.max_iter, n_init=args.n_init)
            assignment = model.fit_predict(feats)
            methods[name] = assignment

        cx, ca = methods["X"], methods["A_tilde_X"]
        mx = cx[src] == cx[dst]
        ma = ca[src] == ca[dst]
        conditions = {
            "edge_only": np.ones(total_edges, dtype=bool),
            "edge_and_X_cluster": mx,
            "edge_and_A_tilde_X_cluster": ma,
            "edge_and_both_clusters": mx & ma,
            "edge_and_either_cluster": mx | ma,
        }
        stats = {name: calculate_stats(mask, same_gt, total_edges) for name, mask in conditions.items()}
        report["results"][str(k)] = {
            "edge_conditions": stats,
            "X_cluster_pairs_without_edge_requirement": within_cluster_pairs(cx, y),
            "A_tilde_X_cluster_pairs_without_edge_requirement": within_cluster_pairs(ca, y),
            "cluster_sizes": {"X": np.bincount(cx, minlength=k).tolist(), "A_tilde_X": np.bincount(ca, minlength=k).tolist()},
        }
        print(f"K={k}, edge-only GT same = {100 * same_gt.mean():.2f}%")
        for name, s in stats.items():
            print(
                f"  {name}: edges={s['selected_edges']}, "
                f"GT match={s['gt_same_rate_pct']:.2f}% "
                f"(lift={s['lift_over_edge_baseline']:.3f}, "
                f"GT-edge recall={s['recall_of_gt_same_edges']:.3f})"
                if s["gt_same_rate_pct"] is not None
                else f"  {name}: no selected edges"
            )

    output = Path(__file__).resolve().parent / "exp059.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
