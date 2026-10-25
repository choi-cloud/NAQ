"""exp067: 4-level nested clustering and pairwise GT agreement diagnostics.

Run in the original NAQ repository. No labels are used to fit clusters.
No 1-hop edge condition is imposed in any pair diagnostic.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize

from data import load_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Cora_full")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--k", nargs=4, type=int, default=[10, 20, 40, 80])
    parser.add_argument("--svd_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--n_init", type=int, default=3)
    return parser.parse_args()


def choose2(counts):
    counts = np.asarray(counts, dtype=np.int64)
    return int(np.sum(counts * (counts - 1) // 2))


def pair_counts(labels, gt):
    _, inverse, sizes = np.unique(labels, return_inverse=True, return_counts=True)
    chosen = choose2(sizes)
    _, gt_inverse = np.unique(gt, return_inverse=True)
    num_gt = int(gt_inverse.max()) + 1
    joint = np.bincount(inverse.astype(np.int64) * num_gt + gt_inverse, minlength=len(sizes) * num_gt)
    return chosen, choose2(joint), sizes


def stats(labels, gt, baseline, all_pairs, all_same, requested_k):
    selected, same, sizes = pair_counts(labels, gt)
    p = sizes / sizes.sum()
    purity = same / selected if selected else None
    return {
        "requested_k": int(requested_k),
        "occupied_clusters": int(len(sizes)),
        "cluster_sizes": sizes.astype(int).tolist(),
        "singleton_clusters": int(np.sum(sizes == 1)),
        "largest_cluster_fraction": float(sizes.max() / len(gt)),
        "effective_cluster_count": float(np.exp(-np.sum(p * np.log(p)))),
        "nmi_gt_diagnostic": float(normalized_mutual_info_score(gt, labels)),
        "ari_gt_diagnostic": float(adjusted_rand_score(gt, labels)),
        "same_cluster_pairs": selected,
        "gt_same_pairs": same,
        "gt_different_pairs": selected - same,
        "gt_same_rate": purity,
        "gt_same_rate_pct": 100 * purity if purity is not None else None,
        "pair_coverage": selected / all_pairs,
        "recall_of_all_gt_same_pairs": same / all_same if all_same else None,
        "lift_over_random_pair": purity / baseline if purity is not None else None,
    }


def fit_kmeans(features, k, seed, args):
    model = MiniBatchKMeans(
        n_clusters=k, random_state=seed, batch_size=min(args.batch_size, len(features)), max_iter=args.max_iter, n_init=args.n_init
    )
    return model.fit_predict(features)


def nested_partitions(features, levels, args):
    # L1: K=10. Later levels split *every* parent into two children.
    current = fit_kmeans(features, levels[0], args.seed, args)
    history = [current.copy()]
    for depth, target_k in enumerate(levels[1:], start=1):
        parent_ids = np.unique(current)
        if target_k != len(parent_ids) * 2:
            raise ValueError("Nested implementation requires each level to double the occupied parent count")
        next_labels = np.empty_like(current)
        for parent in parent_ids:
            idx = np.flatnonzero(current == parent)
            if len(idx) < 2:
                raise ValueError(f"Cannot split parent cluster {parent} at level {depth + 1}: " f"only {len(idx)} node(s)")
            child = fit_kmeans(features[idx], 2, args.seed + depth * 1000 + int(parent), args)
            if len(np.unique(child)) != 2:
                raise RuntimeError(f"KMeans returned fewer than two occupied children for parent {parent}")
            next_labels[idx] = int(parent) * 2 + child
        current = next_labels
        history.append(current.copy())
        print(f"Nested L{depth + 1}: {len(np.unique(current))} occupied clusters", flush=True)
    return history


def exact_depth_stats(history, gt):
    # Because partitions are nested, pairs sharing exactly d levels = S_d - S_(d+1).
    n = len(gt)
    total = n * (n - 1) // 2
    same_gt = choose2(np.unique(gt, return_counts=True)[1])
    counts = [(total, same_gt)]
    for labels in history:
        selected, same, _ = pair_counts(labels, gt)
        counts.append((selected, same))
    result = {}
    for depth in range(len(history) + 1):
        if depth < len(history):
            count = counts[depth][0] - counts[depth + 1][0]
            same = counts[depth][1] - counts[depth + 1][1]
        else:
            count, same = counts[depth]
        result[str(depth)] = {
            "shared_levels_exactly": depth,
            "pairs": count,
            "gt_same_pairs": same,
            "gt_same_rate": same / count if count else None,
            "gt_same_rate_pct": 100 * same / count if count else None,
        }
    return result


def main():
    args = parse_args()
    if args.dataset != "Cora_full":
        raise ValueError("exp067 is configured for Cora_full")
    levels = list(args.k)
    if levels != [10, 20, 40, 80]:
        raise ValueError("This 4-level binary-split experiment uses --k 10 20 40 80")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_data(args)
    gt = data.labels.detach().cpu().numpy().reshape(-1)
    x = data.features.detach().cpu().numpy().astype(np.float32, copy=False)
    n = len(gt)
    adj = data.adj.coalesce().cpu()
    indices = adj.indices().numpy()
    a = sp.coo_matrix((adj.values().numpy(), (indices[0], indices[1])), shape=(n, n)).tocsr()
    ax = np.asarray(a @ x, dtype=np.float32)
    dim = min(args.svd_dim, min(ax.shape) - 1)
    reduced = TruncatedSVD(n_components=dim, n_iter=5, random_state=args.seed).fit_transform(ax).astype(np.float32)
    features = normalize(reduced, norm="l2", copy=True)
    del ax, reduced, x

    total = n * (n - 1) // 2
    gt_sizes = np.unique(gt, return_counts=True)[1]
    all_same = choose2(gt_sizes)
    base_rate = all_same / total
    report = {
        "experiment": "exp067",
        "status": "running",
        "args": vars(args),
        "definitions": {
            "representation": "L2_normalize(TruncatedSVD_128(NAQ data.adj @ raw X))",
            "hierarchical": "MiniBatchKMeans K10, then binary splitting each parent at K20, K40, K80",
            "independent": "Separate MiniBatchKMeans fits at K10, K20, K40, K80",
            "pair_population": "All unordered distinct node pairs; no graph edge restriction",
            "cumulative_level": "Both nodes are in the same cluster at that level (and all earlier levels for hierarchical)",
            "exact_shared_depth": "Number of consecutive nested levels both nodes share, from 0 to 4",
            "GT": "Only diagnostics; never used in SVD or clustering",
            "note": "L2-normalized MiniBatchKMeans is a cosine-style proxy, not exact spherical k-means",
        },
        "graph": {
            "num_nodes": n,
            "num_gt_classes": int(np.unique(gt).size),
            "all_unordered_pairs": total,
            "all_gt_same_pairs": all_same,
            "random_pair_gt_same_rate": base_rate,
        },
        "hierarchical": {},
        "independent": {},
        "hierarchical_exact_shared_depth": {},
    }
    output = Path(__file__).resolve().parent / "exp067.json"
    print("Fitting nested hierarchy...", flush=True)
    history = nested_partitions(features, levels, args)
    for depth, labels in enumerate(history):
        k = levels[depth]
        report["hierarchical"][str(k)] = stats(labels, gt, base_rate, total, all_same, k)
        s = report["hierarchical"][str(k)]
        print(f"Nested K{k}: GT same={s['gt_same_rate_pct']:.2f}% coverage={s['pair_coverage']:.4f}", flush=True)
    report["hierarchical_exact_shared_depth"] = exact_depth_stats(history, gt)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k in levels:
        print(f"Independent K{k}...", flush=True)
        labels = fit_kmeans(features, k, args.seed, args)
        report["independent"][str(k)] = stats(labels, gt, base_rate, total, all_same, k)
        s = report["independent"][str(k)]
        print(f"Independent K{k}: GT same={s['gt_same_rate_pct']:.2f}% coverage={s['pair_coverage']:.4f}", flush=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["status"] = "completed"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"DONE: {output}", flush=True)


if __name__ == "__main__":
    main()
