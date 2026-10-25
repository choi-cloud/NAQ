"""exp060: Compare cluster quality and 1-hop GT homophily on Cora-Full.

All clustering is unsupervised. GT labels are used only for diagnostics.
Run from the original NAQ repository root; original source files are untouched.
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
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Cora_full")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--k", type=int, nargs="+", default=[70])
    p.add_argument("--svd_dim", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--max_iter", type=int, default=200)
    p.add_argument("--n_init", type=int, default=3)
    return p.parse_args()


def edge_stats(mask, same_gt):
    n = len(same_gt)
    count = int(mask.sum())
    pos = int(np.count_nonzero(mask & same_gt))
    true_total = int(same_gt.sum())
    precision = pos / count if count else None
    baseline = true_total / n
    return {
        "selected_edges": count,
        "edge_coverage": count / n,
        "gt_same_edges": pos,
        "gt_different_edges": count - pos,
        "gt_same_rate": precision,
        "gt_same_rate_pct": 100 * precision if precision is not None else None,
        "lift_over_edge_baseline": precision / baseline if precision is not None else None,
        "recall_of_gt_same_edges": pos / true_total if true_total else None,
    }


def pair_purity(c, y):
    n = len(y)
    selected = 0
    same = 0
    for cl in np.unique(c):
        ys = y[c == cl]
        sz = len(ys)
        selected += sz * (sz - 1) // 2
        _, cnt = np.unique(ys, return_counts=True)
        same += int(np.sum(cnt * (cnt - 1) // 2))
    all_cnt = np.unique(y, return_counts=True)[1]
    base = np.sum(all_cnt * (all_cnt - 1) // 2) / (n * (n - 1) / 2)
    precision = same / selected if selected else None
    return {
        "pairs": int(selected),
        "gt_same_pairs": same,
        "gt_same_rate": precision,
        "random_pair_gt_same_rate": float(base),
        "lift_over_random": precision / base if precision is not None else None,
    }


def cluster_diagnostics(c, k, y):
    sizes = np.bincount(c, minlength=k)
    occupied = sizes[sizes > 0]
    p = occupied / len(c)
    entropy = float(-(p * np.log(p)).sum())
    return {
        "sizes": sizes.tolist(),
        "occupied_clusters": int(len(occupied)),
        "singleton_clusters": int((sizes == 1).sum()),
        "largest_cluster": int(sizes.max()),
        "largest_cluster_fraction": float(sizes.max() / len(c)),
        "top2_cluster_fraction": float(np.sort(sizes)[-2:].sum() / len(c)),
        "effective_cluster_count_exp_entropy": float(np.exp(entropy)),
        "normalized_mutual_info_gt_diagnostic": float(normalized_mutual_info_score(y, c)),
        "adjusted_rand_gt_diagnostic": float(adjusted_rand_score(y, c)),
        "nonedge_restricted_pair_purity": pair_purity(c, y),
    }


def build_cluster(feat, k, args):
    model = MiniBatchKMeans(n_clusters=k, random_state=args.seed, batch_size=args.batch_size, max_iter=args.max_iter, n_init=args.n_init)
    return model.fit_predict(feat)


def main():
    args = parse_args()
    if args.dataset != "Cora_full":
        raise ValueError("exp060 is intended for Cora_full")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = load_data(args)
    y = np.asarray(data.labels.detach().cpu().numpy()).reshape(-1)
    x = data.features.detach().cpu().numpy().astype(np.float32, copy=False)
    n = len(y)
    raw_edges = data.edge_index.detach().cpu().numpy()
    u = np.minimum(raw_edges[0], raw_edges[1]).astype(np.int64)
    v = np.maximum(raw_edges[0], raw_edges[1]).astype(np.int64)
    good = u != v
    pairs = np.unique(np.stack([u[good], v[good]], axis=1), axis=0)
    src, dst = pairs[:, 0], pairs[:, 1]
    same_gt = y[src] == y[dst]
    adj = data.adj.coalesce().cpu()
    ix = adj.indices().numpy()
    a = sp.coo_matrix((adj.values().numpy(), (ix[0], ix[1])), shape=(n, n)).tocsr()
    ax = np.asarray(a @ x, dtype=np.float32)

    # SVD is fitted separately to each feature space; no labels are involved.
    representations = {}
    for name, feats in [("X", x), ("A_tilde_X", ax)]:
        dim = min(args.svd_dim, min(feats.shape) - 1)
        print(f"Preparing representations: {name}, SVD dim={dim}", flush=True)
        svd = TruncatedSVD(n_components=dim, n_iter=5, random_state=args.seed)
        reduced = svd.fit_transform(feats).astype(np.float32)
        representations[name] = {
            "raw_euclidean": feats,
            "raw_spherical": normalize(feats, norm="l2", copy=True),
            "svd_euclidean": reduced,
            "svd_spherical": normalize(reduced, norm="l2", copy=True),
        }
        del svd

    report = {
        "experiment": "exp060",
        "status": "running",
        "args": vars(args),
        "definition": {
            "purpose": "GT diagnostics of clustering quality and homophilic 1-hop edge selection",
            "X": "Raw node features",
            "A_tilde_X": "Original NAQ data.adj (with self loops) @ raw X",
            "raw_spherical": "L2 row normalization followed by MiniBatchKMeans (cosine-style proxy, not exact spherical k-means)",
            "svd_euclidean": "TruncatedSVD then MiniBatchKMeans",
            "svd_spherical": "TruncatedSVD then L2 normalization and MiniBatchKMeans (cosine-style proxy)",
            "edge": "Unique undirected non-self 1-hop pair",
            "GT_use": "Only for post-clustering diagnostics; no GT in SVD/KMeans",
            "joint": "Same cluster in both independently fitted X and AX assignments; cluster IDs need not align",
            "warning": "L2-normalized Euclidean KMeans approximates angular grouping but is not an exact spherical KMeans objective",
        },
        "graph": {"nodes": n, "gt_classes": int(np.unique(y).size), "edges": int(len(src)), "edge_gt_same_rate": float(same_gt.mean())},
        "results": {},
    }
    output = Path(__file__).resolve().parent / "exp060.json"
    for k in args.k:
        if not (2 <= k <= n):
            raise ValueError(f"Invalid k={k}")
        result_k = {"methods": {}, "joint_by_representation": {}}
        labels_by_rep = {}
        for rep in ["raw_euclidean", "raw_spherical", "svd_euclidean", "svd_spherical"]:
            labels_by_rep[rep] = {}
            for space in ["X", "A_tilde_X"]:
                print(f"K={k} {space}/{rep}: clustering...", flush=True)
                clusters = build_cluster(representations[space][rep], k, args)
                labels_by_rep[rep][space] = clusters
                mask = clusters[src] == clusters[dst]
                result_k["methods"][f"{space}/{rep}"] = {
                    "cluster": cluster_diagnostics(clusters, k, y),
                    "edge_and_same_cluster": edge_stats(mask, same_gt),
                }
                s = result_k["methods"][f"{space}/{rep}"]
                print(
                    f"  precision={s['edge_and_same_cluster']['gt_same_rate_pct']:.2f}% "
                    f"coverage={s['edge_and_same_cluster']['edge_coverage']:.3f} "
                    f"largest={s['cluster']['largest_cluster_fraction']:.3f}",
                    flush=True,
                )
            cx = labels_by_rep[rep]["X"]
            ca = labels_by_rep[rep]["A_tilde_X"]
            mx = cx[src] == cx[dst]
            ma = ca[src] == ca[dst]
            result_k["joint_by_representation"][rep] = {"both": edge_stats(mx & ma, same_gt), "either": edge_stats(mx | ma, same_gt)}
        report["results"][str(k)] = result_k
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"K={k} saved intermediate results", flush=True)
    report["status"] = "completed"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"DONE: {output}", flush=True)


if __name__ == "__main__":
    main()
