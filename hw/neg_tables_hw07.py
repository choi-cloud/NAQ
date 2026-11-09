"""HW07 G-table: seed-2 k-means clusters, far_strong candidates, shuffled cluster-size control. Before training.

    python -m hw.neg_tables_hw07 [--out results/hw_iislab4/HW07_tables.json]

Builds (once) cache/kmeans/<ds>_k100_s20261109.pt and cache/rawcos/<ds>_top1000.pt; existing caches are only read.
measurement-only label use: same-class ratios. Episode exclusions (≤ 55 nodes per query) are not applied here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from data import load_data
from hw.config import build_args, load_config
from hw.negatives import (
    FAR_STRONG_TABLE_K, KMEANS_DEFAULT_K, KMEANS_SEED, adjacency_torch, kmeans_labels, kmeans_labels_seed, raw_cos_table, within_two_hops,
)

DATASETS = ("Cora_full", "Amazon_clothing")
KMEANS_SEED_2 = 20261109
EPISODE_EXCLUDED_MAX = 55
FAR_MIN_CANDIDATES = 1000
SHUFFLE_DRAWS = 10000
SHUFFLE_CHECK_SEED = 20261109  # dedicated CPU generator for the G-table draw
SHUFFLE_STOP_REL = 0.10
CHUNK = 1024


def _dist(x):
    x = np.asarray(x, dtype=np.float64)
    return {"mean": float(x.mean()), "median": float(np.median(x)), "q90": float(np.quantile(x, 0.9))}


def _cluster_stats(clus, labels, k):
    sizes = torch.bincount(clus, minlength=k)
    n_class = int(labels.max()) + 1
    cont = torch.zeros((k, n_class), dtype=torch.long)
    cont.index_put_((clus, labels), torch.ones(labels.numel(), dtype=torch.long), accumulate=True)  # measurement-only label use
    n_kc = cont[clus, labels].float()
    size_k = sizes[clus].float()
    same_in = torch.where(size_k > 1, (n_kc - 1) / (size_k - 1).clamp_min(1), torch.zeros_like(n_kc))
    return {
        "size_min": int(sizes.min()), "size_median": float(sizes.float().median()), "size_max": int(sizes.max()),
        "same_class_within_cluster_node_macro": float(same_in.mean()),  # measurement-only label use
    }


@torch.no_grad()
def tables(dataset):
    args = build_args(load_config("naq_base"), dataset)
    data = load_data(args)  # CPU
    features = data.features.float()
    labels = torch.as_tensor(np.asarray(data.labels.cpu() if torch.is_tensor(data.labels) else data.labels)).long()
    n = features.shape[0]
    k = KMEANS_DEFAULT_K

    # seed-2 clusters
    ref = kmeans_labels(dataset, features, data.adj, k)
    c2 = kmeans_labels_seed(dataset, features, data.adj, k, KMEANS_SEED_2)
    seed2 = {"kmeans_seed": KMEANS_SEED_2, **_cluster_stats(c2, labels, k), "ari_vs_seed_%d" % KMEANS_SEED: float(adjusted_rand_score(ref.numpy(), c2.numpy()))}
    ref_stats = _cluster_stats(ref, labels, k)

    # far_strong
    t1000 = raw_cos_table(dataset, features, k=FAR_STRONG_TABLE_K)
    t500 = raw_cos_table(dataset, features)
    a = adjacency_torch(data.edge_index, n, "cpu")
    top = t1000["ids"]
    remaining = torch.empty(n, dtype=torch.long)
    for start in range(0, n, CHUNK):
        rows = torch.arange(start, min(n, start + CHUNK))
        banned = within_two_hops(a, rows)
        banned.scatter_(1, top[rows], True)
        remaining[rows] = n - banned.sum(dim=1)
    head = top[:, :500]
    exact = (head == t500["ids"]).all(dim=1)
    same_set = (torch.sort(head, dim=1).values == torch.sort(t500["ids"], dim=1).values).all(dim=1)
    x = torch.nn.functional.normalize(features, p=2, dim=1)
    tie_only = 0
    bad = (~same_set).nonzero().flatten().tolist()
    for i in bad:
        diff = list(set(head[i].tolist()) ^ set(t500["ids"][i].tolist()))
        cs = x[i] @ x[diff].T
        tie_only += bool(torch.allclose(cs, torch.full_like(cs, float(t500["cos"][i, 499])), atol=1e-6))
    cos_close = float((t1000["cos"][:, :500] - t500["cos"]).abs().max())
    far_strong = {
        "far_top": FAR_STRONG_TABLE_K,
        "remaining_min": int(remaining.min()),
        "remaining_median": float(remaining.float().median()),
        "remaining_min_after_max_episode_exclusion": int(remaining.min()) - EPISODE_EXCLUDED_MAX,
        "top1000_head500_equal_order_frac": float(exact.float().mean()),
        "top1000_head500_equal_set_frac": float(same_set.float().mean()),
        "set_mismatch_nodes": len(bad),
        "set_mismatch_nodes_rank500_tie_only": tie_only,
        "head500_cos_max_abs_diff": cos_close,
    }
    far_strong["stop"] = far_strong["remaining_min_after_max_episode_exclusion"] < FAR_MIN_CANDIDATES

    # shuffled control: node-weighted |C(q)| vs |C(u)|, u uniform outside C(q)
    sizes = torch.bincount(ref, minlength=k)
    node_weighted = sizes[ref].numpy()
    g = torch.Generator()
    g.manual_seed(SHUFFLE_CHECK_SEED)
    q = torch.randint(0, n, (SHUFFLE_DRAWS,), generator=g)
    drawn = []
    for start in range(0, SHUFFLE_DRAWS, 500):
        qs = q[start : start + 500]
        outside = ref[None, :] != ref[qs][:, None]
        u = torch.rand((qs.numel(), n), generator=g).masked_fill(~outside, -1.0).argmax(dim=1)
        drawn.append(sizes[ref[u]])
    drawn = torch.cat(drawn).numpy()
    shuffled = {"node_weighted_cluster_size": _dist(node_weighted), "shuffled_cluster_size": _dist(drawn), "draws": SHUFFLE_DRAWS, "draw_seed": SHUFFLE_CHECK_SEED}
    shuffled["mean_rel_diff"] = abs(shuffled["shuffled_cluster_size"]["mean"] - shuffled["node_weighted_cluster_size"]["mean"]) / shuffled["node_weighted_cluster_size"]["mean"]
    shuffled["stop"] = shuffled["mean_rel_diff"] > SHUFFLE_STOP_REL
    return {"dataset": dataset, "num_nodes": n, "reference_clusters": ref_stats, "seed2_clusters": seed2, "far_strong": far_strong, "shuffled": shuffled}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/hw_iislab4/HW07_tables.json")
    out = {ds: tables(ds) for ds in DATASETS}
    stop = any(out[ds]["far_strong"]["stop"] or out[ds]["shuffled"]["stop"] for ds in DATASETS)
    out["G_table"] = {"far_min_candidates": FAR_MIN_CANDIDATES, "shuffled_mean_rel_diff_max": SHUFFLE_STOP_REL, "stop": stop, "pass": not stop}
    path = Path(parser.parse_args().out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
