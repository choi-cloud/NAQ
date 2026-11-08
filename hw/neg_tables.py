"""HW06 G-table: candidate statistics of extra_neg far / band / cluster, before training.

    python -m hw.neg_tables [--out results/hw_iislab4/HW06_tables.json]

Builds (or loads) the caches cache/rawcos/<ds>_top500.pt and cache/kmeans/<ds>_k100.pt (hw.negatives) and writes
the G-table values for both datasets. Deterministic, no training. measurement-only label use (class ratios).
Episode exclusions (5 supports + 50 pseudo-queries) are not applied here; at most 55 more nodes are excluded per query.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import load_data
from hw import structure
from hw.config import build_args, load_config
from hw.negatives import BAND_DEFAULT, FAR_TOP, KMEANS_DEFAULT_K, adjacency_torch, hard_table, kmeans_labels, raw_cos_table, within_two_hops

DATASETS = ("Cora_full", "Amazon_clothing")
EPISODE_EXCLUDED_MAX = 55
BAND_STOP_ZERO_FRAC = 0.20
FAR_MIN_CANDIDATES = 1000
CHUNK = 1024


def _quartiles(x):
    q = np.quantile(np.asarray(x, dtype=np.float64), [0.0, 0.25, 0.5, 0.75, 1.0])
    return {"min": float(q[0]), "q1": float(q[1]), "median": float(q[2]), "q3": float(q[3]), "max": float(q[4])}


@torch.no_grad()
def tables(dataset):
    cfg = load_config("naq_base")
    args = build_args(cfg, dataset)
    data = load_data(args)  # CPU
    features = data.features.float()
    labels = torch.as_tensor(np.asarray(data.labels.cpu() if torch.is_tensor(data.labels) else data.labels)).long()
    n = features.shape[0]
    lo, hi = BAND_DEFAULT
    raw = raw_cos_table(dataset, features)
    ids, cos = raw["ids"], raw["cos"]

    # band (ranks lo..hi, 1-based; self excluded by the table)
    band_ids, band_cos = ids[:, lo - 1 : hi], cos[:, lo - 1 : hi]
    positive_in_band = (band_cos > 0).sum(dim=1)
    same_band = (labels[band_ids] == labels[:, None]).float().mean(dim=1)  # measurement-only label use
    band = {
        "ranks": [lo, hi],
        "candidates_per_node": hi - lo + 1,
        "after_max_episode_exclusion": hi - lo + 1 - EPISODE_EXCLUDED_MAX,
        "cos_positive_count_min": int(positive_in_band.min()),
        "cos_positive_count_median": float(positive_in_band.float().median()),
        "cos_at_rank_lo": _quartiles(cos[:, lo - 1].numpy()),
        "cos_at_rank_hi": _quartiles(cos[:, hi - 1].numpy()),
        "rank_hi_cos_zero_frac": float((cos[:, hi - 1] <= 0).float().mean()),
        "same_class_node_macro": float(same_band.mean()),  # measurement-only label use
    }
    band["stop"] = band["rank_hi_cos_zero_frac"] > BAND_STOP_ZERO_FRAC

    # far: N - |{≤2 hops of q} ∪ top-100 raw cos of q|
    a = adjacency_torch(data.edge_index, n, "cpu")
    top = ids[:, :FAR_TOP]
    remaining = torch.empty(n, dtype=torch.long)
    for start in range(0, n, CHUNK):
        rows = torch.arange(start, min(n, start + CHUNK))
        banned = within_two_hops(a, rows)
        banned.scatter_(1, top[rows], True)
        remaining[rows] = n - banned.sum(dim=1)
    # cross-check within_two_hops against hw.structure.hop_sets_dense on 200 nodes
    a_sp = structure.normalize_graph(data.edge_index.detach().cpu().numpy(), n)
    check_rows = np.arange(0, n, max(1, n // 200))[:200]
    h1, h2, _ = structure.hop_sets_dense(a_sp, check_rows)
    ref = h1 | h2
    ref[np.arange(len(check_rows)), check_rows] = True
    hop_match = bool((within_two_hops(a, torch.as_tensor(check_rows)).numpy() == ref).all())
    hard_top = hard_table(structure.l2_rows(features), k=FAR_TOP)
    far = {
        "remaining_min": int(remaining.min()),
        "remaining_median": float(remaining.float().median()),
        "remaining_min_after_max_episode_exclusion": int(remaining.min()) - EPISODE_EXCLUDED_MAX,
        "hop_check_vs_structure_hop_sets_dense": hop_match,
        "top100_equal_to_hard_raw_table_frac": float((torch.sort(hard_top, dim=1).values == torch.sort(top, dim=1).values).all(dim=1).float().mean()),
    }
    far["stop"] = far["remaining_min"] - EPISODE_EXCLUDED_MAX < FAR_MIN_CANDIDATES

    # cluster
    k = KMEANS_DEFAULT_K
    clus = kmeans_labels(dataset, features, data.adj, k)
    sizes = torch.bincount(clus, minlength=k)
    n_class = int(labels.max()) + 1
    cont = torch.zeros((k, n_class), dtype=torch.long)
    cont.index_put_((clus, labels), torch.ones(n, dtype=torch.long), accumulate=True)  # measurement-only label use
    class_sizes = cont.sum(dim=0)
    n_kc = cont[clus, labels].float()
    size_k = sizes[clus].float()
    same_in = torch.where(size_k > 1, (n_kc - 1) / (size_k - 1).clamp_min(1), torch.zeros_like(n_kc))
    same_out = (class_sizes[labels].float() - n_kc) / (n - size_k)
    cluster = {
        "k": k,
        "size_min": int(sizes.min()),
        "size_median": float(sizes.float().median()),
        "size_max": int(sizes.max()),
        "same_class_within_cluster_node_macro": float(same_in.mean()),  # measurement-only label use
        "same_class_other_clusters_node_macro": float(same_out.mean()),  # measurement-only label use
    }
    p = class_sizes.float() / n
    random_fn = float((p * p).sum())  # measurement-only label use
    return {"dataset": dataset, "num_nodes": n, "band": band, "far": far, "cluster": cluster, "random_expected_fn_sum_pc2": random_fn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/hw_iislab4/HW06_tables.json")
    out = {ds: tables(ds) for ds in DATASETS}
    out["G_table"] = {
        "band_rank_hi_cos_zero_frac_max": BAND_STOP_ZERO_FRAC,
        "far_min_candidates": FAR_MIN_CANDIDATES,
        "stop": any(out[ds]["band"]["stop"] or out[ds]["far"]["stop"] for ds in DATASETS),
    }
    out["G_table"]["pass"] = not out["G_table"]["stop"]
    path = Path(parser.parse_args().out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
