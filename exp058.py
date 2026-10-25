"""exp058: label-only diagnostics for HUGE HALO query filtering on Cora-Full.

No training; GT is used only to evaluate query purity and pair classification.
NAQ random anchors and NAQ-Feat cosine ranking are preserved.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score

from data import load_data
from task_generator import spt_sampling
from utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Cora_full")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--q_query", type=int, default=10)
    p.add_argument("--num_episodes", type=int, default=200)
    p.add_argument("--full_anchors", type=int, default=50)
    p.add_argument("--pool_sizes", nargs="+", type=int, default=[20, 50, 100])
    p.add_argument("--halo_chunk", type=int, default=2048)
    p.add_argument("--epsilon", type=float, default=1e-8)
    p.add_argument("--tie_tolerance", type=float, default=1e-6)
    return p.parse_args()


def save_json(path, obj):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def halo_for_candidates(x, anchor, indices, chunk=2048, epsilon=1e-8):
    """Exact HUGE Eq. (2) on original input features, via CSR sparse operations."""
    center = x.getrow(int(anchor))
    results = np.empty(len(indices), dtype=np.float64)
    for lo in range(0, len(indices), chunk):
        hi = min(lo + chunk, len(indices))
        other = x[indices[lo:hi]]
        center_batch = sp.vstack([center] * other.shape[0], format="csr")
        diff_abs = abs(other - center_batch)
        xa = diff_abs.multiply(center_batch)
        xb = diff_abs.multiply(other)
        numer_sq = np.asarray((xa - xb).power(2).sum(axis=1)).reshape(-1)
        denom_sq = np.asarray(xa.power(2).sum(axis=1) + xb.power(2).sum(axis=1)).reshape(-1)
        results[lo:hi] = np.sqrt(np.maximum(numer_sq, 0)) / np.sqrt(denom_sq + epsilon)
    return results


def calc_stats(records):
    if not records:
        return None
    fields = [
        "base_precision",
        "halo_precision",
        "base_true",
        "halo_true",
        "base_false",
        "halo_false",
        "removed_false",
        "added_true",
        "false_removal_fraction",
        "retained_true_fraction",
        "mean_halo",
        "std_halo",
        "tied_fraction",
        "pair_auc",
        "selection_overlap",
    ]
    out = {"anchors": len(records)}
    for key in fields:
        vals = np.array([r[key] for r in records if r[key] is not None], dtype=float)
        out[key + "_mean"] = float(vals.mean()) if len(vals) else None
    out["pooled_precision_base"] = float(sum(r["base_true"] for r in records) / sum(r["k"] for r in records))
    out["pooled_precision_halo"] = float(sum(r["halo_true"] for r in records) / sum(r["k"] for r in records))
    out["precision_gain_pp"] = 100 * (out["pooled_precision_halo"] - out["pooled_precision_base"])
    out["number_pair_auc_defined"] = sum(r["pair_auc"] is not None for r in records)
    out["total_base_false_queries"] = int(sum(r["base_false"] for r in records))
    out["total_halo_false_queries"] = int(sum(r["halo_false"] for r in records))
    return out


def main():
    args = parse_args()
    if args.dataset != "Cora_full" or args.n_way != 5 or args.q_query != 10:
        raise ValueError("exp058 expects Cora_full, 5-way, 10 queries per anchor")
    if args.num_episodes <= 0 or args.full_anchors < 0 or args.halo_chunk <= 0:
        raise ValueError("invalid argument")
    if any(k < args.q_query for k in args.pool_sizes):
        raise ValueError("pool_sizes must be >= q_query")
    args.pool_sizes = sorted(set(args.pool_sizes))
    args.setting = "unsup"
    args.query_generation = "NaQ"
    args.type = "feature"
    args.k_shot = 1
    args.k_shot_test = 1
    args.epoch = 1
    root = Path(__file__).resolve().parent
    output = root / "exp058.json"
    report = {
        "experiment": "exp058",
        "status": "initializing",
        "args": vars(args).copy(),
        "definition": {
            "source": "HUGE Eq. (2): HALO(x_i,x_j) = ||abs(x_i-x_j)*x_i - abs(x_i-x_j)*x_j|| / sqrt(||abs(x_i-x_j)*x_i||^2 + ||abs(x_i-x_j)*x_j||^2 + epsilon)",
            "features": "Original NAQ data.features (no GNN embedding)",
            "sampling": "Original NAQ spt_sampling, GT class collisions allowed",
            "baseline": "Original NAQ feature-cosine top-10 candidates (excluding all five supports)",
            "reranking": "Choose 10 smallest HALO scores from cosine top-20/50/100; also full eligible graph on limited anchors",
            "GT": "Used only for diagnostics; never candidate filtering or training",
            "tie_breaking": "HALO ascending, cosine descending, node ID ascending",
            "caveat": "Full-pool results are on the first full_anchors sampled anchor instances; compare each to its own baseline",
        },
        "results": {},
    }
    save_json(output, report)
    seed_everything(args.seed)
    data = load_data(args)
    labels = np.asarray(torch.as_tensor(data.labels).detach().cpu()).reshape(-1)
    features = data.features.detach().cpu().numpy().astype(np.float64, copy=False)
    x = sp.csr_matrix(features)
    n = x.shape[0]
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        data.set_device(f"cuda:{args.device}")
    data.add_diffusion(args)
    sim = data.Diffusion_matrix
    if sim.layout == torch.sparse_coo:
        raise ValueError("Expected dense NAQ-Feat similarity")
    if sim.device.type == "cuda":
        torch.cuda.synchronize()
    supports = [spt_sampling(data.id_train, args.n_way, 1)[0] for _ in range(args.num_episodes)]
    records = {str(k): [] for k in args.pool_sizes}
    records["full"] = []
    num_full = 0
    num_anchor = 0
    all_ids = np.arange(n, dtype=np.int64)
    for ep, ids in enumerate(supports, 1):
        selected = np.asarray(ids, dtype=np.int64).reshape(-1)
        for anchor in selected:
            anchor = int(anchor)
            num_anchor += 1
            # Same feature-cosine rank as NAQ-Feat; exclude all episode supports.
            sims = sim[anchor].detach().cpu().numpy().astype(np.float64, copy=True)
            sims[selected] = -np.inf
            max_pool = max(args.pool_sizes)
            top_ids = np.argpartition(-sims, max_pool - 1)[:max_pool]
            top_ids = top_ids[np.lexsort((top_ids, -sims[top_ids]))]
            # For pools, compute HALO once on the largest candidate set.
            top_halo = halo_for_candidates(x, anchor, top_ids, args.halo_chunk, args.epsilon)
            scopes = [(str(k), top_ids[:k], top_halo[:k]) for k in args.pool_sizes]
            if num_full < args.full_anchors:
                eligible = all_ids[~np.isin(all_ids, selected)]
                full_halo = halo_for_candidates(x, anchor, eligible, args.halo_chunk, args.epsilon)
                scopes.append(("full", eligible, full_halo))
                num_full += 1
            for name, pool, halo in scopes:
                k = args.q_query
                pool_sim = sims[pool]
                baseline_order = np.lexsort((pool, -pool_sim))
                filtered_order = np.lexsort((pool, -pool_sim, halo))
                base = pool[baseline_order[:k]]
                chosen = pool[filtered_order[:k]]
                base_good = labels[base] == labels[anchor]
                chosen_good = labels[chosen] == labels[anchor]
                actual = labels[pool] == labels[anchor]
                auc = None
                if actual.any() and (~actual).any():
                    auc = float(roc_auc_score(actual.astype(int), -halo))
                tol = args.tie_tolerance
                # fraction of pool pairs indistinguishable from a typical score (rounded bins)
                bins = np.round(halo / tol).astype(np.int64)
                _, cnt = np.unique(bins, return_counts=True)
                tied_fraction = float(cnt[cnt > 1].sum() / len(pool))
                records[name].append(
                    {
                        "k": k,
                        "base_precision": float(base_good.mean()),
                        "halo_precision": float(chosen_good.mean()),
                        "base_true": int(base_good.sum()),
                        "halo_true": int(chosen_good.sum()),
                        "base_false": int((~base_good).sum()),
                        "halo_false": int((~chosen_good).sum()),
                        "removed_false": int(len(set(base[~base_good]) - set(chosen))),
                        "added_true": int(len(set(chosen[chosen_good]) - set(base))),
                        "false_removal_fraction": float(len(set(base[~base_good]) - set(chosen)) / max((~base_good).sum(), 1)),
                        "retained_true_fraction": float(len(set(base[base_good]) & set(chosen)) / max(base_good.sum(), 1)),
                        "mean_halo": float(np.mean(halo)),
                        "std_halo": float(np.std(halo)),
                        "tied_fraction": tied_fraction,
                        "pair_auc": auc,
                        "selection_overlap": float(len(set(base) & set(chosen)) / k),
                    }
                )
        if ep % 20 == 0 or ep == args.num_episodes:
            print(f"[exp058] episodes={ep}/{args.num_episodes} anchors={num_anchor} full={num_full}", flush=True)
            report["status"] = "running"
            report["results"] = {key: calc_stats(value) for key, value in records.items()}
            save_json(output, report)
    report["status"] = "completed"
    report["results"] = {key: calc_stats(value) for key, value in records.items()}
    report["num_anchor_instances"] = num_anchor
    report["num_full_anchor_instances"] = num_full
    report["feature_diagnostics"] = {
        "nonzero_fraction": float(x.nnz / (x.shape[0] * x.shape[1])),
        "all_nonzero_values_equal_one": bool(np.allclose(x.data, 1)),
        "distinct_nonzero_values_rounded_6dp": int(np.unique(x.data.round(6)).size),
    }
    save_json(output, report)
    print(
        "[exp058] DONE",
        {key: round(value["pooled_precision_halo"], 4) for key, value in report["results"].items() if value},
        f"json={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
