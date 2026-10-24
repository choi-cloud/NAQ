"""exp043: label-free negative filtering using multi-depth feature disagreement.

For each support, construct an X-cosine Top-P candidate pool, compute X/AX/A2X
cosine similarities and within-pool ranks, filter candidate pairs by four
label-free strategies. GT labels are used *only* for offline diagnostics.
Requires NAQ data.py, task_generator.py, utils.py (same as exp042).
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from data import load_data
from task_generator import spt_sampling
from utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Amazon_clothing")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--num_tasks", type=int, default=200)
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--pool_sizes", default="200,1000")
    p.add_argument("--remove_fractions", default="0.25,0.5,0.75,0.9")
    p.add_argument("--query_count", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--adj_norm", choices=["symmetric", "row"], default="symmetric")
    p.add_argument("--output_dir", default="results/Amazon_clothing/exp043")
    return p.parse_args()


def make_adjacency(edge_index, n, device, norm):
    edges = edge_index.detach().cpu().numpy()
    a = sp.csr_matrix((np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])), shape=(n, n))
    a = ((a + a.T) > 0).astype(np.float32).tocsr()
    a.setdiag(1)
    a.eliminate_zeros()
    degree = np.asarray(a.sum(axis=1)).ravel()
    if norm == "symmetric":
        inv = np.power(np.maximum(degree, 1), -0.5)
        a = sp.diags(inv) @ a @ sp.diags(inv)
    else:
        a = sp.diags(1.0 / np.maximum(degree, 1)) @ a
    coo = a.tocoo()
    idx = torch.tensor(np.stack((coo.row, coo.col)), dtype=torch.long, device=device)
    val = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()


def ranks_desc(scores):
    """Per-row 0-based descending ranks; stable sort for deterministic ties."""
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(scores.shape[1], device=scores.device).expand_as(order))
    return ranks


def calculate_methods(r0, r1, r2, generator):
    # All ranks are within the same X-selected candidate pool; larger = worse.
    # Rank disagreement: evidence deteriorates from X to smoothed views.
    # Structural rejection: both smoothed views rank the candidate poorly.
    # Consensus: poor average rank across all three views.
    pool_size = len(r0)
    random_score = generator.random(pool_size)
    return {
        "random": random_score,
        "rank_disagreement": (r1 + r2 - 2 * r0).astype(float),
        "structural_rejection": np.minimum(r1, r2).astype(float),
        "consensus_ranking": (r0 + r1 + r2).astype(float),
    }


def summarize(records):
    total_removed = sum(x["removed_count"] for x in records)
    removed_neg = sum(x["removed_negative_count"] for x in records)
    total_neg = sum(x["candidate_negative_count"] for x in records)
    total_pos = sum(x["candidate_positive_count"] for x in records)
    kept_pos = sum(x["retained_positive_count"] for x in records)
    kept_total = sum(x["remaining_size"] for x in records)
    middle_pos = sum(x["middle_correct_count"] for x in records)
    middle_total = sum(x["middle_size"] for x in records)
    return {
        "num_supports": len(records),
        "candidate_precision_micro": total_pos / (total_pos + total_neg),
        "removed_negative_precision": removed_neg / total_removed if total_removed else None,
        "negative_removal_rate": removed_neg / total_neg if total_neg else None,
        "positive_retention_rate": kept_pos / total_pos if total_pos else None,
        "remaining_precision": kept_pos / kept_total if kept_total else None,
        "middle_query_precision": middle_pos / middle_total if middle_total else None,
        "middle_query_precision_macro": float(np.mean([x["middle_query_precision"] for x in records])),
        "remaining_size_mean": float(np.mean([x["remaining_size"] for x in records])),
        "removed_count_mean": float(np.mean([x["removed_count"] for x in records])),
    }


def main():
    a = parse_args()
    a.setting, a.type, a.query_generation = "unsup", "feature", "NaQ"
    seed_everything(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device(f"cuda:{a.device}" if torch.cuda.is_available() else "cpu")
    pool_sizes = sorted(set(int(v) for v in a.pool_sizes.split(",")))
    fractions = sorted(set([0.0] + [float(v) for v in a.remove_fractions.split(",")]))
    if not all(0 <= r < 1 for r in fractions):
        raise ValueError("All removal fractions must be in [0, 1)")
    if not all(p >= a.query_count for p in pool_sizes):
        raise ValueError("Pool size must be at least query_count")
    started = time.perf_counter()
    print("[exp043] Loading data", flush=True)
    data = load_data(a)
    labels = np.asarray(data.labels.cpu()).reshape(-1)
    x = data.features.float().to(device)
    n = x.shape[0]
    if max(pool_sizes) >= n:
        raise ValueError("pool_sizes must be less than node count")
    print("[exp043] Propagating X, AX, A2X", flush=True)
    adj = make_adjacency(data.edge_index, n, device, a.adj_norm)
    h1 = torch.sparse.mm(adj, x)
    h2 = torch.sparse.mm(adj, h1)
    z0, z1, z2 = (F.normalize(h, dim=1) for h in (x, h1, h2))

    supports = []
    for _ in range(a.num_tasks):
        ids, _ = spt_sampling(data.id_train, a.n_way, 1)
        supports.extend(np.asarray(ids).reshape(-1).astype(int).tolist())
    supports = np.asarray(list(dict.fromkeys(supports)), dtype=np.int64)
    print(f"[exp043] {len(supports)} unique supports", flush=True)

    methods = ("random", "rank_disagreement", "structural_rejection", "consensus_ranking")
    records = {(str(p), m, str(r)): [] for p in pool_sizes for m in methods for r in fractions}
    rows = []
    max_pool = max(pool_sizes)
    with torch.no_grad():
        for start in tqdm(range(0, len(supports), a.batch_size), desc="Filtering", unit="batch"):
            ids_np = supports[start : start + a.batch_size]
            ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)
            sx = z0[ids] @ z0.T
            sx[torch.arange(len(ids), device=device), ids] = -float("inf")
            candidates = sx.topk(max_pool, dim=1, sorted=True).indices
            # Scores for the *same candidates* from all three views.
            s0 = sx.gather(1, candidates)
            # Compute similarities by matrix multiplication without materializing
            # [batch_size, max_pool, feature_dim] gathered feature tensors.
            s1 = (z1[ids] @ z1.T).gather(1, candidates)
            s2 = (z2[ids] @ z2.T).gather(1, candidates)
            candidates_np = candidates.cpu().numpy()
            s0_np, s1_np, s2_np = (s.cpu().numpy() for s in (s0, s1, s2))

            for j, support in enumerate(ids_np):
                for p in pool_sizes:
                    selected = candidates_np[j, :p]
                    correct = labels[selected] == labels[support]
                    # X's original top-k order is 0,1,...,p-1.
                    r0 = np.arange(p)
                    r1 = np.argsort(np.argsort(-s1_np[j, :p], kind="stable"), kind="stable")
                    r2 = np.argsort(np.argsort(-s2_np[j, :p], kind="stable"), kind="stable")
                    score_dict = calculate_methods(r0, r1, r2, rng)
                    for method, scores in score_dict.items():
                        # Stable tie break using X rank, reproducible across runs.
                        removal_order = np.lexsort((r0, -scores))
                        for fraction in fractions:
                            remove_count = int(round(p * fraction))
                            removed = removal_order[:remove_count]
                            # Important: middle query is selected AFTER sorting survivors by X similarity.
                            # Since candidates are already X sorted, ascending local index is sufficient.
                            kept = np.sort(removal_order[remove_count:])
                            mid = max(0, (len(kept) - a.query_count) // 2)
                            middle = kept[mid : mid + a.query_count]
                            row = {
                                "support_id": int(support),
                                "pool_size": p,
                                "method": method,
                                "remove_fraction": fraction,
                                "candidate_positive_count": int(correct.sum()),
                                "candidate_negative_count": int((~correct).sum()),
                                "removed_count": len(removed),
                                "removed_negative_count": int((~correct[removed]).sum()),
                                "retained_positive_count": int(correct[kept].sum()),
                                "remaining_size": len(kept),
                                "middle_size": len(middle),
                                "middle_correct_count": int(correct[middle].sum()),
                                "middle_query_precision": float(correct[middle].mean()),
                            }
                            records[(str(p), method, str(fraction))].append(row)
                            rows.append(row)

    metrics = {str(p): {method: {str(r): summarize(records[(str(p), method, str(r))]) for r in fractions} for method in methods} for p in pool_sizes}
    result = {
        "experiment": "exp043",
        "description": "Negative filtering from rank agreement/disagreement on X, AX, A2X; GT only for evaluation",
        "config": vars(a),
        "nodes": int(n),
        "num_supports": len(supports),
        "adjacency_nnz": int(adj._nnz()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "methods": {
            "random": "randomly remove candidates (control)",
            "rank_disagreement": "remove largest (rank_AX + rank_A2X - 2*rank_X)",
            "structural_rejection": "remove largest min(rank_AX, rank_A2X): both structural views rank poorly",
            "consensus_ranking": "remove largest (rank_X + rank_AX + rank_A2X)",
        },
        "metrics": metrics,
        "notes": [
            "All ranks computed within the fixed X Top-P pool for each pool size.",
            "GT is not used for ranking, removal, or query selection.",
            "Middle-K is the center of remaining candidates ordered by descending X cosine.",
            "At fraction zero, all methods are identical.",
            "This is query label consistency, not downstream GFSL accuracy.",
        ],
    }
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "exp043.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    with (out / "exp043_per_anchor.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("[exp043] SUMMARY (middle_query_precision, removed_negative_precision)", flush=True)
    for p in pool_sizes:
        for method in methods:
            numbers = [metrics[str(p)][method][str(r)] for r in fractions]
            print(
                f"  Pool={p} {method}: "
                + " | ".join(
                    f"{r:.0%}: mid={v['middle_query_precision']:.3f}, neg={v['removed_negative_precision']}" for r, v in zip(fractions, numbers)
                ),
                flush=True,
            )
    print(f"[exp043] Saved to {out}", flush=True)


if __name__ == "__main__":
    main()
