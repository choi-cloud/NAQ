"""exp045: structure-first filtering over ALL nodes, then X-middle query mining.

For every sampled NAQ support, rank every other node by cosine similarity in
AX and A^2X. Remove nodes ranked poorly in BOTH structural views (or random
control), sort survivors by original X cosine, and select the middle K.
Ground-truth labels are used for diagnostics ONLY.
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
    p.add_argument("--query_count", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--adj_norm", choices=["symmetric", "row"], default="symmetric")
    p.add_argument("--remove_fractions", default="0,0.5,0.9,0.95,0.99,0.995,0.999")
    p.add_argument("--output_dir", default="results/Amazon_clothing/exp045")
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


def rank_desc(arr):
    order = np.argsort(-arr, kind="stable")
    ranks = np.empty(len(arr), dtype=np.int32)
    ranks[order] = np.arange(len(arr), dtype=np.int32)
    return ranks


def summarize(rows):
    def ratio(num, den):
        return float(num / den) if den else None

    candidate_n = sum(r["candidate_count"] for r in rows)
    candidate_pos = sum(r["candidate_positive_count"] for r in rows)
    removed_n = sum(r["removed_count"] for r in rows)
    removed_neg = sum(r["removed_negative_count"] for r in rows)
    kept_n = sum(r["retained_count"] for r in rows)
    kept_pos = sum(r["retained_positive_count"] for r in rows)
    query_n = sum(r["middle_size"] for r in rows)
    query_pos = sum(r["middle_correct_count"] for r in rows)

    def mean_key(key):
        return float(np.mean([r[key] for r in rows]))

    return {
        "num_supports": len(rows),
        "candidate_gt_match_rate": ratio(candidate_pos, candidate_n),
        "removed_negative_precision": ratio(removed_neg, removed_n),
        "negative_removal_recall": ratio(removed_neg, candidate_n - candidate_pos),
        "positive_retention_rate": ratio(kept_pos, candidate_pos),
        "retained_gt_match_rate": ratio(kept_pos, kept_n),
        "middle_query_gt_match_rate": ratio(query_pos, query_n),
        "middle_query_gt_match_macro": mean_key("middle_gt_match"),
        "mean_retained_count": mean_key("retained_count"),
        "mean_query_original_x_rank": mean_key("middle_original_x_rank_mean"),
        "mean_query_original_x_rank_percentile": mean_key("middle_original_x_rank_percentile"),
        "mean_query_x_cosine": mean_key("middle_x_cosine_mean"),
        "mean_query_x_cosine_min": mean_key("middle_x_cosine_min"),
        "mean_query_x_cosine_max": mean_key("middle_x_cosine_max"),
        "mean_query_ax_cosine": mean_key("middle_ax_cosine_mean"),
        "mean_query_a2x_cosine": mean_key("middle_a2x_cosine_mean"),
    }


def main():
    args = parse_args()
    args.setting, args.type, args.query_generation = "unsup", "feature", "NaQ"
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    fractions = sorted(set(float(x) for x in args.remove_fractions.split(",")))
    if not fractions or fractions[0] < 0 or fractions[-1] >= 1:
        raise ValueError("Removal fractions must fall in [0, 1)")
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    print("[exp045] Loading data", flush=True)
    data = load_data(args)
    labels = np.asarray(data.labels.cpu()).reshape(-1)
    x = data.features.float().to(device)
    n = x.shape[0]
    candidates_n = n - 1
    if min(candidates_n - round(candidates_n * r) for r in fractions) < args.query_count:
        raise ValueError("At least query_count nodes must survive every removal fraction")
    print("[exp045] Propagating X, AX, A2X", flush=True)
    adj = make_adjacency(data.edge_index, n, device, args.adj_norm)
    h1 = torch.sparse.mm(adj, x)
    h2 = torch.sparse.mm(adj, h1)
    z0, z1, z2 = (F.normalize(h, dim=1) for h in (x, h1, h2))

    supports = []
    for _ in range(args.num_tasks):
        ids, _ = spt_sampling(data.id_train, args.n_way, 1)
        supports.extend(np.asarray(ids).reshape(-1).astype(int).tolist())
    supports = np.asarray(list(dict.fromkeys(supports)), dtype=np.int64)
    print(f"[exp045] {len(supports)} unique supports", flush=True)
    methods = ("random", "structural_rejection")
    records = {(m, str(r)): [] for m in methods for r in fractions}
    per_anchor = []
    neighbor_rows = []
    idx = np.arange(n)
    with torch.no_grad():
        for start in tqdm(range(0, len(supports), args.batch_size), desc="Filtering", unit="batch"):
            ids_np = supports[start : start + args.batch_size]
            ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)
            cosine0 = (z0[ids] @ z0.T).cpu().numpy()
            cosine1 = (z1[ids] @ z1.T).cpu().numpy()
            cosine2 = (z2[ids] @ z2.T).cpu().numpy()
            for j, anchor in enumerate(ids_np):
                candidates = idx[idx != anchor]
                s0 = cosine0[j, candidates]
                s1 = cosine1[j, candidates]
                s2 = cosine2[j, candidates]
                correct = labels[candidates] == labels[anchor]
                r0, r1, r2 = (rank_desc(s) for s in (s0, s1, s2))
                # Large minimum rank means both structural representations
                # consider the candidate dissimilar to the anchor.
                reject = np.minimum(r1, r2)
                # Deterministic structural ties, random control with seeded RNG.
                remove_orders = {"random": rng.permutation(candidates_n), "structural_rejection": np.lexsort((r0, -reject))}
                # Reference: original NAQ feature Top-K for SAME sampled supports.
                original_top = np.argsort(r0)[: args.query_count]
                original_top_precision = float(correct[original_top].mean())
                for method, order in remove_orders.items():
                    for fraction in fractions:
                        remove_n = round(candidates_n * fraction)
                        removed = order[:remove_n]
                        kept = order[remove_n:]
                        # IMPORTANT: sort survivors ONLY AFTER structural filtering,
                        # using ORIGINAL X feature cosine (not AX/A2X scores).
                        kept_by_x = kept[np.argsort(-s0[kept], kind="stable")]
                        middle_start = (len(kept_by_x) - args.query_count) // 2
                        middle = kept_by_x[middle_start : middle_start + args.query_count]
                        entry = {
                            "support_id": int(anchor),
                            "method": method,
                            "remove_fraction": fraction,
                            "candidate_count": candidates_n,
                            "candidate_positive_count": int(correct.sum()),
                            "removed_count": int(len(removed)),
                            "removed_negative_count": int((~correct[removed]).sum()),
                            "retained_count": int(len(kept)),
                            "retained_positive_count": int(correct[kept].sum()),
                            "middle_size": int(len(middle)),
                            "middle_correct_count": int(correct[middle].sum()),
                            "middle_gt_match": float(correct[middle].mean()),
                            "middle_original_x_rank_mean": float((r0[middle] + 1).mean()),
                            "middle_original_x_rank_percentile": float((r0[middle] / candidates_n).mean()),
                            "middle_x_cosine_mean": float(s0[middle].mean()),
                            "middle_x_cosine_min": float(s0[middle].min()),
                            "middle_x_cosine_max": float(s0[middle].max()),
                            "middle_ax_cosine_mean": float(s1[middle].mean()),
                            "middle_a2x_cosine_mean": float(s2[middle].mean()),
                            "original_top10_gt_match": original_top_precision,
                        }
                        records[(method, str(fraction))].append(entry)
                        per_anchor.append(entry)
                        for pos, local_id in enumerate(middle):
                            neighbor_rows.append(
                                {
                                    "support_id": int(anchor),
                                    "method": method,
                                    "remove_fraction": fraction,
                                    "query_position": pos + 1,
                                    "query_id": int(candidates[local_id]),
                                    "gt_match": int(correct[local_id]),
                                    "original_x_rank": int(r0[local_id] + 1),
                                    "x_cosine": float(s0[local_id]),
                                    "ax_cosine": float(s1[local_id]),
                                    "a2x_cosine": float(s2[local_id]),
                                    "ax_rank": int(r1[local_id] + 1),
                                    "a2x_rank": int(r2[local_id] + 1),
                                }
                            )

    results = {m: {str(r): summarize(records[(m, str(r))]) for r in fractions} for m in methods}
    top10 = float(np.mean([r["original_top10_gt_match"] for r in records[("random", str(fractions[0]))]]))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "exp045",
        "description": "All-nodes AX/A2X structural rejection followed by X-middle-10 query selection",
        "config": vars(args),
        "nodes": int(n),
        "num_supports": len(supports),
        "adjacency_nnz": int(adj._nnz()),
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "candidate_scope": "all_nodes_excluding_support",
        "structural_rejection_rule": "Remove descending min(rank_AX, rank_A2X) among ALL candidates; higher ranks mean lower cosine similarity",
        "original_naq_x_top10_gt_match_rate": top10,
        "results": results,
        "notes": [
            "No GT labels used for ranking, filtering, or query selection.",
            "GT labels only used in diagnostic computations.",
            "Feature original X rank is GLOBAL among all nodes excluding the support.",
            "Middle-K is selected by X cosine AFTER structural filtering.",
            "No GFSL downstream model is trained in this experiment.",
        ],
    }
    (out / "exp045.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    for filename, items in (("exp045_per_anchor.csv", per_anchor), ("exp045_queries.csv", neighbor_rows)):
        with (out / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
    print(f"[exp045] NAQ X Top-10 GT match: {top10:.4%}", flush=True)
    print("[exp045] SUMMARY: fraction | method | removed neg precision | positive retention | middle GT | original X rank | X cosine", flush=True)
    for fraction in fractions:
        for method in methods:
            d = results[method][str(fraction)]
            neg = d["removed_negative_precision"]
            neg_str = f"{neg:.2%}" if neg is not None else "n/a"
            print(
                f"  {fraction:>6.1%} | {method:>20} | neg={neg_str} "
                f"| pos_ret={d['positive_retention_rate']:.2%} "
                f"| middle_gt={d['middle_query_gt_match_rate']:.2%} "
                f"| x_rank={d['mean_query_original_x_rank']:.1f} "
                f"| x_cos={d['mean_query_x_cosine']:.4f}",
                flush=True,
            )
    print(f"[exp045] Saved {out}", flush=True)


if __name__ == "__main__":
    main()
