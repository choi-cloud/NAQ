"""exp047: structure-cosine threshold filtering then X Middle-K (GT diagnostics only).

For each support s, compute cosine(s,v) from X, A~X, A~²X for ALL v != s.
Keep v if cosine_AX >= tau AND cosine_A2X >= tau, then sort by X cosine
and choose the center query_count nodes. No GT is used for selection.
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
    p.add_argument("--taus", default="-0.2,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    p.add_argument("--output_dir", default="results/Amazon_clothing/exp047")
    p.add_argument("--no_save_similarities", action="store_true")
    return p.parse_args()


def make_adjacency(edge_index, n, device, norm):
    edges = edge_index.detach().cpu().numpy()
    adj = sp.csr_matrix(
        (np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])),
        shape=(n, n)
    )
    adj = ((adj + adj.T) > 0).astype(np.float32).tocsr()
    adj.setdiag(1)
    adj.eliminate_zeros()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    if norm == "symmetric":
        d = np.power(np.maximum(deg, 1), -0.5)
        adj = sp.diags(d) @ adj @ sp.diags(d)
    else:
        adj = sp.diags(1.0 / np.maximum(deg, 1)) @ adj
    coo = adj.tocoo()
    idx = torch.tensor(np.stack((coo.row, coo.col)), device=device, dtype=torch.long)
    val = torch.tensor(coo.data, device=device, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()


def describe_group(rows, support_count, candidate_n, query_count):
    valid = [r for r in rows if r["eligible"]]
    removed_n = sum(r["removed_count"] for r in rows)
    removed_neg = sum(r["removed_negative_count"] for r in rows)
    all_pos = sum(r["candidate_positive_count"] for r in rows)
    all_neg = support_count * candidate_n - all_pos
    kept = sum(r["retained_count"] for r in rows)
    kept_pos = sum(r["retained_positive_count"] for r in rows)
    q_pos = sum(r["middle_correct_count"] for r in valid)
    q_n = len(valid) * query_count

    def mean(name, group):
        return float(np.mean([r[name] for r in group])) if group else None

    return {
        "num_supports": support_count,
        "eligible_supports": len(valid),
        "ineligible_supports": support_count - len(valid),
        "coverage_rate": len(valid) / support_count,
        "retained_count_mean": mean("retained_count", rows),
        "retained_count_min": int(min(r["retained_count"] for r in rows)),
        "retained_count_median": float(np.median([r["retained_count"] for r in rows])),
        "retained_count_max": int(max(r["retained_count"] for r in rows)),
        "candidate_gt_match_rate": all_pos / (support_count * candidate_n),
        "removed_negative_precision": removed_neg / removed_n if removed_n else None,
        "negative_removal_recall": removed_neg / all_neg if all_neg else None,
        "positive_retention_rate": kept_pos / all_pos if all_pos else None,
        "retained_gt_match_rate": kept_pos / kept if kept else None,
        "middle_query_gt_match_rate": q_pos / q_n if q_n else None,
        "middle_query_gt_match_macro": mean("middle_gt_match", valid),
        "mean_query_original_x_rank": mean("middle_x_rank", valid),
        "mean_query_original_x_rank_percentile": mean("middle_x_rank_pct", valid),
        "mean_query_x_cosine": mean("middle_x_cosine", valid),
        "mean_query_ax_cosine": mean("middle_ax_cosine", valid),
        "mean_query_a2x_cosine": mean("middle_a2x_cosine", valid)
    }


def main():
    a = parse_args()
    a.setting, a.type, a.query_generation = "unsup", "feature", "NaQ"
    taus = list(dict.fromkeys(float(s) for s in a.taus.split(",")))
    if not taus or min(taus) < -1 or max(taus) > 1:
        raise ValueError("--taus must be in [-1, 1]")
    seed_everything(a.seed)
    dev = torch.device(f"cuda:{a.device}" if torch.cuda.is_available() else "cpu")
    start = time.perf_counter()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("[exp047] Loading data", flush=True)
    data = load_data(a)
    labels = np.asarray(data.labels.cpu()).reshape(-1)
    x = data.features.float().to(dev)
    n = x.shape[0]
    print("[exp047] Propagating X, AX, A2X", flush=True)
    adj = make_adjacency(data.edge_index, n, dev, a.adj_norm)
    h1 = torch.sparse.mm(adj, x)
    h2 = torch.sparse.mm(adj, h1)
    z0, z1, z2 = (F.normalize(h, dim=1) for h in (x, h1, h2))

    supports = []
    for _ in range(a.num_tasks):
        ids, _ = spt_sampling(data.id_train, a.n_way, 1)
        supports.extend(np.asarray(ids).reshape(-1).astype(int).tolist())
    supports = np.asarray(list(dict.fromkeys(supports)), dtype=np.int64)
    print(f"[exp047] {len(supports)} unique supports", flush=True)

    # Raw cosine matrices are indexed [support_row, node_id]. float16 for disk economy.
    sims = None
    if not a.no_save_similarities:
        sims = {
            "cosine_X": np.empty((len(supports), n), dtype=np.float16),
            "cosine_AX": np.empty((len(supports), n), dtype=np.float16),
            "cosine_A2X": np.empty((len(supports), n), dtype=np.float16)
        }

    records = {str(tau): [] for tau in taus}
    per_anchor = []
    queries = []
    original_top_correct = 0
    idx = np.arange(n)
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(supports), a.batch_size), desc="Threshold sweep", unit="batch"):
            bid = supports[batch_start : batch_start + a.batch_size]
            tid = torch.as_tensor(bid, device=dev, dtype=torch.long)
            c0 = (z0[tid] @ z0.T).cpu().numpy()
            c1 = (z1[tid] @ z1.T).cpu().numpy()
            c2 = (z2[tid] @ z2.T).cpu().numpy()
            if sims is not None:
                end = batch_start + len(bid)
                sims["cosine_X"][batch_start:end] = c0.astype(np.float16)
                sims["cosine_AX"][batch_start:end] = c1.astype(np.float16)
                sims["cosine_A2X"][batch_start:end] = c2.astype(np.float16)
            for j, anchor in enumerate(bid):
                candidates = idx[idx != anchor]
                s0, s1, s2 = c0[j, candidates], c1[j, candidates], c2[j, candidates]
                correct = labels[candidates] == labels[anchor]
                x_order = np.argsort(-s0, kind="stable")
                x_rank = np.empty(n - 1, dtype=np.int32)
                x_rank[x_order] = np.arange(1, n, dtype=np.int32)
                original_top_correct += int(correct[x_order[:a.query_count]].sum())

                for tau in taus:
                    # Strict rule: BOTH structural cosines must pass threshold.
                    keep = (s1 >= tau) & (s2 >= tau)
                    kept = np.flatnonzero(keep)
                    retained = len(kept)
                    eligible = retained >= a.query_count
                    entry = {
                        "support_id": int(anchor),
                        "tau": tau,
                        "candidate_count": n - 1,
                        "candidate_positive_count": int(correct.sum()),
                        "removed_count": int((~keep).sum()),
                        "removed_negative_count": int((~correct[~keep]).sum()),
                        "retained_count": retained,
                        "retained_positive_count": int(correct[keep].sum()),
                        "eligible": int(eligible),
                        "middle_correct_count": 0,
                        "middle_gt_match": None,
                        "middle_x_rank": None,
                        "middle_x_rank_pct": None,
                        "middle_x_cosine": None,
                        "middle_ax_cosine": None,
                        "middle_a2x_cosine": None
                    }
                    if eligible:
                        kept_by_x = kept[np.argsort(-s0[kept], kind="stable")]
                        offset = (retained - a.query_count) // 2
                        middle = kept_by_x[offset : offset + a.query_count]
                        entry.update({
                            "middle_correct_count": int(correct[middle].sum()),
                            "middle_gt_match": float(correct[middle].mean()),
                            "middle_x_rank": float(x_rank[middle].mean()),
                            "middle_x_rank_pct": float((x_rank[middle] / (n - 1)).mean()),
                            "middle_x_cosine": float(s0[middle].mean()),
                            "middle_ax_cosine": float(s1[middle].mean()),
                            "middle_a2x_cosine": float(s2[middle].mean())
                        })
                        for position, local in enumerate(middle):
                            queries.append({
                                "support_id": int(anchor),
                                "tau": tau,
                                "query_position": position + 1,
                                "query_id": int(candidates[local]),
                                "gt_match": int(correct[local]),
                                "original_x_rank": int(x_rank[local]),
                                "x_cosine": float(s0[local]),
                                "ax_cosine": float(s1[local]),
                                "a2x_cosine": float(s2[local]),
                                "retained_count": retained
                            })
                    records[str(tau)].append(entry)
                    per_anchor.append(entry)

    results = {str(tau): describe_group(records[str(tau)], len(supports), n - 1, a.query_count) for tau in taus}
    payload = {
        "experiment": "exp047",
        "description": "All-nodes AX/A2X cosine-threshold AND filtering, then X Middle-K",
        "config": vars(a),
        "nodes": n,
        "num_supports": len(supports),
        "adjacency_nnz": adj._nnz(),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "original_naq_x_topk_gt_match_rate": original_top_correct / (len(supports) * a.query_count),
        "filter_rule": "keep candidate iff cosine_AX >= tau AND cosine_A2X >= tau",
        "query_rule": "middle query_count among survivors sorted by descending original X cosine",
        "insufficient_candidates_policy": "exclude support from Middle-K aggregate when survivors < query_count; report coverage",
        "results": results,
        "notes": [
            "GT is used ONLY in diagnostics, never in filtering or selection.",
            "Middle-K accuracy is CONDITIONAL on eligible supports; check coverage.",
            "Saved cosine arrays, when enabled, are float16 approximations; filtering uses float32.",
            "No downstream GFSL training is performed."
        ]
    }
    (out / "exp047.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    for filename, rows in (("exp047_per_anchor.csv", per_anchor), ("exp047_queries.csv", queries)):
        if rows:
            with (out / filename).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    if sims is not None:
        np.savez_compressed(out / "exp047_similarities.npz", support_ids=supports, **sims)
    print(f"[exp047] NAQ X Top-{a.query_count} match: {payload['original_naq_x_topk_gt_match_rate']:.2%}")
    print("[exp047] tau | coverage | retained | positive retention | retained GT | middle GT | original X rank", flush=True)
    for tau in taus:
        r = results[str(tau)]
        match = f"{r['middle_query_gt_match_rate']:.2%}" if r["middle_query_gt_match_rate"] is not None else "n/a"
        rank = f"{r['mean_query_original_x_rank']:.1f}" if r["mean_query_original_x_rank"] is not None else "n/a"
        print(
            f"  {tau:>5.2f} | {r['coverage_rate']:>7.1%} | {r['retained_count_mean']:>8.1f} "
            f"| {r['positive_retention_rate']:>8.2%} | {r['retained_gt_match_rate']:>8.2%} "
            f"| {match:>8} | {rank}",
            flush=True
        )
    print(f"[exp047] Saved to {out}", flush=True)


if __name__ == "__main__":
    main()
