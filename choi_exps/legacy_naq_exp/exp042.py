"""exp042: label-free retrieval with X, A_tilde X, A_tilde^2 X.

Labels are used ONLY for evaluation. Each representation retrieves cosine Top-K
neighbors of the same support nodes; support itself is always excluded.

Run in the original NAQ repository alongside data.py and task_generator.py.
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
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--adj_norm", choices=["symmetric", "row"], default="symmetric")
    p.add_argument("--output_dir", default="results/Amazon_clothing/exp042")
    return p.parse_args()


def make_adjacency(edge_index, n, device, norm="symmetric"):
    """Undirected binary adjacency + self-loops, then standard normalization."""
    edges = edge_index.detach().cpu().numpy()
    a = sp.csr_matrix((np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])), shape=(n, n))
    a = ((a + a.T) > 0).astype(np.float32).tocsr()
    a.setdiag(1)
    a.eliminate_zeros()
    deg = np.asarray(a.sum(axis=1)).reshape(-1)
    if norm == "symmetric":
        d = sp.diags(np.power(np.maximum(deg, 1), -0.5))
        a = d @ a @ d
    else:
        d = sp.diags(1.0 / np.maximum(deg, 1))
        a = d @ a
    coo = a.tocoo()
    indices = torch.from_numpy(np.stack([coo.row, coo.col]).astype(np.int64)).to(device)
    values = torch.from_numpy(coo.data.astype(np.float32)).to(device)
    return torch.sparse_coo_tensor(indices, values, (n, n), device=device).coalesce()


def retrieve_topk(z, supports, k, batch_size):
    """Return [num_supports, k] global IDs, excluding the support itself."""
    z = F.normalize(z, dim=1)
    answers = []
    for start in tqdm(range(0, len(supports), batch_size), desc="Cosine kNN", unit="batch"):
        batch = supports[start : start + batch_size]
        ids = torch.as_tensor(batch, dtype=torch.long, device=z.device)
        similarities = z[ids] @ z.T
        similarities[torch.arange(len(ids), device=z.device), ids] = -float("inf")
        answers.append(similarities.topk(k, dim=1).indices.cpu().numpy())
    return np.concatenate(answers, axis=0)


def safe_mean(items):
    return float(np.mean(items)) if items else None


def main():
    a = parse_args()
    a.setting, a.type, a.query_generation = "unsup", "feature", "NaQ"
    seed_everything(a.seed)
    device = torch.device(f"cuda:{a.device}" if torch.cuda.is_available() else "cpu")
    start = time.perf_counter()
    print("[exp042] Loading data", flush=True)
    data = load_data(a)
    labels = np.asarray(data.labels.cpu()).reshape(-1)
    x = data.features.float().to(device)
    n = x.shape[0]
    if not 1 <= a.k < n:
        raise ValueError(f"k must be between 1 and {n-1}")

    print("[exp042] Building normalized adjacency", flush=True)
    adj = make_adjacency(data.edge_index, n, device, a.adj_norm)
    h0 = x
    print("[exp042] Computing first- and second-order propagation", flush=True)
    h1 = torch.sparse.mm(adj, h0)
    h2 = torch.sparse.mm(adj, h1)

    # Match exp041 support sampling, including de-duplication of repeated anchors.
    supports = []
    for _ in tqdm(range(a.num_tasks), desc="Support sampling", unit="task"):
        support_ids, _ = spt_sampling(data.id_train, a.n_way, 1)
        supports.extend(np.asarray(support_ids).reshape(-1).astype(int).tolist())
    supports = np.asarray(list(dict.fromkeys(supports)), dtype=np.int64)
    print(f"[exp042] Supports: {len(supports)} unique anchors", flush=True)

    names = ("X", "AX", "A2X")
    lists = {}
    for name, embedding in zip(names, (h0, h1, h2)):
        print(f"[exp042] Retrieving {name} Top-{a.k}", flush=True)
        lists[name] = retrieve_topk(embedding, supports, a.k, a.batch_size)

    per_anchor = []
    precision = {name: [] for name in names}
    pair_names = (("X", "AX"), ("X", "A2X"), ("AX", "A2X"))
    overlap = {f"{u}&{v}": [] for u, v in pair_names}
    triple_counts = []
    category = {
        key: {"count": [], "correct": [], "precision": []}
        for key in ("all_three", "at_least_two", "only_X", "only_AX", "only_A2X", "exactly_two", "union")
    }

    for i, support in enumerate(supports):
        buckets = {name: set(lists[name][i].tolist()) for name in names}
        row = {"support_id": int(support), "support_label": int(labels[support])}
        for name in names:
            correct = sum(labels[v] == labels[support] for v in buckets[name])
            row[f"{name}_correct"] = int(correct)
            precision[name].append(correct / a.k)
        for u, v in pair_names:
            common = buckets[u] & buckets[v]
            key = f"{u}&{v}"
            row[f"overlap_{key}"] = len(common)
            overlap[key].append(len(common))

        xset, aset, a2set = (buckets[name] for name in names)
        common3 = xset & aset & a2set
        exact2 = ((xset & aset) | (xset & a2set) | (aset & a2set)) - common3
        groups = {
            "all_three": common3,
            "at_least_two": common3 | exact2,
            "exactly_two": exact2,
            "only_X": xset - aset - a2set,
            "only_AX": aset - xset - a2set,
            "only_A2X": a2set - xset - aset,
            "union": xset | aset | a2set,
        }
        triple_counts.append(len(common3))
        for key, group in groups.items():
            count = len(group)
            correct = sum(labels[v] == labels[support] for v in group)
            row[f"{key}_count"] = count
            row[f"{key}_correct"] = int(correct)
            category[key]["count"].append(count)
            category[key]["correct"].append(int(correct))
            if count:
                category[key]["precision"].append(correct / count)
        per_anchor.append(row)

    summary = {
        "experiment": "exp042",
        "description": "Cosine kNN on X, A_tilde X, A_tilde^2 X; GT only for evaluation",
        "config": vars(a),
        "nodes": int(n),
        "num_supports": len(supports),
        "adjacency_nnz": int(adj._nnz()),
        "preparation_and_evaluation_seconds": round(time.perf_counter() - start, 3),
        "precision_at_k": {name: safe_mean(precision[name]) for name in names},
        "overlap_mean_count": {key: safe_mean(vals) for key, vals in overlap.items()},
        "triple_overlap_mean_count": safe_mean(triple_counts),
        "group_diagnostics": {
            key: {
                "mean_count_per_support": safe_mean(vals["count"]),
                "micro_gt_match_rate": (sum(vals["correct"]) / sum(vals["count"])) if sum(vals["count"]) else None,
                "macro_gt_match_rate_nonempty": safe_mean(vals["precision"]),
                "supports_with_nonempty_group": len(vals["precision"]),
            }
            for key, vals in category.items()
        },
    }
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "exp042.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (out / "exp042_per_anchor.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(per_anchor[0]))
        writer.writeheader()
        writer.writerows(per_anchor)
    np.savez_compressed(out / "exp042_neighbors.npz", support_ids=supports, **lists)
    print("[exp042] Precision@K:", summary["precision_at_k"], flush=True)
    print("[exp042] Overlap:", summary["overlap_mean_count"], flush=True)
    print("[exp042] Group diagnostics:", json.dumps(summary["group_diagnostics"], indent=2), flush=True)
    print(f"[exp042] Saved results to {out}", flush=True)


if __name__ == "__main__":
    main()
