"""exp068: GPS four positive sampling strategies, GT-only diagnostics on Cora-Full.

Run at the root of the original NAQ repository. No model fitting and no GT leakage.
Methods: personalized PageRank, hop-distance, weighted common neighbors, raw cosine.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import normalize

from data import load_data
from task_generator import spt_sampling


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Cora_full")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--q_query", type=int, default=10)
    p.add_argument("--ppr_alpha", type=float, default=0.15)
    p.add_argument("--ppr_iterations", type=int, default=60)
    p.add_argument("--ppr_tolerance", type=float, default=1e-8)
    p.add_argument("--min_hop", type=int, default=1)
    p.add_argument("--max_hop", type=int, default=2)
    p.add_argument("--node_k", type=int, default=10)
    p.add_argument("--relation_k", type=int, default=10)
    p.add_argument("--semantic_k", type=int, default=10)
    p.add_argument("--output", default="exp068.json")
    return p.parse_args()


def sorted_top(scores, eligible, k):
    ids = np.flatnonzero(eligible & np.isfinite(scores))
    if not len(ids):
        return np.empty(0, dtype=np.int64)
    values = scores[ids]
    return ids[np.lexsort((ids, -values))[:k]]


def complete(ids, eligible, base_scores, k):
    """Diagnostic-only fixed-size fallback: raw cosine, when GPS set has <k nodes."""
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) >= k:
        return ids[:k], 0
    available = eligible.copy()
    available[ids] = False
    supplemental = sorted_top(base_scores, available, k - len(ids))
    return np.concatenate([ids, supplemental]), len(supplemental)


def ppr_scores(anchor, transition_transpose, alpha, iterations, tolerance):
    n = transition_transpose.shape[0]
    restart = np.zeros(n, dtype=np.float64)
    restart[anchor] = 1.0
    current = restart.copy()
    for _ in range(iterations):
        nxt = alpha * restart + (1.0 - alpha) * (transition_transpose @ current)
        if np.linalg.norm(nxt - current, ord=1) < tolerance:
            current = nxt
            break
        current = nxt
    return current


def hop_candidates(anchor, graph, min_hop, max_hop, excluded):
    """All reachable nodes in the permitted hop interval, exact shortest hops."""
    dist = np.full(graph.shape[0], -1, dtype=np.int16)
    dist[anchor] = 0
    frontier = np.array([anchor], dtype=np.int64)
    for depth in range(1, max_hop + 1):
        if not len(frontier):
            break
        nbrs = np.unique(np.concatenate([graph.indices[graph.indptr[v] : graph.indptr[v + 1]] for v in frontier]))
        unseen = nbrs[dist[nbrs] == -1]
        dist[unseen] = depth
        frontier = unseen
    eligible = (dist >= min_hop) & (dist <= max_hop) & (~excluded)
    return np.flatnonzero(eligible), dist


def main():
    a = parse_args()
    if a.episodes <= 0 or a.q_query <= 0 or a.n_way <= 0:
        raise ValueError("episodes, n_way, q_query must be positive")
    if not (0 < a.ppr_alpha <= 1) or a.min_hop < 1 or a.max_hop < a.min_hop:
        raise ValueError("Invalid PPR alpha or hop range")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    a.setting = "unsup"
    a.query_generation = "NaQ"
    a.type = "feature"
    a.k_shot = 1
    a.k_shot_test = 1
    a.epoch = 1
    data = load_data(a)
    gt = data.labels.detach().cpu().numpy().reshape(-1)
    x = sp.csr_matrix(data.features.detach().cpu().numpy().astype(np.float64))
    x_unit = normalize(x, norm="l2", axis=1, copy=True)
    n = len(gt)
    edge = data.edge_index.detach().cpu().numpy()
    raw_graph = sp.coo_matrix((np.ones(edge.shape[1]), (edge[0], edge[1])), shape=(n, n)).tocsr()
    graph = ((raw_graph + raw_graph.T) > 0).astype(np.float64).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    inv_degree = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    transition_t = (sp.diags(inv_degree) @ graph).T.tocsr()

    # GPS Eq.(8): shared-neighbor score with inverse-log hub discount.
    # Degree 0/1 is clipped to 2 to avoid division by log(1)=0.
    neighbor_weights = 1.0 / np.log(np.maximum(degree, 2.0))
    weighted_neighbors = sp.diags(neighbor_weights) @ graph

    method_names = ["node_ppr", "topology_hop", "relation_overlap", "semantic_cosine"]
    rows = {name: [] for name in method_names}
    rows.update({"union": [], "intersection": []})
    lengths = {name: [] for name in method_names}
    fill_counts = {name: 0 for name in method_names}
    hop_hist = {}
    rng = np.random.default_rng(a.seed + 6800)
    anchor_total = 0

    for episode in range(a.episodes):
        supports = np.asarray(spt_sampling(data.id_train, a.n_way, 1)[0], dtype=np.int64)
        excluded = np.zeros(n, dtype=bool)
        excluded[supports] = True
        eligible = ~excluded
        for anchor in supports:
            anchor = int(anchor)
            anchor_total += 1
            cosine = np.asarray((x_unit.getrow(anchor) @ x_unit.T).toarray()).reshape(-1)

            ppr = ppr_scores(anchor, transition_t, a.ppr_alpha, a.ppr_iterations, a.ppr_tolerance)
            node = sorted_top(ppr, eligible, a.node_k)

            hops, distances = hop_candidates(anchor, graph, a.min_hop, a.max_hop, excluded)
            # In the GPS paper topology positives form a hop-range set (not a top-k).
            # For fixed K comparison sample k uniformly, without using GT or cosine.
            topology = rng.choice(hops, size=min(a.q_query, len(hops)), replace=False).astype(np.int64)
            for h in distances[topology]:
                hop_hist[str(int(h))] = hop_hist.get(str(int(h)), 0) + 1

            shared = graph.getrow(anchor) @ weighted_neighbors
            relation_score = np.zeros(n, dtype=np.float64)
            if shared.nnz:
                relation_score[shared.indices] = shared.data
            # GPS Eq.(8) includes both endpoints, giving a pair-independent
            # constant for each anchor plus a candidate-specific endpoint term.
            # It is constant-offset/reweighted, but candidate with zero common
            # neighbors is not treated as a discovered relation positive.
            relation_possible = eligible & (relation_score > 0)
            relation = sorted_top(relation_score, relation_possible, a.relation_k)
            semantic = sorted_top(cosine, eligible, a.semantic_k)

            selected = {"node_ppr": node, "topology_hop": topology, "relation_overlap": relation, "semantic_cosine": semantic}
            for name, ids in selected.items():
                lengths[name].append(len(ids))
                chosen, filled = complete(ids, eligible, cosine, a.q_query)
                fill_counts[name] += filled
                rows[name].append(float(np.mean(gt[chosen] == gt[anchor])))

            # GPS fusion diagnostics: union/intersection of the raw GPS sets,
            # evaluated as variable-size sets; no fallback applied.
            sets = [set(ids.tolist()) for ids in selected.values()]
            union = set.union(*sets)
            intersection = set.intersection(*sets)
            for key, pool in [("union", union), ("intersection", intersection)]:
                if pool:
                    pool_ids = np.fromiter(pool, dtype=np.int64)
                    rows[key].append((int(np.sum(gt[pool_ids] == gt[anchor])), len(pool_ids)))
                else:
                    rows[key].append((0, 0))
        if (episode + 1) % 20 == 0:
            print(f"Processed {episode + 1}/{a.episodes} episodes", flush=True)

    results = {}
    for name in method_names:
        vals = np.asarray(rows[name], dtype=float)
        lens = np.asarray(lengths[name], dtype=float)
        results[name] = {
            "anchors": anchor_total,
            "top10_gt_match_rate": float(vals.mean()),
            "top10_gt_match_rate_pct": float(vals.mean() * 100),
            "average_correct_of_10": float(vals.mean() * a.q_query),
            "raw_candidates_mean": float(lens.mean()),
            "raw_candidates_median": float(np.median(lens)),
            "insufficient_anchors": int(np.sum(lens < a.q_query)),
            "fallback_cosine_queries": int(fill_counts[name]),
            "fallback_fraction_of_queries": float(fill_counts[name] / (anchor_total * a.q_query)),
        }
    for name in ("union", "intersection"):
        counts = np.asarray(rows[name], dtype=np.int64)
        total = int(counts[:, 1].sum())
        results[name] = {
            "variable_size_pool_gt_match_rate": float(counts[:, 0].sum() / total) if total else None,
            "mean_pool_size": float(counts[:, 1].mean()),
            "empty_anchors": int(np.sum(counts[:, 1] == 0)),
            "total_pairs": total,
            "note": "Raw set union/intersection; not a fixed Top-10 sampling comparison",
        }
    output = {
        "experiment": "exp068",
        "status": "completed",
        "args": vars(a),
        "graph": {"nodes": n, "classes": int(np.unique(gt).size), "undirected_edges": int(graph.nnz // 2)},
        "definitions": {
            "anchor": "Original NAQ 5-way support sampling, GT collisions allowed",
            "candidate_exclusion": "All supports of the current episode excluded",
            "node_ppr": "GPS Eq.(5)-(6): personalized PageRank, top k",
            "topology_hop": "GPS Eq.(7): uniform sample from shortest hop range [min_hop,max_hop]",
            "relation_overlap": "GPS Eq.(8)-(9): inverse-log-degree weighted common neighbors; zero-overlap pairs excluded",
            "semantic_cosine": "GPS Eq.(10)-(11): raw feature cosine top k; NAQ-Feat baseline",
            "fallback": "If fewer than 10, append best remaining raw cosine candidates; tracked separately",
            "gt": "Evaluation only; never used in positive selection",
            "ppr_convention": "Row-stochastic random walk on undirected simple graph, no self-loop, personalized restart",
            "relation_degree_one": "Degree clipped at 2 in inverse-log weights to prevent division by zero",
            "union_intersection": "Diagnostic-only set fusion of four raw selected sets",
        },
        "results": results,
        "topology_sampled_hop_histogram": hop_hist,
    }
    target = Path(__file__).resolve().parent / a.output
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    print(f"Saved: {target}", flush=True)


if __name__ == "__main__":
    main()
