"""exp014: 1-shot NAQ-Feat bidirectional coverage: PPR / 1-hop / curvature.

Run from the NAQ repository root, with its data.py and utils.py available.
Ground-truth labels are used ONLY for reporting metrics.
"""

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from data import load_data
from utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="1-shot bidirectional NAQ query/subgraph semantic coverage")
    p.add_argument("--dataset", default="Amazon_clothing")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=1)
    p.add_argument("--q_query", type=int, default=10)
    p.add_argument("--num_supports", type=int, default=1000)
    p.add_argument("--context_topk", type=int, default=16)
    p.add_argument("--ppr_steps", type=int, default=20)
    p.add_argument("--PPR_alpha", type=float, default=0.1)
    p.add_argument("--curvature_beta", type=float, default=2.0, help="Strength of normalized Forman curvature edge weighting")
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--json_path", default="exp014.json")
    return p.parse_args()


def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {"mean": None, "std": None, "count": 0}
    return {"mean": float(a.mean()), "std": float(a.std()), "count": int(len(a))}


def main():
    args = parse_args()
    if args.k_shot != 1 or args.q_query < 1 or args.num_supports < 1 or args.context_topk < 1:
        raise ValueError("exp014 requires 1-shot, and positive q_query, num_supports, context_topk")
    start = time.time()
    seed_everything(args.seed)
    args.setting = "unsup"
    args.query_generation = "NaQ"
    args.type = "feature"
    data = load_data(args)
    labels = data.labels.detach().cpu().numpy()
    features = F.normalize(data.features.float(), p=2, dim=1)
    n = features.size(0)
    rng = np.random.default_rng(args.seed)
    candidates = np.unique(np.asarray(data.id_train, dtype=np.int64))
    centers = rng.choice(candidates, size=min(args.num_supports, len(candidates)), replace=False)

    edges = data.edge_index.detach().cpu().numpy()
    adjacency = sp.csr_matrix((np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])), shape=(n, n))
    adjacency.data[:] = 1
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    transition_t = (sp.diags(inv) @ adjacency).transpose().tocsr()
    # Curvature definition: triangle-augmented Forman edge curvature on the
    # underlying simple, undirected graph, F(u,v)=4-deg(u)-deg(v)+3*T(u,v).
    # Curvature is a selection signal only: no ground truth is used.
    # For efficiency compute common-neighbor counts only for existing edges.
    undirected = adjacency.maximum(adjacency.transpose()).tocsr()
    undirected.setdiag(0)
    undirected.eliminate_zeros()
    undirected.sort_indices()
    undir_degree = np.diff(undirected.indptr).astype(np.float64)
    coo = adjacency.tocoo()
    directed_u, directed_v = coo.row, coo.col
    unique_pairs = np.stack((np.minimum(directed_u, directed_v), np.maximum(directed_u, directed_v)), axis=1)
    unique_pairs, inverse_pair = np.unique(unique_pairs, axis=0, return_inverse=True)
    triangle_counts = np.empty(len(unique_pairs), dtype=np.float64)
    for e, (u, v) in enumerate(unique_pairs):
        nu = undirected.indices[undirected.indptr[u] : undirected.indptr[u + 1]]
        nv = undirected.indices[undirected.indptr[v] : undirected.indptr[v + 1]]
        triangle_counts[e] = np.intersect1d(nu, nv, assume_unique=True).size
    forman = 4.0 - undir_degree[unique_pairs[:, 0]] - undir_degree[unique_pairs[:, 1]] + 3.0 * triangle_counts
    edge_forman = forman[inverse_pair]
    # Within-row z-score avoids raw degree scale dominating the softmax.
    directed_order = np.lexsort((directed_v, directed_u))
    if not np.array_equal(directed_u[directed_order], np.repeat(np.arange(n), np.diff(adjacency.indptr))):
        raise RuntimeError("Unexpected adjacency ordering when constructing curvature transition")
    score = edge_forman[directed_order]
    row_avg = np.divide(np.bincount(directed_u[directed_order], weights=score, minlength=n), np.maximum(degree, 1), out=np.zeros(n), where=degree > 0)
    centered = score - np.repeat(row_avg, np.diff(adjacency.indptr))
    row_std = np.sqrt(
        np.divide(np.bincount(directed_u[directed_order], weights=centered**2, minlength=n), np.maximum(degree, 1), out=np.zeros(n), where=degree > 0)
    )
    standardized = centered / np.maximum(np.repeat(row_std, np.diff(adjacency.indptr)), 1.0)
    weights = np.exp(args.curvature_beta * np.clip(standardized, -3.0, 3.0))
    weight_sum = np.bincount(directed_u[directed_order], weights=weights, minlength=n)
    weights = weights / np.maximum(np.repeat(weight_sum, np.diff(adjacency.indptr)), 1e-12)
    curvature_transition_t = sp.csr_matrix((weights, adjacency.indices.copy(), adjacency.indptr.copy()), shape=(n, n)).transpose().tocsr()
    contexts = {"ppr": [], "one_hop": [], "curvature": []}
    rows = []
    metric_keys = [
        "query_class_recall",
        "query_node_coverage",
        "false_query_class_recall",
        "false_query_node_coverage",
        "context_class_precision",
        "context_node_relevance",
        "context_irrelevant_node_ratio",
        "context_non_support_node_ratio",
        "query_classes",
        "context_classes",
        "false_query_count",
        "context_size",
        "full_query_class_coverage",
        "full_false_query_class_coverage",
    ]
    collected = {name: {key: [] for key in metric_keys} for name in contexts}

    def calculate(center, query_ids, context_ids):
        query_labels = labels[query_ids]
        support_label = labels[center]
        context_labels = labels[context_ids]
        query_classes = set(query_labels.tolist())
        context_classes = set(context_labels.tolist())
        false_queries = query_labels[query_labels != support_label]
        false_classes = set(false_queries.tolist())
        covered_classes = query_classes & context_classes
        covered_false = false_classes & context_classes
        result = {
            "query_class_recall": len(covered_classes) / len(query_classes),
            "query_node_coverage": float(np.isin(query_labels, list(context_classes)).mean()),
            "false_query_class_recall": len(covered_false) / len(false_classes) if false_classes else None,
            "false_query_node_coverage": float(np.isin(false_queries, list(context_classes)).mean()) if len(false_queries) else None,
            "context_class_precision": len(covered_classes) / len(context_classes),
            "context_node_relevance": float(np.isin(context_labels, list(query_classes)).mean()),
            "context_irrelevant_node_ratio": float((~np.isin(context_labels, list(query_classes))).mean()),
            "context_non_support_node_ratio": float((context_labels != support_label).mean()),
            "query_classes": len(query_classes),
            "context_classes": len(context_classes),
            "false_query_count": len(false_queries),
            "context_size": len(context_ids),
            "full_query_class_coverage": float(query_classes <= context_classes),
            "full_false_query_class_coverage": float(false_classes <= context_classes) if false_classes else None,
        }
        return result

    for index, center in enumerate(centers, start=1):
        center = int(center)
        # Exactly NAQ-Feat selection: normalized feature cosine, diagonal set to zero,
        # then torch.topk. Use rowwise calculation to avoid N x N allocation.
        scores = features @ features[center]
        scores[center] = 0.0
        query_ids = torch.topk(scores, k=min(args.q_query, n), dim=0).indices.cpu().numpy()
        # A zeroed diagonal may still be top-k on rare zero-feature/tied cases.
        # Preserve this original NAQ behavior, but log self query cases.

        restart = np.zeros(n, dtype=np.float32)
        restart[center] = 1.0
        probability = restart.copy()
        for _ in range(args.ppr_steps):
            probability = args.PPR_alpha * restart + (1 - args.PPR_alpha) * (transition_t @ probability)
        curvature_probability = restart.copy()
        for _ in range(args.ppr_steps):
            curvature_probability = args.PPR_alpha * restart + (1 - args.PPR_alpha) * (curvature_transition_t @ curvature_probability)
        probability[center] = -1
        curvature_probability[center] = -1
        k = min(args.context_topk, n - 1)
        ppr = np.argpartition(probability, -k)[-k:]
        ppr = ppr[np.argsort(-probability[ppr])]
        curvature_neighbors = np.argpartition(curvature_probability, -k)[-k:]
        curvature_neighbors = curvature_neighbors[np.argsort(-curvature_probability[curvature_neighbors])]
        begin, end = adjacency.indptr[center], adjacency.indptr[center + 1]
        hop = adjacency.indices[begin:end]
        if len(hop) > args.context_topk:
            per_node_rng = np.random.default_rng(np.random.SeedSequence([args.seed, center]))
            hop = per_node_rng.choice(hop, size=args.context_topk, replace=False)

        # Include center in BOTH contexts (total capacity topk+1).
        for name, neighbors in (("ppr", ppr), ("one_hop", hop), ("curvature", curvature_neighbors)):
            context_ids = np.concatenate(([center], neighbors)).astype(np.int64)
            stats = calculate(center, query_ids, context_ids)
            for key, value in stats.items():
                if value is not None:
                    collected[name][key].append(value)
            rows.append(
                {
                    "support_id": center,
                    "method": name,
                    "support_class": int(labels[center]),
                    "query_classes": sorted(set(labels[query_ids].tolist())),
                    "context_classes": sorted(set(labels[context_ids].tolist())),
                    **stats,
                }
            )
        if index % 200 == 0:
            print(f"Analyzed {index}/{len(centers)} supports", flush=True)

    methods = {}
    for name, groups in collected.items():
        methods[name] = {key: summarize(values) for key, values in groups.items()}
        methods[name]["supports_with_false_queries"] = len(groups["false_query_node_coverage"])
        methods[name]["query_class_coverage_histogram"] = np.histogram(groups["query_class_recall"], bins=np.linspace(0, 1, 11))[0].tolist()
    report = {
        "experiment": "exp014",
        "description": "1-shot, training-free, bidirectional NAQ-Feat coverage: PPR / one-hop / curvature-weighted PPR",
        "curvature_definition": "triangle-augmented Forman F(u,v)=4-degree(u)-degree(v)+3*common_neighbors(u,v) on undirected graph; weighted directed walks",
        "curvature_stats": {"min": float(forman.min()), "mean": float(forman.mean()), "max": float(forman.max()), "edge_count": int(len(forman))},
        "config": vars(args),
        "num_nodes": n,
        "num_supports_analyzed": len(centers),
        "methods": methods,
        "per_support": rows,
        "notes": [
            "Each support's NAQ-Feat top-10 queries are fixed and compared against all three contexts.",
            "Curvature is triangle-augmented Forman edge curvature; higher curvature edges get larger normalized random-walk transition weight.",
            "Curvature method is curvature-weighted PPR, not a pure curvature ranking; both PPR variants use the same alpha and step count.",
            "PPR and curvature always select topK nodes, including potentially zero-probability nodes on disconnected components (consistent with exp013).",
            "Query-class recall and context-class precision are set-based; node coverage and context relevance are frequency-weighted.",
            "False-positive queries have a ground-truth class different from their selected support. Undefined false-query metrics are excluded when no such queries exist.",
            "Context includes center plus topK selected nodes. 1-hop has no padded nodes in metrics.",
            "Ground-truth labels are used only for analysis, never sampling or optimization.",
            "Context-class precision treats support class as relevant only if a query has that class.",
            "Results are descriptive; class presence alone does not prove attention can align queries.",
        ],
        "elapsed_seconds": round(time.time() - start, 2),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json_path)), exist_ok=True)
    temp = args.json_path + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temp, args.json_path)
    for name, stats in methods.items():
        print(
            name,
            {
                key: round(stats[key]["mean"], 4) if stats[key]["mean"] is not None else None
                for key in [
                    "query_class_recall",
                    "query_node_coverage",
                    "false_query_node_coverage",
                    "context_class_precision",
                    "context_node_relevance",
                    "context_irrelevant_node_ratio",
                ]
            },
            flush=True,
        )
    print(f"Saved: {args.json_path}", flush=True)


if __name__ == "__main__":
    main()
