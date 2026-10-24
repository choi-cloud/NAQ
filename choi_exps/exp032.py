"""exp032: Ground-truth neighbor agreement in support-centered graph contexts.

Run from the NAQ repository root (with data.py, task_generator.py, utils.py).
This is a diagnostic on *training-style NAQ-Feat pseudo-episodes*, NOT a
supervised downstream accuracy evaluation and NOT a model-training experiment.
Ground-truth labels are used only after sampling, for measurement.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from data import load_data
from task_generator import proto_task_generator
from utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="exp032: GT agreement of query nearest context nodes")
    p.add_argument("--dataset", default="Amazon_clothing")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=1)
    p.add_argument("--q_query", type=int, default=10)
    p.add_argument("--num_tasks", type=int, default=200)
    p.add_argument("--context", choices=["1hop", "ppr"], default="1hop")
    p.add_argument("--ppr_topk", type=int, default=20)
    p.add_argument("--ppr_alpha", type=float, default=0.15)
    p.add_argument("--ppr_iter", type=int, default=60)
    p.add_argument("--ppr_tol", type=float, default=1e-7)
    p.add_argument("--topk", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--exclude_query", action="store_true", default=True)
    p.add_argument("--include_query", dest="exclude_query", action="store_false")
    p.add_argument("--json_path", default="exp032.json")
    return p.parse_args()


def edge_neighbors(edge_index, num_nodes):
    edges = edge_index.detach().cpu().numpy()
    neighborhoods = [set() for _ in range(num_nodes)]
    for source, target in zip(edges[0], edges[1]):
        if source != target:
            neighborhoods[int(source)].add(int(target))
    return [sorted(ns) for ns in neighborhoods]


def ppr_transition(edge_index, num_nodes):
    src, dst = edge_index.detach().cpu().numpy()
    adjacency = sp.csr_matrix((np.ones(len(src), dtype=np.float64), (src, dst)), shape=(num_nodes, num_nodes))
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv_degree = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    transition = sp.diags(inv_degree) @ adjacency
    return transition.tocsr(), degree == 0


def ppr_context(center, transition_t, dangling, alpha, iterations, tolerance, topk):
    n = transition_t.shape[0]
    restart = np.zeros(n, dtype=np.float64)
    restart[center] = 1.0
    rank = restart.copy()
    for _ in range(iterations):
        new = alpha * restart + (1.0 - alpha) * (transition_t @ rank + rank[dangling].sum() * restart)
        if np.abs(new - rank).sum() < tolerance:
            rank = new
            break
        rank = new
    rank[center] = -np.inf
    k = min(topk, n - 1)
    indices = np.argpartition(-rank, k - 1)[:k] if k > 0 else np.empty(0, dtype=np.int64)
    indices = indices[np.lexsort((indices, -rank[indices]))]
    return [center, *indices.tolist()]


def summarize(rows):
    # Each query contributes one vote; use valid counts for Top-k when context too small.
    out = {}
    for name, values in rows.items():
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        out[name] = {
            "mean": float(finite.mean()) if len(finite) else None,
            "count": int(len(finite)),
            "std": float(finite.std()) if len(finite) else None,
        }
    return out


def main():
    args = parse_args()
    if args.k_shot != 1 or args.num_tasks <= 0 or args.q_query <= 0:
        raise ValueError("Requires 1-shot, positive num_tasks and q_query")
    if any(k <= 0 for k in args.topk):
        raise ValueError("topk must contain positive integers")
    if not 0 < args.ppr_alpha <= 1 or args.ppr_topk <= 0:
        raise ValueError("Invalid PPR configuration")

    args.setting = "unsup"
    args.query_generation = "NaQ"
    args.type = "feature"
    seed_everything(args.seed)
    torch.set_num_threads(4)
    data = load_data(args)
    # IMPORTANT: match original NAQ-Feat support->query sampling exactly.
    # Uses full raw-feature cosine matrix with diagonal zeroed by add_diffusion.
    data.add_diffusion(args)
    features = F.normalize(data.features.detach().float().cpu(), p=2, dim=-1)
    labels = data.labels.detach().cpu().numpy().reshape(-1)
    n = len(labels)
    neighborhoods = edge_neighbors(data.edge_index, n)

    transition_t = None
    dangling = None
    if args.context == "ppr":
        transition, dangling = ppr_transition(data.edge_index, n)
        transition_t = transition.transpose().tocsr()
    cache = {}

    def get_context(support):
        support = int(support)
        if support not in cache:
            if args.context == "1hop":
                cache[support] = [support, *neighborhoods[support]]
            else:
                cache[support] = ppr_context(support, transition_t, dangling, args.ppr_alpha, args.ppr_iter, args.ppr_tol, args.ppr_topk)
        return cache[support]

    metrics = defaultdict(list)
    topk = sorted(set(args.topk))
    rows_by_task = []
    num_queries = 0
    for task_index in range(args.num_tasks):
        supports, pseudo_support, queries, pseudo_query = proto_task_generator(args, data)
        supports = np.asarray(supports, dtype=np.int64).reshape(-1)
        queries = np.asarray(queries, dtype=np.int64).reshape(-1)
        pseudo_support = torch.as_tensor(pseudo_support).reshape(-1).numpy()
        pseudo_query = torch.as_tensor(pseudo_query).reshape(-1).numpy()
        if len(supports) != args.n_way or len(queries) != args.n_way * args.q_query:
            raise RuntimeError("Unexpected NAQ episode shape")
        if not all(np.count_nonzero(pseudo_query == y) == args.q_query for y in pseudo_support):
            raise RuntimeError("Unexpected NAQ query pseudo-label count")

        # Owner of each query determined by NAQ pseudo-label, NOT actual GT label.
        owners = np.array([np.flatnonzero(pseudo_support == y)[0] for y in pseudo_query])
        contexts = [get_context(s) for s in supports]
        task_metrics = defaultdict(list)
        for q, owner in zip(queries.tolist(), owners.tolist()):
            q_label = int(labels[q])
            s = int(supports[owner])
            metrics["naq_support_query_gt_match"].append(float(labels[s] == q_label))
            task_metrics["naq_support_query_gt_match"].append(float(labels[s] == q_label))
            for name, candidate_ids in [
                ("own", contexts[owner]),
                ("other", list(dict.fromkeys(nid for c, context in enumerate(contexts) if c != owner for nid in context))),
                ("all", list(dict.fromkeys(nid for context in contexts for nid in context))),
            ]:
                candidate_ids = [nid for nid in candidate_ids if not (args.exclude_query and nid == q)]
                if not candidate_ids:
                    for k in topk:
                        metrics[f"{name}_precision_at_{k}"].append(float("nan"))
                    metrics[f"{name}_size"].append(0)
                    metrics[f"{name}_top1_gt_match"].append(float("nan"))
                    continue
                ids = torch.as_tensor(candidate_ids, dtype=torch.long)
                similarities = features[ids] @ features[q]
                ranking = torch.argsort(similarities, descending=True).numpy()
                ranked_ids = ids[ranking].numpy()
                agreement = (labels[ranked_ids] == q_label).astype(np.float64)
                metrics[f"{name}_size"].append(len(candidate_ids))
                metrics[f"{name}_candidate_gt_fraction"].append(float((labels[ids.numpy()] == q_label).mean()))
                metrics[f"{name}_top1_gt_match"].append(float(agreement[0]))
                task_metrics[f"{name}_top1_gt_match"].append(float(agreement[0]))
                for k in topk:
                    # Precision denominator min(k, available candidates), reported as such.
                    metrics[f"{name}_precision_at_{k}"].append(float(agreement[:k].mean()))
                if name == "own":
                    metrics["own_top1_is_support"].append(float(ranked_ids[0] == s))
                    metrics["own_gt_present"].append(float(np.any(labels[ids.numpy()] == q_label)))
            num_queries += 1
        rows_by_task.append({"task": task_index, **{key: float(np.mean(val)) for key, val in task_metrics.items() if val}})
        if (task_index + 1) % 20 == 0:
            print(f"Task {task_index + 1}/{args.num_tasks}", flush=True)

    report = {
        "experiment": "exp032",
        "description": "Training-style NAQ-Feat episode, GT agreement of query->support-context feature nearest neighbors",
        "config": vars(args),
        "num_queries": num_queries,
        "metrics": summarize(metrics),
        "task_means": rows_by_task,
        "notes": [
            "Ground truth is used ONLY for analysis, not for NAQ query sampling.",
            "own: context of the NAQ pseudo-label's support; other: other supports' contexts; all: union.",
            "Query node itself excluded from all context rankings by default to avoid trivial self-match.",
            "Feature nearest means raw feature cosine (same basis as NAQ-Feat), NOT GCN embedding similarity.",
            "1hop follows original directed edge_index; support itself is always included.",
            "PPR uses directed row-normalized random walk with restart, computed separately from NAQ feature sampling.",
            "Training-style pseudo-episodes may contain mixed GT classes; not downstream frozen exp026 tasks.",
            "Top-k precision averages over min(k, number of available candidates).",
        ],
    }
    path = Path(args.json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    out = Path(f"results/{args.dataset}/exp032")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "result.txt").open("w", encoding="utf-8") as f:
        for key, item in sorted(report["metrics"].items()):
            f.write(f"{key}: {item['mean']} (n={item['count']})\n")
    print(json.dumps({"experiment": "exp032", "num_queries": num_queries, "metrics": report["metrics"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
