"""exp036: GT coverage of wrong-class NAQ-Feat queries in PPR/GAE contexts.

Run from repository root. Dataset GT is used ONLY for post-sampling analysis.
NAQ support/query sampling stays exactly the original feature-based protocol.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import load_data
from exp032 import ppr_context, ppr_transition
from task_generator import proto_task_generator
from utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="exp036: cross-GT-class context coverage")
    p.add_argument("--dataset", default="Amazon_clothing")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=1)
    p.add_argument("--q_query", type=int, default=10)
    p.add_argument("--num_tasks", type=int, default=200)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--ppr_alpha", type=float, default=0.15)
    p.add_argument("--ppr_iter", type=int, default=60)
    p.add_argument("--ppr_tol", type=float, default=1e-7)
    p.add_argument("--embedding_path", default="results/Amazon_clothing/exp035/embeddings.pt")
    p.add_argument("--sample_episodes", type=int, default=5)
    p.add_argument("--sample_query_limit", type=int, default=10)
    p.add_argument("--sample_candidate_limit", type=int, default=8)
    p.add_argument("--json_path", default="exp036.json")
    p.add_argument("--output_dir", default="results/Amazon_clothing/exp036")
    return p.parse_args()


def get_embeddings(path, n):
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        saved = torch.load(path, map_location="cpu")
    z = saved["embeddings"] if isinstance(saved, dict) else saved
    z = torch.as_tensor(z).float()
    if z.ndim != 2 or z.shape[0] != n:
        raise ValueError(f"Embedding has shape {tuple(z.shape)}, expected [{n}, D]")
    if isinstance(saved, dict) and "node_ids" in saved:
        node_ids = torch.as_tensor(saved["node_ids"]).cpu()
        if not torch.equal(node_ids, torch.arange(n)):
            raise ValueError("Embedding rows do not follow dataset node order")
    return F.normalize(z, dim=-1)


def make_ranker(normalized, topk, name):
    cache = {}
    n = normalized.shape[0]

    def rank(center):
        center = int(center)
        if center not in cache:
            score = normalized @ normalized[center]
            score[center] = -float("inf")
            selected = torch.topk(score, k=min(topk, n - 1), sorted=True).indices.tolist()
            cache[center] = [center, *map(int, selected)]
        return cache[center]

    return rank, cache


def summarize(vals):
    array = np.asarray(vals, dtype=np.float64)
    return {"mean": float(array.mean()) if len(array) else None, "count": int(len(array))}


def main():
    a = parse_args()
    if a.k_shot != 1 or a.n_way < 1 or a.q_query < 1 or a.num_tasks < 1 or a.topk < 1:
        raise ValueError("Requires 1-shot, positive episode parameters and topk")
    a.setting = "unsup"
    a.query_generation = "NaQ"
    a.type = "feature"
    seed_everything(a.seed)
    torch.set_num_threads(4)
    data = load_data(a)
    # Same original feature-based NAQ sampler as exp032 (GT is not involved).
    data.add_diffusion(a)
    labels = data.labels.detach().cpu().numpy().reshape(-1)
    n = len(labels)
    feature = F.normalize(data.features.detach().cpu().float(), dim=-1)
    embeddings = get_embeddings(a.embedding_path, n)
    transition, dangling = ppr_transition(data.edge_index, n)
    transposed = transition.transpose().tocsr()
    ppr_cache = {}

    def ppr(center):
        center = int(center)
        if center not in ppr_cache:
            ppr_cache[center] = ppr_context(center, transposed, dangling, a.ppr_alpha, a.ppr_iter, a.ppr_tol, a.topk)
        return ppr_cache[center]

    embedding_knn, embedding_cache = make_ranker(embeddings, a.topk, "embedding")
    feature_knn, feature_cache = make_ranker(feature, a.topk, "feature")
    methods = {"support_only": lambda s: [int(s)], "ppr": ppr, "embedding_knn": embedding_knn, "feature_knn": feature_knn}
    results = defaultdict(list)
    samples = []
    example_text = []
    total_wrong = 0
    total_queries = 0

    for ep in range(a.num_tasks):
        supports, pseudo_s, queries, pseudo_q = proto_task_generator(a, data)
        supports = np.asarray(supports, dtype=np.int64).reshape(-1)
        queries = np.asarray(queries, dtype=np.int64).reshape(-1)
        pseudo_s = torch.as_tensor(pseudo_s).cpu().numpy().reshape(-1)
        pseudo_q = torch.as_tensor(pseudo_q).cpu().numpy().reshape(-1)
        owners = np.asarray([int(np.flatnonzero(pseudo_s == y)[0]) for y in pseudo_q])
        contexts = {name: [fn(s) for s in supports] for name, fn in methods.items()}
        # Give support/query GT classes the first letters, then include all context classes.
        # Candidate subgraphs can contain GT classes absent from support/query nodes.
        episode_gt = [int(v) for v in labels[supports]]
        episode_gt.extend(int(v) for v in labels[queries])
        for candidate_lists in contexts.values():
            for candidate_nodes in candidate_lists:
                episode_gt.extend(int(labels[int(v)]) for v in candidate_nodes)
        all_classes = list(dict.fromkeys(episode_gt))
        alphabet = {lab: chr(65 + i) if i < 26 else f"C{i}" for i, lab in enumerate(all_classes)}
        ep_supports = []
        ep_queries = []
        for j, s in enumerate(supports):
            ep_supports.append({"node": int(s), "gt_class": alphabet[int(labels[s])], "pseudo_class": int(pseudo_s[j])})
        for q, owner in zip(queries, owners):
            q = int(q)
            s = int(supports[owner])
            sg = int(labels[s])
            qg = int(labels[q])
            wrong = sg != qg
            total_queries += 1
            total_wrong += int(wrong)
            results["support_query_gt_match"].append(float(not wrong))
            this_q = {
                "node": q,
                "assigned_support": s,
                "support_gt": alphabet[sg],
                "query_gt": alphabet[qg],
                "support_query_gt_mismatch": bool(wrong),
                "methods": {},
            }
            for method, lists in contexts.items():
                # Prevent trivial coverage from query node itself.
                candidate = [int(v) for v in dict.fromkeys(lists[owner]) if int(v) != q]
                candidate_labels = labels[np.asarray(candidate, dtype=np.int64)] if candidate else np.empty(0, dtype=labels.dtype)
                same_q = candidate_labels == qg
                same_s = candidate_labels == sg
                hit = bool(same_q.any())
                non_support_count = int((~same_s).sum())
                ratio = float(same_q.mean()) if len(candidate) else 0.0
                results[f"{method}.all_query_coverage"].append(float(hit))
                results[f"{method}.candidate_size"].append(len(candidate))
                if wrong:
                    results[f"{method}.wrong_query_coverage"].append(float(hit))
                    results[f"{method}.wrong_query_gt_fraction"].append(ratio)
                    results[f"{method}.wrong_query_gt_count"].append(int(same_q.sum()))
                    results[f"{method}.wrong_query_non_support_gt_count"].append(non_support_count)
                    results[f"{method}.wrong_query_non_support_gt_presence"].append(float(non_support_count > 0))
                if len(samples) < a.sample_episodes:
                    this_q["methods"][method] = {
                        "covered": hit,
                        "matching_candidate_count": int(same_q.sum()),
                        "candidate_gt_classes": [alphabet[int(x)] for x in candidate_labels[: a.sample_candidate_limit]],
                        "candidate_count": len(candidate),
                    }
            if len(samples) < a.sample_episodes:
                ep_queries.append(this_q)
        if len(samples) < a.sample_episodes:
            samples.append(
                {
                    "episode": ep + 1,
                    "class_mapping_note": "Letters represent actual GT classes, assigned independently for each episode.",
                    "supports": ep_supports,
                    "queries": (
                        [q for q in ep_queries if q["support_query_gt_mismatch"]] + [q for q in ep_queries if not q["support_query_gt_mismatch"]]
                    )[: a.sample_query_limit],
                    "wrong_query_count_all": int(sum(labels[int(supports[o])] != labels[int(q)] for q, o in zip(queries, owners))),
                    "query_count_all": len(queries),
                }
            )
        if (ep + 1) % 20 == 0:
            print(f"[exp036] {ep + 1}/{a.num_tasks} episodes, wrong-class queries={total_wrong}", flush=True)

    summary = {name: summarize(values) for name, values in sorted(results.items())}
    report = {
        "experiment": "exp036",
        "description": "Cross-class coverage of NAQ-Feat queries from support-centered candidate sets",
        "config": vars(a),
        "counts": {
            "episodes": a.num_tasks,
            "queries": total_queries,
            "support_query_gt_mismatch_queries": total_wrong,
            "same_gt_queries": total_queries - total_wrong,
        },
        "metrics": summary,
        "examples": samples,
        "notes": [
            "The NAQ support/query pairs are generated using ORIGINAL feature-based NAQ, not GAE embeddings.",
            "GT is used ONLY to evaluate coverage and show examples, never for episode generation.",
            "wrong_query_coverage is conditional on GT(query) != GT(assigned NAQ support).",
            "A candidate set covers a query if it contains at least one node of the query's GT class.",
            "Support-center included, query node itself excluded from each candidate set.",
            "Support-only coverage is 0 by definition on the wrong-class subset.",
            "PPR and kNN each use topk neighbors plus the support itself; budget is matched.",
            "A/B/C GT letters are episode-local. Pseudo-class IDs distinguish NAQ assignments and are not GT labels.",
            "Sampled examples are initial training-style episodes; results include every episode.",
        ],
    }
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = Path(a.json_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "exp036 — GT mismatch-focused NAQ coverage",
        "",
        f"Episodes: {a.num_tasks}; queries: {total_queries}; mismatched: {total_wrong} ({total_wrong/total_queries:.2%})",
        "",
    ]
    for method in methods:
        lines.append(
            f"{method}: wrong-class coverage = {summary[f'{method}.wrong_query_coverage']['mean']:.4%}; "
            f"all-query coverage = {summary[f'{method}.all_query_coverage']['mean']:.4%}; "
            f"wrong-query mean matching candidates = {summary[f'{method}.wrong_query_gt_count']['mean']:.3f}"
        )
    lines.append("\nSampled training episodes (GT classes shown as A/B/C/... per episode):")
    for episode in samples:
        lines.append(f"\nEPISODE {episode['episode']} — mismatched {episode['wrong_query_count_all']}/{episode['query_count_all']}")
        for support in episode["supports"]:
            lines.append(f"  Support {support['node']} = {support['gt_class']} (pseudo {support['pseudo_class']})")
        for query in episode["queries"]:
            lines.append(
                f"  Query {query['node']}: Support {query['assigned_support']} "
                f"{query['support_gt']} -> Query {query['query_gt']} "
                f"{'MISMATCH' if query['support_query_gt_mismatch'] else 'MATCH'}"
            )
            if query["support_query_gt_mismatch"]:
                for method, data in query["methods"].items():
                    lines.append(
                        f"    {method}: {'COVER' if data['covered'] else 'MISS'} "
                        f"({data['matching_candidate_count']} matching); "
                        f"first candidate GT classes: {','.join(data['candidate_gt_classes'])}"
                    )
    (out / "result.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8]), flush=True)
    print(f"Saved {target} and {out / 'result.txt'}", flush=True)


if __name__ == "__main__":
    main()
