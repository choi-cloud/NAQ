"""exp005: training-free analysis of support-centered PPR neighborhoods.

Run from the original NAQ repository root (imports its data.py).
True node labels are used for analysis ONLY, never for training or selection.
PPR definition and hyperparameters match exp003/exp004.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import scipy.sparse as sp
import torch

from data import load_data
from utils import seed_everything


class PPRNeighborhoods:
    def __init__(self, data, topk, alpha, steps):
        edge_index = data.edge_index.detach().cpu().numpy()
        n = data.features.size(0)
        adjacency = sp.csr_matrix((np.ones(edge_index.shape[1], dtype=np.float32), (edge_index[0], edge_index[1])), shape=(n, n))
        adjacency.data[:] = 1.0
        degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
        inverse = np.divide(1.0, degrees, out=np.zeros_like(degrees), where=degrees > 0)
        self.transition_t = (sp.diags(inverse) @ adjacency).transpose().tocsr()
        self.n = n
        self.topk = topk
        self.alpha = alpha
        self.steps = steps

    def get(self, center):
        restart = np.zeros(self.n, dtype=np.float32)
        restart[center] = 1.0
        probability = restart.copy()
        for _ in range(self.steps):
            probability = self.alpha * restart + (1 - self.alpha) * (self.transition_t @ probability)
        probability[center] = -1
        k = min(self.topk, self.n - 1)
        neighbors = np.argpartition(probability, -k)[-k:]
        return neighbors[np.argsort(-probability[neighbors])]


def summarize(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std()), "n": int(a.size)}


def aggregate(rows, topk, feature, labels, valid, rng):
    metrics = {key: [] for key in ["ppr", "random", "feature"]}
    per_support = {key: [] for key in metrics}
    rank_matches = [[] for _ in range(topk)]
    rank_cosines = [[] for _ in range(topk)]
    zero_norm_pairs = 0
    for center, ppr_neighbors in rows:
        # Feature NN excludes the center and uses the SAME number of nodes as PPR.
        cos = feature @ feature[center]
        cos[center] = -np.inf
        k = len(ppr_neighbors)
        top_feature = np.argpartition(cos, -k)[-k:]
        top_feature = top_feature[np.argsort(-cos[top_feature])]
        candidates = np.flatnonzero(valid)
        candidates = candidates[candidates != center]
        random_neighbors = rng.choice(candidates, size=k, replace=False)
        for name, ids in [("ppr", ppr_neighbors), ("random", random_neighbors), ("feature", top_feature)]:
            matches = (labels[ids] == labels[center]).astype(np.float64)
            similarities = feature[ids] @ feature[center]
            metrics[name].extend(zip(matches.tolist(), similarities.tolist()))
            per_support[name].append({"match_rate": float(matches.mean()), "feature_cosine": float(similarities.mean())})
            if name == "ppr":
                for r, (match, similarity) in enumerate(zip(matches, similarities)):
                    rank_matches[r].append(float(match))
                    rank_cosines[r].append(float(similarity))
        zero_norm_pairs += int(np.sum(np.linalg.norm(feature[ppr_neighbors], axis=1) == 0))
    report = {"methods": {}, "ppr_by_rank": [], "per_support_distribution": {}}
    for name, pairs in metrics.items():
        match, similarity = np.asarray(pairs, dtype=np.float64).T
        report["methods"][name] = {"class_match_rate": summarize(match), "feature_cosine": summarize(similarity)}
        report["per_support_distribution"][name] = {
            "class_match_rate": summarize([x["match_rate"] for x in per_support[name]]),
            "feature_cosine": summarize([x["feature_cosine"] for x in per_support[name]]),
        }
    for r in range(topk):
        report["ppr_by_rank"].append({"rank": r + 1, "class_match_rate": summarize(rank_matches[r]), "feature_cosine": summarize(rank_cosines[r])})
    for upper in [1, 4, 8, 16, topk]:
        kk = min(upper, topk)
        if kk < 1:
            continue
        report.setdefault("ppr_cumulative_topk", {})[str(kk)] = {
            "class_match_rate": summarize([v for rank in rank_matches[:kk] for v in rank]),
            "feature_cosine": summarize([v for rank in rank_cosines[:kk] for v in rank]),
        }
    return report


def parse_args():
    p = argparse.ArgumentParser(description="exp005: PPR neighbor label purity and raw feature similarity")
    p.add_argument("--dataset", type=str, default="Amazon_clothing")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--ppr_topk", type=int, default=16)
    p.add_argument("--ppr_steps", type=int, default=20)
    p.add_argument("--PPR_alpha", type=float, default=0.1)
    p.add_argument("--num_supports", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--json_path", type=str, default="exp005.json")
    return p.parse_args()


def main():
    args = parse_args()
    if args.ppr_topk < 1 or args.num_supports < 1:
        raise ValueError("--ppr_topk and --num_supports must be positive")
    start = time.time()
    seed_everything(args.seed)
    args.setting = "unsup"
    # load_data does not need NAQ diffusion or training for this analysis.
    data = load_data(args)
    labels = data.labels.detach().cpu().numpy()
    raw = data.features.detach().cpu().numpy()
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    normalized = np.divide(raw, norms, out=np.zeros_like(raw), where=norms > 0)
    all_centers = np.unique(np.asarray(data.id_train, dtype=np.int64))
    # Labels are only used for evaluation, not for choosing supports/neighbors.
    rng = np.random.default_rng(args.seed)
    count = min(args.num_supports, len(all_centers))
    centers = rng.choice(all_centers, size=count, replace=False)
    sampler = PPRNeighborhoods(data, args.ppr_topk, args.PPR_alpha, args.ppr_steps)
    rows = []
    for idx, center in enumerate(centers):
        rows.append((int(center), sampler.get(int(center))))
        if (idx + 1) % 200 == 0:
            print(f"PPR sampling: {idx + 1}/{count}", flush=True)
    # All nodes are eligible: exp003's PPR topk is not label/split filtered.
    report = aggregate(rows, min(args.ppr_topk, sampler.n - 1), normalized, labels, np.ones(sampler.n, dtype=bool), rng)
    result = {
        "experiment": "exp005",
        "description": "training-free PPR neighborhood label purity and raw feature cosine similarity",
        "config": vars(args),
        "num_nodes": int(sampler.n),
        "num_supports_analyzed": int(count),
        "num_zero_feature_norm_nodes": int((norms[:, 0] == 0).sum()),
        "note": "PPR excludes center; baseline random draws uniformly from all other nodes; feature baseline selects top raw cosine neighbors. True labels used for diagnostics only.",
        **report,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    folder = os.path.dirname(os.path.abspath(args.json_path))
    os.makedirs(folder, exist_ok=True)
    tmp_path = args.json_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    os.replace(tmp_path, args.json_path)
    print("\nMethod summaries:")
    for name, stats in report["methods"].items():
        print(f"{name:>8}: class_match={stats['class_match_rate']['mean']:.4f} " f"feature_cos={stats['feature_cosine']['mean']:.4f}")
    print("\nPPR cumulative Top-K:")
    for k, stats in report["ppr_cumulative_topk"].items():
        print(f"Top-{k:>2}: class_match={stats['class_match_rate']['mean']:.4f} " f"feature_cos={stats['feature_cosine']['mean']:.4f}")
    print(f"\nSaved: {args.json_path}")


if __name__ == "__main__":
    main()
