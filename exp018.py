"""exp018: training-free semantic mismatch audit of exp017 pseudo-episodes.

Uses the exact RawFeatureAnchorSampler from exp017: raw-feature cosine,
random disjoint anchors, a shared candidate pool, and sequential exclusion.
Ground-truth classes are accessed ONLY after sampling to compute diagnostics.
Run from the repository root alongside exp017.py, data.py, and utils.py.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from data import load_data
from exp017 import RawFeatureAnchorSampler
from utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="exp018: exp017 pseudo-task semantic mismatch")
    parser.add_argument("--dataset", default="Amazon_clothing")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--q_query", type=int, default=10)
    parser.add_argument("--pool_size", type=int, default=1000)
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--json_path", default="exp018.json")
    return parser.parse_args()


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "std": None, "count": 0}
    return {"mean": float(values.mean()), "std": float(values.std()), "count": int(values.size)}


def main():
    args = parse_args()
    if args.k_shot != 1:
        raise ValueError("exp018 currently targets the exp017 1-shot setting")
    if min(args.n_way, args.q_query, args.num_episodes, args.pool_size) < 1:
        raise ValueError("n_way, q_query, num_episodes, and pool_size must be positive")

    start = time.time()
    seed_everything(args.seed)
    args.setting = "unsup"
    data = load_data(args)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    features = data.features.to(device)
    labels = data.labels.detach().cpu().numpy()
    sampler = RawFeatureAnchorSampler(features, data.id_train, args.n_way, args.k_shot, args.q_query, args.pool_size)

    metric_keys = [
        "support_query_mismatch_rate",
        "support_query_match_rate",
        "mismatched_query_count",
        "distinct_query_classes",
        "distinct_false_query_classes",
        "support_class_in_queries",
        "anchor_support_match",
        "anchor_query_match_rate",
        "anchor_query_mismatch_rate",
        "anchor_vs_support_query_match_gain",
        "full_support_query_match",
        "all_queries_mismatch",
        "query_majority_matches_support",
        "support_class_query_fraction",
    ]
    values = {key: [] for key in metric_keys}
    histogram = np.zeros(args.q_query + 1, dtype=np.int64)
    per_way = []
    per_episode = []

    for episode_id in range(args.num_episodes):
        # Capture anchors without changing exp017's sampler logic or RNG sequence:
        # exp017.sample() uses one randperm to select anchors and candidates.
        # We replay the randperm by saving/restoring the CPU/CUDA RNG state.
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        supports, _, queries, _ = sampler.sample()
        torch.set_rng_state(rng_state)
        if device.type == "cuda":
            torch.cuda.set_rng_state(cuda_rng_state, device)
        perm = torch.randperm(len(sampler.train_ids), device=device)
        anchors = sampler.train_ids[perm[: args.n_way]]

        supports_np = supports.reshape(args.n_way).detach().cpu().numpy()
        queries_np = queries.reshape(args.n_way, args.q_query).detach().cpu().numpy()
        anchors_np = anchors.detach().cpu().numpy()
        episode_mismatch = []

        for way in range(args.n_way):
            support_id = int(supports_np[way])
            anchor_id = int(anchors_np[way])
            query_ids = queries_np[way]
            support_label = int(labels[support_id])
            anchor_label = int(labels[anchor_id])
            query_labels = labels[query_ids]
            matched = query_labels == support_label
            anchor_matched = query_labels == anchor_label
            mismatches = int((~matched).sum())
            histogram[mismatches] += 1
            query_class_ids, counts = np.unique(query_labels, return_counts=True)
            most_frequent = int(counts.max())
            query_majority_matches_support = bool(support_label in query_class_ids[counts == most_frequent])
            row = {
                "episode": episode_id,
                "way": way,
                "anchor_id": anchor_id,
                "support_id": support_id,
                "query_ids": query_ids.tolist(),
                "anchor_class": anchor_label,
                "support_class": support_label,
                "query_classes": query_labels.astype(int).tolist(),
                "support_query_mismatch_rate": mismatches / args.q_query,
                "support_query_match_rate": 1 - mismatches / args.q_query,
                "mismatched_query_count": mismatches,
                "distinct_query_classes": len(query_class_ids),
                "distinct_false_query_classes": int((query_class_ids != support_label).sum()),
                "support_class_in_queries": int(support_label in query_class_ids),
                "anchor_support_match": int(anchor_label == support_label),
                "anchor_query_match_rate": float(anchor_matched.mean()),
                "anchor_query_mismatch_rate": float((~anchor_matched).mean()),
                "anchor_vs_support_query_match_gain": float(anchor_matched.mean() - matched.mean()),
                "full_support_query_match": int(mismatches == 0),
                "all_queries_mismatch": int(mismatches == args.q_query),
                "query_majority_matches_support": int(query_majority_matches_support),
                "support_class_query_fraction": float(matched.mean()),
            }
            for key in metric_keys:
                values[key].append(row[key])
            per_way.append(row)
            episode_mismatch.append(mismatches / args.q_query)

        per_episode.append(
            {
                "episode": episode_id,
                "mean_support_query_mismatch_rate": float(np.mean(episode_mismatch)),
                "ways_with_any_mismatch": int(sum(x > 0 for x in episode_mismatch)),
                "ways_all_queries_mismatch": int(sum(x == 1 for x in episode_mismatch)),
            }
        )
        if (episode_id + 1) % 200 == 0:
            print(f"Analyzed {episode_id + 1}/{args.num_episodes} episodes", flush=True)

    aggregated = {key: summary(items) for key, items in values.items()}
    report = {
        "experiment": "exp018",
        "description": "Training-free ground-truth mismatch audit of exp017 raw-feature anchor-centered pseudo-episodes",
        "config": vars(args).copy(),
        "num_nodes": int(features.size(0)),
        "num_episodes": args.num_episodes,
        "num_ways_analyzed": args.num_episodes * args.n_way,
        "metrics": aggregated,
        "mismatched_query_count_histogram": {str(i): int(count) for i, count in enumerate(histogram)},
        "episodes": per_episode,
        "per_way": per_way,
        "notes": [
            "Exactly the exp017 sampler is imported: anchors are not included in support/query groups.",
            "Ground-truth labels are used only after sampler.sample(), never to select nodes.",
            "Mismatch means query ground-truth class differs from its pseudo-way support ground-truth class.",
            "Sampling is without replacement inside each episode; nodes may reappear across episodes.",
            "Anchor–support and anchor–query match statistics are diagnostic only.",
            "Query majority matches support includes ties for highest frequency.",
            "Statistics average over pseudo-ways; in the 1-shot fixed-10-query setting this also equals the query-weighted mean.",
            "Sampled episodes characterize task-generation mismatch, not model prediction errors.",
        ],
        "elapsed_seconds": round(time.time() - start, 2),
    }
    path = os.path.abspath(args.json_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(path + ".tmp", path)
    for key in [
        "support_query_mismatch_rate",
        "mismatched_query_count",
        "distinct_query_classes",
        "anchor_support_match",
        "anchor_query_mismatch_rate",
        "full_support_query_match",
    ]:
        print(f"{key}: {aggregated[key]['mean']:.4f}", flush=True)
    print(f"Saved: {path}", flush=True)


if __name__ == "__main__":
    main()
