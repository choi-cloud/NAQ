"""exp053: Measure GT-class collisions in original NAQ training support sets.

No model training and no original NAQ files are modified. Support sampling uses
NAQ's original spt_sampling, the same sampler called by proto_task_generator.
These are newly generated episodes, not recovered exp051 episode logs.
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from data import load_data
from task_generator import spt_sampling


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Cora_full")
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--num_episodes", type=int, default=2000)
    parser.add_argument("--diagnostic_interval", type=int, default=160)
    return parser.parse_args()


def summarize(records, n_way):
    n = len(records)
    if not n:
        return None

    def histogram(field):
        counts = Counter(int(r[field]) for r in records)
        return {str(k): int(counts.get(k, 0)) for k in range(n_way + 1) if counts.get(k, 0)}

    patterns = Counter(r["multiplicity_pattern"] for r in records)
    collision = sum(r["num_unique_classes"] < n_way for r in records)
    pair_collision = sum(r["same_class_pairs"] for r in records)
    duplicate = sum(r["duplicate_anchor_count"] for r in records)
    return {
        "num_episodes": n,
        "all_distinct_episodes": n - collision,
        "collision_episodes": collision,
        "collision_episode_rate": collision / n,
        "all_distinct_episode_rate": (n - collision) / n,
        "mean_unique_gt_classes": float(np.mean([r["num_unique_classes"] for r in records])),
        "mean_duplicate_anchors_per_episode": duplicate / n,
        "mean_same_gt_class_pairs_per_episode": pair_collision / n,
        "same_gt_class_pair_rate": pair_collision / (n * (n_way * (n_way - 1) // 2)),
        "unique_gt_class_count_histogram": histogram("num_unique_classes"),
        "duplicate_anchor_count_histogram": histogram("duplicate_anchor_count"),
        "same_gt_class_pair_count_histogram": histogram("same_class_pairs"),
        "max_class_multiplicity_histogram": histogram("max_class_multiplicity"),
        "class_multiplicity_patterns": {
            key: {"count": count, "rate": count / n} for key, count in sorted(patterns.items(), key=lambda x: (-x[1], x[0]))
        },
    }


def main():
    args = parse_args()
    if args.dataset != "Cora_full":
        raise ValueError("This experiment is designed for Cora_full")
    if args.n_way < 2 or args.num_episodes < 1 or args.diagnostic_interval < 1:
        raise ValueError("Invalid n_way, num_episodes or diagnostic_interval")

    # Use the same RNG initialization as the previous experiments and let the
    # original data loader consume randomness for the class split first.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    data = load_data(args)
    labels = data.labels.detach().cpu().numpy().reshape(-1)

    output = Path(__file__).resolve().with_suffix(".json")
    records = []
    windows = []
    for episode_index in range(1, args.num_episodes + 1):
        # Exactly the original 1-shot support sampler used in unsupervised NAQ.
        supports, _pseudo_labels = spt_sampling(data.id_train, args.n_way, 1)
        selected_classes = labels[np.asarray(supports, dtype=np.int64)]
        occurrences = Counter(selected_classes.tolist())
        multiplicities = sorted(occurrences.values(), reverse=True)
        num_unique = len(occurrences)
        records.append(
            {
                "num_unique_classes": num_unique,
                "duplicate_anchor_count": args.n_way - num_unique,
                "same_class_pairs": sum(k * (k - 1) // 2 for k in multiplicities),
                "max_class_multiplicity": multiplicities[0],
                "multiplicity_pattern": "+".join(map(str, multiplicities)),
            }
        )
        if episode_index % args.diagnostic_interval == 0 or episode_index == args.num_episodes:
            start = max(0, episode_index - args.diagnostic_interval)
            stats = summarize(records[start:episode_index], args.n_way)
            stats["episode_start"] = start + 1
            stats["episode_end"] = episode_index
            windows.append(stats)
            print(
                f"[exp053] episodes={start + 1}-{episode_index} "
                f"collision={stats['collision_episode_rate']:.2%} "
                f"mean_unique={stats['mean_unique_gt_classes']:.3f}",
                flush=True,
            )

    report = {
        "experiment": "exp053",
        "dataset": args.dataset,
        "seed": args.seed,
        "settings": vars(args),
        "sampling": {
            "function": "task_generator.spt_sampling(data.id_train, n_way, 1)",
            "node_pool_size": len(data.id_train),
            "num_gt_classes_in_pool": int(np.unique(labels[np.asarray(data.id_train, dtype=np.int64)]).size),
            "sample_without_replacement": True,
            "note": "Fresh samples using the original NAQ support sampler. This does not reconstruct the exact episodes previously used in exp051. No GT labels are used to sample supports.",
        },
        "metric_definitions": {
            "collision_episode": "At least two supports have the same GT class",
            "duplicate_anchor_count": "n_way - number of unique GT classes (excess anchors)",
            "same_class_pairs": "Sum over classes of count*(count-1)/2",
            "multiplicity_pattern": "Sorted GT class frequencies among supports; e.g. 2+2+1",
        },
        "overall": summarize(records, args.n_way),
        "diagnostic_windows": windows,
    }
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, output)
    print(f"[exp053] DONE total={len(records)} " f"collision={report['overall']['collision_episode_rate']:.2%} " f"json={output}", flush=True)


if __name__ == "__main__":
    main()
