"""exp070: GPS-Sum candidate filtering followed by raw-feature Top/Middle/Bottom-k.

Two distinct rules at the same percentage p:
  retain_top: keep highest p% by learned GPS embedding cosine.
  remove_bottom: discard lowest p% by GPS cosine (keep 100-p%).
Then rank retained candidates by raw-feature cosine. GT is diagnostics only.
Run at NAQ repository root with exp069.py and exp069_sum.pt available.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize

from data import load_data
from task_generator import spt_sampling
from exp069 import Encoder, build_graph, torch_graph


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Cora_full")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--query_k", type=int, default=10)
    p.add_argument("--percentages", type=float, nargs="+", default=[1, 5, 10, 20, 50])
    p.add_argument("--checkpoint", default="exp069_sum.pt")
    p.add_argument("--output", default="exp070.json")
    return p.parse_args()


def ranked(ids, scores):
    """Descending score, deterministic ID tie break."""
    return ids[np.lexsort((ids, -scores[ids]))]


def raw_pick(candidate_ids, raw_score, k, position):
    order = ranked(candidate_ids, raw_score)
    if position == "top":
        return order[:k]
    if position == "bottom":
        return order[-k:][::-1]
    if position == "middle":
        start = (len(order) - k) // 2
        return order[start : start + k]
    raise ValueError(position)


def main():
    a = parse_args()
    if a.episodes < 1 or a.n_way < 1 or a.query_k < 1:
        raise ValueError("episodes, n_way and query_k must be positive")
    if any(not (0 < p < 100) for p in a.percentages):
        raise ValueError("percentages must be strictly between 0 and 100")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    a.setting = "unsup"
    a.query_generation = "NaQ"
    a.type = "feature"
    a.k_shot = 1
    a.k_shot_test = 1
    a.epoch = 1
    data = load_data(a)
    raw_x = data.features.detach().cpu().numpy().astype(np.float32)
    labels = data.labels.detach().cpu().numpy().reshape(-1)
    n = len(labels)
    x_unit = normalize(sp.csr_matrix(raw_x), norm="l2", axis=1)
    graph = build_graph(data, n)

    device = torch.device(f"cuda:{a.device}" if a.device >= 0 and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("mode") != "sum":
        raise ValueError("Expected GPS-Sum checkpoint (mode='sum')")
    if checkpoint["input_dim"] != raw_x.shape[1]:
        raise ValueError("Checkpoint input dimension does not match Cora-Full features")
    model = Encoder(checkpoint["input_dim"], checkpoint["hidden"], checkpoint["embedding_dim"], dropout=0).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    adj = torch_graph(graph, device, np.random.default_rng(a.seed), drop=0)
    with torch.no_grad():
        z, _ = model(torch.as_tensor(raw_x, device=device), adj)
    z = F.normalize(z, dim=1).cpu().numpy()
    del model, adj
    print(f"Loaded {a.checkpoint}; embeddings={z.shape}", flush=True)

    episodes = [np.asarray(spt_sampling(data.id_train, a.n_way, 1)[0], dtype=np.int64) for _ in range(a.episodes)]
    percent = sorted(set(a.percentages))
    modes = ("retain_top", "remove_bottom")
    positions = ("top", "middle", "bottom")
    records = {f"{mode}/{p:g}/{pos}": {"correct": 0, "count": 0, "candidates": 0} for mode in modes for p in percent for pos in positions}
    baseline = {key: {"correct": 0, "count": 0} for key in ("raw_top", "raw_middle", "raw_bottom", "gps_top")}
    anchor_instances = 0

    for episode_idx, supports in enumerate(episodes):
        excluded = np.zeros(n, dtype=bool)
        excluded[supports] = True
        ids = np.flatnonzero(~excluded)
        if len(ids) < a.query_k:
            raise ValueError("Not enough nodes after excluding episode supports")
        for anchor in supports:
            anchor = int(anchor)
            anchor_instances += 1
            gps_score = z @ z[anchor]
            feat_score = (x_unit.getrow(anchor) @ x_unit.T).toarray().ravel()
            gps_sorted = ranked(ids, gps_score)
            truth = labels[anchor]
            for name, selected in (
                ("raw_top", raw_pick(ids, feat_score, a.query_k, "top")),
                ("raw_middle", raw_pick(ids, feat_score, a.query_k, "middle")),
                ("raw_bottom", raw_pick(ids, feat_score, a.query_k, "bottom")),
                ("gps_top", gps_sorted[: a.query_k]),
            ):
                baseline[name]["correct"] += int(np.count_nonzero(labels[selected] == truth))
                baseline[name]["count"] += len(selected)
            for p in percent:
                cut = max(a.query_k, int(math.ceil(len(ids) * p / 100)))
                remove = int(math.ceil(len(ids) * p / 100))
                pools = {"retain_top": gps_sorted[:cut], "remove_bottom": gps_sorted[: len(ids) - remove]}
                for mode, pool in pools.items():
                    if len(pool) < a.query_k:
                        raise ValueError("Candidate pool too small; reduce percent")
                    for pos in positions:
                        selected = raw_pick(pool, feat_score, a.query_k, pos)
                        rec = records[f"{mode}/{p:g}/{pos}"]
                        rec["correct"] += int(np.count_nonzero(labels[selected] == truth))
                        rec["count"] += len(selected)
                        rec["candidates"] += len(pool)
        if (episode_idx + 1) % 25 == 0:
            print(f"Evaluated {episode_idx + 1}/{a.episodes} episodes", flush=True)

    results = {}
    for mode in modes:
        results[mode] = {}
        for p in percent:
            results[mode][str(p)] = {}
            for pos in positions:
                r = records[f"{mode}/{p:g}/{pos}"]
                results[mode][str(p)][pos] = {
                    "gt_match_rate": r["correct"] / r["count"],
                    "gt_match_rate_pct": 100 * r["correct"] / r["count"],
                    "correct_pairs": r["correct"],
                    "total_pairs": r["count"],
                    "mean_candidate_count": r["candidates"] / anchor_instances,
                }
    output = {
        "experiment": "exp070",
        "status": "completed",
        "args": vars(a),
        "checkpoint_mode": checkpoint["mode"],
        "graph": {"nodes": n, "classes": int(len(np.unique(labels)))},
        "definitions": {
            "reference": "GPS-Sum encoder normalized node embeddings (not projection head)",
            "candidate_exclusion": "All supports in the NAQ episode excluded",
            "retain_top": "Keep highest N% learned embedding cosine candidates",
            "remove_bottom": "Discard lowest N% learned embedding cosine candidates",
            "rerank": "Raw node-feature cosine ranking within retained pool",
            "middle": "Contiguous centered k items after raw cosine descending sort",
            "bottom": "k lowest raw cosine candidates",
            "nonedge_restriction": False,
            "gt": "Diagnostics only; never used in candidate selection",
        },
        "anchor_instances": anchor_instances,
        "baseline": {
            key: {"gt_match_rate": v["correct"] / v["count"], "gt_match_rate_pct": 100 * v["correct"] / v["count"]} for key, v in baseline.items()
        },
        "results": results,
    }
    Path(a.output).write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print("\nBaseline:", output["baseline"], flush=True)
    for mode in modes:
        print(f"\n[{mode}]", flush=True)
        for p in percent:
            vals = results[mode][str(p)]
            print(
                f"{p:g}%  top={vals['top']['gt_match_rate_pct']:.2f}%  "
                f"middle={vals['middle']['gt_match_rate_pct']:.2f}%  "
                f"bottom={vals['bottom']['gt_match_rate_pct']:.2f}%  "
                f"pool={vals['top']['mean_candidate_count']:.0f}",
                flush=True,
            )
    print(f"Saved {a.output}", flush=True)


if __name__ == "__main__":
    main()
