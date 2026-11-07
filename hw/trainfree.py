"""HW03a-s4: train-free baselines — ProtoNet evaluation directly on X, ÂX, Â²X (raw and L2-row-normalized).

    python -m hw.trainfree --dataset <ds>

Â = data.adj = utils.normalize_adj(A + I) = D^-1/2 (A+I) D^-1/2 (edge lists of both datasets are
symmetric without self-loops/duplicates; checked again below against an independent build).
Evaluation: hw.evaluate.evaluate unchanged, on the cached valid (50) / test (500) episodes
(save/downstream_tasks/<ds>/5way_5shot.pkl). Labels are used only inside evaluation. No training.
The table also lists this server's 5-seed means of naq_base and nbr_raw (results/hw/summary.csv).
Output: $HW_RESULTS_ROOT (default results/hw)/trainfree/HW03a_trainfree_<ds>.{json,md}.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import load_data
from utils import seed_everything

from hw.config import build_args, load_config
from hw.episodes import build_eval_pools
from hw.evaluate import evaluate

RESULTS_ROOT = Path(os.environ.get("HW_RESULTS_ROOT", "results/hw"))
REFERENCE_ARMS = ("naq_base", "nbr_raw")


def parse_cli():
    parser = argparse.ArgumentParser(description="train-free baselines (HW03a-s4)")
    parser.add_argument("--dataset", required=True, choices=["Cora_full", "Amazon_clothing"])
    parser.add_argument("--seed", type=int, default=1212, help="only for load_data's class shuffle; eval episodes come from the cache")
    return parser.parse_args()


@torch.no_grad()
def check_adj(data, num_nodes):
    """max |data.adj - D^-1/2 (A+I) D^-1/2| with A built from edge_index (symmetrized, self-loops added)."""
    ei = data.edge_index
    loops = torch.arange(num_nodes, device=ei.device)
    rows = torch.cat([ei[0], ei[1], loops])
    cols = torch.cat([ei[1], ei[0], loops])
    a = torch.sparse_coo_tensor(torch.stack([rows, cols]), torch.ones(rows.numel(), device=ei.device), (num_nodes, num_nodes)).coalesce()
    idx = a.indices()
    vals = torch.ones(idx.shape[1], device=ei.device)  # binarize (duplicates collapse to 1)
    deg = torch.zeros(num_nodes, device=ei.device).index_add_(0, idx[0], vals)
    dis = deg.pow(-0.5)
    ref = torch.sparse_coo_tensor(idx, dis[idx[0]] * vals * dis[idx[1]], (num_nodes, num_nodes)).coalesce()
    diff = (data.adj.coalesce() - ref).coalesce().values().abs()
    return float(diff.max().item()) if diff.numel() else 0.0


def reference_rows(dataset):
    path = RESULTS_ROOT / "summary.csv"
    out = {}
    if not path.is_file():
        return out
    with open(path, encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for arm in REFERENCE_ARMS:
        sel = [r for r in rows if r["arm"] == arm and r["ds"] == dataset]
        if sel:
            test = np.array([float(r["test"]) for r in sel])
            val = np.array([float(r["val"]) for r in sel])
            out[arm] = {"n": len(sel), "test_mean": float(test.mean()), "test_std": float(test.std(ddof=0)), "val_mean": float(val.mean()), "seeds": sorted(int(r["seed"]) for r in sel)}
    return out


def main():
    cli = parse_cli()
    started = time.time()
    cfg = load_config("naq_base")
    cfg["seed"] = cli.seed
    args = build_args(cfg, cli.dataset)
    seed_everything(cli.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    valid_pool, test_pool = build_eval_pools(args, data)
    x = data.features.float()
    num_nodes = x.shape[0]
    adj_check = check_adj(data, num_nodes)
    ax = torch.sparse.mm(data.adj, x)
    a2x = torch.sparse.mm(data.adj, ax)
    features = {"X": x, "AX": ax, "A2X": a2x}

    result = {
        "dataset": cli.dataset,
        "valid_tasks": len(valid_pool),
        "test_tasks": len(test_pool),
        "adj": "data.adj = utils.normalize_adj(A + I) = D^-1/2 (A+I) D^-1/2",
        "adj_max_abs_diff_vs_independent_build": adj_check,
        "feature_dim": int(x.shape[1]),
        "variants": {},
        "reference_trained": reference_rows(cli.dataset),
        "note": "no training; labels used only inside hw.evaluate.evaluate",
    }
    for name, feat in features.items():
        for norm in ("raw", "l2"):
            emb = F.normalize(feat, p=2, dim=1) if norm == "l2" else feat
            v = evaluate(emb, valid_pool, args.n_way, device)
            t = evaluate(emb, test_pool, args.n_way, device)
            result["variants"][f"{name}_{norm}"] = {"validation": v, "test": t}
    result["elapsed_seconds"] = round(time.time() - started, 2)

    out_dir = RESULTS_ROOT / "trainfree"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"HW03a_trainfree_{cli.dataset}.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    lines = [
        f"# HW03a train-free baselines — {cli.dataset}",
        "",
        f"5-way 5-shot ProtoNet (squared Euclidean), cached episodes: valid {len(valid_pool)} / test {len(test_pool)}. Accuracy %.",
        f"Â = data.adj = D^-1/2 (A+I) D^-1/2 (max |diff| vs independent build: {adj_check:.2e}). No training.",
        "",
        "| 특징 | 행 정규화 | val | test | test ci95 |",
        "|---|---|---|---|---|",
    ]
    labels = {"X": "X (raw)", "AX": "ÂX", "A2X": "Â²X"}
    for key, vals in result["variants"].items():
        name, norm = key.split("_")
        lines.append(f"| {labels[name]} | {'L2' if norm == 'l2' else '없음'} | {100 * vals['validation']['accuracy_mean']:.2f} | {100 * vals['test']['accuracy_mean']:.2f} | {100 * vals['test']['accuracy_ci95_half_width']:.2f} |")
    for arm, ref in result["reference_trained"].items():
        lines.append(f"| {arm} (학습, seio {ref['n']} seed 평균 ± 표준편차) | — | {ref['val_mean']:.2f} | {ref['test_mean']:.2f} ± {ref['test_std']:.2f} | — |")
    (out_dir / f"HW03a_trainfree_{cli.dataset}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"elapsed {result['elapsed_seconds']} s")


if __name__ == "__main__":
    main()
