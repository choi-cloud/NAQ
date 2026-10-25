"""exp050: Original NAQ-Feat ProtoNet on Cora_full, with JSON reporting.

Uses the original repository's episode generator, ProtoNet, and evaluation.
The original evaluates every 160 steps and chooses the best by validation.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from data import load_data
from meta import ProtoNet
from task_generator import downstream_task_generator, proto_task_generator
from utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="exp050: original NAQ-Feat on Cora_full")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", choices=["Cora_full"], default="Cora_full")
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--k_shot_test", type=int, default=1)
    parser.add_argument("--q_query", type=int, default=10)
    parser.add_argument("--q_query_test", type=int, default=8)
    parser.add_argument("--num_valid_tasks", type=int, default=50)
    parser.add_argument("--num_downstream_tasks", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent", type=int, default=256)
    parser.add_argument("--l2_penalty", type=float, default=1e-4)
    parser.add_argument("--eval_interval", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1212)
    return parser.parse_args()


def write_report(path, report):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def diagnostics(tasks, data):
    labels = torch.as_tensor(data.labels).detach().cpu().numpy().reshape(-1)
    distinct = []
    unique_count = []
    query_matches = []
    for supports, _, queries, _ in tasks:
        supports = np.asarray(supports, dtype=np.int64)
        queries = np.asarray(queries, dtype=np.int64)
        num_unique = len(np.unique(labels[supports]))
        distinct.append(num_unique == len(supports))
        unique_count.append(num_unique)
        expected = np.repeat(labels[supports], len(queries) // len(supports))
        query_matches.extend((labels[queries] == expected).tolist())
    return {
        "anchor_gt_distinct_rate": float(np.mean(distinct)),
        "anchor_gt_unique_count": float(np.mean(unique_count)),
        "query_gt_match_rate": float(np.mean(query_matches)),
    }


def evaluate_original(model, episodes, device):
    scores = []
    for ids_support, labels_support, ids_query, labels_query in episodes:
        _, accuracy = model.fine_tuning(ids_support, labels_support.to(device), ids_query, labels_query.to(device))
        scores.append(float(accuracy))
    return np.asarray(scores, dtype=np.float64)


def main():
    args = parse_args()
    if args.k_shot != 1 or args.n_way != 5:
        raise ValueError("This reproduction supports the original 5-way 1-shot training")
    if args.epoch < args.eval_interval or args.eval_interval <= 0:
        raise ValueError("epoch must be >= eval_interval > 0")

    # Required by original proto_task_generator and graph_diffusion.
    args.setting = "unsup"
    args.query_generation = "NaQ"
    args.type = "feature"

    output = Path(__file__).resolve().with_suffix(".json")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    start = time.perf_counter()
    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    data.add_diffusion(args)

    config = [
        ("GCN", (data.features.shape[1], 2 * args.latent), True),
        ("GCN", (2 * args.latent, args.latent), True),
        ("linear", (args.latent, args.n_way), True),
    ]
    model = ProtoNet(args, config, data).to(device)
    # Original order: generate training episodes before downstream episodes.
    train_pool = [proto_task_generator(args, data) for _ in range(args.epoch)]
    valid_pool, test_pool = downstream_task_generator(args, data, args.n_way, args.k_shot_test)

    report = {
        "experiment": "exp050",
        "description": "Original NAQ-Feat ProtoNet reproduction on Cora_full",
        "status": "running",
        "args": vars(args).copy(),
        "device": str(device),
        "protocol": {
            "training": "Original proto_task_generator + meta.ProtoNet.forward",
            "evaluation": "Original ProtoNet.fine_tuning logistic regression; validation and test each evaluation interval",
            "episode_source": "Original downstream_task_generator cached pickle or newly generated",
            "selection": "Maximum validation accuracy over evaluation intervals (earliest tie)",
            "checkpoint_saved": False,
        },
        "diagnostics": diagnostics(train_pool, data),
        "evaluation_history": [],
        "best": None,
    }

    best_valid = float("-inf")
    for epoch, (id_spt, y_spt, id_query, y_query) in enumerate(train_pool, start=1):
        model.network.train()
        loss, accuracy = model(id_spt, y_spt.to(device), id_query, y_query.to(device))
        # Deliberately match original behavior: 2000 epochs evaluates through 1920.
        if epoch % args.eval_interval != 0:
            continue
        valid_scores = evaluate_original(model, valid_pool, device)
        test_scores = evaluate_original(model, test_pool, device)
        val = float(valid_scores.mean())
        test = float(test_scores.mean())
        ci95 = float(1.96 * test_scores.std(ddof=0) / np.sqrt(len(test_scores)))
        row = {
            "epoch": epoch,
            "train_loss": float(loss.detach().cpu()),
            "train_accuracy": float(accuracy),
            "validation_accuracy": val,
            "test_accuracy": test,
            "test_ci95_half_width": ci95,
        }
        report["evaluation_history"].append(row)
        if val > best_valid:
            best_valid = val
            report["best"] = row.copy()
        report["elapsed_seconds"] = round(time.perf_counter() - start, 2)
        write_report(output, report)
        print(f"[exp050] epoch={epoch} valid={val:.4%} test={test:.4%} ± {ci95:.4%}", flush=True)

    report["status"] = "completed"
    report["elapsed_seconds"] = round(time.perf_counter() - start, 2)
    write_report(output, report)
    print(f"[exp050] completed; best={report['best']}; json={output}", flush=True)


if __name__ == "__main__":
    main()
