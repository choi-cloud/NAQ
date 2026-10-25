"""exp051: Original NAQ-Feat training with nearest-prototype evaluation on Cora-full.

Original data.py, task_generator.py, model.py, and meta.py are not modified.
Validation and training diagnostics run every eval_interval epochs.
Test runs once, after loading the best validation checkpoint.
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
    parser = argparse.ArgumentParser(description="exp051: NAQ-Feat + ProtoNet evaluation")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", default="Cora_full")
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


def write_json(path, report):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def episode_diagnostics(tasks, labels, q_query):
    distinct = []
    unique = []
    hits = []
    overlaps = []
    support_in_query = []
    for supports, _, queries, _ in tasks:
        s = np.asarray(supports, dtype=np.int64).reshape(-1)
        q = np.asarray(queries, dtype=np.int64).reshape(-1)
        if q.size != s.size * q_query:
            raise ValueError("Unexpected NAQ query block size")
        k = np.unique(labels[s]).size
        unique.append(int(k))
        distinct.append(k == s.size)
        expected = np.repeat(labels[s], q_query)
        hits.extend((labels[q] == expected).tolist())
        overlaps.append(int(q.size - np.unique(q).size))
        support_in_query.append(int(np.isin(q, s).sum()))
    return {
        "num_episodes": len(tasks),
        "anchor_gt_distinct_rate": float(np.mean(distinct)),
        "anchor_gt_unique_count": float(np.mean(unique)),
        "query_gt_match_rate": float(np.mean(hits)),
        "query_duplicate_occurrences_per_episode": float(np.mean(overlaps)),
        "query_duplicate_fraction": float(sum(overlaps) / (len(tasks) * q_query * len(tasks[0][0]))),
        "support_query_overlap_fraction": float(sum(support_in_query) / (len(tasks) * q_query * len(tasks[0][0]))),
    }


def evaluate_nearest_prototype(model, data, episodes, n_way, k_shot):
    model.network.eval()
    scores = []
    margins = []
    with torch.no_grad():
        emb = model.network(data.features, adj=data.adj)
        for supports, support_labels, queries, query_labels in episodes:
            device = emb.device
            sid = torch.as_tensor(supports, dtype=torch.long, device=device)
            qid = torch.as_tensor(queries, dtype=torch.long, device=device)
            ys = torch.as_tensor(support_labels, dtype=torch.long, device=device).reshape(-1)
            yq = torch.as_tensor(query_labels, dtype=torch.long, device=device).reshape(-1)
            if sid.numel() != n_way * k_shot:
                raise ValueError("Unexpected number of support nodes")
            class_ids = torch.unique(ys, sorted=True)
            if class_ids.numel() != n_way:
                raise ValueError("Episode does not contain n_way distinct classes")
            prototypes = torch.stack([emb[sid[ys == cls]].mean(dim=0) for cls in class_ids])
            dists = (emb[qid, None, :] - prototypes[None, :, :]).square().sum(dim=-1)
            pred = class_ids[dists.argmin(dim=1)]
            scores.append(float((pred == yq).float().mean().item()))
            if n_way > 1:
                smallest = dists.topk(2, largest=False, dim=1).values
                margins.append(float((smallest[:, 1] - smallest[:, 0]).mean().item()))
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "accuracy": float(arr.mean()),
        "ci95_half_width": float(1.96 * arr.std(ddof=0) / np.sqrt(arr.size)),
        "num_episodes": int(arr.size),
        "mean_nearest_distance_margin": float(np.mean(margins)) if margins else None,
    }


def main():
    args = parse_args()
    if args.dataset != "Cora_full" or args.n_way != 5 or args.k_shot != 1:
        raise ValueError("exp051 is the Cora_full 5-way 1-shot NAQ-Feat training test")
    if min(args.epoch, args.eval_interval, args.q_query, args.k_shot_test) <= 0:
        raise ValueError("Epochs, evaluation interval, shots and queries must be positive")

    args.setting = "unsup"
    args.query_generation = "NaQ"
    args.type = "feature"
    torch.set_num_threads(4)
    seed_everything(args.seed)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    output = Path(__file__).resolve().with_suffix(".json")
    checkpoint = Path(__file__).resolve().with_name("exp051_best.pt")
    started = time.perf_counter()
    report = {
        "experiment": "exp051",
        "status": "initializing",
        "args": vars(args).copy(),
        "device": str(device),
        "protocol": {
            "training": "Unmodified NAQ-Feat proto_task_generator and ProtoNet.forward",
            "validation": "Squared Euclidean nearest-prototype classification, every eval_interval",
            "test": "One final evaluation of best validation checkpoint only",
            "selection": "Highest validation accuracy; first checkpoint wins ties",
            "episode_source": "Original downstream_task_generator (cached pickle or generated)",
            "note": "Ground-truth labels are used only in diagnostic statistics and downstream evaluation",
        },
        "checkpoint": str(checkpoint),
        "history": [],
        "best": None,
        "test": None,
    }
    # Start a fresh report on every invocation, even if a previous JSON exists.
    write_json(output, report)

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
    train_pool = [proto_task_generator(args, data) for _ in range(args.epoch)]
    valid_pool, test_pool = downstream_task_generator(args, data, args.n_way, args.k_shot_test)
    labels = torch.as_tensor(data.labels).detach().cpu().numpy().reshape(-1)

    best_accuracy = float("-inf")
    for epoch, (sid, ys, qid, yq) in enumerate(train_pool, start=1):
        model.network.train()
        loss, train_acc = model(sid, ys.to(device), qid, yq.to(device))
        if epoch % args.eval_interval != 0 and epoch != args.epoch:
            continue

        valid = evaluate_nearest_prototype(model, data, valid_pool, args.n_way, args.k_shot_test)
        # Diagnostic sampling statistics from the current evaluation window.
        window = train_pool[max(0, epoch - args.eval_interval) : epoch]
        diag = episode_diagnostics(window, labels, args.q_query)
        row = {
            "epoch": epoch,
            "train_loss": float(loss.detach().cpu().item()),
            "train_accuracy": float(train_acc),
            "validation": valid,
            "sampling_diagnostics": diag,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        report["history"].append(row)
        if valid["accuracy"] > best_accuracy:
            best_accuracy = valid["accuracy"]
            torch.save({"epoch": epoch, "model": model.network.state_dict(), "args": vars(args).copy()}, checkpoint)
            report["best"] = {"epoch": epoch, "validation": valid, "sampling_diagnostics": diag, "checkpoint": str(checkpoint)}
        report["status"] = "training"
        report["elapsed_seconds"] = row["elapsed_seconds"]
        write_json(output, report)
        print(
            f"[exp051] epoch={epoch} loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4%} valid={valid['accuracy']:.4%} "
            f"best={best_accuracy:.4%}@{report['best']['epoch']} "
            f"anchor_distinct={diag['anchor_gt_distinct_rate']:.2%} "
            f"query_match={diag['query_gt_match_rate']:.2%}",
            flush=True,
        )

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.network.load_state_dict(state["model"])
    report["test"] = evaluate_nearest_prototype(model, data, test_pool, args.n_way, args.k_shot_test)
    report["status"] = "completed"
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    write_json(output, report)
    print(
        f"[exp051] DONE best_epoch={report['best']['epoch']} "
        f"valid={report['best']['validation']['accuracy']:.4%} "
        f"test={report['test']['accuracy']:.4%} "
        f"±{report['test']['ci95_half_width']:.4%} "
        f"json={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
