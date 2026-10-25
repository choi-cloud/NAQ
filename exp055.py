"""exp055: Oracle class-conditioned center/boundary/random anchor comparison.

Uses the original NAQ-Feat query generator and ProtoNet training implementation.
Validation is performed each eval_interval; test only at per-shot best checkpoints.
GT labels are used in TRAINING episode construction: this is an oracle diagnostic.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import load_data
from meta import ProtoNet
from task_generator import downstream_task_generator, query_generation_NaQ
from utils import seed_everything


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--dataset", default="Cora_full")
    p.add_argument("--epoch", type=int, default=2000)
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=1)
    p.add_argument("--k_shot_test", type=int, default=1)
    p.add_argument("--q_query", type=int, default=10)
    p.add_argument("--q_query_test", type=int, default=8)
    p.add_argument("--num_valid_tasks", type=int, default=50)
    p.add_argument("--num_downstream_tasks", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--latent", type=int, default=256)
    p.add_argument("--l2_penalty", type=float, default=1e-4)
    p.add_argument("--eval_interval", type=int, default=160)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--fraction", type=float, default=0.2)
    p.add_argument("--strategies", nargs="+", choices=["center", "boundary", "random"], default=["center", "boundary", "random"])
    p.add_argument("--shots", nargs="+", type=int, default=[1, 5])
    return p.parse_args()


def save_report(path, report):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def build_class_candidates(data, labels, fraction):
    """Cosine of original feature vector to the normalized feature centroid per GT class."""
    features = data.features.detach().cpu().float()
    normed = F.normalize(features, dim=1)
    candidates = {}
    scores = {}
    for cls in np.unique(labels):
        ids = np.flatnonzero(labels == cls)
        centroid = F.normalize(features[ids].mean(dim=0).unsqueeze(0), dim=1).squeeze(0)
        sim = (normed[ids] @ centroid).numpy()
        order = np.argsort(-sim, kind="stable")
        count = max(1, int(math.ceil(len(ids) * fraction)))
        candidates[int(cls)] = {"center": ids[order[:count]], "boundary": ids[order[-count:]], "random": ids}
        scores[int(cls)] = {int(idx): float(value) for idx, value in zip(ids, sim)}
    return candidates, scores


def evaluate(model, data, episodes, n_way, shot):
    model.network.eval()
    correct = []
    margins = []
    with torch.no_grad():
        embedding = model.network(data.features, adj=data.adj)
        for sid, ys, qid, yq in episodes:
            sid = torch.as_tensor(sid, dtype=torch.long, device=embedding.device)
            qid = torch.as_tensor(qid, dtype=torch.long, device=embedding.device)
            ys = torch.as_tensor(ys, dtype=torch.long, device=embedding.device).reshape(-1)
            yq = torch.as_tensor(yq, dtype=torch.long, device=embedding.device).reshape(-1)
            classes = torch.unique(ys, sorted=True)
            if sid.numel() != n_way * shot or classes.numel() != n_way:
                raise ValueError("Unexpected downstream episode size")
            prototypes = torch.stack([embedding[sid[ys == cls]].mean(dim=0) for cls in classes])
            distance = (embedding[qid, None] - prototypes[None]).square().sum(dim=-1)
            predictions = classes[distance.argmin(dim=1)]
            correct.append(float((predictions == yq).float().mean().item()))
            smallest = distance.topk(2, dim=1, largest=False).values
            margins.append(float((smallest[:, 1] - smallest[:, 0]).mean().item()))
    x = np.asarray(correct)
    return {
        "accuracy": float(x.mean()),
        "ci95_half_width": float(1.96 * x.std(ddof=0) / np.sqrt(len(x))),
        "num_episodes": len(x),
        "mean_nearest_distance_margin": float(np.mean(margins)),
    }


def diagnostic(tasks, gt_labels, sim_by_class, q_query):
    matches = []
    anchor_sim = []
    duplicate = 0
    support_in_queries = 0
    count = 0
    for sid, _, qid, _ in tasks:
        sid = np.asarray(sid, dtype=np.int64)
        qid = np.asarray(qid, dtype=np.int64)
        if np.unique(gt_labels[sid]).size != len(sid):
            raise AssertionError("Anchor class collision in oracle experiment")
        if qid.size != len(sid) * q_query:
            raise AssertionError("Unexpected NAQ query order")
        matches.extend((gt_labels[qid] == np.repeat(gt_labels[sid], q_query)).tolist())
        anchor_sim.extend(sim_by_class[int(gt_labels[idx])][int(idx)] for idx in sid)
        duplicate += len(qid) - len(np.unique(qid))
        support_in_queries += np.isin(qid, sid).sum()
        count += len(qid)
    return {
        "episodes": len(tasks),
        "anchor_gt_distinct_rate": 1.0,
        "anchor_centroid_cosine_mean": float(np.mean(anchor_sim)),
        "anchor_centroid_cosine_std": float(np.std(anchor_sim)),
        "query_gt_match_rate": float(np.mean(matches)),
        "query_duplicate_fraction": float(duplicate / count),
        "support_query_overlap_fraction": float(support_in_queries / count),
    }


def build_tasks(args, data, labels, candidates, class_schedules, strategy):
    # Separate sampling RNG per strategy; class combinations remain identical.
    rng = np.random.default_rng(args.seed + {"center": 101, "boundary": 202, "random": 303}[strategy])
    tasks = []
    for selected_classes in class_schedules:
        support = np.array([int(rng.choice(candidates[int(cls)][strategy])) for cls in selected_classes], dtype=np.int64)
        # NAQ's pseudo-class IDs are episode-local; GT class IDs are NEVER targets.
        pseudo = torch.as_tensor(rng.permutation(args.n_way), dtype=torch.long)
        query, pseudo_query = query_generation_NaQ(support, pseudo, args.q_query, data, True)
        tasks.append((support, pseudo, query, pseudo_query))
    return tasks


def main():
    args = arguments()
    if args.dataset != "Cora_full" or args.n_way != 5 or args.k_shot != 1:
        raise ValueError("exp055 expects Cora_full 5-way 1-shot NAQ training")
    if not 0 < args.fraction <= 0.5 or args.epoch < 1 or args.eval_interval < 1:
        raise ValueError("fraction must be (0, 0.5], epochs and interval positive")
    if any(shot < 1 for shot in args.shots) or len(set(args.shots)) != len(args.shots):
        raise ValueError("Shots must be distinct positive integers")
    if len(set(args.strategies)) != len(args.strategies):
        raise ValueError("Strategies must be unique")

    args.setting = "unsup"
    args.query_generation = "NaQ"
    args.type = "feature"
    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    output = Path(__file__).resolve().with_suffix(".json")
    started = time.perf_counter()
    report = {
        "experiment": "exp055",
        "status": "initializing",
        "args": vars(args).copy(),
        "oracle_definition": {
            "classes": "GT labels, all 70 classes; choose 5 distinct GT classes uniformly per episode",
            "same_class_schedules": "All methods share the identical sequence of selected GT class combinations",
            "center": "Randomly select from top fraction nearest to the GT-class raw-feature centroid by cosine",
            "boundary": "Randomly select from bottom fraction farthest from the GT-class raw-feature centroid by cosine; this is periphery, not necessarily decision boundary",
            "random": "Uniform random node within each selected GT class",
            "query": "Unmodified original NAQ-Feat cosine Top-K",
            "training": "Original ProtoNet.forward with episode-local pseudo labels",
            "validation": "Squared Euclidean nearest-prototype, every eval_interval epochs, separately for each shot",
            "test": "One test evaluation per shot at its own best validation checkpoint",
            "note": "GT labels used for oracle sampling; NOT an unsupervised approach",
        },
        "results": {},
    }
    save_report(output, report)

    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    data.add_diffusion(args)
    labels = torch.as_tensor(data.labels).detach().cpu().numpy().reshape(-1)
    candidates, centroid_similarity = build_class_candidates(data, labels, args.fraction)

    # Uniform GT class combinations are shared across conditions, NOT drawn proportional to class size.
    # Compared to exp054, class-frequency distribution differs; random is the within-design control.
    schedule_rng = np.random.default_rng(args.seed + 404)
    class_ids = np.unique(labels)
    schedules = [schedule_rng.choice(class_ids, args.n_way, replace=False) for _ in range(args.epoch)]
    report["class_schedule"] = {
        "num_episodes": len(schedules),
        "num_available_classes": len(class_ids),
        "class_sampling": "Uniform without replacement across GT class IDs",
        "node_sampling": "With replacement across episodes, without duplicates within episode by distinct GT class",
    }

    validation_and_test = {}
    for shot in args.shots:
        args.k_shot_test = shot
        validation_and_test[shot] = downstream_task_generator(args, data, args.n_way, shot)
    # Each method receives identical training initialization and fixed downstream tasks.
    for strategy in args.strategies:
        seed_everything(args.seed)
        tasks = build_tasks(args, data, labels, candidates, schedules, strategy)
        valid_and_test = validation_and_test
        config = [
            ("GCN", (data.features.shape[1], 2 * args.latent), True),
            ("GCN", (2 * args.latent, args.latent), True),
            ("linear", (args.latent, args.n_way), True),
        ]
        model = ProtoNet(args, config, data).to(device)
        entry = {
            "status": "training",
            "sampling_diagnostics_all": diagnostic(tasks, labels, centroid_similarity, args.q_query),
            "history": [],
            "best_by_shot": {},
            "test_by_shot": {},
        }
        report["results"][strategy] = entry
        save_report(output, report)
        best_acc = {shot: -float("inf") for shot in args.shots}
        for epoch, (sid, ys, qid, yq) in enumerate(tasks, 1):
            model.network.train()
            loss, train_acc = model(sid, ys.to(device), qid, yq.to(device))
            if epoch % args.eval_interval != 0 and epoch != args.epoch:
                continue
            window = tasks[max(0, epoch - args.eval_interval) : epoch]
            row = {
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu().item()),
                "train_accuracy": float(train_acc),
                "sampling_diagnostics": diagnostic(window, labels, centroid_similarity, args.q_query),
                "validation_by_shot": {},
            }
            for shot in args.shots:
                valid_pool, _ = valid_and_test[shot]
                valid_result = evaluate(model, data, valid_pool, args.n_way, shot)
                row["validation_by_shot"][str(shot)] = valid_result
                if valid_result["accuracy"] > best_acc[shot]:
                    best_acc[shot] = valid_result["accuracy"]
                    checkpoint = Path(__file__).resolve().with_name(f"exp055_{strategy}_{shot}shot_best.pt")
                    torch.save({"epoch": epoch, "model": model.network.state_dict()}, checkpoint)
                    entry["best_by_shot"][str(shot)] = {"epoch": epoch, "validation": valid_result, "checkpoint": str(checkpoint)}
            entry["history"].append(row)
            save_report(output, report)
            print(
                f"[exp055:{strategy}] epoch={epoch} "
                + " ".join(f"{shot}shot_valid={row['validation_by_shot'][str(shot)]['accuracy']:.2%}" for shot in args.shots)
                + f" query_match={row['sampling_diagnostics']['query_gt_match_rate']:.2%}",
                flush=True,
            )
        for shot in args.shots:
            best = entry["best_by_shot"][str(shot)]
            ckpt = torch.load(best["checkpoint"], map_location=device, weights_only=True)
            model.network.load_state_dict(ckpt["model"])
            _, test_pool = valid_and_test[shot]
            result = evaluate(model, data, test_pool, args.n_way, shot)
            entry["test_by_shot"][str(shot)] = result
            print(
                f"[exp055:{strategy}] DONE {shot}shot best_epoch={best['epoch']} "
                f"valid={best['validation']['accuracy']:.4%} "
                f"test={result['accuracy']:.4%} ±{result['ci95_half_width']:.4%}",
                flush=True,
            )
        entry["status"] = "completed"
        save_report(output, report)
    report["status"] = "completed"
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    save_report(output, report)
    print(f"[exp055] Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
