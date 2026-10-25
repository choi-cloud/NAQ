"""exp056: NAQ-Feat with an oracle out-of-episode-class negative node.

Keep original NAQ random anchors and Top-K queries. Train with the original
ProtoNet objective plus a squared-Euclidean hinge repulsion from one sampled
GT-negative node. GT is used only to select the negative (and diagnostics).
Original NAQ source files are not modified.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import load_data
from meta import ProtoNet
from task_generator import downstream_task_generator, proto_task_generator
from utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--dataset", default="Cora_full")
    p.add_argument("--epoch", type=int, default=2000)
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=1)
    p.add_argument("--q_query", type=int, default=10)
    p.add_argument("--q_query_test", type=int, default=8)
    p.add_argument("--num_valid_tasks", type=int, default=50)
    p.add_argument("--num_downstream_tasks", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--latent", type=int, default=256)
    p.add_argument("--l2_penalty", type=float, default=1e-4)
    p.add_argument("--eval_interval", type=int, default=160)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--neg_weight", type=float, default=0.1)
    p.add_argument("--neg_margin", type=float, default=1.0)
    p.add_argument("--shots", type=int, nargs="+", default=[1, 5])
    return p.parse_args()


def save_json(path, obj):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def evaluate(model, data, episodes, n_way, k_shot):
    model.network.eval()
    scores = []
    margins = []
    with torch.no_grad():
        z = model.network(data.features, adj=data.adj)
        dev = z.device
        for spt, ys, qry, yq in episodes:
            spt = torch.as_tensor(spt, device=dev, dtype=torch.long).reshape(-1)
            qry = torch.as_tensor(qry, device=dev, dtype=torch.long).reshape(-1)
            ys = torch.as_tensor(ys, device=dev, dtype=torch.long).reshape(-1)
            yq = torch.as_tensor(yq, device=dev, dtype=torch.long).reshape(-1)
            if spt.numel() != n_way * k_shot:
                raise ValueError("Evaluation support count does not match shot")
            cls = torch.unique(ys, sorted=True)
            if cls.numel() != n_way:
                raise ValueError("Expected n_way classes in downstream episode")
            proto = torch.stack([z[spt[ys == c]].mean(0) for c in cls])
            dist = torch.cdist(z[qry], proto, p=2).square()
            pred = cls[dist.argmin(1)]
            scores.append(float((pred == yq).float().mean().item()))
            if n_way >= 2:
                nearest = torch.topk(dist, k=2, dim=1, largest=False).values
                margins.append(float((nearest[:, 1] - nearest[:, 0]).mean().item()))
    score = np.asarray(scores, dtype=float)
    return {
        "accuracy": float(score.mean()),
        "ci95_half_width": float(1.96 * score.std(ddof=0) / np.sqrt(score.size)),
        "num_episodes": int(score.size),
        "mean_nearest_distance_margin": float(np.mean(margins)) if margins else None,
    }


def summarize_window(window):
    def avg(k):
        return float(np.mean([r[k] for r in window]))

    return {
        "num_episodes": len(window),
        "proto_loss_mean": avg("proto_loss"),
        "negative_loss_mean": avg("negative_loss"),
        "weighted_negative_loss_mean": avg("weighted_negative_loss"),
        "total_loss_mean": avg("total_loss"),
        "train_accuracy_mean": avg("train_accuracy"),
        "negative_anchor_sqdist_mean": avg("negative_anchor_sqdist_mean"),
        "negative_active_hinge_fraction": avg("negative_active_hinge_fraction"),
        "anchor_gt_distinct_rate": avg("anchor_gt_distinct"),
        "anchor_unique_gt_classes_mean": avg("anchor_unique_gt_classes"),
        "query_gt_match_rate": avg("query_gt_match_rate"),
        "negative_gt_collision_rate": avg("negative_gt_collision"),
        "negative_gt_class_count_in_candidates_mean": avg("num_negative_candidate_classes"),
    }


def main():
    args = parse_args()
    if args.dataset != "Cora_full" or args.n_way != 5 or args.k_shot != 1:
        raise ValueError("exp056 is defined for Cora_full 5-way 1-shot training")
    if args.neg_margin <= 0 or args.neg_weight < 0:
        raise ValueError("neg_margin must be positive, neg_weight must be nonnegative")
    if args.eval_interval < 1 or args.epoch < 1 or any(s < 1 for s in args.shots):
        raise ValueError("Invalid epoch, interval or shot")
    if len(set(args.shots)) != len(args.shots):
        raise ValueError("Repeated shot value")

    args.setting = "unsup"
    args.query_generation = "NaQ"
    args.type = "feature"
    # Required by original downstream_task_generator; overwritten for each shot.
    args.k_shot_test = args.shots[0]
    torch.set_num_threads(4)
    seed_everything(args.seed)
    dev = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    root = Path(__file__).resolve().parent
    out = root / "exp056.json"
    start = time.perf_counter()
    report = {
        "experiment": "exp056",
        "status": "initializing",
        "args": vars(args).copy(),
        "protocol": {
            "train_sampling": "Unmodified NAQ random 5 anchors + cosine Top-10 queries",
            "negative": "Uniform choice of one GT class absent from all anchor GT classes; uniform one node in that class",
            "negative_gt_exclusion": "All actual GT classes of current anchors, even if anchors collide",
            "loss": "ProtoNet NLL + neg_weight * mean_i relu(neg_margin - squared_euclidean(anchor_i, negative))",
            "evaluation": "Squared Euclidean nearest prototype; valid every 160 and last epoch, test once at each shot's best valid checkpoint",
            "selection": "Independent validation-selected checkpoint for each shot",
            "oracle_notice": "GT used to sample negative nodes. This is not unsupervised learning.",
        },
        "history": [],
        "best_by_shot": {},
        "test_by_shot": {},
        "checkpoints": {},
    }
    save_json(out, report)

    data = load_data(args)
    if dev.type == "cuda":
        data.set_device(str(dev))
    data.add_diffusion(args)
    config = [
        ("GCN", (data.features.shape[1], 2 * args.latent), True),
        ("GCN", (2 * args.latent, args.latent), True),
        ("linear", (args.latent, args.n_way), True),
    ]
    model = ProtoNet(args, config, data).to(dev)

    # Preserve the original NAQ sampling sequence before using a separate RNG for GT negatives.
    train_pool = [proto_task_generator(args, data) for _ in range(args.epoch)]
    pools = {}
    for shot in args.shots:
        args.k_shot_test = shot
        pools[str(shot)] = downstream_task_generator(args, data, args.n_way, shot)
    labels = torch.as_tensor(data.labels).detach().cpu().numpy().reshape(-1)
    node_ids_by_class = {int(c): np.flatnonzero(labels == c) for c in np.unique(labels)}
    classes = tuple(sorted(node_ids_by_class))
    neg_rng = random.Random(args.seed + 56000)
    negative_ids = []
    negative_labels = []
    num_candidate_classes = []
    for sid, _, _, _ in train_pool:
        anchor_classes = set(labels[np.asarray(sid, dtype=int).reshape(-1)].tolist())
        possible = [c for c in classes if c not in anchor_classes]
        if not possible:
            raise ValueError("No GT-negative class exists outside anchors")
        negative_class = neg_rng.choice(possible)
        node_id = int(neg_rng.choice(node_ids_by_class[negative_class].tolist()))
        negative_ids.append(node_id)
        negative_labels.append(negative_class)
        num_candidate_classes.append(len(possible))

    best_valid = {str(shot): float("-inf") for shot in args.shots}
    checkpoints = {str(shot): root / f"exp056_{shot}shot_best.pt" for shot in args.shots}
    report["checkpoints"] = {shot: str(path) for shot, path in checkpoints.items()}
    window = []
    report["status"] = "training"
    save_json(out, report)

    for epoch, (sid, ys, qid, yq) in enumerate(train_pool, start=1):
        model.network.train()
        enc = model.network(data.features, adj=data.adj)
        sid_t = torch.as_tensor(sid, dtype=torch.long, device=dev).reshape(-1)
        qid_t = torch.as_tensor(qid, dtype=torch.long, device=dev).reshape(-1)
        ys_t = torch.as_tensor(ys, dtype=torch.long, device=dev).reshape(-1)
        yq_t = torch.as_tensor(yq, dtype=torch.long, device=dev).reshape(-1)
        prototypes = enc[sid_t]
        dist = (enc[qid_t, None, :] - prototypes[None, :, :]).square().sum(-1)
        # Match meta.ProtoNet.forward: query pseudo labels map to anchor ORDER.
        mapped = (yq_t[:, None] == ys_t[None, :]).long().argmax(1)
        output = F.log_softmax(-dist, dim=1)
        proto_loss = F.nll_loss(output, mapped)
        neg_dist = (prototypes - enc[negative_ids[epoch - 1]]).square().sum(1)
        negative_hinge = F.relu(args.neg_margin - neg_dist)
        negative_loss = negative_hinge.mean()
        loss = proto_loss + args.neg_weight * negative_loss
        model.meta_optimizer.zero_grad()
        loss.backward()
        model.meta_optimizer.step()

        sid_np = np.asarray(sid, dtype=int).reshape(-1)
        qid_np = np.asarray(qid, dtype=int).reshape(-1)
        row = {
            "proto_loss": float(proto_loss.detach().item()),
            "negative_loss": float(negative_loss.detach().item()),
            "weighted_negative_loss": float(args.neg_weight * negative_loss.detach().item()),
            "total_loss": float(loss.detach().item()),
            "train_accuracy": float((output.argmax(1) == mapped).float().mean().item()),
            "negative_anchor_sqdist_mean": float(neg_dist.detach().mean().item()),
            "negative_active_hinge_fraction": float((neg_dist.detach() < args.neg_margin).float().mean().item()),
            "anchor_gt_distinct": float(np.unique(labels[sid_np]).size == args.n_way),
            "anchor_unique_gt_classes": int(np.unique(labels[sid_np]).size),
            "query_gt_match_rate": float((labels[qid_np] == np.repeat(labels[sid_np], args.q_query)).mean()),
            "negative_gt_collision": float(negative_labels[epoch - 1] in labels[sid_np]),
            "num_negative_candidate_classes": num_candidate_classes[epoch - 1],
        }
        window.append(row)
        if epoch % args.eval_interval != 0 and epoch != args.epoch:
            continue
        diag = summarize_window(window)
        window.clear()
        val_by_shot = {}
        for shot in args.shots:
            key = str(shot)
            val = evaluate(model, data, pools[key][0], args.n_way, shot)
            val_by_shot[key] = val
            if val["accuracy"] > best_valid[key]:
                best_valid[key] = val["accuracy"]
                torch.save({"epoch": epoch, "model": model.network.state_dict()}, checkpoints[key])
                report["best_by_shot"][key] = {"epoch": epoch, "validation": val, "checkpoint": str(checkpoints[key])}
        report["history"].append(
            {"epoch": epoch, "diagnostics": diag, "validation_by_shot": val_by_shot, "elapsed_seconds": round(time.perf_counter() - start, 2)}
        )
        report["elapsed_seconds"] = round(time.perf_counter() - start, 2)
        save_json(out, report)
        print(
            f"[exp056] epoch={epoch} proto={diag['proto_loss_mean']:.4f} "
            f"neg={diag['negative_loss_mean']:.4f} "
            f"hinge_active={diag['negative_active_hinge_fraction']:.2%} "
            f"valid=" + " ".join(f"{k}shot:{v['accuracy']:.2%}" for k, v in val_by_shot.items()),
            flush=True,
        )

    for shot in args.shots:
        key = str(shot)
        state = torch.load(checkpoints[key], map_location=dev, weights_only=True)
        model.network.load_state_dict(state["model"])
        report["test_by_shot"][key] = evaluate(model, data, pools[key][1], args.n_way, shot)
    report["status"] = "completed"
    report["elapsed_seconds"] = round(time.perf_counter() - start, 2)
    save_json(out, report)
    for key, test in report["test_by_shot"].items():
        print(
            f"[exp056] DONE {key}-shot best_epoch={report['best_by_shot'][key]['epoch']} "
            f"valid={report['best_by_shot'][key]['validation']['accuracy']:.4%} "
            f"test={test['accuracy']:.4%} ±{test['ci95_half_width']:.4%}",
            flush=True,
        )
    print(f"[exp056] json={out}", flush=True)


if __name__ == "__main__":
    main()
