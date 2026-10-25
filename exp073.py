"""exp073: NAQ-Feat + pairwise negatives between queries of distinct pseudo-classes.

Uses original NAQ support/query episodes; no ground-truth labels in training.
The original ProtoNet NLL is unchanged, with a normalized-distance hinge
repulsion for all cross-pseudo-class query pairs within each episode.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from data import load_data
from exp057 import evaluate
from meta import ProtoNet
from task_generator import downstream_task_generator, proto_task_generator
from utils import seed_everything


def arguments():
    p = argparse.ArgumentParser(description="NAQ with cross-class query-query negatives")
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
    p.add_argument("--shots", type=int, nargs="+", default=[1, 5])
    p.add_argument("--neg_lambda", type=float, default=0.1)
    p.add_argument("--neg_margin", type=float, default=1.0)
    p.add_argument("--output", default="exp073.json")
    p.add_argument("--save_checkpoints", action="store_true")
    return p.parse_args()


def write_json(path, report):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def step(model, episode, args, device):
    sid, ys, qid, yq = episode
    z = model.network(model.Data.features, adj=model.Data.adj)
    support = z[sid]
    queries = z[qid]
    prototypes = support.reshape(args.n_way, args.k_shot, -1).mean(dim=1)
    distances = model.euclidean_dist(queries, prototypes)
    output = F.log_softmax(-distances, dim=1)

    # Match original ProtoNet.forward's pseudo-label remapping exactly.
    if args.k_shot == 1:
        pseudo = torch.tensor([ys.tolist().index(t) for t in yq.tolist()], device=device)
    else:
        compressed = ys.reshape(-1, args.k_shot).float().mean(dim=1).long()
        pseudo = torch.tensor([compressed.tolist().index(t) for t in yq.tolist()], device=device)
    loss_naq = F.nll_loss(output, pseudo)

    # Pair every query from pseudo-class i with queries from j != i.
    # All pseudo-labels come from NAQ episode construction, never GT.
    query_unit = F.normalize(queries, p=2, dim=-1)
    cross_mask = pseudo[:, None] != pseudo[None, :]
    distances = torch.cdist(query_unit, query_unit, p=2)
    cross_distances = distances[cross_mask]
    if cross_distances.numel() == 0:
        raise ValueError("Episode has no cross-pseudo-class query pairs")
    loss_neg = F.relu(args.neg_margin - cross_distances).mean()
    loss = loss_naq + args.neg_lambda * loss_neg

    model.meta_optimizer.zero_grad()
    loss.backward()
    model.meta_optimizer.step()
    accuracy = float((output.argmax(dim=1) == pseudo).float().mean().item())
    return (loss_naq.item(), loss_neg.item(), loss.item(), accuracy, cross_distances.detach().mean().item(), int(cross_distances.numel()))


def main():
    args = arguments()
    if args.dataset != "Cora_full" or args.n_way != 5 or args.k_shot != 1:
        raise ValueError("Only Cora_full / 5-way / 1-shot training is supported")
    if min(args.epoch, args.eval_interval, args.q_query) < 1:
        raise ValueError("Epoch, interval and q_query must be positive")
    if args.neg_lambda < 0 or not 0 < args.neg_margin <= 2:
        raise ValueError("Require lambda >= 0 and normalized-distance margin in (0,2]")
    args.setting = "unsup"
    args.query_generation = "NaQ"
    args.type = "feature"
    args.k_shot_test = args.shots[0]
    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    root = Path(__file__).resolve().parent
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    report = {
        "experiment": "exp073",
        "status": "initializing",
        "args": vars(args).copy(),
        "protocol": {
            "train_episodes": "Original NAQ-Feat proto_task_generator; unchanged support and query sampling",
            "negative_pairs": "All ordered query pairs assigned to different NAQ pseudo-classes within each episode",
            "objective": "NAQ NLL + lambda * mean relu(margin - L2(normalized query embeddings)) over cross-class query pairs",
            "label_usage": "Only NAQ pseudo-labels are used in training; GT is used in downstream validation/test only",
            "model_selection": "Best validation checkpoint independently for 1-shot and 5-shot; test once",
            "note": "Pairwise query repulsion without oracle information; cross-class pseudo-negative pairs can include GT false negatives",
        },
        "history": [],
        "best_by_shot": {},
        "test_by_shot": {},
    }
    write_json(output, report)
    start = time.perf_counter()
    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    data.add_diffusion(args)
    episodes = [proto_task_generator(args, data) for _ in tqdm(range(args.epoch), desc="NAQ episodes", unit="episode")]
    downstream = {}
    for shot in args.shots:
        args.k_shot_test = shot
        downstream[str(shot)] = downstream_task_generator(args, data, args.n_way, shot)

    config = [
        ("GCN", (data.features.shape[1], 2 * args.latent), True),
        ("GCN", (2 * args.latent, args.latent), True),
        ("linear", (args.latent, args.n_way), True),
    ]
    model = ProtoNet(args, config, data).to(device)
    best_accuracy = {str(s): float("-inf") for s in args.shots}
    best_weights = {}
    bar = tqdm(episodes, desc="exp073 query negatives", unit="epoch", dynamic_ncols=True)
    for epoch, episode in enumerate(bar, 1):
        model.network.train()
        l_naq, l_neg, l_total, accuracy, mean_dist, num_pairs = step(model, episode, args, device)
        if epoch % args.eval_interval != 0 and epoch != args.epoch:
            continue
        row = {
            "epoch": epoch,
            "train_naq_loss": l_naq,
            "train_neg_loss": l_neg,
            "train_total_loss": l_total,
            "train_accuracy": accuracy,
            "train_mean_cross_query_distance": mean_dist,
            "train_num_ordered_negative_pairs": num_pairs,
            "validation": {},
        }
        for shot in args.shots:
            label = str(shot)
            valid = evaluate(model, data, downstream[label][0], args.n_way, shot)
            row["validation"][label] = valid
            if valid["accuracy"] > best_accuracy[label]:
                best_accuracy[label] = valid["accuracy"]
                best_weights[label] = {
                    "epoch": epoch,
                    "validation": valid,
                    "state": {k: v.detach().cpu().clone() for k, v in model.network.state_dict().items()},
                }
        report["history"].append(row)
        report["status"] = "training"
        report["elapsed_seconds"] = round(time.perf_counter() - start, 2)
        write_json(output, report)
        bar.set_postfix(naq=f"{l_naq:.3f}", neg=f"{l_neg:.3f}", val1=f"{best_accuracy.get('1', 0):.3f}", val5=f"{best_accuracy.get('5', 0):.3f}")
    bar.close()
    for shot in args.shots:
        label = str(shot)
        chosen = best_weights[label]
        model.network.load_state_dict(chosen["state"])
        report["test_by_shot"][label] = evaluate(model, data, downstream[label][1], args.n_way, shot)
        report["best_by_shot"][label] = {"epoch": chosen["epoch"], "validation": chosen["validation"]}
        if args.save_checkpoints:
            torch.save({"model": chosen["state"], "epoch": chosen["epoch"], "args": vars(args)}, root / f"exp073_{shot}shot_best.pt")
    report["status"] = "completed"
    report["elapsed_seconds"] = round(time.perf_counter() - start, 2)
    write_json(output, report)
    print("[exp073] " + " ".join(f"{s}shot={report['test_by_shot'][str(s)]['accuracy']:.2%}" for s in args.shots), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
