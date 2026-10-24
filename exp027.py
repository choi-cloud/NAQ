"""exp027: Original NAQ-Feat ProtoNet training + Euclidean ProtoNet evaluation.

Run at the root of the original NAQ repository after running exp026.
Training delegates to the original meta.ProtoNet.forward unchanged.
Validation / test use exp026's immutable episodes. Best checkpoint is selected
using validation ProtoNet accuracy; test is run only once after selection.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from data import load_data
from exp026 import load_fixed_episodes
from meta import ProtoNet
from task_generator import proto_task_generator
from utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="exp027 Original NAQ with ProtoNet evaluation")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", default="Amazon_clothing")
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--k_shot_test", type=int, default=1)
    parser.add_argument("--q_query", type=int, default=10)
    parser.add_argument("--q_query_test", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent", type=int, default=256)
    parser.add_argument("--l2_penalty", type=float, default=1e-4)
    parser.add_argument("--query_generation", default="NaQ")
    parser.add_argument("--type", default="feature")
    parser.add_argument("--num_valid_tasks", type=int, default=50)
    parser.add_argument("--num_downstream_tasks", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--episode_dir", default=None)
    parser.add_argument("--json_path", default="exp027.json")
    return parser.parse_args()


def episode_accuracy(embeddings, task, n_way, k_shot):
    supports, y_support, queries, y_query = task
    device = embeddings.device
    ids_support = torch.as_tensor(supports, dtype=torch.long, device=device)
    ids_query = torch.as_tensor(queries, dtype=torch.long, device=device)
    y_support = torch.as_tensor(y_support, dtype=torch.long, device=device).reshape(-1)
    y_query = torch.as_tensor(y_query, dtype=torch.long, device=device).reshape(-1)

    class_labels = y_support.reshape(n_way, k_shot)[:, 0]
    if torch.unique(class_labels).numel() != n_way:
        raise ValueError("Invalid support classes in fixed episode")
    if not bool((y_support.reshape(n_way, k_shot) == class_labels[:, None]).all()):
        raise ValueError("Supports are not grouped by class")
    targets = y_query[:, None] == class_labels[None, :]
    if not bool(targets.any(dim=1).all()):
        raise ValueError("Query labels missing from support classes")

    prototypes = embeddings[ids_support].reshape(n_way, k_shot, -1).mean(dim=1)
    distances = (embeddings[ids_query, None, :] - prototypes[None, :, :]).square().sum(-1)
    predictions = distances.argmin(dim=-1)
    return float((predictions == targets.long().argmax(dim=1)).float().mean())


def evaluate(meta, data, episodes, args, with_episodes=False):
    meta.network.eval()
    with torch.no_grad():
        embeddings = meta.network(data.features, adj=data.adj)
        scores = np.asarray([episode_accuracy(embeddings, task, args.n_way, args.k_shot_test) for task in episodes], dtype=np.float64)
    result = {
        "accuracy": float(scores.mean()),
        "ci95_half_width": float(1.96 * scores.std(ddof=0) / np.sqrt(len(scores))),
        "num_episodes": int(len(scores)),
    }
    if with_episodes:
        result["episode_accuracies"] = scores.tolist()
    return result


def main():
    args = parse_args()
    args.setting = "unsup"
    if args.query_generation != "NaQ" or args.type != "feature":
        raise ValueError("exp027 is restricted to Original NAQ-Feat")
    if args.k_shot != 1 or args.k_shot_test != 1:
        raise ValueError("exp027 uses 1-shot for both train and evaluation")
    if args.eval_interval <= 0 or args.epoch <= 0:
        raise ValueError("epoch and eval_interval must be positive")
    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    episode_dir = Path(
        args.episode_dir
        or (f"save/fixed_episodes/{args.dataset}/" f"{args.n_way}way_{args.k_shot_test}shot_{args.q_query_test}query_seed{args.seed}")
    )
    valid_path = episode_dir / "valid_episodes.json"
    test_path = episode_dir / "test_episodes.json"
    valid_episodes = load_fixed_episodes(valid_path, dataset=args.dataset, n_way=args.n_way, k_shot=args.k_shot_test, q_query=args.q_query_test)
    test_episodes = load_fixed_episodes(test_path, dataset=args.dataset, n_way=args.n_way, k_shot=args.k_shot_test, q_query=args.q_query_test)
    if len(valid_episodes) != args.num_valid_tasks or len(test_episodes) != args.num_downstream_tasks:
        raise ValueError("Fixed-episode counts differ from requested counts")

    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    data.add_diffusion(args)
    config = [
        ("GCN", (data.features.shape[1], 2 * args.latent), True),
        ("GCN", (2 * args.latent, args.latent), True),
        ("linear", (args.latent, args.n_way), True),
    ]
    meta = ProtoNet(args, config, data).to(device)

    # Original unsup_protonet.py prepares NAQ episodes before training begins.
    train_pool = [proto_task_generator(args, data) for _ in range(args.epoch)]
    output_dir = Path(f"results/{args.dataset}/exp027")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    best_valid = float("-inf")
    best_epoch = None
    trajectory = []
    start = time.time()

    for epoch, (id_spt, y_spt, id_query, y_query) in enumerate(train_pool, 1):
        meta.network.train()
        # Exact original NAQ ProtoNet update: squared Euclidean -> log_softmax -> NLL -> Adam.
        loss, train_acc = meta(id_spt, y_spt.to(device), id_query, y_query.to(device))
        if epoch % args.eval_interval != 0 and epoch != args.epoch:
            continue
        valid = evaluate(meta, data, valid_episodes, args)
        row = {"epoch": epoch, "train_loss": float(loss.detach()), "train_accuracy": float(train_acc), "valid": valid}
        trajectory.append(row)
        if valid["accuracy"] > best_valid:
            best_valid = valid["accuracy"]
            best_epoch = epoch
            torch.save({"epoch": epoch, "model": meta.network.state_dict(), "config": vars(args)}, best_path)
        print(
            f"[exp027][Epoch {epoch}] train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} valid={valid['accuracy']:.4f} "
            f"best={best_valid:.4f}@{best_epoch}",
            flush=True,
        )

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    meta.network.load_state_dict(checkpoint["model"])
    test = evaluate(meta, data, test_episodes, args, with_episodes=True)
    report = {
        "experiment": "exp027",
        "description": "Original NAQ-Feat training via original ProtoNet; Euclidean ProtoNet eval on exp026 frozen episodes",
        "config": vars(args).copy(),
        "protocol": {
            "train": "original meta.ProtoNet.forward (no change)",
            "validation": "Euclidean nearest class mean on exp026 fixed validation episodes",
            "checkpoint_selection": "highest validation ProtoNet accuracy",
            "test": "one final evaluation on exp026 fixed test episodes",
            "valid_file": str(valid_path),
            "test_file": str(test_path),
        },
        "trajectory": trajectory,
        "best": {
            "epoch": best_epoch,
            "valid_accuracy": best_valid,
            "test_accuracy": test["accuracy"],
            "test_ci95_half_width": test["ci95_half_width"],
            "checkpoint": str(best_path),
        },
        "test": test,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    with Path(args.json_path).open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
    summary = (
        f"Best Epoch: {best_epoch}\n"
        f"Valid: {best_valid:.4%}\n"
        f"Test: {test['accuracy']:.4%} ± {test['ci95_half_width']:.4%} (95% CI)\n"
        f"Checkpoint: {best_path}\n"
        f"Episode files: {valid_path} | {test_path}\n"
    )
    (output_dir / "result.txt").write_text(summary, encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
