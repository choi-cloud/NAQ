"""exp025: Paired NAQ-Feat Euclidean ProtoNet vs Cross-part ProtoNet.

Run in NAQ repository root, alongside exp023.py and its dependencies.
One shared pseudo-task sequence, initial encoder weights, validation and test tasks.
Select best checkpoint separately by the SAME evaluation classifier as training,
using validation only; test only once at the end on the paired episodes.
"""

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import load_data
from model import GNNEncoder
from task_generator import downstream_task_generator, proto_task_generator
from utils import seed_everything
from exp023 import CrossPartMetric


class EpisodeModel(nn.Module):
    def __init__(self, config, metric_name, num_parts, attn_tau):
        super().__init__()
        self.encoder = GNNEncoder(config)
        self.metric_name = metric_name
        self.part_metric = CrossPartMetric(num_parts, attn_tau) if metric_name == "cross_part" else None

    def distances(self, h, supports, queries, k_shot):
        supports = torch.as_tensor(supports, device=h.device, dtype=torch.long)
        queries = torch.as_tensor(queries, device=h.device, dtype=torch.long)
        prototypes = h[supports].reshape(-1, k_shot, h.shape[-1]).mean(1)
        query_h = h[queries]
        if self.part_metric is not None:
            return self.part_metric(query_h, prototypes)
        return (query_h[:, None, :] - prototypes[None, :, :]).square().sum(-1)


def targets(y_support, y_query, n_way, k_shot, device):
    labels = torch.as_tensor(y_support).reshape(-1).tolist()
    classes = labels[::k_shot]
    if len(classes) != n_way or len(set(classes)) != n_way:
        raise ValueError("Supports must have n_way consecutive groups of k_shot examples")
    return torch.tensor([classes.index(int(y)) for y in torch.as_tensor(y_query).reshape(-1).tolist()], device=device, dtype=torch.long)


def score(h, model, tasks, args, return_per_episode=False):
    results = []
    for supports, y_support, queries, y_query in tasks:
        y = targets(y_support, y_query, args.n_way, args.k_shot_test, h.device)
        distances = model.distances(h, supports, queries, args.k_shot_test)
        acc = (distances.argmin(-1) == y).float().mean()
        results.append(float(acc))
    vals = np.asarray(results, dtype=np.float64)
    output = {"accuracy": float(vals.mean()), "ci95_half_width": float(1.96 * vals.std(ddof=0) / np.sqrt(len(vals))), "num_episodes": len(vals)}
    if return_per_episode:
        output["episode_accuracies"] = vals.tolist()
    return output


def evaluate(model, data, tasks, args, return_per_episode=False):
    model.eval()
    with torch.no_grad():
        h = model.encoder(data.features, adj=data.adj)
        return score(h, model, tasks, args, return_per_episode)


def train_model(name, model, data, train_tasks, valid_tasks, args, output_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_accuracy = float("-inf")
    best_epoch = None
    best_path = os.path.join(output_dir, f"{name}_best.pt")
    trajectory = []
    started = time.time()

    for epoch, (supports, y_support, queries, y_query) in enumerate(train_tasks, 1):
        model.train()
        h = model.encoder(data.features, adj=data.adj)
        y = targets(y_support, y_query, args.n_way, args.k_shot, h.device)
        distances = model.distances(h, supports, queries, args.k_shot)
        loss = F.cross_entropy(-distances / args.temperature, y)
        train_acc = float((distances.argmin(-1) == y).float().mean().detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch % args.eval_interval != 0 and epoch != args.epoch:
            continue

        valid = evaluate(model, data, valid_tasks, args)
        row = {"epoch": epoch, "train_loss": float(loss.detach()), "train_accuracy": train_acc, "valid": valid}
        trajectory.append(row)
        if valid["accuracy"] > best_accuracy:
            best_accuracy = valid["accuracy"]
            best_epoch = epoch
            torch.save({"epoch": epoch, "model": model.state_dict(), "metric": name, "config": vars(args)}, best_path)
        print(
            f"[{name}][Epoch {epoch}] loss={row['train_loss']:.4f} "
            f"train={train_acc:.4f} valid={valid['accuracy']:.4f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )

    # Reload each model's validation-selected weights before its final test.
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    return {
        "best_epoch": best_epoch,
        "best_valid_accuracy": best_accuracy,
        "trajectory": trajectory,
        "checkpoint": best_path,
        "train_seconds": round(time.time() - started, 2),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="exp025 paired NAQ training/evaluation")
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
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--l2_penalty", type=float, default=1e-4)
    parser.add_argument("--query_generation", default="NaQ")
    parser.add_argument("--type", default="feature")
    parser.add_argument("--num_parts", type=int, default=8)
    parser.add_argument("--attn_tau", type=float, default=0.1)
    parser.add_argument("--num_valid_tasks", type=int, default=50)
    parser.add_argument("--num_downstream_tasks", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--json_path", default="exp025.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.k_shot != 1 or args.k_shot_test != 1:
        raise ValueError("exp025 currently compares NAQ with 1-shot train and 1-shot test")
    if args.query_generation != "NaQ" or args.type != "feature":
        raise ValueError("exp025 is restricted to Original NAQ-Feat pseudo tasks")
    if args.latent % args.num_parts != 0 or args.num_parts < 2:
        raise ValueError("latent must be divisible by num_parts >= 2")
    if args.attn_tau <= 0 or args.temperature <= 0 or args.eval_interval <= 0:
        raise ValueError("temperatures and eval_interval must be positive")

    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    args.setting = "unsup"
    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    data.add_diffusion(args)
    config = [("GCN", (data.features.shape[1], 2 * args.latent), True), ("GCN", (2 * args.latent, args.latent), True)]

    print("Sampling the ONE shared NAQ-Feat training-episode sequence...", flush=True)
    train_tasks = [proto_task_generator(args, data) for _ in range(args.epoch)]
    valid_tasks, test_tasks = downstream_task_generator(args, data, args.n_way, args.k_shot_test)
    if len(valid_tasks) != args.num_valid_tasks or len(test_tasks) != args.num_downstream_tasks:
        raise RuntimeError(f"Unexpected evaluation episode counts: valid={len(valid_tasks)}, " f"test={len(test_tasks)}")

    # Match initialization exactly; model order and RNG advances do not change it.
    seed_everything(args.seed)
    baseline = EpisodeModel(config, "euclidean", args.num_parts, args.attn_tau).to(device)
    ours = EpisodeModel(config, "cross_part", args.num_parts, args.attn_tau).to(device)
    ours.encoder.load_state_dict(copy.deepcopy(baseline.encoder.state_dict()))
    init_state = copy.deepcopy(baseline.encoder.state_dict())
    output_dir = f"results/{args.dataset}/exp025"
    os.makedirs(output_dir, exist_ok=True)

    history = {
        "experiment": "exp025",
        "description": "Paired retraining: same NAQ pseudo tasks, encoder initialization, and downstream episodes",
        "config": vars(args).copy(),
        "protocol": {
            "shared_train_episodes": True,
            "shared_valid_test_episodes": True,
            "shared_initial_gcn_weights": True,
            "checkpoint_selection": "own validation ProtoNet accuracy",
            "test_protocol": "one final test after validation selection",
            "euclidean": "squared Euclidean ProtoNet train/eval",
            "cross_part": "cross-part ProtoNet train/eval",
        },
        "models": {},
    }

    for name, model in [("euclidean", baseline), ("cross_part", ours)]:
        model.encoder.load_state_dict(copy.deepcopy(init_state))
        # Reset RNG for any stochastic training components (e.g. dropout).
        seed_everything(args.seed)
        print(f"=== {name}: training {args.epoch} shared episodes ===", flush=True)
        fit = train_model(name, model, data, train_tasks, valid_tasks, args, output_dir)
        fit["test"] = evaluate(model, data, test_tasks, args, return_per_episode=True)
        # Save selected-validation model diagnostics as a separate metric, with same embeddings.
        history["models"][name] = fit

    base = np.asarray(history["models"]["euclidean"]["test"]["episode_accuracies"])
    part = np.asarray(history["models"]["cross_part"]["test"]["episode_accuracies"])
    differences = part - base
    history["paired_comparison"] = {
        "test_accuracy_delta_pp": float(100 * differences.mean()),
        "delta_ci95_half_width_pp": float(100 * 1.96 * differences.std(ddof=0) / np.sqrt(len(differences))),
        "num_paired_test_episodes": int(len(differences)),
        "episodes_cross_part_better": int((differences > 0).sum()),
        "episodes_euclidean_better": int((differences < 0).sum()),
        "episodes_equal": int((differences == 0).sum()),
    }
    with open(args.json_path, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2, allow_nan=False)
    b = history["models"]["euclidean"]
    o = history["models"]["cross_part"]
    report = (
        f"Euclidean: best={b['best_epoch']} valid={b['best_valid_accuracy']:.4%} "
        f"test={b['test']['accuracy']:.4%}±{b['test']['ci95_half_width']:.4%}\n"
        f"Cross-part: best={o['best_epoch']} valid={o['best_valid_accuracy']:.4%} "
        f"test={o['test']['accuracy']:.4%}±{o['test']['ci95_half_width']:.4%}\n"
        f"Paired delta (Cross-part - Euclidean): "
        f"{history['paired_comparison']['test_accuracy_delta_pp']:.3f} ± "
        f"{history['paired_comparison']['delta_ci95_half_width_pp']:.3f} percentage points"
    )
    with open(f"{output_dir}/result.txt", "w", encoding="utf-8") as file:
        file.write(report + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
