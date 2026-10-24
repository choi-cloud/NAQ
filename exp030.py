"""exp030: NAQ-Feat + 1-hop structural InfoNCE with learned Q/K attention.

Only training adds a contrastive auxiliary loss. Model selection and testing use
unaltered Euclidean ProtoNet on exp026's immutable downstream episodes.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from data import load_data
from exp026 import load_fixed_episodes
from exp027 import episode_accuracy
from exp028 import build_one_hop
from model import GNNEncoder
from task_generator import proto_task_generator
from utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="exp030: NAQ-Feat + projected 1-hop Structural CL")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--dataset", default="Amazon_clothing")
    p.add_argument("--epoch", type=int, default=2000)
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=1)
    p.add_argument("--k_shot_test", type=int, default=1)
    p.add_argument("--q_query", type=int, default=10)
    p.add_argument("--q_query_test", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--latent", type=int, default=256)
    p.add_argument("--l2_penalty", type=float, default=1e-4)
    p.add_argument("--query_generation", default="NaQ")
    p.add_argument("--type", default="feature")
    p.add_argument("--num_valid_tasks", type=int, default=50)
    p.add_argument("--num_downstream_tasks", type=int, default=500)
    p.add_argument("--eval_interval", type=int, default=160)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--episode_dir", default=None)
    p.add_argument("--attn_tau", type=float, default=0.01)
    p.add_argument("--attn_dim", type=int, default=256)
    p.add_argument("--cl_tau", type=float, default=0.5)
    p.add_argument("--cl_weight", type=float, default=0.1)
    p.add_argument("--max_neighbors", type=int, default=0, help="0 uses all 1-hop neighbors")
    p.add_argument("--json_path", default="exp030.json")
    return p.parse_args()


def remap_labels(y_support, y_query, device, n_way, k_shot):
    ys = torch.as_tensor(y_support).reshape(n_way, k_shot)
    classes = ys[:, 0].tolist()
    if len(set(classes)) != n_way or not bool((ys == ys[:, :1]).all()):
        raise ValueError("Support labels must be grouped into distinct classes")
    query_labels = torch.as_tensor(y_query).reshape(-1).tolist()
    if any(y not in classes for y in query_labels):
        raise ValueError("Query labels contain unseen class")
    return torch.tensor([classes.index(y) for y in query_labels], device=device, dtype=torch.long)


class StructuralContrast(nn.Module):
    def __init__(self, neighbors, tau, latent, attn_dim):
        super().__init__()
        self.neighbors = neighbors
        self.tau = tau
        self.query_proj = nn.Linear(latent, attn_dim, bias=False)
        self.key_proj = nn.Linear(latent, attn_dim, bias=False)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)

    def attend(self, embeddings, support_ids, query_ids, k_shot):
        """Query-to-class support-context attention; output shape [num_queries, n_way, dim]."""
        supports = torch.as_tensor(support_ids, dtype=torch.long, device=embeddings.device).reshape(-1, k_shot)
        queries = torch.as_tensor(query_ids, dtype=torch.long, device=embeddings.device).reshape(-1)
        q = embeddings[queries]
        contexts = [
            list(dict.fromkeys(node for center in class_ids for node in [center, *self.neighbors[center]]))
            for class_ids in supports.detach().cpu().tolist()
        ]
        length = max(map(len, contexts))
        indices = torch.zeros((len(contexts), length), device=embeddings.device, dtype=torch.long)
        mask = torch.zeros((len(contexts), length), device=embeddings.device, dtype=torch.bool)
        for c, ids in enumerate(contexts):
            indices[c, : len(ids)] = torch.as_tensor(ids, device=embeddings.device)
            mask[c, : len(ids)] = True
        ctx = embeddings[indices]
        projected_q = F.normalize(self.query_proj(q), dim=-1)
        projected_ctx = F.normalize(self.key_proj(ctx), dim=-1)
        logits = torch.einsum("qd,cmd->qcm", projected_q, projected_ctx) / self.tau
        weights = logits.masked_fill(~mask[None], -torch.inf).softmax(dim=-1)
        attended = torch.einsum("qcm,cmd->qcd", weights, ctx)
        with torch.no_grad():
            entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(-1)
            sizes = mask.sum(-1).float()
            entropy /= torch.where(sizes > 1, sizes.log(), torch.ones_like(sizes))[None]
            stats = {
                "mean_context_size": float(sizes.mean()),
                "attention_entropy": float(entropy.mean()),
                "attention_top1": float(weights.max(-1).values.mean()),
                "attention_center": float(weights[..., 0].mean()),
            }
        return q, attended, stats


def evaluate(encoder, data, tasks, args, include_episodes=False):
    encoder.eval()
    with torch.no_grad():
        h = encoder(data.features, adj=data.adj)
        scores = np.asarray([episode_accuracy(h, task, args.n_way, args.k_shot_test) for task in tasks], dtype=np.float64)
    out = {
        "accuracy": float(scores.mean()),
        "ci95_half_width": float(1.96 * scores.std(ddof=0) / np.sqrt(len(scores))),
        "num_episodes": int(len(scores)),
    }
    if include_episodes:
        out["episode_accuracies"] = scores.tolist()
    return out


def main():
    args = parse_args()
    args.setting = "unsup"
    if args.query_generation != "NaQ" or args.type != "feature" or args.k_shot != 1 or args.k_shot_test != 1:
        raise ValueError("exp030 requires NAQ-Feat, 1-shot")
    if args.attn_tau <= 0 or args.attn_dim <= 0 or args.cl_tau <= 0 or args.cl_weight < 0 or args.max_neighbors < 0:
        raise ValueError("Invalid CL/attention parameters")
    if args.epoch <= 0 or args.eval_interval <= 0:
        raise ValueError("Epoch and evaluation interval must be positive")
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
    valid_tasks = load_fixed_episodes(valid_path, dataset=args.dataset, n_way=args.n_way, k_shot=args.k_shot_test, q_query=args.q_query_test)
    test_tasks = load_fixed_episodes(test_path, dataset=args.dataset, n_way=args.n_way, k_shot=args.k_shot_test, q_query=args.q_query_test)
    if len(valid_tasks) != args.num_valid_tasks or len(test_tasks) != args.num_downstream_tasks:
        raise ValueError("Fixed evaluation episode count mismatch")

    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    data.add_diffusion(args)
    neighbors = build_one_hop(data.adj, args.max_neighbors)
    context = StructuralContrast(neighbors, args.attn_tau, args.latent, args.attn_dim).to(device)
    config = [
        ("GCN", (data.features.shape[1], 2 * args.latent), True),
        ("GCN", (2 * args.latent, args.latent), True),
        ("linear", (args.latent, args.n_way), True),
    ]
    encoder = GNNEncoder(config).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(context.parameters()), lr=args.lr)
    train_tasks = [proto_task_generator(args, data) for _ in range(args.epoch)]
    output_dir = Path(f"results/{args.dataset}/exp030")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    best_valid = float("-inf")
    best_epoch = None
    trajectory = []
    start = time.time()

    for epoch, (s, ys, q, yq) in enumerate(train_tasks, 1):
        encoder.train()
        context.train()
        h = encoder(data.features, adj=data.adj)
        s_ids = torch.as_tensor(s, device=device, dtype=torch.long)
        q_ids = torch.as_tensor(q, device=device, dtype=torch.long)
        targets = remap_labels(ys, yq, device, args.n_way, args.k_shot)
        prototypes = h[s_ids].reshape(args.n_way, args.k_shot, -1).mean(dim=1)
        q_embed = h[q_ids]
        distances = (q_embed[:, None] - prototypes[None]).square().sum(-1)
        proto_loss = F.cross_entropy(-distances, targets)

        query_embedding, attended, attn_stats = context.attend(h, s_ids, q_ids, args.k_shot)
        similarities = F.cosine_similarity(query_embedding[:, None], attended, dim=-1)
        cl_logits = similarities / args.cl_tau
        cl_loss = F.cross_entropy(cl_logits, targets)
        loss = proto_loss + args.cl_weight * cl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % args.eval_interval != 0 and epoch != args.epoch:
            continue
        valid = evaluate(encoder, data, valid_tasks, args)
        with torch.no_grad():
            positive = similarities.gather(1, targets[:, None]).mean()
            negative = similarities.masked_fill(F.one_hot(targets, args.n_way).bool(), 0.0).sum(-1).div(args.n_way - 1).mean()
        row = {
            "epoch": epoch,
            "train_loss": float(loss.detach()),
            "proto_loss": float(proto_loss.detach()),
            "cl_loss": float(cl_loss.detach()),
            "train_accuracy": float((distances.argmin(-1) == targets).float().mean()),
            "mean_positive_similarity": float(positive),
            "mean_negative_similarity": float(negative),
            "attention": attn_stats,
            "valid": valid,
        }
        trajectory.append(row)
        if valid["accuracy"] > best_valid:
            best_valid = valid["accuracy"]
            best_epoch = epoch
            torch.save({"epoch": epoch, "model": encoder.state_dict(), "attention": context.state_dict(), "config": vars(args)}, best_path)
        print(
            f"[exp030][Epoch {epoch}] loss={row['train_loss']:.4f} "
            f"proto={row['proto_loss']:.4f} cl={row['cl_loss']:.4f} "
            f"train={row['train_accuracy']:.4f} valid={valid['accuracy']:.4f} "
            f"best={best_valid:.4f}@{best_epoch} "
            f"pos_sim={row['mean_positive_similarity']:.3f} "
            f"neg_sim={row['mean_negative_similarity']:.3f}",
            flush=True,
        )

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(checkpoint["model"])
    context.load_state_dict(checkpoint["attention"])
    test = evaluate(encoder, data, test_tasks, args, include_episodes=True)
    report = {
        "experiment": "exp030",
        "description": "NAQ-Feat Euclidean ProtoNet + 1-hop structural InfoNCE with learned Q/K attention",
        "config": vars(args),
        "protocol": {
            "train": "Original NAQ-Feat squared Euclidean ProtoNet CE + weighted 1-hop structural InfoNCE",
            "attention": "learned separate Q/K linear projections, projected cosine softmax; values are original GCN embeddings",
            "positive": "same pseudo-class support context",
            "negative": "other pseudo-class support contexts",
            "validation": "raw Euclidean ProtoNet on exp026 fixed episodes",
            "checkpoint_selection": "highest validation accuracy",
            "test": "one final test after validation selection",
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
    with Path(args.json_path).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
    summary = (
        f"Best Epoch: {best_epoch}\n"
        f"Valid: {best_valid:.4%}\n"
        f"Test: {test['accuracy']:.4%} ± {test['ci95_half_width']:.4%} (95% CI)\n"
        f"Checkpoint: {best_path}\n"
    )
    (output_dir / "result.txt").write_text(summary, encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
