"""exp031: Original NAQ-Feat + projected-dot-product 1-hop prototype adaptation.

Training uses original GCN, NAQ pseudo-tasks, optimizer and ProtoNet NLL objective;
the query-conditioned squared-Euclidean distance uses learned Q/K attention. Validation/Test
reuse immutable exp026 episodes and select checkpoints only by validation accuracy.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import load_data
from exp026 import load_fixed_episodes
from model import GNNEncoder
from task_generator import proto_task_generator
from utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="exp031: NAQ with 1-hop prototype correction")
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
    p.add_argument("--correction_lambda", type=float, default=0.1)
    p.add_argument("--max_neighbors", type=int, default=0, help="0 means ALL 1-hop neighbors")
    p.add_argument("--json_path", default="exp031.json")
    return p.parse_args()


def build_one_hop(adj, max_neighbors=0):
    """Construct true 1-hop context IDs, including center node, from graph adjacency.

    Ignores normalized edge weights; zero-weight entries are excluded. Neighbor
    truncation, when requested, uses ascending node IDs (not PPR/top-k affinity).
    """
    if adj.layout == torch.sparse_csr:
        coo = adj.to_sparse_coo().coalesce()
    elif adj.layout == torch.sparse_coo:
        coo = adj.coalesce()
    else:
        raise TypeError(f"Expected sparse adjacency, got {adj.layout}")
    n = adj.size(0)
    edges = coo.indices().cpu().numpy()
    values = coo.values().detach().cpu().numpy()
    rows, cols = edges[:, values != 0]
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    neighbors = [[] for _ in range(n)]
    for row, col in zip(rows.tolist(), cols.tolist()):
        if row != col and (not neighbors[row] or neighbors[row][-1] != col):
            neighbors[row].append(col)
    if max_neighbors > 0:
        neighbors = [ns[:max_neighbors] for ns in neighbors]
    return neighbors


class SupportContextMetric(nn.Module):
    def __init__(self, neighbors, tau, lam, latent, attn_dim):
        super().__init__()
        self.neighbors = neighbors
        self.tau = tau
        self.lam = lam
        self.query_proj = nn.Linear(latent, attn_dim, bias=False)
        self.key_proj = nn.Linear(latent, attn_dim, bias=False)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)

    def distances(self, embeddings, supports, queries, k_shot=1, return_stats=False):
        device = embeddings.device
        support_ids = torch.as_tensor(supports, device=device, dtype=torch.long).reshape(-1, k_shot)
        query_ids = torch.as_tensor(queries, device=device, dtype=torch.long).reshape(-1)
        q = embeddings[query_ids]
        p = embeddings[support_ids].mean(dim=1)
        contexts = [
            list(dict.fromkeys(node for center in class_ids for node in [center, *self.neighbors[center]]))
            for class_ids in support_ids.detach().cpu().tolist()
        ]
        max_len = max(map(len, contexts))
        indices = torch.zeros((len(contexts), max_len), device=device, dtype=torch.long)
        mask = torch.zeros((len(contexts), max_len), device=device, dtype=torch.bool)
        for c, ids in enumerate(contexts):
            indices[c, : len(ids)] = torch.as_tensor(ids, device=device)
            mask[c, : len(ids)] = True

        ctx = embeddings[indices]
        projected_q = self.query_proj(q)
        projected_ctx = self.key_proj(ctx)
        logits = torch.einsum("qd,cmd->qcm", projected_q, projected_ctx) / self.tau
        weights = logits.masked_fill(~mask[None], -torch.inf).softmax(dim=-1)
        attended = torch.einsum("qcm,cmd->qcd", weights, ctx)
        corrected = p[None] + self.lam * attended
        distances = (q[:, None] - corrected).square().sum(dim=-1)
        if not return_stats:
            return distances
        with torch.no_grad():
            raw = (q[:, None] - p[None]).square().sum(-1)
            entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(-1)
            sizes = mask.sum(-1).float()
            entropy /= torch.where(sizes > 1, sizes.log(), torch.ones_like(sizes))[None]
            stats = {
                "mean_context_size": float(sizes.mean()),
                "max_context_size": int(sizes.max()),
                "mean_attention_entropy": float(entropy.mean()),
                "top1_attention": float(weights.max(-1).values.mean()),
                "center_attention": float(weights[..., 0].mean()),
                "mean_raw_distance": float(raw.mean()),
                "mean_corrected_distance": float(distances.mean()),
                "raw_accuracy_available": True,
            }
        return distances, raw, stats


def remap_labels(y_support, y_query, device):
    labels = torch.as_tensor(y_support).reshape(-1).tolist()
    classes = list(dict.fromkeys(labels))
    return torch.tensor([classes.index(int(y)) for y in torch.as_tensor(y_query).reshape(-1).tolist()], device=device)


def evaluate(encoder, data, metric, episodes, args, include_episodes=False):
    encoder.eval()
    metric.eval()
    scores = []
    raw_scores = []
    with torch.no_grad():
        embeddings = encoder(data.features, adj=data.adj)
        for s, ys, q, yq in episodes:
            target = remap_labels(ys, yq, embeddings.device)
            d, raw, _ = metric.distances(embeddings, s, q, args.k_shot_test, True)
            scores.append(float((d.argmin(-1) == target).float().mean()))
            raw_scores.append(float((raw.argmin(-1) == target).float().mean()))
    scores = np.asarray(scores)
    raw_scores = np.asarray(raw_scores)
    report = {
        "accuracy": float(scores.mean()),
        "ci95_half_width": float(1.96 * scores.std(ddof=0) / np.sqrt(len(scores))),
        "raw_accuracy": float(raw_scores.mean()),
        "raw_ci95_half_width": float(1.96 * raw_scores.std(ddof=0) / np.sqrt(len(raw_scores))),
        "num_episodes": len(scores),
    }
    if include_episodes:
        report["episode_accuracies"] = scores.tolist()
        report["raw_episode_accuracies"] = raw_scores.tolist()
    return report


def main():
    args = parse_args()
    args.setting = "unsup"
    if args.query_generation != "NaQ" or args.type != "feature":
        raise ValueError("exp031 is only for original NAQ-Feat")
    if args.k_shot != 1 or args.k_shot_test != 1:
        raise ValueError("exp031 first study is 1-shot only")
    if args.attn_tau <= 0 or args.attn_dim <= 0 or args.correction_lambda < 0 or args.max_neighbors < 0:
        raise ValueError("Invalid attention parameters")
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
        raise ValueError("Frozen evaluation episode counts do not match args")

    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    data.add_diffusion(args)
    neighbors = build_one_hop(data.adj, args.max_neighbors)
    metric = SupportContextMetric(neighbors, args.attn_tau, args.correction_lambda, args.latent, args.attn_dim).to(device)
    config = [
        ("GCN", (data.features.shape[1], 2 * args.latent), True),
        ("GCN", (2 * args.latent, args.latent), True),
        ("linear", (args.latent, args.n_way), True),
    ]
    encoder = GNNEncoder(config).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(metric.parameters()), lr=args.lr)
    train_tasks = [proto_task_generator(args, data) for _ in range(args.epoch)]
    output_dir = Path(f"results/{args.dataset}/exp031")
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "best.pt"
    best_valid = -1.0
    best_epoch = None
    trajectory = []
    start = time.time()

    for epoch, (ids_s, ys, ids_q, yq) in enumerate(train_tasks, 1):
        encoder.train()
        metric.train()
        h = encoder(data.features, adj=data.adj)
        d, raw_d, stats = metric.distances(h, ids_s, ids_q, args.k_shot, True)
        targets = remap_labels(ys, yq, device)
        loss = F.cross_entropy(-d, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % args.eval_interval != 0 and epoch != args.epoch:
            continue
        valid = evaluate(encoder, data, metric, valid_tasks, args)
        row = {
            "epoch": epoch,
            "train_loss": float(loss.detach()),
            "train_accuracy": float((d.argmin(-1) == targets).float().mean()),
            "train_raw_accuracy": float((raw_d.argmin(-1) == targets).float().mean()),
            "attention": stats,
            "valid": valid,
        }
        trajectory.append(row)
        if valid["accuracy"] > best_valid:
            best_valid = valid["accuracy"]
            best_epoch = epoch
            torch.save({"epoch": epoch, "model": encoder.state_dict(), "metric": metric.state_dict(), "config": vars(args)}, ckpt_path)
        print(
            f"[exp031][Epoch {epoch}] loss={row['train_loss']:.4f} "
            f"train={row['train_accuracy']:.4f} "
            f"valid={valid['accuracy']:.4f} raw_valid={valid['raw_accuracy']:.4f} "
            f"best={best_valid:.4f}@{best_epoch} "
            f"attn_top1={stats['top1_attention']:.4f}",
            flush=True,
        )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(checkpoint["model"])
    metric.load_state_dict(checkpoint["metric"])
    test = evaluate(encoder, data, metric, test_tasks, args, include_episodes=True)
    report = {
        "experiment": "exp031",
        "description": "NAQ-Feat with learned Q/K dot-product 1-hop prototype adaptation",
        "config": vars(args),
        "protocol": {
            "train": "NAQ-Feat, original 2-layer GCN, projected dot-product attention and corrected squared Euclidean ProtoNet CE (no auxiliary CL)",
            "attention": "un-normalized dot product of separate learned Q/K linear projections",
            "adaptation": "query-conditioned support prototype = raw prototype + lambda * attended 1-hop context",
            "valid_file": str(valid_path),
            "test_file": str(test_path),
            "selection": "best corrected-ProtoNet validation accuracy",
            "test": "one final test at validation-selected checkpoint",
            "subgraph": "center and all 1-hop graph neighbors; no PPR",
        },
        "trajectory": trajectory,
        "best": {
            "epoch": best_epoch,
            "valid_accuracy": best_valid,
            "test_accuracy": test["accuracy"],
            "test_ci95_half_width": test["ci95_half_width"],
            "checkpoint": str(ckpt_path),
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
        f"Raw Euclidean Test (same GCN): {test['raw_accuracy']:.4%}\n"
        f"Checkpoint: {ckpt_path}\n"
    )
    (output_dir / "result.txt").write_text(summary, encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
