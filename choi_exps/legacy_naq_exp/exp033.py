"""exp033: NAQ-Feat + feature-guided PPR attention prototype adaptation.

Training uses original GCN, NAQ pseudo-tasks, optimizer and ProtoNet NLL objective;
the query-conditioned squared-Euclidean distance uses learned Q/K and raw-feature cosine prior. Validation/Test
reuse immutable exp026 episodes and select checkpoints only by validation accuracy.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import load_data
from exp026 import load_fixed_episodes
from model import GNNEncoder
from task_generator import proto_task_generator
from utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="exp033: NAQ with feature-guided PPR attention")
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
    p.add_argument("--ppr_topk", type=int, default=20)
    p.add_argument("--ppr_alpha", type=float, default=0.15)
    p.add_argument("--ppr_iter", type=int, default=60)
    p.add_argument("--ppr_tol", type=float, default=1e-7)
    p.add_argument("--feature_prior_lambda", type=float, default=1.0)
    p.add_argument("--mask_query_from_context", action="store_true", default=True)
    p.add_argument("--allow_query_in_context", dest="mask_query_from_context", action="store_false")
    p.add_argument("--json_path", default="exp033.json")
    return p.parse_args()


def build_ppr_sampler(edge_index, num_nodes, alpha, iterations, tol, topk):
    """Directed PPR, matching exp032's original edge_index convention."""
    edges = edge_index.detach().cpu().numpy()
    adjacency = sp.csr_matrix((np.ones(edges.shape[1], dtype=np.float64), (edges[0], edges[1])), shape=(num_nodes, num_nodes))
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv = np.divide(1, degree, out=np.zeros_like(degree), where=degree > 0)
    transpose = (sp.diags(inv) @ adjacency).transpose().tocsr()
    dangling = degree == 0
    cache = {}

    def context(center):
        center = int(center)
        if center in cache:
            return cache[center]
        restart = np.zeros(num_nodes, dtype=np.float64)
        restart[center] = 1.0
        rank = restart.copy()
        for _ in range(iterations):
            updated = alpha * restart + (1 - alpha) * (transpose @ rank + rank[dangling].sum() * restart)
            if np.abs(updated - rank).sum() < tol:
                rank = updated
                break
            rank = updated
        rank[center] = -np.inf
        k = min(topk, num_nodes - 1)
        selected = np.argpartition(-rank, k - 1)[:k] if k else np.empty(0, dtype=np.int64)
        selected = selected[np.lexsort((selected, -rank[selected]))]
        cache[center] = [center, *selected.tolist()]
        return cache[center]

    return context, cache


class SupportContextMetric(nn.Module):
    def __init__(self, context_fn, raw_features, tau, lam, feature_prior_lambda, latent, attn_dim, mask_query):
        super().__init__()
        self.context_fn = context_fn
        self.register_buffer("raw_features", F.normalize(raw_features.float(), dim=-1))
        self.tau = tau
        self.lam = lam
        self.feature_prior_lambda = feature_prior_lambda
        self.mask_query = mask_query
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
            list(dict.fromkeys(node for center in class_ids for node in self.context_fn(center))) for class_ids in support_ids.detach().cpu().tolist()
        ]
        max_len = max(map(len, contexts))
        indices = torch.zeros((len(contexts), max_len), device=device, dtype=torch.long)
        mask = torch.zeros((len(contexts), max_len), device=device, dtype=torch.bool)
        for c, ids in enumerate(contexts):
            indices[c, : len(ids)] = torch.as_tensor(ids, device=device)
            mask[c, : len(ids)] = True
        if self.mask_query:
            mask = mask[None] & (indices[None] != query_ids[:, None, None])
            # Support center remains as a valid fallback unless support and query coincide.
            if not mask.any(-1).all():
                raise RuntimeError("Empty context after excluding query; check episode sampling")
        else:
            mask = mask[None].expand(len(query_ids), -1, -1)

        ctx = embeddings[indices]

        projected_q = self.query_proj(q)
        projected_ctx = self.key_proj(ctx)

        # Scaled dot-product attention
        learned_logits = torch.einsum("qd,cmd->qcm", projected_q, projected_ctx) / (projected_q.size(-1) ** 0.5)

        # Feature similarity prior
        prior_logits = torch.einsum("qd,cmd->qcm", self.raw_features[query_ids], self.raw_features[indices]) / self.tau

        # Feature-guided attention
        guided_logits = learned_logits + self.feature_prior_lambda * prior_logits
        weights = guided_logits.masked_fill(~mask, -torch.inf).softmax(dim=-1)

        attended = torch.einsum("qcm,cmd->qcd", weights, ctx)
        corrected = p[None] + self.lam * attended
        distances = (q[:, None] - corrected).square().sum(dim=-1)
        if not return_stats:
            return distances
        with torch.no_grad():
            raw = (q[:, None] - p[None]).square().sum(-1)
            normalized_entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(-1)
            sizes = mask.sum(-1).float()
            normalized_entropy /= torch.where(sizes > 1, sizes.log(), torch.ones_like(sizes))
            attention_top = weights.argmax(-1)
            feature_only_weights = (self.feature_prior_lambda * prior_logits).masked_fill(~mask, -torch.inf).softmax(-1)
            learned_only_weights = learned_logits.masked_fill(~mask, -torch.inf).softmax(-1)
            entropy_prior = -(feature_only_weights.clamp_min(1e-12) * feature_only_weights.clamp_min(1e-12).log()).sum(-1)
            entropy_prior /= torch.where(sizes > 1, sizes.log(), torch.ones_like(sizes))
            valid_learn = learned_logits[mask]
            valid_prior = prior_logits[mask]
            stats = {
                "mean_context_size": float(sizes.mean()),
                "mean_attention_entropy": float(normalized_entropy.mean()),
                "mean_prior_only_entropy": float(entropy_prior.mean()),
                "top1_attention": float(weights.max(-1).values.mean()),
                "center_attention": float(weights[..., 0].mean()),
                "attention_top1_is_center": float((attention_top == 0).float().mean()),
                "attention_top1_matches_feature_only": float((attention_top == feature_only_weights.argmax(-1)).float().mean()),
                "attention_top1_matches_learned_only": float((attention_top == learned_only_weights.argmax(-1)).float().mean()),
                "learned_logit_std": float(valid_learn.std(unbiased=False)),
                "prior_cosine_std": float(valid_prior.std(unbiased=False)),
                "scaled_prior_logit_std": float((self.feature_prior_lambda * valid_prior).std(unbiased=False)),
                "mean_raw_distance": float(raw.mean()),
                "mean_corrected_distance": float(distances.mean()),
                "mean_proto_correction_norm": float((self.lam * attended).norm(dim=-1).mean()),
                "mean_prototype_norm": float(p.norm(dim=-1).mean()),
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
        raise ValueError("exp033 is only for original NAQ-Feat")
    if args.k_shot != 1 or args.k_shot_test != 1:
        raise ValueError("exp033 first study is 1-shot only")
    if (
        args.attn_tau <= 0
        or args.attn_dim <= 0
        or args.correction_lambda < 0
        or args.ppr_topk <= 0
        or not 0 < args.ppr_alpha <= 1
        or args.ppr_iter <= 0
        or args.feature_prior_lambda < 0
    ):
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
    get_context, context_cache = build_ppr_sampler(
        data.edge_index, data.features.shape[0], args.ppr_alpha, args.ppr_iter, args.ppr_tol, args.ppr_topk
    )
    metric = SupportContextMetric(
        get_context,
        data.features,
        args.attn_tau,
        args.correction_lambda,
        args.feature_prior_lambda,
        args.latent,
        args.attn_dim,
        args.mask_query_from_context,
    ).to(device)
    config = [
        ("GCN", (data.features.shape[1], 2 * args.latent), True),
        ("GCN", (2 * args.latent, args.latent), True),
        ("linear", (args.latent, args.n_way), True),
    ]
    encoder = GNNEncoder(config).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(metric.parameters()), lr=args.lr)
    train_tasks = [proto_task_generator(args, data) for _ in range(args.epoch)]
    # Cache PPR once per unique support used by training and validation.
    centers = set()
    for s, _, _, _ in train_tasks:
        centers.update(np.asarray(s, dtype=np.int64).reshape(-1).tolist())
    for s, _, _, _ in valid_tasks:
        centers.update(np.asarray(s, dtype=np.int64).reshape(-1).tolist())
    print(f"[exp033] Building PPR for {len(centers)} train/valid support centers", flush=True)
    for i, center in enumerate(sorted(centers), 1):
        get_context(center)
        if i % 1000 == 0:
            print(f"[exp033] PPR cached {i}/{len(centers)}", flush=True)
    output_dir = Path(f"results/{args.dataset}/exp033")
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
        q_grad = float(metric.query_proj.weight.grad.norm()) if metric.query_proj.weight.grad is not None else 0.0
        k_grad = float(metric.key_proj.weight.grad.norm()) if metric.key_proj.weight.grad is not None else 0.0
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
            "gradient": {"query_proj_norm": q_grad, "key_proj_norm": k_grad},
            "valid": valid,
        }
        trajectory.append(row)
        if valid["accuracy"] > best_valid:
            best_valid = valid["accuracy"]
            best_epoch = epoch
            torch.save({"epoch": epoch, "model": encoder.state_dict(), "metric": metric.state_dict(), "config": vars(args)}, ckpt_path)
        print(
            f"[exp033][Epoch {epoch}] loss={row['train_loss']:.4f} "
            f"train={row['train_accuracy']:.4f} "
            f"valid={valid['accuracy']:.4f} raw_valid={valid['raw_accuracy']:.4f} "
            f"best={best_valid:.4f}@{best_epoch} "
            f"attn_top1={stats['top1_attention']:.4f} "
            f"entropy={stats['mean_attention_entropy']:.4f} "
            f"center={stats['center_attention']:.4f} "
            f"logit_std={stats['learned_logit_std']:.3f}/{stats['scaled_prior_logit_std']:.3f} "
            f"qk_grad={q_grad:.3e}/{k_grad:.3e}",
            flush=True,
        )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(checkpoint["model"])
    metric.load_state_dict(checkpoint["metric"])
    test = evaluate(encoder, data, metric, test_tasks, args, include_episodes=True)
    report = {
        "experiment": "exp033",
        "description": "NAQ-Feat with feature-prior-guided Q/K dot-product PPR prototype adaptation",
        "config": vars(args),
        "protocol": {
            "train": "NAQ-Feat, original 2-layer GCN, projected dot-product + cosine-prior attention and corrected Euclidean ProtoNet CE (no auxiliary CL)",
            "attention": "Q/K projected dot product / tau plus feature_prior_lambda * raw-feature cosine",
            "adaptation": "query-conditioned support prototype = raw prototype + lambda * attended PPR context",
            "valid_file": str(valid_path),
            "test_file": str(test_path),
            "selection": "best corrected-ProtoNet validation accuracy",
            "test": "one final test at validation-selected checkpoint",
            "subgraph": "directed PPR Top-K from original edge_index plus support center, query itself masked by default",
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
        "ppr_unique_centers": len(context_cache),
        "elapsed_seconds": round(time.time() - start, 2),
    }
    with Path(args.json_path).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
    summary = (
        f"Best Epoch: {best_epoch}\n"
        f"Valid: {best_valid:.4%}\n"
        f"Test: {test['accuracy']:.4%} ± {test['ci95_half_width']:.4%} (95% CI)\n"
        f"Raw Euclidean Test (same GCN): {test['raw_accuracy']:.4%}\n"
        f"Feature prior lambda: {args.feature_prior_lambda}\n"
        f"PPR topk: {args.ppr_topk}\n"
        f"Checkpoint: {ckpt_path}\n"
    )
    (output_dir / "result.txt").write_text(summary, encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
