"""exp007: NAQ-Feat + PPR-local GCN-embedding attention.

Training: each pseudo-query attends ONLY to the PPR neighborhood of its
corresponding pseudo-support. The routed support prototype is query-conditionally shifted; other supports are unchanged.
Evaluation: each query is routed to the raw-feature-nearest support, without
using query labels. The routed support prototype is corrected; all queries are
classified by argmin squared Euclidean distance to the five prototypes.
Training uses Euclidean pairwise contrastive loss: positive distance squared
plus squared hinge on negative Euclidean distances (not CE/softmax).
Attention Q/K/V all originate from trainable GCN node embeddings.
Requires the NAQ repository's data.py, model.py, task_generator.py, utils.py.
"""

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data import load_data
from model import GNNEncoder
from task_generator import downstream_task_generator, proto_task_generator
from utils import seed_everything


class PPRNeighborhoods:
    def __init__(self, data, topk, alpha, steps):
        edge_index = data.edge_index.detach().cpu().numpy()
        n = data.features.size(0)
        adjacency = sp.csr_matrix((np.ones(edge_index.shape[1], dtype=np.float32), (edge_index[0], edge_index[1])), shape=(n, n))
        adjacency.data[:] = 1.0
        degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
        inverse = np.divide(1.0, degrees, out=np.zeros_like(degrees), where=degrees > 0)
        self.transition_t = (sp.diags(inverse) @ adjacency).transpose().tocsr()
        self.n = n
        self.topk = topk
        self.alpha = alpha
        self.steps = steps
        self.cache = {}

    def get(self, center):
        center = int(center)
        if center not in self.cache:
            restart = np.zeros(self.n, dtype=np.float32)
            restart[center] = 1
            probability = restart.copy()
            for _ in range(self.steps):
                probability = self.alpha * restart + (1 - self.alpha) * (self.transition_t @ probability)
            probability[center] = -1
            k = min(self.topk, self.n - 1)
            neighbors = np.argpartition(probability, -k)[-k:]
            neighbors = neighbors[np.argsort(-probability[neighbors])]
            self.cache[center] = np.concatenate(([center], neighbors)).astype(np.int64)
        return self.cache[center]


class NAQSupportPrototypeAttention(nn.Module):
    def __init__(self, args, config, data, sampler):
        super().__init__()
        self.args = args
        self.data = data
        self.sampler = sampler
        self.encoder = GNNEncoder(config)
        self.attention = nn.MultiheadAttention(embed_dim=args.latent, num_heads=args.attn_heads, dropout=args.attn_dropout, batch_first=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

    def adjusted_distances(self, h, query_ids, support_ids, routing, diagnostics=False):
        """Change ONLY the prototype of the support routed to each query.

        All non-routed support prototypes remain their original GCN embedding.
        The routing indices are pseudo-labels in training and feature-nearest
        supports at evaluation; test labels are never used to determine routing.
        """
        device = h.device
        query_ids = torch.as_tensor(query_ids, dtype=torch.long, device=device)
        support_ids = torch.as_tensor(support_ids, dtype=torch.long, device=device)
        routing = torch.as_tensor(routing, dtype=torch.long, device=device)
        queries = h[query_ids]
        supports = h[support_ids]
        baseline_distances = ((queries[:, None, :] - supports[None, :, :]) ** 2).sum(-1)
        distances = baseline_distances.clone()
        statistics = []
        for group in torch.unique(routing).tolist():
            selection = (routing == group).nonzero(as_tuple=True)[0]
            context_ids = torch.as_tensor(self.sampler.get(int(support_ids[group].item())), dtype=torch.long, device=device)
            query_embeddings = h[query_ids[selection]]
            context_embeddings = h[context_ids]
            output, weights = self.attention(
                query_embeddings.unsqueeze(0),
                context_embeddings.unsqueeze(0),
                context_embeddings.unsqueeze(0),
                need_weights=diagnostics,
                average_attn_weights=True,
            )
            delta = self.args.attn_lambda * output.squeeze(0)
            original_prototype = supports[group].unsqueeze(0).expand_as(delta)
            updated_prototype = original_prototype + delta
            distances[selection, group] = ((queries[selection] - updated_prototype) ** 2).sum(-1)
            if diagnostics:
                with torch.no_grad():
                    w = weights.squeeze(0).clamp_min(1e-12)
                    entropy = -(w * w.log()).sum(-1)
                    a = F.normalize(output.squeeze(0), dim=-1)
                    if a.size(0) > 1:
                        pairwise = a @ a.T
                        mask = ~torch.eye(a.size(0), dtype=torch.bool, device=device)
                        pair_cos = float(pairwise[mask].mean())
                    else:
                        pair_cos = None
                    statistics.append(
                        {
                            "count": int(selection.numel()),
                            "prototype_cos": float(F.cosine_similarity(original_prototype, updated_prototype).mean()),
                            "delta_ratio": float((delta.norm(dim=-1) / original_prototype.norm(dim=-1).clamp_min(1e-8)).mean()),
                            "delta_norm": float(delta.norm(dim=-1).mean()),
                            "prototype_norm": float(original_prototype.norm(dim=-1).mean()),
                            "output_cos_prototype": float(F.cosine_similarity(output.squeeze(0), original_prototype).mean()),
                            "entropy": float(entropy.mean()),
                            "normalized_entropy": float((entropy / np.log(w.size(-1))).mean()),
                            "top1_attn": float(w.max(dim=-1).values.mean()),
                            "center_attn": float(w[:, 0].mean()),
                            "effective_neighbors": float(entropy.exp().mean()),
                            "within_support_output_cos": pair_cos,
                            "distance_shift": float((distances[selection, group] - baseline_distances[selection, group]).mean()),
                            "improvement_fraction": float((distances[selection, group] < baseline_distances[selection, group]).float().mean()),
                        }
                    )
        return distances, baseline_distances, self.metrics(statistics)

    @staticmethod
    def metrics(statistics):
        if not statistics:
            return {}
        total = sum(r["count"] for r in statistics)
        result = {"num_queries": total}
        for key in statistics[0]:
            if key == "count":
                continue
            selected = [r for r in statistics if r[key] is not None]
            result[key] = float(sum(r["count"] * r[key] for r in selected) / sum(r["count"] for r in selected)) if selected else None
        return result

    @staticmethod
    def remap(y_support, y_query):
        labels = y_support.detach().cpu().tolist()
        return torch.tensor([labels.index(int(label)) for label in y_query.detach().cpu().tolist()], device=y_query.device, dtype=torch.long)

    @staticmethod
    def margin(distances, target):
        target = target.to(device=distances.device, dtype=torch.long)
        correct = distances.gather(1, target[:, None]).squeeze(1)
        others = distances.scatter(1, target[:, None], float("inf")).min(1).values
        return float((others - correct).mean())

    def train_episode(self, id_spt, y_spt, id_query, y_query, diagnostics=False):
        self.train()
        h = self.encoder(self.data.features, adj=self.data.adj)
        target = self.remap(y_spt, y_query)
        distances, baseline, stats = self.adjusted_distances(h, id_query, id_spt, target, diagnostics)
        # Explicit pairwise contrastive loss over Euclidean distances:
        # positives are the NAQ pseudo-paired supports, negatives are the
        # other N-1 supports in the same episode. Unlike distance-softmax CE,
        # this uses independent positive pull and margin-based negative push.
        euclidean = distances.clamp_min(1e-12).sqrt()
        positive = euclidean.gather(1, target[:, None]).squeeze(1)
        negative_mask = torch.ones_like(distances, dtype=torch.bool)
        negative_mask.scatter_(1, target[:, None], False)
        negative = euclidean[negative_mask].reshape(euclidean.size(0), -1)
        positive_loss = positive.square().mean()
        negative_loss = F.relu(self.args.cl_margin - negative).square().mean()
        loss = self.args.cl_pos_weight * positive_loss + self.args.cl_neg_weight * negative_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        accuracy = float((distances.argmin(1) == target).float().mean())
        if diagnostics:
            stats["raw_proto_accuracy"] = float((baseline.argmin(1) == target).float().mean())
            stats["raw_margin"] = self.margin(baseline, target)
            stats["adjusted_margin"] = self.margin(distances, target)
            stats["positive_euclidean"] = float(positive.mean())
            stats["negative_euclidean"] = float(negative.mean())
            stats["positive_loss"] = float(positive_loss)
            stats["negative_loss"] = float(negative_loss)
            stats["margin_violation_fraction"] = float((negative < self.args.cl_margin).float().mean())
        return float(loss), accuracy, stats

    def evaluate_pools(self, tasks):
        self.eval()
        with torch.no_grad():
            h = self.encoder(self.data.features, adj=self.data.adj)
            feature_norm = F.normalize(self.data.features, dim=-1)
            results = {"raw_proto": [], "attn_proto": []}
            statistics = []
            routing_correct = 0
            routing_total = 0
            raw_margins = []
            adjusted_margins = []
            for id_spt, y_spt, id_query, y_query in tasks:
                y_s = y_spt.detach().cpu().numpy()
                y_q = y_query.detach().cpu().numpy()
                # Labels are only used for evaluation and diagnostics.
                routing = (feature_norm[id_query] @ feature_norm[id_spt].T).argmax(1)
                predicted_label = y_s[routing.detach().cpu().numpy()]
                routing_correct += int((predicted_label == y_q).sum())
                routing_total += len(y_q)
                distances, baseline, stats = self.adjusted_distances(h, id_query, id_spt, routing, diagnostics=True)
                true_target = self.remap(y_spt, y_query).to(h.device)
                raw_margins.append(self.margin(baseline, true_target))
                adjusted_margins.append(self.margin(distances, true_target))
                results["raw_proto"].append(float((baseline.argmin(1) == true_target).float().mean()))
                results["attn_proto"].append(float((distances.argmin(1) == true_target).float().mean()))
                statistics.append(stats)
        # Weight task diagnostics by number of queries.
        count = sum(r["num_queries"] for r in statistics)
        keys = [k for k in statistics[0] if k != "num_queries"]
        diagnostics = {
            k: (
                float(
                    sum(r["num_queries"] * r[k] for r in statistics if r[k] is not None)
                    / sum(r["num_queries"] for r in statistics if r[k] is not None)
                )
                if any(r[k] is not None for r in statistics)
                else None
            )
            for k in keys
        }
        diagnostics.update(
            {
                "num_queries": count,
                "routing_label_accuracy": float(routing_correct / max(1, routing_total)),
                "raw_margin": float(np.mean(raw_margins)),
                "adjusted_margin": float(np.mean(adjusted_margins)),
            }
        )
        return results, diagnostics


def parse_args():
    parser = argparse.ArgumentParser(
        description="exp007: NAQ-Feat + corresponding support PPR GCN EMBEDDING attention with Euclidean contrastive loss"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="Amazon_clothing")
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--k_shot_test", type=int, default=1)
    parser.add_argument("--q_query", type=int, default=10)
    parser.add_argument("--q_query_test", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent", type=int, default=256)
    parser.add_argument("--num_valid_tasks", type=int, default=50)
    parser.add_argument("--num_downstream_tasks", type=int, default=500)
    parser.add_argument("--query_generation", type=str, default="NaQ")
    parser.add_argument("--type", type=str, default="feature")
    parser.add_argument("--PPR_alpha", type=float, default=0.1)
    parser.add_argument("--ppr_topk", type=int, default=16)
    parser.add_argument("--ppr_steps", type=int, default=20)
    parser.add_argument("--attn_lambda", type=float, default=0.1)
    parser.add_argument("--cl_margin", type=float, default=2.0)
    parser.add_argument("--cl_pos_weight", type=float, default=1.0)
    parser.add_argument("--cl_neg_weight", type=float, default=1.0)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--attn_dropout", type=float, default=0)
    parser.add_argument("--eval_interval", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--json_path", type=str, default="exp007.json")
    return parser.parse_args()


def summarize(values):
    array = np.asarray(values)
    return float(array.mean()), float(1.96 * array.std() / np.sqrt(len(array)))


def main():
    args = parse_args()
    if args.n_way < 2:
        raise ValueError("exp007 requires at least two classes")
    if args.k_shot != 1 or args.k_shot_test != 1:
        raise ValueError("exp007 currently supports 1-shot training and evaluation only")
    if args.latent % args.attn_heads:
        raise ValueError("--latent must be divisible by --attn_heads")
    if args.cl_margin <= 0 or args.cl_pos_weight < 0 or args.cl_neg_weight < 0:
        raise ValueError("contrastive margin must be positive, and weights nonnegative")
    if args.type != "feature":
        raise ValueError("exp007 currently requires --type feature for NAQ-Feat")
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
    sampler = PPRNeighborhoods(data, args.ppr_topk, args.PPR_alpha, args.ppr_steps)
    config = [("GCN", (data.features.shape[1], 2 * args.latent), True), ("GCN", (2 * args.latent, args.latent), True)]
    model = NAQSupportPrototypeAttention(args, config, data, sampler).to(device)
    train_pool = [proto_task_generator(args, data) for _ in range(args.epoch)]
    valid_pool, test_pool = downstream_task_generator(args, data, args.n_way, args.k_shot_test)
    directory = f"results/{args.dataset}/exp007"
    os.makedirs(directory, exist_ok=True)
    best = None
    start = time.time()
    print(
        f"exp007 | GCN Q/K/V | Euclidean contrastive margin={args.cl_margin} | "
        f"Support prototype residual | PPR topk={args.ppr_topk} (+ center) | "
        f"lambda={args.attn_lambda}",
        flush=True,
    )
    history = {
        "experiment": "exp007",
        "description": "GCN-embedding PPR attention, query-conditioned support prototype residual, Euclidean pairwise contrastive loss",
        "config": vars(args).copy(),
        "evaluations": [],
        "best": None,
    }

    def save_json():
        path = args.json_path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(tmp, path)

    save_json()
    for epoch, (id_spt, y_spt, id_query, y_query) in tqdm(enumerate(train_pool, 1), total=len(train_pool), desc="exp007"):
        diagnostics = epoch % args.eval_interval == 0
        loss, train_acc, training_stats = model.train_episode(id_spt, y_spt.to(device), id_query, y_query.to(device), diagnostics)
        if diagnostics:
            valid, val_stats = model.evaluate_pools(valid_pool)
            test, test_stats = model.evaluate_pools(test_pool)
            valid_proto, _ = summarize(valid["raw_proto"])
            valid_attn, _ = summarize(valid["attn_proto"])
            test_proto, test_proto_ci = summarize(test["raw_proto"])
            test_attn, test_attn_ci = summarize(test["attn_proto"])
            print(
                f"\n[Epoch {epoch}] loss={loss:.4f} train={train_acc:.4f} "
                f"val_proto={valid_proto:.4f} val_attn={valid_attn:.4f} "
                f"test_proto={test_proto:.4f}±{test_proto_ci:.4f} "
                f"test_attn={test_attn:.4f}±{test_attn_ci:.4f}",
                flush=True,
            )
            print(f"  train diagnostics: {training_stats}", flush=True)
            print(f"  valid diagnostics: {val_stats}", flush=True)
            print(f"  test diagnostics: {test_stats}", flush=True)
            print(f"  PPR cache entries: {len(sampler.cache)}", flush=True)
            entry = {
                "epoch": epoch,
                "train": {"loss": float(loss), "accuracy": float(train_acc)},
                "valid": {"raw_proto_accuracy": valid_proto, "attn_proto_accuracy": valid_attn, "diagnostics": val_stats},
                "test": {
                    "raw_proto_accuracy": test_proto,
                    "raw_proto_ci95_half_width": test_proto_ci,
                    "attn_proto_accuracy": test_attn,
                    "attn_proto_ci95_half_width": test_attn_ci,
                    "diagnostics": test_stats,
                },
                "train_diagnostics": training_stats,
                "ppr_cache_entries": len(sampler.cache),
                "elapsed_seconds": round(time.time() - start, 2),
            }
            history["evaluations"].append(entry)
            if best is None or valid_attn > best[1]:
                best = (epoch, valid_attn, test_attn, test_attn_ci, valid_proto, test_proto)
                history["best"] = {
                    "epoch": epoch,
                    "valid_attn_proto": valid_attn,
                    "test_attn_proto": test_attn,
                    "test_attn_proto_ci95_half_width": test_attn_ci,
                    "valid_raw_proto": valid_proto,
                    "test_raw_proto": test_proto,
                }
                torch.save({"epoch": epoch, "model": model.state_dict(), "args": vars(args)}, f"{directory}/best.pt")
            save_json()
    if best is not None:
        ep, val, tst, ci, val_proto, test_proto = best
        summary = (
            f"Best epoch (attn prototype validation): {ep} | Valid attn: {val:.4f} | "
            f"Test attn: {tst:.4f} ± {ci:.4f} | "
            f"Valid raw prototype: {val_proto:.4f} | Test raw prototype: {test_proto:.4f} | "
            f"Elapsed: {time.time() - start:.1f}s"
        )
        print(summary, flush=True)
        history["elapsed_seconds"] = round(time.time() - start, 2)
        save_json()
        with open(f"{directory}/result.txt", "w") as file:
            file.write(summary + "\n")


if __name__ == "__main__":
    main()
