"""exp019: exp017 raw-feature anchor tasks + query-conditioned support PPR.

Training and evaluation share support-centered PPR context union, projection-free
scaled dot-product attention, prototype residual, and squared Euclidean ProtoNet CE.
Raw features are used only to construct unlabeled training pseudo-episodes.
"""

import argparse
import json
import os
import random

import scipy.sparse as sp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data import load_data
from model import GNNEncoder
from task_generator import downstream_task_generator
from utils import seed_everything


class RawFeatureAnchorSampler:
    def __init__(self, features, train_ids, n_way, k_shot, q_query, pool_size):
        self.device = features.device
        self.train_ids = torch.as_tensor(np.asarray(train_ids), dtype=torch.long, device=self.device).unique()
        self.features = F.normalize(features.detach(), p=2, dim=-1)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.pool_size = pool_size
        self.samples_per_way = k_shot + q_query
        if len(self.train_ids) < n_way * (1 + self.samples_per_way):
            raise ValueError("Insufficient training nodes for disjoint anchors and support/query nodes")

    @torch.no_grad()
    def sample(self):
        # All sampled nodes are restricted to the unsupervised training index set.
        perm = torch.randperm(len(self.train_ids), device=self.device)
        anchors = self.train_ids[perm[: self.n_way]]
        remaining = self.train_ids[perm[self.n_way :]]
        required = self.n_way * self.samples_per_way
        pool_len = min(self.pool_size, len(remaining))
        if pool_len < required:
            pool_len = len(remaining)
        candidate_nodes = remaining[:pool_len]
        similarity = self.features[anchors] @ self.features[candidate_nodes].T
        # Anchor order is randomized by the initial permutation.
        rankings = similarity.argsort(dim=1, descending=True)
        available = torch.ones(pool_len, device=self.device, dtype=torch.bool)
        groups = []
        for i in range(self.n_way):
            options = rankings[i][available[rankings[i]]]
            chosen = options[: self.samples_per_way]
            if len(chosen) != self.samples_per_way:
                raise RuntimeError("Not enough distinct candidates to construct an episode")
            available[chosen] = False
            groups.append(candidate_nodes[chosen])
        groups = torch.stack(groups, dim=0)
        supports = groups[:, : self.k_shot].reshape(-1)
        queries = groups[:, self.k_shot :].reshape(-1)
        query_labels = torch.arange(self.n_way, device=self.device).repeat_interleave(self.q_query)
        support_labels = torch.arange(self.n_way, device=self.device).repeat_interleave(self.k_shot)
        return supports, support_labels, queries, query_labels


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
            restart[center] = 1.0
            probability = restart.copy()
            for _ in range(self.steps):
                probability = self.alpha * restart + (1 - self.alpha) * (self.transition_t @ probability)
            probability[center] = -1.0
            k = min(self.topk, self.n - 1)
            neighbors = np.argpartition(probability, -k)[-k:]
            neighbors = neighbors[np.argsort(-probability[neighbors])]
            self.cache[center] = np.concatenate(([center], neighbors)).astype(np.int64)
        return self.cache[center]


class AnchorProtoNet(nn.Module):
    def __init__(self, args, config, data, neighborhood_sampler):
        super().__init__()
        self.args = args
        self.data = data
        self.encoder = GNNEncoder(config)
        self.neighborhood_sampler = neighborhood_sampler
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

    @staticmethod
    def remap(y_support, y_query, device):
        classes = torch.as_tensor(y_support).cpu().tolist()
        return torch.tensor([classes.index(int(y)) for y in torch.as_tensor(y_query).cpu().tolist()], dtype=torch.long, device=device)

    def distances(self, h, support_ids, query_ids, k_shot, diagnostics=False):
        support_ids = torch.as_tensor(support_ids, dtype=torch.long, device=h.device)
        query_ids = torch.as_tensor(query_ids, dtype=torch.long, device=h.device)
        grouped_supports = support_ids.reshape(-1, k_shot)
        prototypes = h[grouped_supports].mean(dim=1)
        queries = h[query_ids]
        raw_squared_distances = (queries[:, None, :] - prototypes[None, :, :]).square().sum(dim=-1)
        adjusted_squared_distances = []
        attention_stats = []

        for c, support_group in enumerate(grouped_supports):
            # Union of per-support PPR contexts; deduplicate in stable order.
            context_ids = []
            seen = set()
            for support_id in support_group.tolist():
                for node_id in self.neighborhood_sampler.get(support_id):
                    node_id = int(node_id)
                    if node_id not in seen:
                        seen.add(node_id)
                        context_ids.append(node_id)
            context = h[torch.as_tensor(context_ids, device=h.device, dtype=torch.long)]

            # No W_Q/W_K/W_V/W_O parameters; differentiable through GCN embeddings.
            # logits = queries @ context.T / (h.shape[-1] ** 0.5)
            # weights = F.softmax(logits, dim=-1)
            # output = weights @ context

            # Attention temperature
            tau_attn = self.args.attn_tau

            logits = queries @ context.T / tau_attn
            weights = torch.softmax(logits, dim=-1)
            output = weights @ context

            adjusted = prototypes[c][None, :] + self.args.attn_lambda * output
            adjusted_squared_distances.append((queries - adjusted).square().sum(dim=-1))

            if diagnostics:
                with torch.no_grad():
                    safe = weights.clamp_min(1e-12)
                    entropy = -(safe * safe.log()).sum(dim=-1)
                    norm_entropy = entropy / np.log(len(context_ids)) if len(context_ids) > 1 else torch.zeros_like(entropy)
                    if len(queries) > 1:
                        pair = F.normalize(output, dim=-1) @ F.normalize(output, dim=-1).T
                        within_cos = float(pair[~torch.eye(len(queries), dtype=torch.bool, device=h.device)].mean())
                    else:
                        within_cos = None
                    delta = self.args.attn_lambda * output
                    attention_stats.append(
                        {
                            "normalized_entropy": float(norm_entropy.mean()),
                            "top1_attn": float(weights.max(dim=-1).values.mean()),
                            "center_attn": float(weights[:, 0].mean()),
                            "within_support_output_cos": within_cos,
                            "delta_norm": float(delta.norm(dim=-1).mean()),
                            "delta_ratio": float((delta.norm(dim=-1) / prototypes[c].norm().clamp_min(1e-8)).mean()),
                            "context_size": len(context_ids),
                            "logit_std": float(logits.std(dim=-1, unbiased=False).mean()),
                        }
                    )

        distances = torch.stack(adjusted_squared_distances, dim=1)
        stats = {}
        if attention_stats:
            for key in attention_stats[0]:
                vals = [r[key] for r in attention_stats if r[key] is not None]
                stats[key] = float(np.mean(vals)) if vals else None
        return distances, raw_squared_distances, queries, prototypes, stats

    def train_episode(self, supports, support_labels, queries, query_labels):
        self.train()
        h = self.encoder(self.data.features, adj=self.data.adj)
        squared_distances, raw_squared_distances, query_h, prototypes, attn_stats = self.distances(
            h, supports, queries, self.args.k_shot, diagnostics=True
        )
        logits = -squared_distances / self.args.temperature
        loss = F.cross_entropy(logits, query_labels)
        accuracy = float((logits.argmax(dim=1) == query_labels).float().mean())
        with torch.no_grad():
            positive_dist = squared_distances.gather(1, query_labels[:, None]).mean()
            negative_mask = ~F.one_hot(query_labels, num_classes=self.args.n_way).bool()
            negative_dist = squared_distances[negative_mask].mean()
            raw_acc = float((raw_squared_distances.argmin(dim=1) == query_labels).float().mean())
            stats = {
                "positive_squared_distance": float(positive_dist),
                "negative_squared_distance": float(negative_dist),
                "raw_accuracy": raw_acc,
                "loss": float(loss.detach()),
                "query_norm": float(query_h.detach().norm(dim=-1).mean()),
                "prototype_norm": float(prototypes.detach().norm(dim=-1).mean()),
                "attention": attn_stats,
            }
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach()), accuracy, stats

    @torch.no_grad()
    def evaluate_pools(self, tasks):
        self.eval()
        h = self.encoder(self.data.features, adj=self.data.adj)
        corrected_accuracies = []
        raw_accuracies = []
        for support_ids, y_support, query_ids, y_query in tasks:
            target = self.remap(y_support, y_query, h.device)
            distances, raw_distances, _, _, _ = self.distances(h, support_ids, query_ids, self.args.k_shot_test)
            corrected_accuracies.append(float((distances.argmin(dim=1) == target).float().mean()))
            raw_accuracies.append(float((raw_distances.argmin(dim=1) == target).float().mean()))
        return corrected_accuracies, raw_accuracies


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(1.96 * values.std() / np.sqrt(len(values)))


def parse_args():
    parser = argparse.ArgumentParser(description="exp019: exp017 sampling + support PPR query-aligned ProtoNet CE")
    parser.add_argument("--attn_tau", type=float, default=0.1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", default="Amazon_clothing")
    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--k_shot_test", type=int, default=1)
    parser.add_argument("--q_query", type=int, default=10)
    parser.add_argument("--q_query_test", type=int, default=8)
    parser.add_argument("--pool_size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--ppr_topk", type=int, default=16)
    parser.add_argument("--PPR_alpha", type=float, default=0.1)
    parser.add_argument("--ppr_steps", type=int, default=20)
    parser.add_argument("--attn_lambda", type=float, default=0.1)
    parser.add_argument("--num_valid_tasks", type=int, default=50)
    parser.add_argument("--num_downstream_tasks", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--json_path", default="exp019.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_way < 2 or args.k_shot < 1 or args.k_shot_test < 1:
        raise ValueError("n_way >= 2 and shots >= 1 are required")
    if args.temperature <= 0 or args.eval_interval <= 0 or args.pool_size < 1:
        raise ValueError("temperature, eval_interval, and pool_size must be positive")
    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    args.setting = "unsup"
    # Raw-feature pseudo-task sampling is unchanged; PPR is only for support contexts.
    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    config = [("GCN", (data.features.shape[1], 2 * args.latent), True), ("GCN", (2 * args.latent, args.latent), True)]
    sampler = RawFeatureAnchorSampler(data.features, data.id_train, args.n_way, args.k_shot, args.q_query, args.pool_size)
    neighborhood_sampler = PPRNeighborhoods(data, args.ppr_topk, args.PPR_alpha, args.ppr_steps)
    model = AnchorProtoNet(args, config, data, neighborhood_sampler).to(device)
    valid_pool, test_pool = downstream_task_generator(args, data, args.n_way, args.k_shot_test)
    result_dir = f"results/{args.dataset}/exp019"
    os.makedirs(result_dir, exist_ok=True)
    history = {
        "experiment": "exp019",
        "description": "Raw-feature anchor pseudo-episodes + support-union PPR context + projection-free query-conditioned residual ProtoNet",
        "task_sampling": "VISION-style random anchor, raw feature cosine, shared candidate pool without replacement",
        "objective": "ProtoNet CE on squared Euclidean distances to query-adjusted prototypes, with unchanged exp017 training sampler",
        "config": vars(args).copy(),
        "evaluations": [],
        "best": None,
        "selection_metric": "valid.accuracy (attention-corrected)",
    }

    def save_json():
        path = args.json_path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(tmp, path)

    start = time.time()
    save_json()
    print("exp019 | exp017 sampler | PPR support-union dot-product attention | train+eval corrected", flush=True)
    for epoch in tqdm(range(1, args.epoch + 1), desc="exp019"):
        supports, support_labels, queries, query_labels = sampler.sample()
        loss, train_acc, train_stats = model.train_episode(supports, support_labels, queries, query_labels)
        if epoch % args.eval_interval != 0:
            continue
        valid_corrected, valid_raw = model.evaluate_pools(valid_pool)
        test_corrected, test_raw = model.evaluate_pools(test_pool)
        valid_acc, valid_ci = summarize(valid_corrected)
        test_acc, test_ci = summarize(test_corrected)
        valid_raw_acc, valid_raw_ci = summarize(valid_raw)
        test_raw_acc, test_raw_ci = summarize(test_raw)
        history["evaluations"].append(
            {
                "epoch": epoch,
                "train": {"loss": loss, "accuracy": train_acc, "diagnostics": train_stats},
                "valid": {"accuracy": valid_acc, "ci95_half_width": valid_ci, "raw_accuracy": valid_raw_acc, "raw_ci95_half_width": valid_raw_ci},
                "test": {"accuracy": test_acc, "ci95_half_width": test_ci, "raw_accuracy": test_raw_acc, "raw_ci95_half_width": test_raw_ci},
                "elapsed_seconds": round(time.time() - start, 2),
            }
        )
        if history["best"] is None or valid_acc > history["best"]["valid_accuracy"]:
            history["best"] = {
                "epoch": epoch,
                "valid_accuracy": valid_acc,
                "test_accuracy": test_acc,
                "test_ci95_half_width": test_ci,
                "test_raw_accuracy": test_raw_acc,
            }
            torch.save({"epoch": epoch, "model": model.state_dict(), "args": vars(args)}, f"{result_dir}/best.pt")
        save_json()
        print(
            f"[Epoch {epoch}] loss={loss:.4f} train={train_acc:.4f} "
            f"val={valid_acc:.4f}±{valid_ci:.4f} (raw={valid_raw_acc:.4f}) "
            f"test={test_acc:.4f}±{test_ci:.4f} (raw={test_raw_acc:.4f})",
            flush=True,
        )
    history["elapsed_seconds"] = round(time.time() - start, 2)
    save_json()
    if history["best"] is not None:
        best = history["best"]
        result = (
            f"Best epoch: {best['epoch']} | Valid {best['valid_accuracy']:.4f} | "
            f"Test {best['test_accuracy']:.4f}±{best['test_ci95_half_width']:.4f}"
        )
        print(result, flush=True)
        with open(f"{result_dir}/result.txt", "w", encoding="utf-8") as file:
            file.write(result + "\n")


if __name__ == "__main__":
    main()
