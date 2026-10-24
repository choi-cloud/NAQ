"""exp016: raw-feature VISION-style anchor episode sampling + COLA-style support contrastive loss.

Training: disjoint random anchors; pool candidates ranked by L2-normalized raw feature
cosine. For each anchor select K+Q nodes, with no reuse among episode ways;
the anchor itself is NOT a support or query. Contrast each GCN query with all
K*N individual GCN support nodes via dot-product supervised contrastive loss.
No attention, structural context, feature smoothing, EMA, or learned projection.
Evaluation: same cached labeled validation/test episodes; GNN-only Euclidean ProtoNet.
"""

import argparse
import json
import os
import random
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


class AnchorCOLA(nn.Module):
    def __init__(self, args, config, data):
        super().__init__()
        self.args = args
        self.data = data
        self.encoder = GNNEncoder(config)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

    @staticmethod
    def remap(y_support, y_query, device):
        classes = torch.as_tensor(y_support).cpu().tolist()
        return torch.tensor([classes.index(int(y)) for y in torch.as_tensor(y_query).cpu().tolist()], dtype=torch.long, device=device)

    def train_episode(self, supports, support_labels, queries, query_labels):
        self.train()
        h = self.encoder(self.data.features, adj=self.data.adj)
        support_h = h[supports]
        query_h = h[queries]
        # COLA-style individual-support multi-positive contrastive objective.
        logits = query_h @ support_h.T / self.args.temperature
        positive_mask = query_labels[:, None] == support_labels[None, :]
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(log_probs * positive_mask.float()).sum(dim=1).div(positive_mask.sum(dim=1)).mean()
        # Training episode classification for context, not model selection.
        prototypes = support_h.reshape(self.args.n_way, self.args.k_shot, -1).mean(dim=1)
        distances = torch.cdist(query_h.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0)
        accuracy = float((distances.argmin(dim=1) == query_labels).float().mean())
        stats = {
            "positive_logit": float(logits[positive_mask].detach().mean()),
            "negative_logit": float(logits[~positive_mask].detach().mean()),
            "loss": float(loss.detach()),
            "query_norm": float(query_h.detach().norm(dim=-1).mean()),
            "support_norm": float(support_h.detach().norm(dim=-1).mean()),
        }
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach()), accuracy, stats

    @torch.no_grad()
    def evaluate_pools(self, tasks):
        self.eval()
        h = self.encoder(self.data.features, adj=self.data.adj)
        accuracies = []
        for support_ids, y_support, query_ids, y_query in tasks:
            target = self.remap(y_support, y_query, h.device)
            support_ids = torch.as_tensor(support_ids, dtype=torch.long, device=h.device)
            query_ids = torch.as_tensor(query_ids, dtype=torch.long, device=h.device)
            prototypes = h[support_ids].reshape(-1, self.args.k_shot_test, h.size(-1)).mean(dim=1)
            distances = torch.cdist(h[query_ids][None], prototypes[None]).squeeze(0)
            accuracies.append(float((distances.argmin(dim=1) == target).float().mean()))
        return accuracies


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(1.96 * values.std() / np.sqrt(len(values)))


def parse_args():
    parser = argparse.ArgumentParser(description="exp016: raw-feature anchor tasks + COLA support-level CL")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", default="Amazon_clothing")
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--k_shot_test", type=int, default=1)
    parser.add_argument("--q_query", type=int, default=10)
    parser.add_argument("--q_query_test", type=int, default=8)
    parser.add_argument("--pool_size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--num_valid_tasks", type=int, default=50)
    parser.add_argument("--num_downstream_tasks", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--json_path", default="exp016.json")
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
    # No add_diffusion(): no smoothing, random walk, or PPR is used for sampling.
    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    config = [("GCN", (data.features.shape[1], 2 * args.latent), True), ("GCN", (2 * args.latent, args.latent), True)]
    sampler = RawFeatureAnchorSampler(data.features, data.id_train, args.n_way, args.k_shot, args.q_query, args.pool_size)
    model = AnchorCOLA(args, config, data).to(device)
    valid_pool, test_pool = downstream_task_generator(args, data, args.n_way, args.k_shot_test)
    result_dir = f"results/{args.dataset}/exp016"
    os.makedirs(result_dir, exist_ok=True)
    history = {
        "experiment": "exp016",
        "description": "Raw-feature anchor-centered disjoint pseudo-episodes; COLA-style individual-support CL; GNN-only Euclidean evaluation",
        "task_sampling": "VISION-style random anchor, raw feature cosine, shared candidate pool without replacement",
        "objective": "Mean positive log-softmax over all individual supports; dot-product / temperature",
        "cola_components_not_used": ["three-view graph augmentation", "momentum encoder", "projection head"],
        "config": vars(args).copy(),
        "evaluations": [],
        "best": None,
        "selection_metric": "valid.accuracy (GNN-only)",
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
    print("exp016 | raw feature VISION-style anchor tasks | COLA-style CL | GNN-only test", flush=True)
    for epoch in tqdm(range(1, args.epoch + 1), desc="exp016"):
        supports, support_labels, queries, query_labels = sampler.sample()
        loss, train_acc, train_stats = model.train_episode(supports, support_labels, queries, query_labels)
        if epoch % args.eval_interval != 0:
            continue
        valid_acc, valid_ci = summarize(model.evaluate_pools(valid_pool))
        test_acc, test_ci = summarize(model.evaluate_pools(test_pool))
        history["evaluations"].append(
            {
                "epoch": epoch,
                "train": {"loss": loss, "accuracy": train_acc, "diagnostics": train_stats},
                "valid": {"accuracy": valid_acc, "ci95_half_width": valid_ci},
                "test": {"accuracy": test_acc, "ci95_half_width": test_ci},
                "elapsed_seconds": round(time.time() - start, 2),
            }
        )
        if history["best"] is None or valid_acc > history["best"]["valid_accuracy"]:
            history["best"] = {"epoch": epoch, "valid_accuracy": valid_acc, "test_accuracy": test_acc, "test_ci95_half_width": test_ci}
            torch.save({"epoch": epoch, "model": model.state_dict(), "args": vars(args)}, f"{result_dir}/best.pt")
        save_json()
        print(
            f"[Epoch {epoch}] loss={loss:.4f} train={train_acc:.4f} " f"val={valid_acc:.4f}±{valid_ci:.4f} test={test_acc:.4f}±{test_ci:.4f}",
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
