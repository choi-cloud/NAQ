"""exp022: Cross-part matching distance for anchor-sampled graph few-shot learning.

Keeps exp017's GCN, raw-feature anchor pseudo episodes and episodic protocol.
Changes only the classifier distance: query latent chunks softly match ALL
support-prototype chunks (rather than only equal-index chunks).
The SAME cross-part metric is used in training, validation and test.
No PPR, subgraph sampling, DGCF routing, projections or auxiliary losses.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data import load_data
from exp017 import RawFeatureAnchorSampler
from model import GNNEncoder
from task_generator import downstream_task_generator
from utils import seed_everything


class CrossPartMetric(nn.Module):
    def __init__(self, num_parts, attn_tau):
        super().__init__()
        self.num_parts = num_parts
        self.attn_tau = attn_tau

    def forward(self, queries, prototypes, return_stats=False):
        """queries [Q,D], prototypes [C,D] -> distances [Q,C].

        Each query part attends over ALL candidate-prototype parts.
        Key/query cosine attention is scale-stable; values are raw chunks.
        Sum of part distances preserves exp017 Euclidean logit scale.
        """
        num_queries, dim = queries.shape
        num_classes = prototypes.shape[0]
        if dim % self.num_parts != 0:
            raise ValueError("latent must be divisible by num_parts")
        part_dim = dim // self.num_parts
        q = queries.reshape(num_queries, self.num_parts, part_dim)
        s = prototypes.reshape(num_classes, self.num_parts, part_dim)

        q_unit = F.normalize(q, p=2, dim=-1, eps=1e-8)
        s_unit = F.normalize(s, p=2, dim=-1, eps=1e-8)
        logits = torch.einsum("qkd,cjd->qckj", q_unit, s_unit) / self.attn_tau
        weights = logits.softmax(dim=-1)
        aligned = torch.einsum("qckj,cjd->qckd", weights, s)
        distances = (q[:, None] - aligned).square().sum(dim=(-1, -2))

        if not return_stats:
            return distances

        with torch.no_grad():
            uniform = 1.0 / self.num_parts
            entropy = -(weights.detach().clamp_min(1e-12) * weights.detach().clamp_min(1e-12).log()).sum(-1)
            entropy = entropy / np.log(self.num_parts) if self.num_parts > 1 else torch.zeros_like(entropy)
            peak_index = weights.detach().argmax(dim=-1)
            diagonal_index = torch.arange(self.num_parts, device=weights.device)
            diagonal_weight = weights.detach().diagonal(dim1=-2, dim2=-1).mean()
            stats = {
                "normalized_entropy": float(entropy.mean()),
                "top1_weight": float(weights.detach().amax(dim=-1).mean()),
                "uniform_top1_weight": uniform,
                "diagonal_attention_mass": float(diagonal_weight),
                "diagonal_top1_rate": float((peak_index == diagonal_index).float().mean()),
                "logit_std": float(logits.detach().std(dim=-1, unbiased=False).mean()),
                "query_part_norm": float(q.detach().norm(dim=-1).mean()),
                "prototype_part_norm": float(s.detach().norm(dim=-1).mean()),
            }
        return distances, stats


class CrossPartProtoNet(nn.Module):
    def __init__(self, args, config, data):
        super().__init__()
        self.args = args
        self.data = data
        self.encoder = GNNEncoder(config)
        self.metric = CrossPartMetric(args.num_parts, args.attn_tau)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

    @staticmethod
    def remap(y_support, y_query, device):
        labels = torch.as_tensor(y_support).cpu().tolist()
        return torch.tensor([labels.index(int(value)) for value in torch.as_tensor(y_query).cpu().tolist()], dtype=torch.long, device=device)

    def calculate(self, h, supports, queries, k_shot, return_stats=False):
        supports = torch.as_tensor(supports, dtype=torch.long, device=h.device)
        queries = torch.as_tensor(queries, dtype=torch.long, device=h.device)
        prototypes = h[supports].reshape(-1, k_shot, h.shape[-1]).mean(dim=1)
        query_h = h[queries]
        if return_stats:
            matched, stats = self.metric(query_h, prototypes, return_stats=True)
        else:
            matched = self.metric(query_h, prototypes)
            stats = None
        # Direct baseline, from the SAME encoder representations.
        raw = (query_h[:, None, :] - prototypes[None, :, :]).square().sum(-1)
        return matched, raw, stats

    def train_episode(self, supports, queries, labels):
        self.train()
        h = self.encoder(self.data.features, adj=self.data.adj)
        matched, raw, attn_stats = self.calculate(h, supports, queries, self.args.k_shot, True)
        logits = -matched / self.args.temperature
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            negative = ~F.one_hot(labels, num_classes=self.args.n_way).bool()
            stats = {
                "positive_distance": float(matched.gather(1, labels[:, None]).mean()),
                "negative_distance": float(matched[negative].mean()),
                "raw_accuracy": float((raw.argmin(dim=-1) == labels).float().mean()),
                "matched_distance_mean": float(matched.mean()),
                "raw_distance_mean": float(raw.mean()),
                "attention": attn_stats,
            }
            acc = float((matched.argmin(dim=-1) == labels).float().mean())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach()), acc, stats

    @torch.no_grad()
    def evaluate(self, pool):
        self.eval()
        h = self.encoder(self.data.features, adj=self.data.adj)
        correct, raw_correct = [], []
        for supports, y_support, queries, y_query in pool:
            target = self.remap(y_support, y_query, h.device)
            matched, raw, _ = self.calculate(h, supports, queries, self.args.k_shot_test)
            correct.append(float((matched.argmin(-1) == target).float().mean()))
            raw_correct.append(float((raw.argmin(-1) == target).float().mean()))
        return correct, raw_correct


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(1.96 * values.std() / np.sqrt(len(values)))


def parse_args():
    parser = argparse.ArgumentParser(description="exp022: Cross-part matching ProtoNet")
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
    parser.add_argument("--num_parts", type=int, default=8)
    parser.add_argument("--attn_tau", type=float, default=0.1)
    parser.add_argument("--num_valid_tasks", type=int, default=50)
    parser.add_argument("--num_downstream_tasks", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--json_path", default="exp022.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_way < 2 or args.k_shot < 1 or args.k_shot_test < 1:
        raise ValueError("n_way >= 2 and shots >= 1 required")
    if args.num_parts < 2 or args.latent % args.num_parts != 0:
        raise ValueError("num_parts >= 2 and latent divisible by num_parts required")
    if args.attn_tau <= 0 or args.temperature <= 0 or args.eval_interval <= 0:
        raise ValueError("temperatures and eval_interval must be positive")
    if args.pool_size < args.n_way * (args.k_shot + args.q_query):
        raise ValueError("pool_size too small for the requested pseudo episode")

    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    args.setting = "unsup"
    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))

    config = [("GCN", (data.features.shape[1], 2 * args.latent), True), ("GCN", (2 * args.latent, args.latent), True)]
    sampler = RawFeatureAnchorSampler(data.features, data.id_train, args.n_way, args.k_shot, args.q_query, args.pool_size)
    model = CrossPartProtoNet(args, config, data).to(device)
    valid_pool, test_pool = downstream_task_generator(args, data, args.n_way, args.k_shot_test)

    result_dir = f"results/{args.dataset}/exp022"
    os.makedirs(result_dir, exist_ok=True)
    history = {
        "experiment": "exp022",
        "description": "exp017 GCN / sampler; Cross-part Attention distance at train and eval",
        "classifier": "Query K parts attend to all K prototype parts; sum squared part differences",
        "attention": "Cosine Q/K, no projection, values are raw part chunks",
        "config": vars(args).copy(),
        "evaluations": [],
        "best": None,
        "selection_metric": "valid.accuracy (cross-part)",
    }

    def save_json():
        path = args.json_path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(path + ".tmp", path)

    start = time.time()
    save_json()
    print("exp022 | exp017 raw-feature pseudo sampling | train/eval cross-part distance", flush=True)
    for epoch in tqdm(range(1, args.epoch + 1), desc="exp022"):
        supports, _, queries, labels = sampler.sample()
        loss, train_acc, diagnostics = model.train_episode(supports, queries, labels)
        if epoch % args.eval_interval != 0:
            continue
        val, raw_val = model.evaluate(valid_pool)
        test, raw_test = model.evaluate(test_pool)
        val_acc, val_ci = summarize(val)
        raw_val_acc, raw_val_ci = summarize(raw_val)
        test_acc, test_ci = summarize(test)
        raw_test_acc, raw_test_ci = summarize(raw_test)
        history["evaluations"].append(
            {
                "epoch": epoch,
                "train": {"loss": loss, "accuracy": train_acc, "diagnostics": diagnostics},
                "valid": {"accuracy": val_acc, "ci95_half_width": val_ci, "raw_accuracy": raw_val_acc, "raw_ci95_half_width": raw_val_ci},
                "test": {"accuracy": test_acc, "ci95_half_width": test_ci, "raw_accuracy": raw_test_acc, "raw_ci95_half_width": raw_test_ci},
                "elapsed_seconds": round(time.time() - start, 2),
            }
        )
        if history["best"] is None or val_acc > history["best"]["valid_accuracy"]:
            history["best"] = {
                "epoch": epoch,
                "valid_accuracy": val_acc,
                "test_accuracy": test_acc,
                "test_ci95_half_width": test_ci,
                "raw_valid_accuracy": raw_val_acc,
                "raw_test_accuracy": raw_test_acc,
            }
            torch.save({"epoch": epoch, "model": model.state_dict(), "args": vars(args)}, f"{result_dir}/best.pt")
        save_json()
        print(
            f"[Epoch {epoch}] loss={loss:.4f} train={train_acc:.4f} "
            f"val={val_acc:.4f} raw_val={raw_val_acc:.4f} "
            f"test={test_acc:.4f}±{test_ci:.4f} raw_test={raw_test_acc:.4f} "
            f"entropy={diagnostics['attention']['normalized_entropy']:.4f}",
            flush=True,
        )

    history["elapsed_seconds"] = round(time.time() - start, 2)
    save_json()
    if history["best"] is not None:
        best = history["best"]
        result = (
            f"Best epoch: {best['epoch']} | Valid {best['valid_accuracy']:.4f} | "
            f"Test {best['test_accuracy']:.4f}±{best['test_ci95_half_width']:.4f} | "
            f"Raw Test {best['raw_test_accuracy']:.4f}"
        )
        print(result, flush=True)
        with open(f"{result_dir}/result.txt", "w", encoding="utf-8") as file:
            file.write(result + "\n")


if __name__ == "__main__":
    main()
