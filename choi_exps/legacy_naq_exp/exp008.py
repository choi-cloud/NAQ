"""exp008: NAQ-Feat GCN baseline with Euclidean pairwise contrastive loss.

No attention, PPR, routing, classifier training, or logistic regression.
Train on NAQ pseudo-episodes and classify held-out episodes by nearest
support embedding, using the same Euclidean distance in both stages.
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
from model import GNNEncoder
from task_generator import downstream_task_generator, proto_task_generator
from utils import seed_everything


class NAQEuclideanCL(nn.Module):
    def __init__(self, args, config, data):
        super().__init__()
        self.args = args
        self.data = data
        self.encoder = GNNEncoder(config)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

    @staticmethod
    def remap(y_support, y_query, device):
        classes = y_support.detach().cpu().tolist()
        return torch.tensor([classes.index(int(y)) for y in y_query.detach().cpu().tolist()], device=device, dtype=torch.long)

    @staticmethod
    def distances(h, support_ids, query_ids, k_shot):
        support_ids = torch.as_tensor(support_ids, dtype=torch.long, device=h.device)
        query_ids = torch.as_tensor(query_ids, dtype=torch.long, device=h.device)
        supports = h[support_ids]
        queries = h[query_ids]
        prototypes = supports.reshape(-1, k_shot, h.size(-1)).mean(dim=1)
        return torch.cdist(queries.unsqueeze(0), prototypes.unsqueeze(0), p=2).squeeze(0), queries, prototypes

    @staticmethod
    def diagnostics(distances, target, queries, prototypes, margin):
        positive = distances.gather(1, target[:, None]).squeeze(1)
        negative_mask = torch.ones_like(distances, dtype=torch.bool)
        negative_mask.scatter_(1, target[:, None], False)
        negatives = distances[negative_mask].reshape(len(target), -1)
        nearest_negative = negatives.min(dim=1).values
        return {
            "positive_euclidean": float(positive.mean()),
            "negative_euclidean": float(negatives.mean()),
            "positive_loss": float(positive.square().mean()),
            "negative_loss": float(F.relu(margin - negatives).square().mean()),
            "margin_violation_fraction": float((negatives < margin).float().mean()),
            "classification_margin": float((nearest_negative - positive).mean()),
            "positive_closer_fraction": float((positive < nearest_negative).float().mean()),
            "query_norm": float(queries.norm(dim=-1).mean()),
            "prototype_norm": float(prototypes.norm(dim=-1).mean()),
        }

    def train_episode(self, id_spt, y_spt, id_query, y_query, do_diagnostics):
        self.train()
        h = self.encoder(self.data.features, adj=self.data.adj)
        target = self.remap(y_spt, y_query, h.device)
        distances, queries, prototypes = self.distances(h, id_spt, id_query, self.args.k_shot)
        positive = distances.gather(1, target[:, None]).squeeze(1)
        negative_mask = torch.ones_like(distances, dtype=torch.bool)
        negative_mask.scatter_(1, target[:, None], False)
        negatives = distances[negative_mask].reshape(len(target), -1)
        pos_loss = positive.square().mean()
        neg_loss = F.relu(self.args.cl_margin - negatives).square().mean()
        loss = self.args.cl_pos_weight * pos_loss + self.args.cl_neg_weight * neg_loss
        accuracy = float((distances.argmin(1) == target).float().mean())
        stats = self.diagnostics(distances, target, queries, prototypes, self.args.cl_margin) if do_diagnostics else {}
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach()), accuracy, stats

    @torch.no_grad()
    def evaluate_pools(self, tasks):
        self.eval()
        h = self.encoder(self.data.features, adj=self.data.adj)
        accuracies = []
        total_queries = 0
        diagnostic_sums = {}
        for id_spt, y_spt, id_query, y_query in tasks:
            target = self.remap(y_spt, y_query, h.device)
            distances, queries, prototypes = self.distances(h, id_spt, id_query, self.args.k_shot_test)
            accuracies.append(float((distances.argmin(1) == target).float().mean()))
            stats = self.diagnostics(distances, target, queries, prototypes, self.args.cl_margin)
            n = len(target)
            total_queries += n
            for key, value in stats.items():
                diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + n * value
        aggregated = {k: v / total_queries for k, v in diagnostic_sums.items()}
        aggregated["num_queries"] = total_queries
        return accuracies, aggregated


def parse_args():
    parser = argparse.ArgumentParser(description="exp008: NAQ-Feat, Euclidean contrastive learning, nearest-prototype evaluation")
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
    parser.add_argument("--num_valid_tasks", type=int, default=50)
    parser.add_argument("--num_downstream_tasks", type=int, default=500)
    parser.add_argument("--query_generation", default="NaQ")
    parser.add_argument("--type", default="feature")
    parser.add_argument("--cl_margin", type=float, default=2.0)
    parser.add_argument("--cl_pos_weight", type=float, default=1.0)
    parser.add_argument("--cl_neg_weight", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--json_path", default="exp008.json")
    return parser.parse_args()


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(1.96 * array.std() / np.sqrt(len(array)))


def main():
    args = parse_args()
    if args.n_way < 2 or args.k_shot != 1 or args.k_shot_test != 1:
        raise ValueError("exp008 supports N-way (N>=2), 1-shot training and evaluation")
    if args.type != "feature" or args.query_generation != "NaQ":
        raise ValueError("exp008 is the NAQ-Feat setting")
    if args.cl_margin <= 0 or args.cl_pos_weight < 0 or args.cl_neg_weight < 0:
        raise ValueError("invalid contrastive margin or weights")
    if args.eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
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
    model = NAQEuclideanCL(args, config, data).to(device)
    train_pool = [proto_task_generator(args, data) for _ in range(args.epoch)]
    valid_pool, test_pool = downstream_task_generator(args, data, args.n_way, args.k_shot_test)
    result_dir = f"results/{args.dataset}/exp008"
    os.makedirs(result_dir, exist_ok=True)
    history = {
        "experiment": "exp008",
        "description": "NAQ-Feat GCN; Euclidean positive-pull negative-margin CL; nearest support evaluation; no attention/PPR/LR",
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

    start = time.time()
    save_json()
    print("exp008 | NAQ-Feat | Euclidean CL | nearest prototype | no attention/PPR/LR", flush=True)
    for epoch, (id_spt, y_spt, id_query, y_query) in tqdm(enumerate(train_pool, 1), total=len(train_pool), desc="exp008"):
        do_diagnostics = epoch % args.eval_interval == 0
        loss, train_acc, train_stats = model.train_episode(id_spt, y_spt, id_query, y_query, do_diagnostics)
        if not do_diagnostics:
            continue
        valid_accuracies, valid_stats = model.evaluate_pools(valid_pool)
        test_accuracies, test_stats = model.evaluate_pools(test_pool)
        valid_acc, valid_ci = summarize(valid_accuracies)
        test_acc, test_ci = summarize(test_accuracies)
        record = {
            "epoch": epoch,
            "train": {"loss": loss, "accuracy": train_acc, "diagnostics": train_stats},
            "valid": {"accuracy": valid_acc, "ci95_half_width": valid_ci, "diagnostics": valid_stats},
            "test": {"accuracy": test_acc, "ci95_half_width": test_ci, "diagnostics": test_stats},
            "elapsed_seconds": round(time.time() - start, 2),
        }
        history["evaluations"].append(record)
        if history["best"] is None or valid_acc > history["best"]["valid_accuracy"]:
            history["best"] = {"epoch": epoch, "valid_accuracy": valid_acc, "test_accuracy": test_acc, "test_ci95_half_width": test_ci}
            torch.save({"epoch": epoch, "model": model.state_dict(), "args": vars(args)}, f"{result_dir}/best.pt")
        save_json()
        print(f"\n[Epoch {epoch}] loss={loss:.4f} train={train_acc:.4f} " f"val={valid_acc:.4f} test={test_acc:.4f}±{test_ci:.4f}", flush=True)
        print(f"  train diagnostics: {train_stats}", flush=True)
        print(f"  valid diagnostics: {valid_stats}", flush=True)
        print(f"  test diagnostics: {test_stats}", flush=True)
    history["elapsed_seconds"] = round(time.time() - start, 2)
    save_json()
    best = history["best"]
    if best is not None:
        result = (
            f"Best epoch (validation): {best['epoch']} | "
            f"Valid: {best['valid_accuracy']:.4f} | "
            f"Test: {best['test_accuracy']:.4f} ± {best['test_ci95_half_width']:.4f} | "
            f"Elapsed: {history['elapsed_seconds']:.1f}s"
        )
        print(result, flush=True)
        with open(f"{result_dir}/result.txt", "w", encoding="utf-8") as fp:
            fp.write(result + "\n")


if __name__ == "__main__":
    main()
