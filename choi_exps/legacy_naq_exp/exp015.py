"""exp015: exp009 PPR query-conditioned attention with distance-based InfoNCE.

Each query independently attends to EACH candidate support neighborhood,
and adds the output to that candidate support prototype. Thus there is no
label-dependent routing during training or evaluation, no raw-feature
attention, no LR, and no attention mixing across support neighborhoods.
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


class NAQInfoNCE(nn.Module):
    def __init__(self, args, config, data, sampler):
        super().__init__()
        self.args = args
        self.data = data
        self.encoder = GNNEncoder(config)
        self.sampler = sampler
        self.attention = nn.MultiheadAttention(embed_dim=args.latent, num_heads=args.attn_heads, dropout=args.attn_dropout, batch_first=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

    @staticmethod
    def remap(y_support, y_query, device):
        classes = y_support.detach().cpu().tolist()
        return torch.tensor([classes.index(int(y)) for y in y_query.detach().cpu().tolist()], device=device, dtype=torch.long)

    def distances(self, h, support_ids, query_ids, k_shot, do_diagnostics=False):
        support_ids = torch.as_tensor(support_ids, dtype=torch.long, device=h.device)
        query_ids = torch.as_tensor(query_ids, dtype=torch.long, device=h.device)
        supports = h[support_ids].reshape(-1, k_shot, h.size(-1))
        queries = h[query_ids]
        prototypes = supports.mean(dim=1)

        raw_distances = torch.cdist(queries[None], prototypes[None], p=2).squeeze(0)

        candidate_distances = []
        stats = []

        for c in range(prototypes.size(0)):
            ids = []
            seen = set()

            for support_id in support_ids.reshape(-1, k_shot)[c].tolist():
                for node_id in self.sampler.get(support_id):
                    if int(node_id) not in seen:
                        ids.append(int(node_id))
                        seen.add(int(node_id))

            context = h[torch.as_tensor(ids, dtype=torch.long, device=h.device)]

            # Projection-free scaled dot-product attention
            logits = queries @ context.T / (h.size(-1) ** 0.5)
            weights = torch.softmax(logits, dim=-1)
            output = weights @ context

            delta = self.args.attn_lambda * output
            adjusted = prototypes[c][None, :] + delta

            candidate_distances.append((queries - adjusted).norm(dim=-1))

            if do_diagnostics:
                with torch.no_grad():
                    weight = weights.clamp_min(1e-12)
                    entropy = -(weight * weight.log()).sum(dim=-1)

                    unit_output = F.normalize(output, dim=-1)
                    pairwise = unit_output @ unit_output.T
                    mask = ~torch.eye(len(queries), dtype=torch.bool, device=h.device)

                    same_output_cos = float(pairwise[mask].mean()) if mask.any() else None

                    proto = prototypes[c][None, :].expand_as(adjusted)

                    stats.append(
                        {
                            "entropy": float(entropy.mean()),
                            "normalized_entropy": float((entropy / np.log(len(ids))).mean()) if len(ids) > 1 else 0.0,
                            "top1_attn": float(weight.max(dim=-1).values.mean()),
                            "center_attn": float(weight[:, 0].mean()),
                            "effective_neighbors": float(entropy.exp().mean()),
                            "within_support_output_cos": same_output_cos,
                            "prototype_cos": float(F.cosine_similarity(proto, adjusted).mean()),
                            "delta_ratio": float((delta.norm(dim=-1) / proto.norm(dim=-1).clamp_min(1e-8)).mean()),
                            "delta_norm": float(delta.norm(dim=-1).mean()),
                            "prototype_norm": float(proto.norm(dim=-1).mean()),
                            "distance_shift": float((candidate_distances[-1] - raw_distances[:, c]).mean()),
                            "closer_fraction": float((candidate_distances[-1] < raw_distances[:, c]).float().mean()),
                        }
                    )

        distances = torch.stack(candidate_distances, dim=1)

        diagnostics = {k: float(np.mean([r[k] for r in stats])) for k in stats[0] if k != "within_support_output_cos"} if stats else {}

        if stats:
            vals = [r["within_support_output_cos"] for r in stats if r["within_support_output_cos"] is not None]
            diagnostics["within_support_output_cos"] = float(np.mean(vals)) if vals else None
            diagnostics["num_context_classes"] = len(stats)

        return distances, raw_distances, queries, prototypes, diagnostics

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
        distances, raw, queries, prototypes, attn_stats = self.distances(h, id_spt, id_query, self.args.k_shot, do_diagnostics)
        # Each pseudo-query is matched to its pseudo-support among all N candidates.
        # Negative squared Euclidean distance preserves the nearest-prototype decision.
        logits = -distances.square() / self.args.temperature
        loss = F.cross_entropy(logits, target)
        accuracy = float((distances.argmin(1) == target).float().mean())
        stats = self.diagnostics(distances, target, queries, prototypes, self.args.cl_margin) if do_diagnostics else {}
        if do_diagnostics:
            stats.update(attn_stats)
            stats["raw_accuracy"] = float((raw.argmin(1) == target).float().mean())
            stats["infonce_loss"] = float(loss.detach())
            stats["positive_probability"] = float(F.softmax(logits.detach(), dim=1).gather(1, target[:, None]).mean())
            stats["logit_gap"] = float(
                (
                    logits.detach().gather(1, target[:, None]).squeeze(1)
                    - logits.detach().masked_fill(F.one_hot(target, logits.size(1)).bool(), float("-inf")).max(dim=1).values
                ).mean()
            )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach()), accuracy, stats

    @torch.no_grad()
    def evaluate_pools(self, tasks):
        """GNN-only Euclidean nearest-prototype evaluation (NO attention calls)."""
        self.eval()
        h = self.encoder(self.data.features, adj=self.data.adj)
        accuracies = []
        total_queries = 0
        diagnostic_sums = {}
        for id_spt, y_spt, id_query, y_query in tasks:
            target = self.remap(y_spt, y_query, h.device)
            support_ids = torch.as_tensor(id_spt, dtype=torch.long, device=h.device)
            query_ids = torch.as_tensor(id_query, dtype=torch.long, device=h.device)
            queries = h[query_ids]
            prototypes = h[support_ids].reshape(-1, self.args.k_shot_test, h.size(-1)).mean(dim=1)
            distances = torch.cdist(queries.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0)
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
    parser = argparse.ArgumentParser(description="exp015: NAQ-Feat, PPR attention, distance InfoNCE, raw GNN evaluation")
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
    parser.add_argument("--ppr_topk", type=int, default=16)
    parser.add_argument("--ppr_steps", type=int, default=20)
    parser.add_argument("--PPR_alpha", type=float, default=0.1)
    parser.add_argument("--attn_lambda", type=float, default=0.1)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    # Margin is used only for unchanged distance diagnostics, NOT for the loss.
    parser.add_argument("--cl_margin", type=float, default=2.0)
    parser.add_argument("--eval_interval", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--json_path", default="exp009.json")
    return parser.parse_args()


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(1.96 * array.std() / np.sqrt(len(array)))


def main():
    args = parse_args()
    if args.n_way < 2 or args.k_shot != 1 or args.k_shot_test != 1:
        raise ValueError("exp015 supports N-way (N>=2), 1-shot training and evaluation")
    if args.type != "feature" or args.query_generation != "NaQ":
        raise ValueError("exp015 is the NAQ-Feat setting")
    if args.temperature <= 0 or args.cl_margin <= 0:
        raise ValueError("temperature and diagnostic margin must be positive")
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
    if args.latent % args.attn_heads:
        raise ValueError("latent must be divisible by attn_heads")
    sampler = PPRNeighborhoods(data, args.ppr_topk, args.PPR_alpha, args.ppr_steps)
    model = NAQInfoNCE(args, config, data, sampler).to(device)
    train_pool = [proto_task_generator(args, data) for _ in range(args.epoch)]
    valid_pool, test_pool = downstream_task_generator(args, data, args.n_way, args.k_shot_test)
    result_dir = f"results/{args.dataset}/exp015"
    os.makedirs(result_dir, exist_ok=True)
    history = {
        "experiment": "exp015",
        "description": "NAQ-Feat, GCN-PPR per-support attention, squared-Euclidean InfoNCE; select and test using plain GNN prototypes",
        "objective": "CrossEntropy(logits=-squared_aligned_euclidean_distance/temperature, target=pseudo_class)",
        "config": vars(args).copy(),
        "evaluations": [],
        "best": None,
        "selection_metric": "valid.accuracy (GNN-only)",
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
    print("exp015 | NAQ-Feat | PPR GCN attention | squared Euclidean InfoNCE | GNN-only evaluation", flush=True)
    for epoch, (id_spt, y_spt, id_query, y_query) in tqdm(enumerate(train_pool, 1), total=len(train_pool), desc="exp015"):
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
        history["ppr_cache_entries"] = len(sampler.cache)
        save_json()
        print(
            f"\n[Epoch {epoch}] loss={loss:.4f} train={train_acc:.4f} " f"val={valid_acc:.4f} test={test_acc:.4f}±{test_ci:.4f} (GNN-only)",
            flush=True,
        )
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
