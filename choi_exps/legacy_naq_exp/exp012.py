"""exp012: NAQ-Feat GCN + support-set query-conditioned attention.

No PPR or neighborhood sampling. Each query attends the current episode's
N support prototypes. Each query-conditioned attention output is added to
candidate support prototypes before Euclidean nearest-prototype prediction.

The training objective retains exp008's inter-class Euclidean contrastive
loss and adds within-pseudo-class instance discrimination: query i is
positive only to its own aligned prototype p_{y_i}(q_i); for the same
pseudo-support, aligned prototypes constructed using peer queries are
negatives. This auxiliary term is train-only because query labels are
unknown at test time. No labels are used in evaluation attention/routing.
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


class NAQIntraAlignedCL(nn.Module):
    def __init__(self, args, config, data):
        super().__init__()
        self.args = args
        self.data = data
        self.encoder = GNNEncoder(config)
        self.attention = nn.MultiheadAttention(embed_dim=args.latent, num_heads=args.attn_heads, dropout=args.attn_dropout, batch_first=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

    @staticmethod
    def remap(y_support, y_query, device):
        classes = y_support.detach().cpu().tolist()
        return torch.tensor([classes.index(int(y)) for y in y_query.detach().cpu().tolist()], device=device, dtype=torch.long)

    def aligned_prototypes(self, h, support_ids, query_ids, k_shot, diagnostics=False):
        support_ids = torch.as_tensor(support_ids, dtype=torch.long, device=h.device)
        query_ids = torch.as_tensor(query_ids, dtype=torch.long, device=h.device)
        queries = h[query_ids]
        prototypes = h[support_ids].reshape(-1, k_shot, h.size(-1)).mean(dim=1)
        # Attention is query-dependent and episode-conditioned, with no graph sampler.
        output, weights = self.attention(queries.unsqueeze(0), prototypes.unsqueeze(0), prototypes.unsqueeze(0), need_weights=diagnostics)
        output = output.squeeze(0)
        delta = self.args.attn_lambda * output
        aligned = prototypes.unsqueeze(0) + delta.unsqueeze(1)  # [Q, N, D]
        distances = (queries.unsqueeze(1) - aligned).norm(dim=-1)
        raw_distances = torch.cdist(queries.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0)
        stats = {}
        if diagnostics:
            with torch.no_grad():
                w = weights.squeeze(0).clamp_min(1e-12)
                entropy = -(w * w.log()).sum(dim=1)
                norms = prototypes.norm(dim=1).mean().clamp_min(1e-8)
                output_unit = F.normalize(output, dim=-1)
                output_cos = output_unit @ output_unit.T
                n = queries.size(0)
                mask = ~torch.eye(n, dtype=torch.bool, device=h.device)
                stats = {
                    "entropy": float(entropy.mean()),
                    "normalized_entropy": float((entropy / np.log(prototypes.size(0))).mean()),
                    "top1_attn": float(w.max(dim=1).values.mean()),
                    "delta_norm": float(delta.norm(dim=1).mean()),
                    "delta_ratio": float(delta.norm(dim=1).mean() / norms),
                    "output_pair_cos": float(output_cos[mask].mean()) if mask.any() else None,
                    "raw_accuracy_available": 1,
                    "num_attention_keys": int(prototypes.size(0)),
                }
        return distances, raw_distances, queries, prototypes, aligned, stats

    @staticmethod
    def inter_loss(distances, target, margin, pos_weight, neg_weight):
        positive = distances.gather(1, target[:, None]).squeeze(1)
        mask = torch.ones_like(distances, dtype=torch.bool)
        mask.scatter_(1, target[:, None], False)
        negatives = distances[mask].reshape(len(target), -1)
        pos_loss = positive.square().mean()
        neg_loss = F.relu(margin - negatives).square().mean()
        return pos_weight * pos_loss + neg_weight * neg_loss, pos_loss, neg_loss

    def intra_loss(self, queries, aligned, target):
        # For a fixed pseudo-support c, compare q_i to p_c(q_j), where
        # i,j range only over pseudo-queries assigned to c.
        positive_parts = []
        negative_parts = []
        diagonal_acc = []
        groups = 0
        for c in range(aligned.size(1)):
            indices = (target == c).nonzero(as_tuple=True)[0]
            if indices.numel() < 2:
                continue
            groups += 1
            q = queries[indices]
            aligned_c = aligned[indices, c]
            pair_dist = torch.cdist(q.unsqueeze(0), aligned_c.unsqueeze(0)).squeeze(0)
            positive_parts.append(pair_dist.diagonal())
            mask = ~torch.eye(len(indices), dtype=torch.bool, device=q.device)
            neg = pair_dist[mask].reshape(len(indices), -1)
            negative_parts.append(neg.reshape(-1))
            diagonal_acc.append((pair_dist.argmin(1) == torch.arange(len(indices), device=q.device)).float())
        if not groups:
            zero = queries.sum() * 0.0
            return zero, {"intra_num_groups": 0, "intra_num_positives": 0, "intra_num_negatives": 0}
        pos = torch.cat(positive_parts)
        neg = torch.cat(negative_parts)
        pos_loss = pos.square().mean()
        neg_loss = F.relu(self.args.intra_margin - neg).square().mean()
        loss = self.args.intra_pos_weight * pos_loss + self.args.intra_neg_weight * neg_loss
        stats = {
            "intra_positive_distance": float(pos.detach().mean()),
            "intra_negative_distance": float(neg.detach().mean()),
            "intra_positive_loss": float(pos_loss.detach()),
            "intra_negative_loss": float(neg_loss.detach()),
            "intra_margin_violation": float((neg.detach() < self.args.intra_margin).float().mean()),
            "intra_retrieval_accuracy": float(torch.cat(diagonal_acc).mean()),
            "intra_num_groups": groups,
            "intra_num_positives": int(pos.numel()),
            "intra_num_negatives": int(neg.numel()),
        }
        return loss, stats

    @staticmethod
    def diagnostics(distances, target, queries, prototypes, margin):
        pos = distances.gather(1, target[:, None]).squeeze(1)
        mask = torch.ones_like(distances, dtype=torch.bool)
        mask.scatter_(1, target[:, None], False)
        neg = distances[mask].reshape(len(target), -1)
        return {
            "positive_euclidean": float(pos.mean()),
            "negative_euclidean": float(neg.mean()),
            "positive_loss": float(pos.square().mean()),
            "negative_loss": float(F.relu(margin - neg).square().mean()),
            "margin_violation_fraction": float((neg < margin).float().mean()),
            "classification_margin": float((neg.min(dim=1).values - pos).mean()),
            "positive_closer_fraction": float((pos < neg.min(dim=1).values).float().mean()),
            "query_norm": float(queries.norm(dim=-1).mean()),
            "prototype_norm": float(prototypes.norm(dim=-1).mean()),
        }

    def train_episode(self, id_spt, y_spt, id_query, y_query, do_diagnostics):
        self.train()
        h = self.encoder(self.data.features, adj=self.data.adj)
        target = self.remap(y_spt, y_query, h.device)
        distances, raw, queries, prototypes, aligned, attn_stats = self.aligned_prototypes(h, id_spt, id_query, self.args.k_shot, do_diagnostics)
        inter, inter_pos, inter_neg = self.inter_loss(distances, target, self.args.cl_margin, self.args.cl_pos_weight, self.args.cl_neg_weight)
        intra, intra_stats = self.intra_loss(queries, aligned, target)
        loss = inter + self.args.intra_weight * intra
        accuracy = float((distances.argmin(1) == target).float().mean())
        stats = {}
        if do_diagnostics:
            stats = self.diagnostics(distances.detach(), target, queries.detach(), prototypes.detach(), self.args.cl_margin)
            stats.update(attn_stats)
            stats.update(intra_stats)
            stats.update(
                {
                    "inter_loss": float(inter.detach()),
                    "intra_loss": float(intra.detach()),
                    "inter_positive_loss": float(inter_pos.detach()),
                    "inter_negative_loss": float(inter_neg.detach()),
                    "raw_accuracy": float((raw.argmin(1) == target).float().mean()),
                }
            )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach()), accuracy, stats

    @torch.no_grad()
    def evaluate_pools(self, tasks):
        self.eval()
        h = self.encoder(self.data.features, adj=self.data.adj)
        accuracies, raw_accuracies = [], []
        total_queries, sums = 0, {}
        for id_spt, y_spt, id_query, y_query in tasks:
            target = self.remap(y_spt, y_query, h.device)
            distances, raw, queries, prototypes, aligned, attn_stats = self.aligned_prototypes(h, id_spt, id_query, self.args.k_shot_test, True)
            accuracies.append(float((distances.argmin(1) == target).float().mean()))
            raw_accuracies.append(float((raw.argmin(1) == target).float().mean()))
            stats = self.diagnostics(distances, target, queries, prototypes, self.args.cl_margin)
            stats.update(attn_stats)
            n = len(target)
            total_queries += n
            for key, value in stats.items():
                if value is not None:
                    sums[key] = sums.get(key, 0.0) + n * value
        result = {key: val / total_queries for key, val in sums.items()}
        result["num_queries"] = total_queries
        return accuracies, raw_accuracies, result


def parse_args():
    parser = argparse.ArgumentParser(description="exp012: NAQ-Feat, support-set attention, inter/intra Euclidean CL")
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
    parser.add_argument("--attn_lambda", type=float, default=0.1)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--cl_margin", type=float, default=2.0)
    parser.add_argument("--cl_pos_weight", type=float, default=1.0)
    parser.add_argument("--cl_neg_weight", type=float, default=1.0)
    parser.add_argument("--intra_margin", type=float, default=0.5)
    parser.add_argument("--intra_pos_weight", type=float, default=1.0)
    parser.add_argument("--intra_neg_weight", type=float, default=1.0)
    parser.add_argument("--intra_weight", type=float, default=0.1)
    parser.add_argument("--eval_interval", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--json_path", default="exp012.json")
    return parser.parse_args()


def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(1.96 * a.std() / np.sqrt(len(a)))


def main():
    args = parse_args()
    if args.n_way < 2 or args.k_shot != 1 or args.k_shot_test != 1:
        raise ValueError("exp012 currently supports N-way, 1-shot")
    if args.type != "feature" or args.query_generation != "NaQ":
        raise ValueError("exp012 uses NAQ-Feat")
    if args.latent % args.attn_heads:
        raise ValueError("latent must be divisible by attention heads")
    if min(args.cl_margin, args.intra_margin) <= 0:
        raise ValueError("margins must be positive")
    if min(args.cl_pos_weight, args.cl_neg_weight, args.intra_weight, args.intra_pos_weight, args.intra_neg_weight) < 0:
        raise ValueError("loss weights cannot be negative")
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
    model = NAQIntraAlignedCL(args, config, data).to(device)
    train_pool = [proto_task_generator(args, data) for _ in range(args.epoch)]
    valid_pool, test_pool = downstream_task_generator(args, data, args.n_way, args.k_shot_test)
    result_dir = f"results/{args.dataset}/exp012"
    os.makedirs(result_dir, exist_ok=True)
    history = {
        "experiment": "exp012",
        "description": "NAQ-Feat, GCN support-set attention, inter CL + intra aligned-instance CL, nearest corrected prototype",
        "attention_context": "all episode support prototypes; no PPR and no raw feature routing",
        "intra_objective": "q_i positive to p_{y_i}(q_i), negative to p_{y_i}(q_j) for other pseudo-queries j from same support",
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
    print("exp012 | no PPR | support-set attention | inter + intra Euclidean CL", flush=True)
    for epoch, (id_spt, y_spt, id_query, y_query) in tqdm(enumerate(train_pool, 1), total=len(train_pool), desc="exp012"):
        diagnostic = epoch % args.eval_interval == 0
        loss, train_acc, train_stats = model.train_episode(id_spt, y_spt, id_query, y_query, diagnostic)
        if not diagnostic:
            continue
        valid_accs, valid_raws, valid_stats = model.evaluate_pools(valid_pool)
        test_accs, test_raws, test_stats = model.evaluate_pools(test_pool)
        valid_acc, valid_ci = summarize(valid_accs)
        test_acc, test_ci = summarize(test_accs)
        valid_raw, _ = summarize(valid_raws)
        test_raw, test_raw_ci = summarize(test_raws)
        history["evaluations"].append(
            {
                "epoch": epoch,
                "train": {"loss": loss, "accuracy": train_acc, "diagnostics": train_stats},
                "valid": {"accuracy": valid_acc, "raw_accuracy": valid_raw, "ci95_half_width": valid_ci, "diagnostics": valid_stats},
                "test": {
                    "accuracy": test_acc,
                    "raw_accuracy": test_raw,
                    "raw_ci95_half_width": test_raw_ci,
                    "ci95_half_width": test_ci,
                    "diagnostics": test_stats,
                },
                "elapsed_seconds": round(time.time() - start, 2),
            }
        )
        if history["best"] is None or valid_acc > history["best"]["valid_accuracy"]:
            history["best"] = {
                "epoch": epoch,
                "valid_accuracy": valid_acc,
                "test_accuracy": test_acc,
                "test_raw_accuracy": test_raw,
                "test_ci95_half_width": test_ci,
            }
            torch.save({"epoch": epoch, "model": model.state_dict(), "args": vars(args)}, f"{result_dir}/best.pt")
        save_json()
        print(
            f"\n[Epoch {epoch}] loss={loss:.4f} train={train_acc:.4f} "
            f"valid={valid_acc:.4f} (raw={valid_raw:.4f}) "
            f"test={test_acc:.4f}±{test_ci:.4f} (raw={test_raw:.4f})",
            flush=True,
        )
        print(f"  train diagnostics: {train_stats}", flush=True)
    history["elapsed_seconds"] = round(time.time() - start, 2)
    save_json()
    best = history["best"]
    if best is not None:
        line = (
            f"Best epoch: {best['epoch']} | Valid: {best['valid_accuracy']:.4f} | "
            f"Test: {best['test_accuracy']:.4f} ± {best['test_ci95_half_width']:.4f}"
        )
        print(line, flush=True)
        with open(f"{result_dir}/result.txt", "w", encoding="utf-8") as fp:
            fp.write(line + "\n")


if __name__ == "__main__":
    main()
