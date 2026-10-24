"""exp004: NAQ-Feat + PPR-local raw-feature attention.

Training: each pseudo-query attends ONLY to the PPR neighborhood of its
corresponding pseudo-support. Original GCN support prototypes remain fixed.
Evaluation: query labels are unknown, so each query is routed to the support
with highest RAW feature cosine similarity (never via ground-truth labels).
Attention Q/K/V originate from raw node features (learned projections); its
output is added to the original GCN embedding. Reports raw GCN+LR and
attention+LR, plus periodic diagnostics in exp004.json.
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
from sklearn.linear_model import LogisticRegression
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


class NAQFeatureAttention(nn.Module):
    def __init__(self, args, config, data, sampler):
        super().__init__()
        self.args = args
        self.data = data
        self.sampler = sampler
        self.encoder = GNNEncoder(config)
        feature_dim = data.features.shape[1]
        # Q is learned from raw features, while the MHA layer internally
        # projects raw-feature K and V to the latent attention dimension.
        self.query_projection = nn.Linear(feature_dim, args.latent)
        self.attention = nn.MultiheadAttention(
            embed_dim=args.latent, kdim=feature_dim, vdim=feature_dim, num_heads=args.attn_heads, dropout=args.attn_dropout, batch_first=True
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

    def augment(self, queries, query_ids, support_ids, group_indices, diagnostics=False):
        """Each query only attends to the raw features of its assigned support PPR neighborhood.

        queries are ORIGINAL GCN embeddings. query_ids are original graph node IDs.
        group_indices select the corresponding support in support_ids.
        """

        updated = torch.empty_like(queries)
        statistics = []
        raw_features = self.data.features
        query_ids = torch.as_tensor(query_ids, dtype=torch.long, device=raw_features.device)
        support_ids = torch.as_tensor(support_ids, dtype=torch.long, device=raw_features.device)
        for group in torch.unique(group_indices).tolist():
            selection = (group_indices == group).nonzero(as_tuple=True)[0]
            context_ids = torch.as_tensor(self.sampler.get(support_ids[group]), dtype=torch.long, device=queries.device)
            query_features = raw_features[query_ids[selection]]
            context_features = raw_features[context_ids]
            projected_queries = self.query_projection(query_features)
            output, weights = self.attention(
                projected_queries.unsqueeze(0),
                context_features.unsqueeze(0),
                context_features.unsqueeze(0),
                need_weights=diagnostics,
                average_attn_weights=True,
            )
            delta = self.args.attn_lambda * output.squeeze(0)
            updated[selection] = queries[selection] + delta
            if diagnostics:
                with torch.no_grad():
                    w = weights.squeeze(0).clamp_min(1e-12)
                    base = queries[selection]
                    shifted = updated[selection]
                    entropy = -(w * w.log()).sum(-1)
                    # Pairwise similarity of attention outputs from queries
                    # assigned to the same center: high similarity = similar corrections.
                    a = F.normalize(output.squeeze(0), dim=-1)
                    if a.size(0) > 1:
                        pairwise = a @ a.T
                        mask = ~torch.eye(a.size(0), dtype=torch.bool, device=a.device)
                        pair_cos = pairwise[mask].mean().item()
                    else:
                        pair_cos = None
                    statistics.append(
                        {
                            "count": int(selection.numel()),
                            "q_cos": float(F.cosine_similarity(base, shifted).mean()),
                            "delta_ratio": float((delta.norm(dim=-1) / base.norm(dim=-1).clamp_min(1e-8)).mean()),
                            "delta_norm": float(delta.norm(dim=-1).mean()),
                            "gnn_norm": float(base.norm(dim=-1).mean()),
                            "output_cos_query": float(F.cosine_similarity(output.squeeze(0), base).mean()),
                            "entropy": float(entropy.mean()),
                            "normalized_entropy": float((entropy / np.log(w.size(-1))).mean()),
                            "top1_attn": float(w.max(dim=-1).values.mean()),
                            "center_attn": float(w[:, 0].mean()),
                            "effective_neighbors": float(entropy.exp().mean()),
                            "within_support_output_cos": pair_cos,
                        }
                    )
        return updated, statistics

    @staticmethod
    def metrics(statistics):
        if not statistics:
            return {}
        total = sum(r["count"] for r in statistics)
        keys = [key for key in statistics[0] if key != "count"]
        return {
            key: (
                float(sum(r["count"] * r[key] for r in statistics if r[key] is not None) / sum(r["count"] for r in statistics if r[key] is not None))
                if any(r[key] is not None for r in statistics)
                else None
            )
            for key in keys
        } | {"num_queries": total}

    @staticmethod
    def remap(y_support, y_query):
        labels = y_support.detach().cpu().tolist()
        return torch.tensor([labels.index(int(label)) for label in y_query.detach().cpu().tolist()], device=y_query.device, dtype=torch.long)

    def train_episode(self, id_spt, y_spt, id_query, y_query, diagnostics=False):
        self.train()
        h = self.encoder(self.data.features, adj=self.data.adj)
        support = h[id_spt]
        queries = h[id_query]
        target = self.remap(y_spt, y_query)
        enhanced, stats = self.augment(queries, id_query, id_spt, target, diagnostics)
        logits = -((enhanced[:, None, :] - support[None, :, :]) ** 2).sum(dim=-1)
        loss = F.cross_entropy(logits, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        accuracy = (logits.argmax(dim=1) == target).float().mean().item()
        return loss.item(), accuracy, self.metrics(stats)

    def evaluate_pools(self, tasks):
        self.eval()
        with torch.no_grad():
            h = self.encoder(self.data.features, adj=self.data.adj)
            feature_norm = F.normalize(self.data.features, dim=-1)
            results = {"raw": [], "attn": []}
            statistics = []
            routing_correct = 0
            routing_total = 0
            for id_spt, y_spt, id_query, y_query in tasks:
                support = h[id_spt]
                queries = h[id_query]
                y_s = y_spt.detach().cpu().numpy()
                y_q = y_query.detach().cpu().numpy()
                # Evaluation-time routing MUST NOT use y_query.
                routing = (feature_norm[id_query] @ feature_norm[id_spt].T).argmax(dim=1)
                # Diagnostic only: never uses query labels for routing.
                predicted_label = y_s[routing.detach().cpu().numpy()]
                routing_correct += int((predicted_label == y_q).sum())
                routing_total += len(y_q)
                support_routing = torch.arange(len(id_spt), device=h.device)
                enhanced_q, stats = self.augment(queries, id_query, id_spt, routing, True)
                enhanced_s, _ = self.augment(support, id_spt, id_spt, support_routing)
                raw_clf = LogisticRegression(max_iter=1000, C=1 / self.args.l2_penalty)
                attn_clf = LogisticRegression(max_iter=1000, C=1 / self.args.l2_penalty)
                raw_clf.fit(support.cpu().numpy(), y_s)
                attn_clf.fit(enhanced_s.cpu().numpy(), y_s)
                results["raw"].append(raw_clf.score(queries.cpu().numpy(), y_q))
                results["attn"].append(attn_clf.score(enhanced_q.cpu().numpy(), y_q))
                statistics.extend(stats)
        diagnostics = self.metrics(statistics)
        diagnostics["routing_label_accuracy"] = float(routing_correct / max(routing_total, 1))
        return results, diagnostics


def parse_args():
    parser = argparse.ArgumentParser(description="exp004: NAQ-Feat + corresponding support PPR RAW FEATURE attention")
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
    parser.add_argument("--l2_penalty", type=float, default=1e-4)
    parser.add_argument("--num_valid_tasks", type=int, default=50)
    parser.add_argument("--num_downstream_tasks", type=int, default=500)
    parser.add_argument("--query_generation", type=str, default="NaQ")
    parser.add_argument("--type", type=str, default="feature")
    parser.add_argument("--PPR_alpha", type=float, default=0.1)
    parser.add_argument("--ppr_topk", type=int, default=16)
    parser.add_argument("--ppr_steps", type=int, default=20)
    parser.add_argument("--attn_lambda", type=float, default=0.1)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--attn_dropout", type=float, default=0)
    parser.add_argument("--eval_interval", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--json_path", type=str, default="exp004.json")
    return parser.parse_args()


def summarize(values):
    array = np.asarray(values)
    return float(array.mean()), float(1.96 * array.std() / np.sqrt(len(array)))


def main():
    args = parse_args()
    if args.k_shot != 1 or args.k_shot_test != 1:
        raise ValueError("exp004 currently supports 1-shot training and evaluation only")
    if args.latent % args.attn_heads:
        raise ValueError("--latent must be divisible by --attn_heads")
    if args.l2_penalty <= 0:
        raise ValueError("--l2_penalty must be positive")
    if args.type != "feature":
        raise ValueError("exp004 currently requires --type feature for NAQ-Feat")
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
    model = NAQFeatureAttention(args, config, data, sampler).to(device)
    train_pool = [proto_task_generator(args, data) for _ in range(args.epoch)]
    valid_pool, test_pool = downstream_task_generator(args, data, args.n_way, args.k_shot_test)
    directory = f"results/{args.dataset}/exp004"
    os.makedirs(directory, exist_ok=True)
    best = None
    start = time.time()
    print(f"exp004 | Raw-feature Q/K/V PPR topk={args.ppr_topk} (+ center) | lambda={args.attn_lambda}", flush=True)
    history = {
        "experiment": "exp004",
        "description": "raw-feature PPR-neighborhood attention, query GCN residual",
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
    for epoch, (id_spt, y_spt, id_query, y_query) in tqdm(enumerate(train_pool, 1), total=len(train_pool), desc="exp004"):
        diagnostics = epoch % args.eval_interval == 0
        loss, train_acc, training_stats = model.train_episode(id_spt, y_spt.to(device), id_query, y_query.to(device), diagnostics)
        if diagnostics:
            valid, val_stats = model.evaluate_pools(valid_pool)
            test, test_stats = model.evaluate_pools(test_pool)
            valid_raw, _ = summarize(valid["raw"])
            valid_attn, _ = summarize(valid["attn"])
            test_raw, test_raw_ci = summarize(test["raw"])
            test_attn, test_attn_ci = summarize(test["attn"])
            print(
                f"\n[Epoch {epoch}] loss={loss:.4f} train={train_acc:.4f} "
                f"val_raw={valid_raw:.4f} val_attn={valid_attn:.4f} "
                f"test_raw={test_raw:.4f}±{test_raw_ci:.4f} "
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
                "valid": {"raw_accuracy": valid_raw, "attn_accuracy": valid_attn, "diagnostics": val_stats},
                "test": {
                    "raw_accuracy": test_raw,
                    "raw_ci95_half_width": test_raw_ci,
                    "attn_accuracy": test_attn,
                    "attn_ci95_half_width": test_attn_ci,
                    "diagnostics": test_stats,
                },
                "train_diagnostics": training_stats,
                "ppr_cache_entries": len(sampler.cache),
                "elapsed_seconds": round(time.time() - start, 2),
            }
            history["evaluations"].append(entry)
            if best is None or valid_attn > best[1]:
                best = (epoch, valid_attn, test_attn, test_attn_ci, valid_raw, test_raw, test_raw_ci)
                history["best"] = {
                    "epoch": epoch,
                    "valid_attn": valid_attn,
                    "test_attn": test_attn,
                    "test_attn_ci95_half_width": test_attn_ci,
                    "valid_raw": valid_raw,
                    "test_raw": test_raw,
                    "test_raw_ci95_half_width": test_raw_ci,
                }
                torch.save({"epoch": epoch, "model": model.state_dict(), "args": vars(args)}, f"{directory}/best.pt")
            save_json()
    if best is not None:
        ep, val, tst, ci, val_raw, test_raw, raw_ci = best
        summary = (
            f"Best epoch (attn validation): {ep} | Valid attn: {val:.4f} | "
            f"Test attn: {tst:.4f} ± {ci:.4f} | "
            f"Valid raw: {val_raw:.4f} | Test raw: {test_raw:.4f} ± {raw_ci:.4f} | "
            f"Elapsed: {time.time() - start:.1f}s"
        )
        print(summary, flush=True)
        history["elapsed_seconds"] = round(time.time() - start, 2)
        save_json()
        with open(f"{directory}/result.txt", "w") as file:
            file.write(summary + "\n")


if __name__ == "__main__":
    main()
