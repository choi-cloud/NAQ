"""exp012: NaQ with query-conditioned support-neighborhood prototype.

Single original graph, one 2-layer GCN. For every query and every episode
support class, build a neighbor prototype using query-to-neighbor attention.
Candidate positives for the query's pseudo-class: direct support embedding and
the query-conditioned neighborhood prototype. Other-class candidates are
negatives. Original graph 1-hop neighbors only, no graph augmentation, no BGRL.

Run in the original NAQ project root beside arg.py/data.py/meta.py, etc.
"""

import argparse
import json
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from arg import parse_args_proto
from data import load_data
from meta import ProtoNet
from task_generator import downstream_task_generator, proto_task_generator
from utils import seed_everything

EXP = "exp012"
SEED = 1212
EVAL_EVERY = 160


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--attention_temperature", type=float, default=1.0)
    parser.add_argument("--max_neighbors", type=int, default=10)
    exp, remaining = parser.parse_known_args()
    if exp.temperature <= 0 or exp.attention_temperature <= 0:
        raise ValueError("Both temperatures must be positive")
    if exp.max_neighbors < 0:
        raise ValueError("max_neighbors must be non-negative (0 means all)")
    original = sys.argv
    try:
        sys.argv = [original[0], *remaining]
        args = parse_args_proto("unsup")
    finally:
        sys.argv = original
    return args, exp


def build_neighbor_lookup(edge_index, num_nodes):
    """Unique 1-hop neighbors of original graph; exclude self-loops."""
    u, v = edge_index.detach().cpu().long().tolist()
    lookup = [set() for _ in range(num_nodes)]
    for a, b in zip(u, v):
        if a != b:
            lookup[a].add(b)
            lookup[b].add(a)
    return [list(neighbors) for neighbors in lookup]


def episode_contrastive_loss(embeddings, id_spt, y_spt, id_query, y_query, neighbor_lookup, temperature, attention_temperature, max_neighbors):
    device = embeddings.device
    support_ids = torch.as_tensor(id_spt, dtype=torch.long, device=device).flatten()
    query_ids = torch.as_tensor(id_query, dtype=torch.long, device=device).flatten()
    support_labels = torch.as_tensor(y_spt, dtype=torch.long, device=device).flatten()
    query_labels = torch.as_tensor(y_query, dtype=torch.long, device=device).flatten()
    if support_ids.numel() != support_labels.numel():
        raise ValueError("Support IDs/labels mismatch")
    if query_ids.numel() != query_labels.numel():
        raise ValueError("Query IDs/labels mismatch")

    supports = embeddings[support_ids]  # [S, D]
    queries = embeddings[query_ids]  # [Q, D]
    # Randomly cap each support's neighbors for efficiency, as in exp009.
    neighbor_lists = []
    for sid in support_ids.detach().cpu().tolist():
        neighbors = neighbor_lookup[sid]
        if max_neighbors > 0 and len(neighbors) > max_neighbors:
            picked = torch.randperm(len(neighbors))[:max_neighbors].tolist()
            neighbors = [neighbors[i] for i in picked]
        neighbor_lists.append(neighbors)

    sizes = [len(nodes) for nodes in neighbor_lists]
    largest = max(1, max(sizes))
    neighbor_ids = torch.zeros((len(neighbor_lists), largest), dtype=torch.long)
    neighbor_valid = torch.zeros_like(neighbor_ids, dtype=torch.bool)
    for i, nodes in enumerate(neighbor_lists):
        count = len(nodes)
        if count:
            neighbor_ids[i, :count] = torch.tensor(nodes, dtype=torch.long)
            neighbor_valid[i, :count] = True
    neighbor_ids = neighbor_ids.to(device)
    neighbor_valid = neighbor_valid.to(device)
    neighbor_vectors = embeddings[neighbor_ids]  # [S, K, D]

    # For each query and each support, attend to THAT support's neighbors.
    # Exclude the query itself from its candidate neighborhood to prevent
    # trivial identity copying. Fall back to original support if none remain.
    valid = neighbor_valid.unsqueeze(0) & (neighbor_ids.unsqueeze(0) != query_ids[:, None, None])  # [Q, S, K]
    scores = torch.einsum("qd,skd->qsk", queries, neighbor_vectors)
    scores = scores / (math.sqrt(embeddings.shape[1]) * attention_temperature)
    scores = scores.masked_fill(~valid, -torch.inf)
    has_neighbors = valid.any(dim=-1)  # [Q, S]
    weights = torch.softmax(scores.masked_fill(~has_neighbors[:, :, None], 0), dim=-1)
    weights = weights.masked_fill(~valid, 0)
    neighborhood_prototypes = torch.einsum("qsk,skd->qsd", weights, neighbor_vectors)
    neighborhood_prototypes = torch.where(
        has_neighbors[:, :, None], neighborhood_prototypes, supports.unsqueeze(0).expand_as(neighborhood_prototypes)
    )

    # Each class has exactly two candidate vectors per query:
    # (1) direct support, (2) query-conditioned neighborhood prototype.
    candidate = torch.stack([supports.unsqueeze(0).expand(queries.shape[0], -1, -1), neighborhood_prototypes], dim=2)  # [Q, S, 2, D]
    sq_dist = (queries[:, None, None, :] - candidate).square().sum(-1)
    logits = -sq_dist.reshape(queries.shape[0], -1) / temperature
    candidate_labels = support_labels.repeat_interleave(2)
    positives = query_labels[:, None] == candidate_labels[None, :]
    if not bool(positives.any(dim=1).all()):
        raise ValueError("Every query requires a positive support class")
    positive_log_mean = torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1) - positives.sum(dim=1).float().log()
    loss = (torch.logsumexp(logits, dim=1) - positive_log_mean).mean()

    direct_dist = sq_dist[:, :, 0]
    classes = support_labels.unique(sorted=True)
    class_scores = torch.stack([-direct_dist[:, support_labels == cls].mean(dim=1) for cls in classes], dim=1)
    prediction = classes[class_scores.argmax(dim=1)]
    accuracy = float((prediction == query_labels).float().mean().detach().item())
    diagnostics = {
        "mean_support_neighbors_sampled": float(np.mean(sizes)),
        "mean_valid_attention_neighborhoods_per_query": float(has_neighbors.sum(dim=1).float().mean().detach().item()),
    }
    return loss, accuracy, diagnostics


@torch.no_grad()
def evaluate(embeddings, task_pool, n_way, device):
    accuracies = []
    margins = []
    for id_spt, y_spt, id_query, y_query in task_pool:
        supports = embeddings[id_spt]
        queries = embeddings[id_query]
        labels_s = y_spt.to(device)
        labels_q = y_query.to(device)
        prototypes = torch.stack([supports[labels_s == cls].mean(dim=0) for cls in range(n_way)])
        dists = torch.cdist(queries, prototypes).square()
        prediction = dists.argmin(dim=1)
        accuracies.append(float((prediction == labels_q).float().mean().item()))
        correct = dists.gather(1, labels_q[:, None]).squeeze(1)
        wrong = dists.clone()
        wrong.scatter_(1, labels_q[:, None], float("inf"))
        margins.append(float((wrong.min(dim=1).values - correct).mean().item()))
    x = np.asarray(accuracies, dtype=np.float64)
    return {
        "accuracy_mean": float(x.mean()),
        "accuracy_std": float(x.std(ddof=0)),
        "accuracy_sem": float(x.std(ddof=0) / math.sqrt(len(x))),
        "accuracy_ci95_half_width": float(1.96 * x.std(ddof=0) / math.sqrt(len(x))),
        "mean_true_class_distance_margin": float(np.mean(margins)),
        "num_tasks": len(x),
        "n_way": n_way,
        "k_shot": int(task_pool[0][0].size // n_way),
    }


def main():
    args, exp = parse_args()
    if (args.n_way, args.k_shot, args.k_shot_test) != (5, 1, 5):
        raise ValueError("Only 5-way/1-shot train, 5-way/5-shot test supported")
    if args.dataset not in ("Cora_full", "Amazon_clothing"):
        raise ValueError("Unsupported dataset")
    if args.query_generation != "NaQ" or args.type != "feature":
        raise ValueError("Requires original NaQ-Feat pseudo-queries")
    torch.set_num_threads(4)
    seed_everything(SEED)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    output = Path("out") / EXP
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / f"{EXP}_{args.dataset}.json"
    result = {
        "experiment": EXP,
        "dataset": args.dataset,
        "status": "running",
        "seed": SEED,
        "device": str(device),
        "settings": vars(args),
        "neighborhood_attention": vars(exp),
        "protocol": {
            "sampling": "original NaQ-Feat top-10 raw-feature pseudo-queries",
            "augmentation": "none; single original graph",
            "encoder": "one randomly initialized 2-layer GCN",
            "positives": ["own support", "query-conditioned own support neighborhood prototype"],
            "negatives": "other pseudo-class supports and their query-conditioned neighborhood prototypes",
            "neighbor_source": "original graph 1-hop neighbors, optional random cap; query node excluded",
            "loss": "multi-positive logmeanexp, negative squared Euclidean logits",
            "attention": "scaled query-neighbor dot product softmax over each support neighborhood",
            "inference": "single clean original graph, original 5-way/5-shot mean ProtoNet",
            "eval_every_epochs": EVAL_EVERY,
            "checkpoint_selection": "best validation of 50 tasks; test once on 500 tasks",
            "pretrained_embeddings": "not used; trainable embeddings for attention",
            "code_changes": "none to original NaQ modules",
        },
        "epochs": [],
        "best": None,
    }
    save_json(json_path, result)
    started = time.time()
    try:
        data = load_data(args)
        if device.type == "cuda":
            data.set_device(str(device))
        data.add_diffusion(args)
        features = data.features
        num_nodes, d = features.shape
        config = [("GCN", (d, 2 * args.latent), True), ("GCN", (2 * args.latent, args.latent), True), ("linear", (args.latent, args.n_way), True)]
        model = ProtoNet(args, config, data).to(device)
        neighbor_lookup = build_neighbor_lookup(data.edge_index, num_nodes)
        train_pool = [proto_task_generator(args, data) for _ in range(args.epoch)]
        valid_pool, test_pool = downstream_task_generator(args, data, args.n_way, args.k_shot_test, args.q_query_test)
        print(f"[{EXP}] {args.dataset} train={len(train_pool)} " f"valid={len(valid_pool)} test={len(test_pool)}", flush=True)

        best_validation = -float("inf")
        best_state = None
        losses, accuracies, neighbor_counts = [], [], []
        for epoch, (id_spt, y_spt, id_query, y_query) in tqdm(enumerate(train_pool, 1), total=len(train_pool), desc=args.dataset):
            model.train()
            embeddings = model.network(features, adj=data.adj)
            loss, accuracy, diagnostics = episode_contrastive_loss(
                embeddings, id_spt, y_spt, id_query, y_query, neighbor_lookup, exp.temperature, exp.attention_temperature, exp.max_neighbors
            )
            model.meta_optimizer.zero_grad()
            loss.backward()
            model.meta_optimizer.step()
            losses.append(float(loss.detach().item()))
            accuracies.append(accuracy)
            neighbor_counts.append(diagnostics["mean_support_neighbors_sampled"])
            if epoch % EVAL_EVERY != 0 and epoch != args.epoch:
                continue
            model.eval()
            with torch.no_grad():
                clean = model.network(features, adj=data.adj)
                validation = evaluate(clean, valid_pool, args.n_way, device)
            record = {
                "epoch": epoch,
                "train_loss_mean": float(np.mean(losses)),
                "train_accuracy_support_euclidean_mean": float(np.mean(accuracies)),
                "mean_support_neighbors_sampled": float(np.mean(neighbor_counts)),
                "validation": validation,
                "elapsed_seconds": round(time.time() - started, 2),
            }
            result["epochs"].append(record)
            print(
                f"[Epoch {epoch:5d}] loss={record['train_loss_mean']:.4f} "
                f"acc={record['train_accuracy_support_euclidean_mean']:.4f} "
                f"val={validation['accuracy_mean']:.4f}",
                flush=True,
            )
            losses.clear()
            accuracies.clear()
            neighbor_counts.clear()
            if validation["accuracy_mean"] > best_validation:
                best_validation = validation["accuracy_mean"]
                best_state = deepcopy(model.network.state_dict())
                result["best"] = {"epoch": epoch, "validation": validation}
            save_json(json_path, result)

        model.network.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            clean = model.network(features, adj=data.adj)
            testing = evaluate(clean, test_pool, args.n_way, device)
        result["best"]["test"] = testing
        result["status"] = "completed"
        result["elapsed_seconds"] = round(time.time() - started, 2)
        save_json(json_path, result)
        print(
            f"[{EXP}] {args.dataset} best={result['best']['epoch']} "
            f"test={testing['accuracy_mean']:.4f} "
            f"±{testing['accuracy_ci95_half_width']:.4f}; {json_path}",
            flush=True,
        )
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        result["elapsed_seconds"] = round(time.time() - started, 2)
        save_json(json_path, result)
        raise


if __name__ == "__main__":
    main()
