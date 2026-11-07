"""exp015: independent support and raw-feature-neighbor contrastive losses.

Train one 2-layer GCN on the original graph. For each pseudo-query, L_original
contrasts its own pseudo-class support against other class supports. Separately,
for each support class, pick the single original-graph 1-hop neighbor with the
highest RAW-FEATURE cosine similarity to this query (excluding query identity),
then L_neighbor contrasts the selected correct-class neighbor against selected
other-class neighbors. Total = L_original + lambda_neighbor * L_neighbor.

No graph augmentation, BGRL, QKV projections, or query attention.
Evaluation is the unchanged clean-graph 5-way/5-shot ProtoNet procedure.
Run in the original NaQ project root alongside arg.py/data.py/meta.py.
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

EXP = "exp015"
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
    parser.add_argument("--lambda_neighbor", type=float, default=1.0)
    parser.add_argument("--max_neighbors", type=int, default=0)
    exp, remaining = parser.parse_known_args()
    if exp.temperature <= 0 or exp.lambda_neighbor < 0:
        raise ValueError("temperature must be positive; lambda_neighbor must be nonnegative")
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


def episode_contrastive_loss(
    embeddings, raw_features_normalized, id_spt, y_spt, id_query, y_query, neighbor_lookup, temperature, lambda_neighbor, max_neighbors
):
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

    # (1) Original support-vs-query contrastive objective (unchanged metric).
    direct_dists = (queries[:, None, :] - supports[None, :, :]).square().sum(-1)
    direct_logits = -direct_dists / temperature  # [Q, S]
    positive_mask = query_labels[:, None] == support_labels[None, :]
    if not bool(positive_mask.any(dim=1).all()):
        raise ValueError("Every query needs a positive support")
    direct_log_pos = torch.logsumexp(direct_logits.masked_fill(~positive_mask, -torch.inf), dim=1) - positive_mask.sum(dim=1).float().log()
    original_loss = (torch.logsumexp(direct_logits, dim=1) - direct_log_pos).mean()

    # (2) Select a per-query, per-support neighbor by frozen RAW FEATURES.
    # Default: look through ALL original-graph 1-hop neighbors of each support.
    # Optionally limit the candidate pool using max_neighbors > 0.
    neighbor_lists = []
    for sid in support_ids.detach().cpu().tolist():
        neighbors = neighbor_lookup[sid]
        if max_neighbors > 0 and len(neighbors) > max_neighbors:
            picked = torch.randperm(len(neighbors))[:max_neighbors].tolist()
            neighbors = [neighbors[i] for i in picked]
        neighbor_lists.append(neighbors)

    sizes = [len(nodes) for nodes in neighbor_lists]
    largest = max(1, max(sizes))
    neighbor_ids = torch.zeros((len(sizes), largest), dtype=torch.long, device=device)
    neighbor_valid = torch.zeros((len(sizes), largest), dtype=torch.bool, device=device)
    for i, nodes in enumerate(neighbor_lists):
        if nodes:
            neighbor_ids[i, : len(nodes)] = torch.as_tensor(nodes, device=device)
            neighbor_valid[i, : len(nodes)] = True

    # [Q, S, K]; exclude query itself so it cannot trivially select its own embedding.
    valid = neighbor_valid[None, :, :] & (neighbor_ids[None, :, :] != query_ids[:, None, None])
    with torch.no_grad():
        feature_sim = torch.einsum("qd,skd->qsk", raw_features_normalized[query_ids], raw_features_normalized[neighbor_ids])
        feature_sim = feature_sim.masked_fill(~valid, -torch.inf)
        selected_positions = feature_sim.argmax(dim=-1)  # [Q, S]
        selected_neighbor_ids = neighbor_ids.unsqueeze(0).expand(query_ids.numel(), -1, -1).gather(2, selected_positions.unsqueeze(-1)).squeeze(-1)
        has_neighbor = valid.any(dim=-1)
        # If support has no eligible 1-hop neighbor, use its original ID.
        # This keeps the same number of class candidates and avoids NaN logits.
        selected_neighbor_ids = torch.where(has_neighbor, selected_neighbor_ids, support_ids[None, :].expand_as(selected_neighbor_ids))

    neighbor_vectors = embeddings[selected_neighbor_ids]  # [Q, S, D]
    neighbor_dists = (queries[:, None, :] - neighbor_vectors).square().sum(-1)
    neighbor_logits = -neighbor_dists / temperature
    neighbor_log_pos = torch.logsumexp(neighbor_logits.masked_fill(~positive_mask, -torch.inf), dim=1) - positive_mask.sum(dim=1).float().log()
    neighbor_loss = (torch.logsumexp(neighbor_logits, dim=1) - neighbor_log_pos).mean()
    loss = original_loss + lambda_neighbor * neighbor_loss

    classes = support_labels.unique(sorted=True)
    class_scores = torch.stack([-direct_dists[:, support_labels == cls].mean(dim=1) for cls in classes], dim=1)
    prediction = classes[class_scores.argmax(dim=1)]
    accuracy = float((prediction == query_labels).float().mean().detach().item())
    diagnostics = {
        "original_loss": float(original_loss.detach().item()),
        "neighbor_loss": float(neighbor_loss.detach().item()),
        "mean_support_neighbors_considered": float(np.mean(sizes)),
        "fraction_valid_neighbor_selections": float(has_neighbor.float().mean().item()),
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
        "neighbor_contrastive": vars(exp),
        "protocol": {
            "sampling": "original NaQ-Feat top-10 raw-feature pseudo-queries",
            "augmentation": "none; single original graph",
            "encoder": "one randomly initialized 2-layer GCN",
            "loss": "original_support_InfoNCE + lambda_neighbor * feature_selected_neighbor_InfoNCE",
            "original_candidates": "episode class support embeddings",
            "neighbor_candidates": "one per class; query raw-feature cosine Top-1 within each support original 1-hop neighborhood",
            "neighbor_query_identity": "excluded from neighbor selection",
            "neighbor_without_candidates": "fallback to that class support embedding",
            "selection_embedding": "original node features, frozen and cosine-normalized",
            "loss_embedding": "trainable GNN embeddings; negative squared Euclidean logits",
            "inference": "single clean original graph, original 5-way/5-shot mean ProtoNet",
            "eval_every_epochs": EVAL_EVERY,
            "checkpoint_selection": "best validation of 50 tasks; test once on 500 tasks",
            "pretrained_embeddings": "not used",
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
        raw_features_normalized = F.normalize(features.detach(), p=2, dim=1)
        num_nodes, d = features.shape
        config = [("GCN", (d, 2 * args.latent), True), ("GCN", (2 * args.latent, args.latent), True), ("linear", (args.latent, args.n_way), True)]
        model = ProtoNet(args, config, data).to(device)
        neighbor_lookup = build_neighbor_lookup(data.edge_index, num_nodes)
        train_pool = [proto_task_generator(args, data) for _ in range(args.epoch)]
        valid_pool, test_pool = downstream_task_generator(args, data, args.n_way, args.k_shot_test, args.q_query_test)
        print(f"[{EXP}] {args.dataset} train={len(train_pool)} " f"valid={len(valid_pool)} test={len(test_pool)}", flush=True)

        best_validation = -float("inf")
        best_state = None
        losses, accuracies, original_losses, neighbor_losses, neighbor_counts, valid_fractions = [], [], [], [], [], []
        for epoch, (id_spt, y_spt, id_query, y_query) in tqdm(enumerate(train_pool, 1), total=len(train_pool), desc=args.dataset):
            model.train()
            embeddings = model.network(features, adj=data.adj)
            loss, accuracy, diagnostics = episode_contrastive_loss(
                embeddings,
                raw_features_normalized,
                id_spt,
                y_spt,
                id_query,
                y_query,
                neighbor_lookup,
                exp.temperature,
                exp.lambda_neighbor,
                exp.max_neighbors,
            )
            model.meta_optimizer.zero_grad()
            loss.backward()
            model.meta_optimizer.step()
            losses.append(float(loss.detach().item()))
            accuracies.append(accuracy)
            original_losses.append(diagnostics["original_loss"])
            neighbor_losses.append(diagnostics["neighbor_loss"])
            neighbor_counts.append(diagnostics["mean_support_neighbors_considered"])
            valid_fractions.append(diagnostics["fraction_valid_neighbor_selections"])
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
                "train_original_loss_mean": float(np.mean(original_losses)),
                "train_neighbor_loss_mean": float(np.mean(neighbor_losses)),
                "mean_support_neighbors_considered": float(np.mean(neighbor_counts)),
                "fraction_valid_neighbor_selections": float(np.mean(valid_fractions)),
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
            original_losses.clear()
            neighbor_losses.clear()
            valid_fractions.clear()
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
