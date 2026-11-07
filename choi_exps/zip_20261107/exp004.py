"""exp004: BGRL pretraining and pool-purity comparison against baselines.

Place next to original NAQ modules and exp003.py. No labels are used for
learning, model selection, or candidate retrieval: labels are diagnostics only.
The CL encoder is NOT used to initialize the downstream few-shot encoder.
"""

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from data import load_data
from exp003 import dropped_normalized_adj
from model import GNNEncoder
from utils import seed_everything

EXP = "exp004"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["Cora_full", "Amazon_clothing"], required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=400)
    parser.add_argument("--latent", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--edge_drop", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--predictor_hidden", type=int, default=512)
    parser.add_argument("--diagnostic_anchors", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--infonce_dir", type=str, default="out/exp003")
    args = parser.parse_args()
    if args.epoch < 1 or args.eval_every < 1:
        parser.error("epoch and eval_every must be >= 1")
    if args.batch_size < 2 or args.diagnostic_anchors < 1:
        parser.error("batch_size >= 2 and diagnostic_anchors >= 1 are required")
    if not 0 <= args.edge_drop < 1 or not 0 <= args.momentum < 1:
        parser.error("edge_drop and momentum must be in [0, 1)")
    return args


def save_json(path, result):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)


@torch.no_grad()
def knn_purity(reps, labels, anchors, ks=(10, 50, 100)):
    """Exact cosine top-K neighbors, excluding self; O(chunk_size*N) memory."""
    n = reps.size(0)
    ks = [min(k, n - 1) for k in ks]
    max_k = max(ks)
    reps = F.normalize(reps, dim=-1)
    labels = labels.to(reps.device)
    counts = {k: 0 for k in ks}
    similarity_sums = {k: 0.0 for k in ks}
    for chunk in anchors.split(32):
        sim = reps[chunk] @ reps.T
        sim[torch.arange(chunk.numel(), device=reps.device), chunk] = -torch.inf
        values, ids = sim.topk(max_k, dim=1)
        same = labels[ids].eq(labels[chunk].unsqueeze(1))
        for k in ks:
            counts[k] += int(same[:, :k].sum().item())
            similarity_sums[k] += float(values[:, :k].sum().item())
    return {f"top{k}": {"class_match_rate": counts[k] / (anchors.numel() * k), "cosine_mean": similarity_sums[k] / (anchors.numel() * k)} for k in ks}


@torch.no_grad()
def update_target(online, target, momentum):
    for online_param, target_param in zip(online.parameters(), target.parameters()):
        target_param.mul_(momentum).add_(online_param, alpha=1 - momentum)
    for online_buffer, target_buffer in zip(online.buffers(), target.buffers()):
        target_buffer.copy_(online_buffer)


def bgrl_loss(p1, t2, p2, t1):
    """Negative-free bootstrapped cosine regression; targets are stop-gradient."""
    loss1 = 2 - 2 * F.cosine_similarity(p1, t2.detach(), dim=-1).mean()
    loss2 = 2 - 2 * F.cosine_similarity(p2, t1.detach(), dim=-1).mean()
    return (loss1 + loss2) / 2


def main():
    args = parse_args()
    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    out_dir = Path("out") / EXP
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{EXP}_{args.dataset}"
    json_path = out_dir / f"{prefix}.json"
    model_path = out_dir / f"{prefix}_encoder.pt"
    embedding_path = out_dir / f"{prefix}_embeddings.pt"
    start_time = time.time()
    result = {
        "experiment": EXP,
        "dataset": args.dataset,
        "status": "running",
        "seed": args.seed,
        "device": str(device),
        "settings": vars(args),
        "protocol": {
            "pretraining": "BGRL, two independent edge-drop views, shared original features",
            "loss": "symmetric negative-free cosine prediction of EMA target; target stop-gradient",
            "encoder": "NAQ-compatible 2-layer GCN; separate randomly initialized few-shot model",
            "model_selection": "final epoch without using downstream labels",
            "evaluation": "fixed anchor IDs, cosine Top-10/50/100 true-class match rates; no downstream labels in training",
            "random_baseline": "epoch-0 untrained GCN with same architecture",
            "infonce_baseline": "load exp003 final embeddings if available, otherwise mark missing",
            "raw_baseline": "original node features",
            "augmentation": "edge drop 20% by default, no feature masking",
            "outputs": "online encoder checkpoint and unperturbed-graph embeddings",
        },
        "artifacts": {"encoder_checkpoint": str(model_path), "embeddings": str(embedding_path)},
        "baselines": {},
        "epochs": [],
    }
    save_json(json_path, result)

    try:
        data = load_data(args)
        data.set_device(str(device))
        x = data.features
        labels = data.labels.to(device)
        n = x.size(0)
        config = [("GCN", (x.size(1), 2 * args.latent), True), ("GCN", (2 * args.latent, args.latent), True)]
        online = GNNEncoder(config).to(device)
        target = copy.deepcopy(online).to(device)
        target.requires_grad_(False)
        predictor = nn.Sequential(nn.Linear(args.latent, args.predictor_hidden), nn.PReLU(), nn.Linear(args.predictor_hidden, args.latent)).to(device)
        optimizer = torch.optim.Adam(list(online.parameters()) + list(predictor.parameters()), lr=args.lr)
        anchors = torch.linspace(0, n - 1, min(args.diagnostic_anchors, n), device=device).long().unique()
        with torch.no_grad():
            random_embeddings = online(x, adj=data.adj)
            result["baselines"]["random_gcn"] = knn_purity(random_embeddings, labels, anchors)
            result["baselines"]["raw_features"] = knn_purity(x, labels, anchors)
        del random_embeddings

        infonce_path = Path(args.infonce_dir) / f"exp003_{args.dataset}_embeddings.pt"
        if infonce_path.is_file():
            payload = torch.load(infonce_path, map_location="cpu", weights_only=True)
            if payload.get("dataset") != args.dataset:
                raise ValueError(f"Mismatch in exp003 embedding dataset: {infonce_path}")
            embeddings = payload["embeddings"].to(device)
            if embeddings.size(0) != n:
                raise ValueError("exp003 embeddings have a different number of nodes")
            expected_ids = torch.arange(n)
            if not torch.equal(payload["node_ids"].cpu(), expected_ids):
                raise ValueError("exp003 embeddings are not aligned with original node IDs")
            result["baselines"]["infonce_exp003"] = knn_purity(embeddings, labels, anchors)
            result["baselines"]["infonce_exp003_status"] = "loaded"
            del embeddings
        else:
            result["baselines"]["infonce_exp003_status"] = "missing"
            result["baselines"]["infonce_exp003_expected_path"] = str(infonce_path)
            print(f"[Warning] Missing exp003 embedding: {infonce_path}", flush=True)

        result["diagnostic_anchor_ids"] = anchors.cpu().tolist()
        save_json(json_path, result)
        print(f"[{EXP}] {args.dataset} nodes={n} anchor_count={anchors.numel()}", flush=True)
        for name in ["raw_features", "random_gcn", "infonce_exp003"]:
            if name in result["baselines"]:
                summary = result["baselines"][name]
                print(f"[Baseline {name}] " + " ".join(f"top{k}={summary[f'top{k}']['class_match_rate']:.4f}" for k in (10, 50, 100)), flush=True)

        for epoch in trange(1, args.epoch + 1, desc=f"BGRL {args.dataset}"):
            online.train()
            predictor.train()
            view1 = dropped_normalized_adj(data.edge_index, n, args.edge_drop)
            view2 = dropped_normalized_adj(data.edge_index, n, args.edge_drop)
            sample_ids = torch.randperm(n, device=device)[: min(n, args.batch_size)]
            online1 = online(x, adj=view1)[sample_ids]
            online2 = online(x, adj=view2)[sample_ids]
            with torch.no_grad():
                target1 = target(x, adj=view1)[sample_ids]
                target2 = target(x, adj=view2)[sample_ids]
            prediction1 = predictor(online1)
            prediction2 = predictor(online2)
            loss = bgrl_loss(prediction1, target2, prediction2, target1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Momentum starts at args.momentum and approaches 1 by epoch N.
            momentum = 1 - (1 - args.momentum) * (math.cos(math.pi * epoch / args.epoch) + 1) / 2
            update_target(online, target, momentum)

            if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epoch:
                online.eval()
                with torch.no_grad():
                    final_embeddings = online(x, adj=data.adj)
                    diagnostics = knn_purity(final_embeddings, labels, anchors)
                record = {
                    "epoch": epoch,
                    "train_loss": float(loss.item()),
                    "momentum": momentum,
                    "pool_diagnostics": diagnostics,
                    "elapsed_seconds": round(time.time() - start_time, 2),
                }
                result["epochs"].append(record)
                save_json(json_path, result)
                print(
                    f"[Epoch {epoch:4d}] loss={loss.item():.4f} "
                    + " ".join(f"top{k}={diagnostics[f'top{k}']['class_match_rate']:.4f}" for k in (10, 50, 100)),
                    flush=True,
                )

        online.eval()
        with torch.no_grad():
            final_embeddings = online(x, adj=data.adj).cpu()
        torch.save(
            {
                "state_dict": online.state_dict(),
                "config": config,
                "dataset": args.dataset,
                "epoch": args.epoch,
                "seed": args.seed,
                "settings": vars(args),
            },
            model_path,
        )
        torch.save(
            {
                "embeddings": final_embeddings,
                "node_ids": torch.arange(n),
                "dataset": args.dataset,
                "epoch": args.epoch,
                "embedding_type": "BGRL online encoder on original unperturbed graph",
            },
            embedding_path,
        )
        result["status"] = "completed"
        result["final"] = result["epochs"][-1]
        result["elapsed_seconds"] = round(time.time() - start_time, 2)
        save_json(json_path, result)
        print(f"Saved model: {model_path}\nSaved embeddings: {embedding_path}", flush=True)
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        save_json(json_path, result)
        raise


if __name__ == "__main__":
    main()
