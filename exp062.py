"""exp062: Hierarchical pseudo-pair pretraining for GFSL.

Same backbone, clustering and evaluation protocol as exp061. Three tiers:
positive: edge and both clusters agree;
weak negative: other edges OR nonedges sharing at least one cluster;
strong negative: nonedge and neither cluster agrees.
GT is exclusively used for diagnostics, not training or checkpoint selection.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import normalize

from data import load_data
from exp061 import GraphPairModel, candidate_precision, fit_clusters, get_edges, pair_scores, sample_negatives, sparse_adjacency, torch_adj


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Cora_full")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--k", type=int, default=70)
    p.add_argument("--svd_dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--cluster_batch_size", type=int, default=2048)
    p.add_argument("--cluster_max_iter", type=int, default=200)
    p.add_argument("--cluster_n_init", type=int, default=3)
    p.add_argument("--pairs_per_epoch", type=int, default=12000)
    p.add_argument("--pair_batch", type=int, default=4096)
    p.add_argument("--eval_anchors", type=int, default=200)
    p.add_argument("--eval_pairs", type=int, default=10000)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--rank_weight", type=float, default=0.5)
    p.add_argument("--margin_pos_weak", type=float, default=0.15)
    p.add_argument("--margin_weak_strong", type=float, default=0.15)
    p.add_argument("--weak_edge_fraction", type=float, default=0.5)
    return p.parse_args()


def save_json(path, obj):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sample_weak_nonedges(rng, count, cx, ca, edge_keys):
    """Nonedge pairs sharing >=1 cluster, sampled without GT information."""
    n = len(cx)
    groups_x = [np.flatnonzero(cx == k) for k in np.unique(cx)]
    groups_a = [np.flatnonzero(ca == k) for k in np.unique(ca)]
    groups_x = [g for g in groups_x if len(g) > 1]
    groups_a = [g for g in groups_a if len(g) > 1]
    pools = groups_x + groups_a
    weights = np.array([len(g) * (len(g) - 1) for g in pools], dtype=np.float64)
    weights /= weights.sum()
    results = []
    left = count
    for _ in range(100):
        if left == 0:
            break
        m = max(left * 3, 1024)
        indices = rng.choice(len(pools), size=m, p=weights)
        u = np.empty(m, dtype=np.int64)
        v = np.empty(m, dtype=np.int64)
        for gid in np.unique(indices):
            locs = np.flatnonzero(indices == gid)
            group = pools[gid]
            u[locs] = group[rng.integers(len(group), size=len(locs))]
            v[locs] = group[rng.integers(len(group), size=len(locs))]
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        valid = (lo != hi) & ~np.isin(lo * n + hi, edge_keys)
        batch = np.column_stack([lo[valid], hi[valid]])[:left]
        if len(batch):
            results.append(batch)
            left -= len(batch)
    if left:
        raise RuntimeError(f"Weak nonedge sampling failed: remaining={left}")
    return np.concatenate(results, axis=0) if results else np.empty((0, 2), dtype=np.int64)


def sample_weak(rng, count, weak_edges, cx, ca, edge_keys, edge_fraction):
    ecount = min(count, round(count * edge_fraction))
    if not len(weak_edges):
        ecount = 0
    edge_pairs = weak_edges[rng.choice(len(weak_edges), size=ecount, replace=True)] if ecount else np.empty((0, 2), dtype=np.int64)
    other = sample_weak_nonedges(rng, count - ecount, cx, ca, edge_keys)
    return np.concatenate([edge_pairs, other], axis=0)


def metrics(gt, scores):
    if np.unique(gt).size < 2:
        return {"gt_base_rate": float(gt.mean()), "roc_auc": None, "average_precision": None}
    return {
        "gt_base_rate": float(gt.mean()),
        "roc_auc": float(roc_auc_score(gt, scores)),
        "average_precision": float(average_precision_score(gt, scores)),
    }


def random_nonedge_pairs(rng, n, count, edge_keys):
    out = []
    remain = count
    while remain:
        q = rng.integers(0, n, size=(max(remain * 2, 1024), 2), dtype=np.int64)
        lo, hi = np.minimum(q[:, 0], q[:, 1]), np.maximum(q[:, 0], q[:, 1])
        valid = (lo != hi) & ~np.isin(lo * n + hi, edge_keys)
        batch = np.column_stack([lo[valid], hi[valid]])[:remain]
        out.append(batch)
        remain -= len(batch)
    return np.concatenate(out, axis=0)


def main():
    args = parse_args()
    if args.dataset != "Cora_full":
        raise ValueError("Configured for Cora_full")
    if not 0 <= args.weak_edge_fraction <= 1:
        raise ValueError("weak_edge_fraction must be in [0,1]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    data = load_data(args)
    y = data.labels.detach().cpu().numpy().reshape(-1)
    x = data.features.detach().cpu().numpy().astype(np.float32, copy=False)
    n = len(y)
    x_sparse = sp.csr_matrix(x)
    adj_cpu = sparse_adjacency(data, n)
    ax = np.asarray(adj_cpu @ x, dtype=np.float32)
    edges = get_edges(data)
    edge_keys = edges[:, 0] * n + edges[:, 1]
    cx = fit_clusters(x, args)
    ca = fit_clusters(ax, args)
    agree_x = cx[edges[:, 0]] == cx[edges[:, 1]]
    agree_a = ca[edges[:, 0]] == ca[edges[:, 1]]
    positives = edges[agree_x & agree_a]
    weak_edges = edges[~(agree_x & agree_a)]
    if not len(positives):
        raise RuntimeError("Empty positive tier")

    views = []
    for mat in [x, ax]:
        dim = min(args.svd_dim, min(mat.shape) - 1)
        reduced = TruncatedSVD(n_components=dim, n_iter=5, random_state=args.seed).fit_transform(mat)
        views.append(normalize(reduced.astype(np.float32), norm="l2", copy=True))
    features = torch.as_tensor(np.concatenate(views, axis=1), dtype=torch.float32, device=device)
    adj = torch_adj(data, device)
    model = GraphPairModel(features.shape[1], args.hidden, args.embed_dim, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    diagnostic_count = min(args.eval_pairs, 10000)
    weak_diag = sample_weak(rng, diagnostic_count, weak_edges, cx, ca, edge_keys, args.weak_edge_fraction)
    strong_diag = sample_negatives(rng, n, diagnostic_count, cx, ca, edge_keys)
    eval_pos = positives[rng.choice(len(positives), size=min(args.eval_pairs // 2, len(positives)), replace=False)]
    eval_neg = sample_negatives(rng, n, len(eval_pos), cx, ca, edge_keys)
    pseudo_eval = np.concatenate([eval_pos, eval_neg])
    pseudo_gt = (y[pseudo_eval[:, 0]] == y[pseudo_eval[:, 1]]).astype(np.int64)
    random_eval = rng.integers(0, n, size=(args.eval_pairs, 2), dtype=np.int64)
    random_eval = random_eval[random_eval[:, 0] != random_eval[:, 1]]
    random_gt = (y[random_eval[:, 0]] == y[random_eval[:, 1]]).astype(np.int64)
    nonedge_eval = random_nonedge_pairs(rng, n, args.eval_pairs, edge_keys)
    nonedge_gt = (y[nonedge_eval[:, 0]] == y[nonedge_eval[:, 1]]).astype(np.int64)
    anchors = rng.choice(n, size=min(args.eval_anchors, n), replace=False)

    report = {
        "experiment": "exp062",
        "status": "running",
        "args": vars(args),
        "design": {
            "positive": "1-hop edge AND both raw_spherical clusters agree",
            "weak_negative": "other 1-hop edge OR nonedge sharing at least one of the two clusters",
            "strong_negative": "nonedge AND neither raw_spherical cluster agrees",
            "training": "exp061 BCE(pos/strong) plus relative hinge rankings pos>weak>strong on predicted probabilities",
            "weak_negative_sampling": "mixture of edge and nonedge weak pools; sampling with replacement",
            "gt_use": "diagnostics only; no GT training or checkpoint selection",
            "caveat": "ranked tiers are a hypothesis, not a GT-guaranteed ordering; weak edge/nonedge rates are measured separately",
        },
        "pseudo_label_diagnostics": {
            "positive_count": int(len(positives)),
            "positive_gt_same_rate": float(np.mean(y[positives[:, 0]] == y[positives[:, 1]])),
            "strong_negative_gt_different_rate": float(np.mean(y[strong_diag[:, 0]] != y[strong_diag[:, 1]])),
            "weak_negative_gt_same_rate": float(np.mean(y[weak_diag[:, 0]] == y[weak_diag[:, 1]])),
            "weak_edge_count": int(len(weak_edges)),
            "weak_edge_gt_same_rate": float(np.mean(y[weak_edges[:, 0]] == y[weak_edges[:, 1]])),
            "weak_nonedge_gt_same_rate": (
                float(
                    np.mean(
                        y[weak_diag[round(diagnostic_count * args.weak_edge_fraction) :, 0]]
                        == y[weak_diag[round(diagnostic_count * args.weak_edge_fraction) :, 1]]
                    )
                )
                if args.weak_edge_fraction < 1
                else None
            ),
            "edge_gt_same_rate": float(np.mean(y[edges[:, 0]] == y[edges[:, 1]])),
        },
        "history": [],
    }
    out = Path(__file__).resolve().parent / "exp062.json"
    save_json(out, report)
    for epoch in range(1, args.epochs + 1):
        model.train()
        count = args.pairs_per_epoch // 2
        pos = positives[rng.choice(len(positives), size=count, replace=len(positives) < count)]
        weak = sample_weak(rng, count, weak_edges, cx, ca, edge_keys, args.weak_edge_fraction)
        strong = sample_negatives(rng, n, count, cx, ca, edge_keys)
        tp = torch.as_tensor(pos, dtype=torch.long, device=device)
        tw = torch.as_tensor(weak, dtype=torch.long, device=device)
        ts = torch.as_tensor(strong, dtype=torch.long, device=device)
        optimizer.zero_grad()
        z = model.encode(features, adj)
        pos_logits = model.pair_logits(z, tp)
        weak_logits = model.pair_logits(z, tw)
        strong_logits = model.pair_logits(z, ts)
        bce = (F.softplus(-pos_logits).mean() + F.softplus(strong_logits).mean()) / 2
        p = torch.sigmoid(pos_logits)
        w = torch.sigmoid(weak_logits)
        s = torch.sigmoid(strong_logits)
        rank_pw = F.relu(args.margin_pos_weak - p + w).mean()
        rank_ws = F.relu(args.margin_weak_strong - w + s).mean()
        rank = rank_pw + rank_ws
        loss = bce + args.rank_weight * rank
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                z = model.encode(features, adj)
            rec = {
                "epoch": epoch,
                "total_loss": float(loss.item()),
                "pseudo_bce": float(bce.item()),
                "rank_loss": float(rank.item()),
                "rank_pos_weak": float(rank_pw.item()),
                "rank_weak_strong": float(rank_ws.item()),
                "score_means_at_train_step": {
                    "positive": float(p.detach().mean().item()),
                    "weak_negative": float(w.detach().mean().item()),
                    "strong_negative": float(s.detach().mean().item()),
                },
                "pseudo_training_pool_gt_auc": metrics(pseudo_gt, pair_scores(model, z, pseudo_eval, device, args.pair_batch)),
                "random_pairs_gt_auc": metrics(random_gt, pair_scores(model, z, random_eval, device, args.pair_batch)),
                "random_nonedge_pairs_gt_auc": metrics(nonedge_gt, pair_scores(model, z, nonedge_eval, device, args.pair_batch)),
            }
            report["history"].append(rec)
            print(
                f"epoch={epoch} bce={bce.item():.4f} rank={rank.item():.4f} nonedge_auc={rec['random_nonedge_pairs_gt_auc']['roc_auc']}", flush=True
            )
            save_json(out, report)

    model.eval()
    with torch.no_grad():
        z = model.encode(features, adj)
    print("Evaluating full and nonedge Query P@10...", flush=True)
    report["query_evaluation"] = candidate_precision(model, z, x_sparse, y, anchors, edge_keys, device, args.pair_batch)
    report["status"] = "completed"
    save_json(out, report)
    torch.save({"model": model.state_dict(), "args": vars(args)}, out.with_suffix(".pt"))
    print(f"DONE: {out}", flush=True)


if __name__ == "__main__":
    main()
