"""exp063: Anchor-aligned, three-tier ranking-only homophily pretraining.

Strong positive: edge and both X / AX raw-spherical clusters agree.
Weak positive: other edge, or a nonedge sharing at least one cluster.
Negative: nonedge and neither cluster agrees.

No BCE and no GT supervision. Existing NAQ files and exp061 are not modified.
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
from exp062 import random_nonedge_pairs, sample_weak


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
    p.add_argument("--margin_pos_weak", type=float, default=0.5)
    p.add_argument("--margin_weak_strong", type=float, default=0.5)
    p.add_argument("--weak_edge_fraction", type=float, default=0.5)
    return p.parse_args()


def save_json(path, obj):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def metrics(gt, scores):
    if np.unique(gt).size < 2:
        return {"gt_base_rate": float(gt.mean()), "roc_auc": None, "average_precision": None}
    return {
        "gt_base_rate": float(gt.mean()),
        "roc_auc": float(roc_auc_score(gt, scores)),
        "average_precision": float(average_precision_score(gt, scores)),
    }


def make_neighbor_lists(n, pairs):
    neighbors = [[] for _ in range(n)]
    for u, v in pairs:
        neighbors[u].append(v)
        neighbors[v].append(u)
    return [np.asarray(row, dtype=np.int64) for row in neighbors]


def make_cluster_members(labels):
    return {int(k): np.flatnonzero(labels == k) for k in np.unique(labels)}


def sample_anchor_triples(rng, count, positives, weak_neighbors, cx, ca, members_x, members_a, edge_keys, n, edge_fraction):
    """Sample (anchor, strong-pos, weak-pos, negative) sharing each anchor.

    Edge-based weak samples are preferred at the requested rate when available.
    Weak nonedges are sampled by shared cluster with fallback to the other view.
    A negative is accepted only if distinct, nonadjacent, and disagrees in both views.
    """
    selection = positives[rng.integers(len(positives), size=count)]
    swap = rng.random(count) < 0.5
    anchors = np.where(swap, selection[:, 1], selection[:, 0])
    strong = np.where(swap, selection[:, 0], selection[:, 1])
    weak = np.full(count, -1, dtype=np.int64)
    neg = np.full(count, -1, dtype=np.int64)
    want_weak_edge = rng.random(count) < edge_fraction

    for idx in np.flatnonzero(want_weak_edge):
        candidates = weak_neighbors[int(anchors[idx])]
        if len(candidates):
            weak[idx] = candidates[rng.integers(len(candidates))]

    remaining = np.flatnonzero(weak < 0)
    for attempt in range(100):
        if not len(remaining):
            break
        anchor_batch = anchors[remaining]
        choose_x = rng.random(len(remaining)) < 0.5
        choices = np.empty(len(remaining), dtype=np.int64)
        for view, labels, groups in [(True, cx, members_x), (False, ca, members_a)]:
            loc = np.flatnonzero(choose_x == view)
            for local in loc:
                anchor = int(anchor_batch[local])
                group = groups[int(labels[anchor])]
                choices[local] = group[rng.integers(len(group))]
        lo = np.minimum(anchor_batch, choices)
        hi = np.maximum(anchor_batch, choices)
        valid = (anchor_batch != choices) & ~np.isin(lo * n + hi, edge_keys)
        weak[remaining[valid]] = choices[valid]
        remaining = remaining[~valid]
    if len(remaining):
        raise RuntimeError(f"Could not sample weak nonedge for {len(remaining)} anchors")

    remaining = np.arange(count)
    for attempt in range(100):
        if not len(remaining):
            break
        anchor_batch = anchors[remaining]
        choices = rng.integers(n, size=len(remaining), dtype=np.int64)
        lo = np.minimum(anchor_batch, choices)
        hi = np.maximum(anchor_batch, choices)
        valid = (anchor_batch != choices) & (cx[anchor_batch] != cx[choices]) & (ca[anchor_batch] != ca[choices]) & ~np.isin(lo * n + hi, edge_keys)
        neg[remaining[valid]] = choices[valid]
        remaining = remaining[~valid]
    if len(remaining):
        raise RuntimeError(f"Could not sample strong negative for {len(remaining)} anchors")

    return (np.column_stack([anchors, strong]), np.column_stack([anchors, weak]), np.column_stack([anchors, neg]))


def main():
    args = parse_args()
    if args.dataset != "Cora_full":
        raise ValueError("Configured for Cora_full")
    if not 0 <= args.weak_edge_fraction <= 1:
        raise ValueError("weak_edge_fraction must be in [0, 1]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 9000)
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

    print("Fitting raw spherical-proxy clusters...", flush=True)
    cx = fit_clusters(x, args)
    ca = fit_clusters(ax, args)
    agree_x = cx[edges[:, 0]] == cx[edges[:, 1]]
    agree_a = ca[edges[:, 0]] == ca[edges[:, 1]]
    positives = edges[agree_x & agree_a]
    weak_edges = edges[~(agree_x & agree_a)]
    if not len(positives):
        raise RuntimeError("Empty strong-positive pool")
    weak_neighbors = make_neighbor_lists(n, weak_edges)
    members_x = make_cluster_members(cx)
    members_a = make_cluster_members(ca)

    views = []
    for mat in [x, ax]:
        dim = min(args.svd_dim, min(mat.shape) - 1)
        reduced = TruncatedSVD(n_components=dim, n_iter=5, random_state=args.seed).fit_transform(mat)
        views.append(normalize(reduced.astype(np.float32), norm="l2", copy=True))
    features = torch.as_tensor(np.concatenate(views, axis=1), dtype=torch.float32, device=device)
    adj = torch_adj(data, device)
    model = GraphPairModel(features.shape[1], args.hidden, args.embed_dim, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Fixed evaluation: GT is used only to compute diagnostic metrics.
    diagnostic_count = min(args.eval_pairs, 10000)
    weak_diag = sample_weak(eval_rng, diagnostic_count, weak_edges, cx, ca, edge_keys, args.weak_edge_fraction)
    strong_diag = sample_negatives(eval_rng, n, diagnostic_count, cx, ca, edge_keys)
    eval_pos = positives[eval_rng.choice(len(positives), size=min(args.eval_pairs // 2, len(positives)), replace=False)]
    eval_neg = sample_negatives(eval_rng, n, len(eval_pos), cx, ca, edge_keys)
    pseudo_eval = np.concatenate([eval_pos, eval_neg])
    pseudo_gt = (y[pseudo_eval[:, 0]] == y[pseudo_eval[:, 1]]).astype(np.int64)
    random_eval = eval_rng.integers(0, n, size=(args.eval_pairs, 2), dtype=np.int64)
    random_eval = random_eval[random_eval[:, 0] != random_eval[:, 1]]
    random_gt = (y[random_eval[:, 0]] == y[random_eval[:, 1]]).astype(np.int64)
    nonedge_eval = random_nonedge_pairs(eval_rng, n, args.eval_pairs, edge_keys)
    nonedge_gt = (y[nonedge_eval[:, 0]] == y[nonedge_eval[:, 1]]).astype(np.int64)
    anchors = eval_rng.choice(n, size=min(args.eval_anchors, n), replace=False)

    report = {
        "experiment": "exp063",
        "status": "running",
        "args": vars(args),
        "design": {
            "strong_positive": "1-hop edge AND both X/AX raw-spherical clusters agree",
            "weak_positive": "Other 1-hop edge OR nonedge sharing >=1 cluster",
            "negative": "Nonedge AND neither cluster agrees",
            "objective": "Ranking-only shared-anchor triples: relu(m1 - logit(SP) + logit(WP)) + relu(m2 - logit(WP) + logit(N))",
            "bce": False,
            "note": "This is a pure three-tier ranking experiment, not the original RCL InfoNCE loss",
            "gt_use": "Diagnostics only; not used in cluster fitting, training, or checkpoint selection",
            "evaluation": "Independent fixed-seed anchors/pairs; compare directly with the included cosine baseline",
        },
        "pseudo_label_diagnostics": {
            "positive_count": int(len(positives)),
            "positive_gt_same_rate": float(np.mean(y[positives[:, 0]] == y[positives[:, 1]])),
            "weak_edge_count": int(len(weak_edges)),
            "weak_edge_gt_same_rate": float(np.mean(y[weak_edges[:, 0]] == y[weak_edges[:, 1]])),
            "weak_sample_gt_same_rate": float(np.mean(y[weak_diag[:, 0]] == y[weak_diag[:, 1]])),
            "strong_negative_gt_different_rate": float(np.mean(y[strong_diag[:, 0]] != y[strong_diag[:, 1]])),
        },
        "history": [],
    }
    out = Path(__file__).resolve().parent / "exp063.json"
    save_json(out, report)
    print(f"Strong positives: {len(positives)}; weak edges: {len(weak_edges)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pair_count = args.pairs_per_epoch // 2
        sp_pairs, wp_pairs, neg_pairs = sample_anchor_triples(
            rng, pair_count, positives, weak_neighbors, cx, ca, members_x, members_a, edge_keys, n, args.weak_edge_fraction
        )
        tp = torch.as_tensor(sp_pairs, dtype=torch.long, device=device)
        tw = torch.as_tensor(wp_pairs, dtype=torch.long, device=device)
        tn = torch.as_tensor(neg_pairs, dtype=torch.long, device=device)
        optimizer.zero_grad()
        z = model.encode(features, adj)
        sp_logit = model.pair_logits(z, tp)
        wp_logit = model.pair_logits(z, tw)
        neg_logit = model.pair_logits(z, tn)
        rank_sw = F.relu(args.margin_pos_weak - sp_logit + wp_logit).mean()
        rank_wn = F.relu(args.margin_weak_strong - wp_logit + neg_logit).mean()
        loss = rank_sw + rank_wn
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                z = model.encode(features, adj)
            rec = {
                "epoch": epoch,
                "rank_loss": float(loss.item()),
                "rank_strong_weak": float(rank_sw.item()),
                "rank_weak_negative": float(rank_wn.item()),
                "score_means_at_train_step": {
                    "strong_positive": float(torch.sigmoid(sp_logit.detach()).mean().item()),
                    "weak_positive": float(torch.sigmoid(wp_logit.detach()).mean().item()),
                    "negative": float(torch.sigmoid(neg_logit.detach()).mean().item()),
                },
                "pseudo_training_pool_gt_auc": metrics(pseudo_gt, pair_scores(model, z, pseudo_eval, device, args.pair_batch)),
                "random_pairs_gt_auc": metrics(random_gt, pair_scores(model, z, random_eval, device, args.pair_batch)),
                "random_nonedge_pairs_gt_auc": metrics(nonedge_gt, pair_scores(model, z, nonedge_eval, device, args.pair_batch)),
            }
            report["history"].append(rec)
            print(f"epoch={epoch} rank={loss.item():.4f} " f"nonedge_auc={rec['random_nonedge_pairs_gt_auc']['roc_auc']}", flush=True)
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
