"""exp061: Pseudo-supervised pairwise homophily pre-training for NAQ.

Only X and A are used to form pseudo labels and train the model. Ground-truth
labels are accessed exclusively for diagnostics. Original NAQ files untouched.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import normalize

from data import load_data


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
    return p.parse_args()


def save_json(path, obj):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def get_edges(data):
    edges = data.edge_index.detach().cpu().numpy()
    u = np.minimum(edges[0], edges[1]).astype(np.int64)
    v = np.maximum(edges[0], edges[1]).astype(np.int64)
    valid = u != v
    pairs = np.unique(np.column_stack([u[valid], v[valid]]), axis=0)
    return pairs


def sparse_adjacency(data, n):
    adj = data.adj.coalesce().cpu()
    ij = adj.indices().numpy()
    return sp.coo_matrix((adj.values().numpy(), (ij[0], ij[1])), shape=(n, n)).tocsr()


def torch_adj(data, device):
    return data.adj.coalesce().to(device)


def fit_clusters(features, args):
    model = MiniBatchKMeans(
        n_clusters=args.k, random_state=args.seed, batch_size=args.cluster_batch_size, max_iter=args.cluster_max_iter, n_init=args.cluster_n_init
    )
    return model.fit_predict(normalize(features, norm="l2", copy=True))


def sample_negatives(rng, n, count, cx, ca, edge_keys, max_attempts=100):
    result = []
    remaining = count
    for _ in range(max_attempts):
        if remaining == 0:
            break
        batch = max(remaining * 4, 2048)
        u = rng.integers(0, n, size=batch, dtype=np.int64)
        v = rng.integers(0, n, size=batch, dtype=np.int64)
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        mask = (lo != hi) & (cx[lo] != cx[hi]) & (ca[lo] != ca[hi])
        keys = lo * n + hi
        mask &= ~np.isin(keys, edge_keys, assume_unique=False)
        pair = np.column_stack([lo[mask], hi[mask]])[:remaining]
        if len(pair):
            result.append(pair)
            remaining -= len(pair)
    if remaining:
        raise RuntimeError(f"Negative pair sampling failed, remaining={remaining}")
    return np.concatenate(result, axis=0)


class GraphPairModel(nn.Module):
    def __init__(self, input_dim, hidden, embedding_dim, dropout):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden)
        self.lin2 = nn.Linear(hidden, embedding_dim)
        self.pair_head = nn.Sequential(nn.Linear(embedding_dim * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.dropout = dropout

    def encode(self, features, adj):
        h = F.dropout(features, self.dropout, training=self.training)
        h = torch.sparse.mm(adj, h)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.lin2(torch.sparse.mm(adj, h))
        return F.normalize(h, dim=-1)

    def pair_logits(self, z, pairs):
        a = z[pairs[:, 0]]
        b = z[pairs[:, 1]]
        inp = torch.cat([torch.abs(a - b), a * b], dim=1)
        return self.pair_head(inp).squeeze(-1)


def pair_scores(model, embeddings, pairs, device, batch_size):
    results = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = torch.as_tensor(pairs[start : start + batch_size], dtype=torch.long, device=device)
            results.append(torch.sigmoid(model.pair_logits(embeddings, batch)).cpu().numpy())
    return np.concatenate(results) if results else np.empty(0)


def candidate_precision(model, z, x_sparse, y, anchors, edge_keys, device, batch_size):
    n = len(y)
    xnorm = np.sqrt(np.asarray(x_sparse.multiply(x_sparse).sum(axis=1)).reshape(-1))
    results = {"model_full": [], "cosine_full": [], "model_nonedge": [], "cosine_nonedge": []}
    edge_selected = []
    for anchor in anchors:
        candidates = np.arange(n, dtype=np.int64)
        candidates = candidates[candidates != anchor]
        pairs = np.column_stack([np.full(len(candidates), anchor, dtype=np.int64), candidates])
        scores = pair_scores(model, z, pairs, device, batch_size)
        sims = (x_sparse @ x_sparse[anchor].T).toarray().reshape(-1)
        sims = sims[candidates] / np.maximum(xnorm[anchor] * xnorm[candidates], 1e-12)
        near = np.minimum(candidates, anchor) * n + np.maximum(candidates, anchor)
        is_edge = np.isin(near, edge_keys)
        edge_selected.append(float(is_edge[np.argsort(-scores, kind="stable")[:10]].mean()))
        for label, mask in [("full", np.ones(len(candidates), dtype=bool)), ("nonedge", ~is_edge)]:
            ids = np.flatnonzero(mask)
            if len(ids) < 10:
                continue
            for method, values in [("model", scores), ("cosine", sims)]:
                chosen = ids[np.argsort(-values[ids], kind="stable")[:10]]
                results[f"{method}_{label}"].append(float(np.mean(y[candidates[chosen]] == y[anchor])))
    return {key + "_p_at_10": float(np.mean(val)) if val else None for key, val in results.items()} | {
        "model_full_selected_edge_fraction": float(np.mean(edge_selected)),
        "anchors": len(anchors),
    }


def main():
    args = parse_args()
    if args.dataset != "Cora_full":
        raise ValueError("exp061 is configured for Cora_full")
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

    print("Fitting exp060 raw_spherical clusters...", flush=True)
    cx = fit_clusters(x, args)
    ca = fit_clusters(ax, args)
    pos_mask = (cx[edges[:, 0]] == cx[edges[:, 1]]) & (ca[edges[:, 0]] == ca[edges[:, 1]])
    positives = edges[pos_mask]
    if not len(positives):
        raise RuntimeError("No pseudo-positive pairs")
    print(f"Pseudo-positive pairs={len(positives)}", flush=True)

    # SVD compression is input preprocessing for the GNN encoder, NOT pseudo-label construction.
    views = []
    for name, mat in [("X", x), ("AX", ax)]:
        d = min(args.svd_dim, min(mat.shape) - 1)
        svd = TruncatedSVD(n_components=d, n_iter=5, random_state=args.seed)
        reduced = svd.fit_transform(mat).astype(np.float32)
        views.append(normalize(reduced, norm="l2", copy=True))
        print(f"Input {name}: {d} SVD dimensions", flush=True)
    features = torch.as_tensor(np.concatenate(views, axis=1), dtype=torch.float32, device=device)
    adj = torch_adj(data, device)
    model = GraphPairModel(features.shape[1], args.hidden, args.embed_dim, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # GT read only in diagnostics; never used by fitting, pseudo-labeling, or optimization.
    pseudo_neg_eval = sample_negatives(rng, n, len(positives), cx, ca, edge_keys)
    heldout_pos = positives[rng.choice(len(positives), size=min(args.eval_pairs // 2, len(positives)), replace=False)]
    heldout_neg = sample_negatives(rng, n, len(heldout_pos), cx, ca, edge_keys)
    gt_eval_pairs = np.concatenate([heldout_pos, heldout_neg])
    gt_eval_targets = (y[gt_eval_pairs[:, 0]] == y[gt_eval_pairs[:, 1]]).astype(np.int64)
    random_eval = np.column_stack([rng.integers(0, n, size=args.eval_pairs), rng.integers(0, n, size=args.eval_pairs)])
    random_eval = random_eval[random_eval[:, 0] != random_eval[:, 1]]
    random_gt = (y[random_eval[:, 0]] == y[random_eval[:, 1]]).astype(np.int64)
    nonedge_eval = sample_negatives(rng, n, args.eval_pairs, cx, ca, edge_keys)
    # Independently sampled random nonedges without cluster restrictions:
    nonedge_random = []
    while sum(map(len, nonedge_random)) < args.eval_pairs:
        q = rng.integers(0, n, size=(args.eval_pairs, 2), dtype=np.int64)
        lo, hi = np.minimum(q[:, 0], q[:, 1]), np.maximum(q[:, 0], q[:, 1])
        valid = (lo != hi) & ~np.isin(lo * n + hi, edge_keys)
        nonedge_random.append(np.column_stack([lo[valid], hi[valid]]))
    nonedge_random = np.concatenate(nonedge_random, axis=0)[: args.eval_pairs]
    nonedge_gt = (y[nonedge_random[:, 0]] == y[nonedge_random[:, 1]]).astype(np.int64)
    anchors = rng.choice(n, size=min(args.eval_anchors, n), replace=False)

    report = {
        "experiment": "exp061",
        "status": "running",
        "args": vars(args),
        "design": {
            "positive": "1-hop edge AND same cluster in both X/raw_spherical and AX/raw_spherical",
            "negative": "non-edge AND different cluster in both X/raw_spherical and AX/raw_spherical",
            "uncertain": "other pairs omitted from training",
            "encoder": "2-layer GCN on concatenated SVD(X), SVD(AX)",
            "pair_head": "shared symmetric MLP(abs(z_i-z_j), z_i*z_j)",
            "gt": "evaluation only; NOT used to build pseudo labels, update parameters, or select checkpoints",
            "warning": "Pseudo-positive pairs are noisy; nonedge generalization must be measured independently",
        },
        "pseudo_label_diagnostics": {
            "positive_count": int(len(positives)),
            "positive_gt_same_rate": float(np.mean(y[positives[:, 0]] == y[positives[:, 1]])),
            "negative_gt_different_rate": float(np.mean(y[pseudo_neg_eval[:, 0]] != y[pseudo_neg_eval[:, 1]])),
            "edge_gt_same_rate": float(np.mean(y[edges[:, 0]] == y[edges[:, 1]])),
        },
        "history": [],
    }
    out = Path(__file__).resolve().parent / "exp061.json"

    for epoch in range(1, args.epochs + 1):
        model.train()
        npos = args.pairs_per_epoch // 2
        idx = rng.choice(len(positives), size=npos, replace=len(positives) < npos)
        ps = positives[idx]
        ns = sample_negatives(rng, n, npos, cx, ca, edge_keys)
        pairs = np.concatenate([ps, ns])
        targets = np.concatenate([np.ones(npos, dtype=np.float32), np.zeros(npos, dtype=np.float32)])
        perm = rng.permutation(len(pairs))
        pair_tensor = torch.as_tensor(pairs[perm], dtype=torch.long, device=device)
        target_tensor = torch.as_tensor(targets[perm], dtype=torch.float32, device=device)
        # Recompute the graph encoding once per optimizer step; mini-batch pair
        # chunks would require retain_graph and greater memory consumption.
        optimizer.zero_grad()
        z = model.encode(features, adj)
        logits = model.pair_logits(z, pair_tensor)
        loss = F.binary_cross_entropy_with_logits(logits, target_tensor)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                z = model.encode(features, adj)
            p = pair_scores(model, z, gt_eval_pairs, device, args.pair_batch)
            r = pair_scores(model, z, random_eval, device, args.pair_batch)
            ne = pair_scores(model, z, nonedge_random, device, args.pair_batch)

            def metrics(gt, scores):
                if np.unique(gt).size < 2:
                    return {"gt_base_rate": float(gt.mean()), "roc_auc": None, "average_precision": None}
                return {
                    "gt_base_rate": float(gt.mean()),
                    "roc_auc": float(roc_auc_score(gt, scores)),
                    "average_precision": float(average_precision_score(gt, scores)),
                }

            rec = {
                "epoch": epoch,
                "pseudo_bce": float(loss.item()),
                "pseudo_training_pool_gt_auc": metrics(gt_eval_targets, p),
                "random_pairs_gt_auc": metrics(random_gt, r),
                "random_nonedge_pairs_gt_auc": metrics(nonedge_gt, ne),
            }
            report["history"].append(rec)
            print(
                f"epoch={epoch} loss={loss.item():.4f} random_auc={rec['random_pairs_gt_auc']['roc_auc']} nonedge_auc={rec['random_nonedge_pairs_gt_auc']['roc_auc']}",
                flush=True,
            )
            save_json(out, report)

    model.eval()
    with torch.no_grad():
        z = model.encode(features, adj)
    print("Evaluating global Query P@10...", flush=True)
    report["query_evaluation"] = candidate_precision(model, z, x_sparse, y, anchors, edge_keys, device, args.pair_batch)
    report["status"] = "completed"
    save_json(out, report)
    torch.save({"model": model.state_dict(), "args": vars(args)}, Path(__file__).resolve().parent / "exp061.pt")
    print(f"DONE: {out}", flush=True)


if __name__ == "__main__":
    main()
