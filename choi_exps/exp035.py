"""exp035: Unsupervised graph-structure embeddings for future NAQ task sampling.

Train a GCN graph autoencoder to predict observed directed edges from node embeddings.
GT class labels and NAQ pseudo-tasks are not used. Export one vector per node.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from data import load_data
from utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="exp035 structural graph autoencoder")
    p.add_argument("--dataset", default="Amazon_clothing")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--latent", type=int, default=256)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--epoch", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_edges", type=int, default=8192)
    p.add_argument("--valid_ratio", type=float, default=0.1)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--negative_ratio", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--output_dir", default="results/Amazon_clothing/exp035")
    p.add_argument("--json_path", default="exp035.json")
    return p.parse_args()


def unique_edges(edge_index, n):
    edges = edge_index.cpu().numpy().T.astype(np.int64)
    edges = edges[edges[:, 0] != edges[:, 1]]
    codes = np.unique(edges[:, 0] * n + edges[:, 1])
    return np.column_stack((codes // n, codes % n)), codes


def draw_negatives(rng, count, n, forbidden):
    codes = []
    remaining = count
    while remaining:
        size = max(1024, int(remaining * 1.5))
        source = rng.integers(0, n, size=size, dtype=np.int64)
        target = rng.integers(0, n, size=size, dtype=np.int64)
        ids = source * n + target
        valid = (source != target) & ~np.isin(ids, forbidden, assume_unique=False)
        picked = ids[valid][:remaining]
        codes.append(picked)
        remaining -= len(picked)
    result = np.concatenate(codes)
    return np.column_stack((result // n, result % n))


def make_adj(edges, n, device):
    # Directed outgoing message passing as in the original edge_index convention.
    a = sp.csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    a = a + sp.eye(n, format="csr")
    a.data[:] = 1.0
    degree = np.asarray(a.sum(axis=1)).ravel()
    inv_sqrt = np.power(degree, -0.5, where=degree > 0)
    inv_sqrt[degree == 0] = 0
    a = sp.diags(inv_sqrt) @ a @ sp.diags(inv_sqrt)
    coo = a.tocoo()
    indices = torch.as_tensor(np.stack((coo.row, coo.col)), dtype=torch.long, device=device)
    values = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


class StructuralGAE(nn.Module):
    def __init__(self, features_dim, hidden, latent, dropout):
        super().__init__()
        self.conv1 = nn.Linear(features_dim, hidden, bias=False)
        self.conv2 = nn.Linear(hidden, latent, bias=False)
        self.dropout = dropout
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, adj):
        h = torch.sparse.mm(adj, x)
        h = F.relu(self.conv1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.sparse.mm(adj, h)
        return self.conv2(h)

    def predict(self, z, pairs, temperature):
        # Cosine decoder: prevents embedding-norm escalation as a shortcut.
        normalized = F.normalize(z, dim=-1)
        return self.scale.exp().clamp(max=100) * (normalized[pairs[:, 0]] * normalized[pairs[:, 1]]).sum(dim=-1) / temperature


def evaluate(model, x, adj, pos, neg, temperature, device):
    model.eval()
    with torch.no_grad():
        z = model(x, adj)
        positive = model.predict(z, torch.as_tensor(pos, device=device), temperature).cpu().numpy()
        negative = model.predict(z, torch.as_tensor(neg, device=device), temperature).cpu().numpy()
        truth = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))
        logits = np.concatenate((positive, negative))
        auc = roc_auc_score(truth, logits)
        ap = average_precision_score(truth, logits)
        ids = np.random.default_rng(123).choice(z.shape[0], size=min(1024, z.shape[0]), replace=False)
        vectors = F.normalize(z[torch.as_tensor(ids, device=device)], dim=-1)
        cos = vectors @ vectors.T
        tri = torch.triu_indices(len(ids), len(ids), offset=1, device=device)
        cosine_mean = float(cos[tri[0], tri[1]].mean()) if len(ids) > 1 else 0.0
        norm = z.norm(dim=-1)
    return {
        "auc": float(auc),
        "ap": float(ap),
        "positive_logit_mean": float(positive.mean()),
        "negative_logit_mean": float(negative.mean()),
        "embedding_norm_mean": float(norm.mean()),
        "embedding_norm_std": float(norm.std(unbiased=False)),
        "random_pair_cosine": cosine_mean,
    }


def main():
    args = parse_args()
    args.setting = "unsup"
    if not 0 < args.valid_ratio < 1 or args.negative_ratio < 1 or args.batch_edges < 1 or args.temperature <= 0:
        raise ValueError("Invalid split/training hyperparameters")
    torch.set_num_threads(4)
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    data = load_data(args)
    x = data.features.to(device)
    n = x.shape[0]
    all_edges, forbidden = unique_edges(data.edge_index, n)
    perm = rng.permutation(len(all_edges))
    num_valid = max(1, int(len(perm) * args.valid_ratio))
    train_edges = all_edges[perm[num_valid:]]
    valid_edges = all_edges[perm[:num_valid]]
    if len(train_edges) == 0:
        raise ValueError("Not enough edges to split")
    valid_neg = draw_negatives(rng, len(valid_edges), n, forbidden)
    train_adj = make_adj(train_edges, n, device)
    model = StructuralGAE(x.size(-1), args.hidden, args.latent, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = []
    best_auc = -1.0
    best_epoch = 0
    bad = 0
    start = time.time()
    for epoch in range(1, args.epoch + 1):
        model.train()
        pos_idx = rng.integers(0, len(train_edges), size=min(args.batch_edges, len(train_edges)))
        pos = train_edges[pos_idx]
        neg = draw_negatives(rng, len(pos) * args.negative_ratio, n, forbidden)
        z = model(x, train_adj)
        pos_score = model.predict(z, torch.as_tensor(pos, device=device), args.temperature)
        neg_score = model.predict(z, torch.as_tensor(neg, device=device), args.temperature)
        loss = F.softplus(-pos_score).mean() + F.softplus(neg_score).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % args.eval_interval != 0 and epoch != args.epoch:
            continue
        metrics = evaluate(model, x, train_adj, valid_edges, valid_neg, args.temperature, device)
        row = {"epoch": epoch, "loss": float(loss.detach()), **metrics}
        trajectory.append(row)
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_epoch = epoch
            bad = 0
            torch.save({"model": model.state_dict(), "config": vars(args), "epoch": epoch}, output_dir / "best.pt")
        else:
            bad += 1
        print(
            f"[exp035][{epoch}] loss={row['loss']:.4f} val_AUC={metrics['auc']:.4f} "
            f"val_AP={metrics['ap']:.4f} cos={metrics['random_pair_cosine']:.3f} "
            f"norm={metrics['embedding_norm_mean']:.3f} best={best_auc:.4f}@{best_epoch}",
            flush=True,
        )
        if bad >= args.patience:
            break

    ckpt = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    # Export full-graph embeddings for later unsupervised task construction.
    # Full adjacency can include held-out validation edges AFTER checkpoint selection.
    full_adj = make_adj(all_edges, n, device)
    with torch.no_grad():
        z = model(x, full_adj).cpu()
    torch.save(
        {"embeddings": z, "node_ids": torch.arange(n), "dataset": args.dataset, "method": "GCN-GAE directed link prediction", "epoch": best_epoch},
        output_dir / "embeddings.pt",
    )
    np.save(output_dir / "embeddings.npy", z.numpy())
    report = {
        "experiment": "exp035",
        "description": "Unsupervised structure-reconstructing node embeddings",
        "config": vars(args),
        "protocol": {
            "labels_used": False,
            "objective": "Observed directed edge vs sampled nonedge binary link prediction",
            "validation": "Held-out 10% directed edges; excluded from training adjacency",
            "negative_sampling": "Sample ordered node pairs absent from the full original edge_index",
            "embedding_export": "GCN encoder on full graph after selection; node index matches dataset",
            "note": "This is topology prediction from embeddings, not an explicit decoded PPR subgraph",
        },
        "graph": {"nodes": n, "edges_total": len(all_edges), "train_edges": len(train_edges), "valid_edges": len(valid_edges)},
        "trajectory": trajectory,
        "best": {"epoch": best_epoch, "valid_auc": best_auc, "checkpoint": str(output_dir / "best.pt")},
        "embedding_path": str(output_dir / "embeddings.pt"),
        "elapsed_seconds": round(time.time() - start, 2),
    }
    with Path(args.json_path).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    (output_dir / "result.txt").write_text(
        f"Best epoch: {best_epoch}\nHeld-out edge AUC: {best_auc:.4f}\n" f"Embeddings: {output_dir / 'embeddings.pt'}\n", encoding="utf-8"
    )
    print(f"Saved embeddings: {output_dir / 'embeddings.pt'} shape={tuple(z.shape)}", flush=True)


if __name__ == "__main__":
    main()
