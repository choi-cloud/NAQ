"""exp069: GPS Sum / Union / Intersection contrastive pretraining on NAQ Cora-Full.

Positive mining uses GPS sampling definitions (GPS Sec IV-A). Training uses
multi-positive sampled-negative InfoNCE (memory-scalable approximation of
full-negative GPS Eq. 2). No ground truth in mining/training/checkpointing.
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
from sklearn.preprocessing import normalize

from data import load_data
from task_generator import spt_sampling
from exp068 import ppr_scores, hop_candidates, sorted_top

METHODS = ("node", "topology", "relation", "semantic")


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Cora_full")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--eval_episodes", type=int, default=200)
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--q_query", type=int, default=10)
    p.add_argument("--positive_k", type=int, default=10)
    p.add_argument("--min_hop", type=int, default=1)
    p.add_argument("--max_hop", type=int, default=2)
    p.add_argument("--ppr_alpha", type=float, default=0.15)
    p.add_argument("--ppr_iterations", type=int, default=40)
    p.add_argument("--ppr_tolerance", type=float, default=1e-7)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--edge_drop", type=float, default=0.2)
    p.add_argument("--feature_drop", type=float, default=0.2)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--train_anchors", type=int, default=1024)
    p.add_argument("--negative_count", type=int, default=256)
    p.add_argument("--positive_per_strategy", type=int, default=2)
    p.add_argument("--positive_per_fused", type=int, default=8)
    p.add_argument("--cache", default="exp069_gps_positives.npz")
    p.add_argument("--output", default="exp069.json")
    p.add_argument("--checkpoint_prefix", default="exp069")
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_naqloder(a):
    a.setting = "unsup"
    a.query_generation = "NaQ"
    a.type = "feature"
    a.k_shot = 1
    a.k_shot_test = 1
    a.epoch = 1
    return load_data(a)


def build_graph(data, n):
    edge = data.edge_index.detach().cpu().numpy()
    graph = sp.coo_matrix((np.ones(edge.shape[1]), (edge[0], edge[1])), shape=(n, n)).tocsr()
    graph = ((graph + graph.T) > 0).astype(np.float64).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def mine_sets(x_unit, graph, a, path):
    n = graph.shape[0]
    if path.is_file():
        archive = np.load(path, allow_pickle=False)
        keys = ("node", "topology", "relation", "semantic", "union", "intersection")
        if all(key in archive.files for key in keys):
            arrays = {key: archive[key] for key in keys}
            if all(val.shape[0] == n for val in arrays.values()):
                print(f"Loaded GPS positive cache: {path}", flush=True)
                return arrays
    deg = np.asarray(graph.sum(axis=1)).ravel()
    inv_deg = np.divide(1, deg, out=np.zeros_like(deg), where=deg > 0)
    walk_t = (sp.diags(inv_deg) @ graph).T.tocsr()
    weights = 1.0 / np.log(np.maximum(deg, 2.0))
    weighted_neighbors = sp.diags(weights) @ graph
    arr = {key: np.full((n, a.positive_k), -1, dtype=np.int32) for key in METHODS}
    rng = np.random.default_rng(a.seed + 6900)
    for i in range(n):
        allow = np.ones(n, dtype=bool)
        allow[i] = False
        ppr = ppr_scores(i, walk_t, a.ppr_alpha, a.ppr_iterations, a.ppr_tolerance)
        arr["node"][i] = sorted_top(ppr, allow, a.positive_k)
        hop, _ = hop_candidates(i, graph, a.min_hop, a.max_hop, ~allow)
        if len(hop):
            chosen = rng.choice(hop, size=min(a.positive_k, len(hop)), replace=False)
            arr["topology"][i, : len(chosen)] = chosen
        shared = graph.getrow(i) @ weighted_neighbors
        score = np.zeros(n, dtype=np.float64)
        if shared.nnz:
            score[shared.indices] = shared.data
        rel = sorted_top(score, allow & (score > 0), a.positive_k)
        arr["relation"][i, : len(rel)] = rel
        cos = np.asarray((x_unit.getrow(i) @ x_unit.T).toarray()).ravel()
        arr["semantic"][i] = sorted_top(cos, allow, a.positive_k)
        if (i + 1) % 1000 == 0:
            print(f"GPS mined {i + 1}/{n} nodes", flush=True)
    fused = {"union": np.full((n, a.positive_k * 4), -1, dtype=np.int32), "intersection": np.full((n, a.positive_k), -1, dtype=np.int32)}
    for i in range(n):
        sets = [set(arr[k][i][arr[k][i] >= 0].tolist()) for k in METHODS]
        union = sorted(set.union(*sets))
        # Original GPS experimental intersection uses node and topology only.
        inter = sorted(sets[0] & sets[1])
        fused["union"][i, : len(union)] = union
        fused["intersection"][i, : len(inter)] = inter
    arrays = {**arr, **fused}
    np.savez_compressed(path, **arrays)
    print(f"Saved GPS positive cache: {path}", flush=True)
    return arrays


class Encoder(nn.Module):
    def __init__(self, d, h, z, dropout):
        super().__init__()
        self.one = nn.Linear(d, h)
        self.two = nn.Linear(h, z)
        self.project = nn.Sequential(nn.Linear(z, z), nn.ELU(), nn.Linear(z, z))
        self.dropout = dropout

    def forward(self, x, a):
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.relu(self.one(torch.sparse.mm(a, h)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.two(torch.sparse.mm(a, h))
        return F.normalize(z, dim=1), F.normalize(self.project(z), dim=1)


def torch_graph(graph, device, rng, drop=0):
    coo = graph.tocoo()
    edges = np.column_stack((coo.row, coo.col))
    mask = rng.random(len(edges)) >= drop if drop > 0 else np.ones(len(edges), dtype=bool)
    edges = edges[mask]
    indices = torch.as_tensor(edges.T.copy(), dtype=torch.long, device=device)
    vals = torch.ones(len(edges), device=device)
    n = graph.shape[0]
    sparse = torch.sparse_coo_tensor(indices, vals, (n, n)).coalesce()
    diag = torch.arange(n, device=device)
    loops = torch.sparse_coo_tensor(torch.stack([diag, diag]), torch.ones(n, device=device), (n, n))
    sparse = (sparse + loops).coalesce()
    row, col = sparse.indices()
    degree = torch.zeros(n, device=device).scatter_add_(0, row, sparse.values())
    scale = degree.clamp_min(1).pow(-0.5)
    norm_vals = sparse.values() * scale[row] * scale[col]
    return torch.sparse_coo_tensor(sparse.indices(), norm_vals, (n, n)).coalesce()


def augment_features(x, fraction):
    if fraction <= 0:
        return x
    drop_cols = torch.rand(x.shape[1], device=x.device) < fraction
    return x.masked_fill(drop_cols[None, :], 0)


def sample_positive(pool, anchors, count, rng):
    out = np.full((len(anchors), count), -1, dtype=np.int64)
    valid = np.zeros(len(anchors), dtype=bool)
    for r, anchor in enumerate(anchors):
        candidates = pool[anchor]
        candidates = candidates[candidates >= 0]
        if len(candidates):
            out[r] = rng.choice(candidates, size=count, replace=True)
            valid[r] = True
    return out, valid


def nce_loss(online, target, anchor, positive, negatives, temperature):
    # All sampled negatives are from outside this anchor's union of GPS positives.
    # Positive loss is averaged over each sampled positive (GPS Eq.2 form).
    a = online[anchor]
    pos = target[positive]
    neg = torch.einsum("bd,bnd->bn", a, target[negatives]) / temperature
    positive_logits = torch.einsum("bd,bpd->bp", a, pos) / temperature
    negative_logsum = torch.logsumexp(neg, dim=1, keepdim=True)
    denominator = torch.logaddexp(positive_logits, negative_logsum)
    return (denominator - positive_logits).mean()


def sample_negatives(anchor_ids, union_pool, n, count, rng):
    out = rng.integers(0, n, size=(len(anchor_ids), count), dtype=np.int64)
    for r, anchor in enumerate(anchor_ids):
        forbidden = union_pool[anchor]
        forbidden = set(forbidden[forbidden >= 0].tolist())
        forbidden.add(int(anchor))
        bad = np.fromiter((v in forbidden for v in out[r]), dtype=bool, count=count)
        while np.any(bad):
            out[r, bad] = rng.integers(0, n, size=int(bad.sum()))
            bad = np.fromiter((v in forbidden for v in out[r]), dtype=bool, count=count)
    return out


def batch_objective(mode, z1, z2, pools, anchors, negatives, rng, a, device):
    losses = []
    for source in METHODS if mode == "sum" else (mode,):
        count = a.positive_per_strategy if mode == "sum" else a.positive_per_fused
        sampled, valid = sample_positive(pools[source], anchors, count, rng)
        if not valid.any():
            continue
        anc = torch.tensor(anchors[valid], dtype=torch.long, device=device)
        pos = torch.tensor(sampled[valid], dtype=torch.long, device=device)
        neg = torch.tensor(negatives[valid], dtype=torch.long, device=device)
        losses.append((nce_loss(z1, z2, anc, pos, neg, a.temperature) + nce_loss(z2, z1, anc, pos, neg, a.temperature)) / 2)
    if not losses:
        raise RuntimeError("No nonempty GPS positive sets")
    return torch.stack(losses).sum() if mode == "sum" else losses[0]


def choose_top10(scores, excluded, k):
    valid = ~excluded
    scores = scores.copy()
    scores[~valid] = -np.inf
    ids = np.flatnonzero(valid)
    return ids[np.lexsort((ids, -scores[ids]))[:k]]


def evaluate_embeddings(z, x_unit, labels, episodes, graph, k):
    z = normalize(z)
    n = len(z)
    scores = z @ z.T if n <= 10000 else None
    cosine_rows = []
    learned_rows = []
    learned_nonedge = []
    cosine_nonedge = []
    for support in episodes:
        excluded = np.zeros(n, dtype=bool)
        excluded[support] = True
        for anchor in support:
            anchor = int(anchor)
            s = scores[anchor] if scores is not None else z @ z[anchor]
            raw = (x_unit.getrow(anchor) @ x_unit.T).toarray().ravel()
            eids = graph.indices[graph.indptr[anchor] : graph.indptr[anchor + 1]]
            noedge_ex = excluded.copy()
            noedge_ex[eids] = True
            for vals, full, nonedge in ((s, learned_rows, learned_nonedge), (raw, cosine_rows, cosine_nonedge)):
                chosen = choose_top10(vals, excluded, k)
                full.append(np.mean(labels[chosen] == labels[anchor]))
                chosen_ne = choose_top10(vals, noedge_ex, k)
                nonedge.append(np.mean(labels[chosen_ne] == labels[anchor]))
    return {
        "model_full_p_at_10": float(np.mean(learned_rows)),
        "cosine_full_p_at_10": float(np.mean(cosine_rows)),
        "model_nonedge_p_at_10": float(np.mean(learned_nonedge)),
        "cosine_nonedge_p_at_10": float(np.mean(cosine_nonedge)),
        "anchor_instances": len(learned_rows),
    }


def write_json(path, output):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def main():
    a = arguments()
    if a.positive_k < 1 or a.negative_count < 1 or a.train_anchors < 1:
        raise ValueError("Positive, negative and anchor counts must be >= 1")
    seed_everything(a.seed)
    data = setup_naqloder(a)
    x = data.features.detach().cpu().numpy().astype(np.float32)
    labels = data.labels.detach().cpu().numpy().ravel()
    n = len(labels)
    graph = build_graph(data, n)
    x_unit = normalize(sp.csr_matrix(x), norm="l2", axis=1)
    root = Path(__file__).resolve().parent
    pools = mine_sets(x_unit, graph, a, root / a.cache)
    episodes = [np.asarray(spt_sampling(data.id_train, a.n_way, 1)[0], dtype=np.int64) for _ in range(a.eval_episodes)]
    device = torch.device(f"cuda:{a.device}" if a.device >= 0 and torch.cuda.is_available() else "cpu")
    feature = torch.as_tensor(x, device=device)
    raw_adj = torch_graph(graph, device, np.random.default_rng(a.seed), drop=0)
    output = {
        "experiment": "exp069",
        "status": "running",
        "args": vars(a),
        "design": {
            "positive_sets": "GPS node PPR / topology hops / relation common neighbors / semantic raw cosine",
            "sum": "Sum four per-strategy multi-positive InfoNCE objectives",
            "union": "Union of all four GPS sets",
            "intersection": "PPR intersection topology ONLY, as in GPS paper experiments",
            "self_positive": "Include anchor itself in every strategy positive pool",
            "deviation": "Sampled-negative InfoNCE, not the original full-N GPS InfoNCE due to memory; raw adjacency symmetrized and GCN-normalized",
            "gt": "Diagnostics only; never training, positive mining or checkpoint selection",
        },
        "graph": {"nodes": n, "classes": int(np.unique(labels).size), "edges": int(graph.nnz // 2)},
        "positive_pool_stats": {
            key: {"avg_size": float((val >= 0).sum(axis=1).mean()), "empty_nodes": int(((val >= 0).sum(axis=1) == 0).sum())}
            for key, val in pools.items()
        },
        "results": {},
    }
    write_json(root / a.output, output)
    sampled_pools = {key: np.concatenate([np.arange(n, dtype=np.int32)[:, None], arr], axis=1) for key, arr in pools.items()}
    for mode in ("sum", "union", "intersection"):
        seed_everything(a.seed + 1)  # identical model initialization across modes
        rng = np.random.default_rng(a.seed + 6901)  # identical anchor schedules
        model = Encoder(x.shape[1], a.hidden, a.embedding_dim, a.dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        history = []
        for epoch in range(1, a.epochs + 1):
            model.train()
            anchors = rng.choice(n, size=min(n, a.train_anchors), replace=False)
            negatives = sample_negatives(anchors, pools["union"], n, a.negative_count, rng)
            adj1 = torch_graph(graph, device, rng, a.edge_drop)
            adj2 = torch_graph(graph, device, rng, a.edge_drop)
            _, z1 = model(augment_features(feature, a.feature_drop), adj1)
            _, z2 = model(augment_features(feature, a.feature_drop), adj2)
            # Self-positive always available, including empty intersection cases.
            loss = batch_objective(mode, z1, z2, sampled_pools, anchors, negatives, rng, a, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch == 1 or epoch % a.eval_every == 0 or epoch == a.epochs:
                model.eval()
                with torch.no_grad():
                    embeddings, _ = model(feature, raw_adj)
                ev = evaluate_embeddings(embeddings.cpu().numpy(), x_unit, labels, episodes, graph, a.q_query)
                history.append({"epoch": epoch, "loss": float(loss.item()), "query": ev})
                print(
                    f"[{mode}] epoch={epoch} loss={loss.item():.4f} full={ev['model_full_p_at_10']:.4f} nonedge={ev['model_nonedge_p_at_10']:.4f}",
                    flush=True,
                )
                output["results"][mode] = {"history": history, "final": ev}
                write_json(root / a.output, output)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hidden": a.hidden,
                "embedding_dim": a.embedding_dim,
                "input_dim": x.shape[1],
                "seed": a.seed,
                "mode": mode,
            },
            root / f"{a.checkpoint_prefix}_{mode}.pt",
        )
    output["status"] = "completed"
    write_json(root / a.output, output)
    print(f"Saved {root / a.output}", flush=True)


if __name__ == "__main__":
    main()
