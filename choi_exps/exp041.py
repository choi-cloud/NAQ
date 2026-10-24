"""exp041: original GREET discriminator + two-channel alternating training.

Requires original NAQ repository (data.py/task_generator.py/utils.py) and the
bundled exp041_greet directory (original GREET model/utils, import isolated).
Uses original Edge_Discriminator and original SGC encoders; replaces the
original dense N x N contrastive masks with a sparse mini-batch equivalent.
GT is never used for representation learning, edge discrimination or pruning.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import kneighbors_graph
import time
from data import load_data
from task_generator import spt_sampling
from utils import seed_everything
from exp041_greet.model import Edge_Discriminator, GCL, sparse_edge_dropout


def args_parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Amazon_clothing")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--cl_rounds", type=int, default=2)
    p.add_argument("--lr_gcl", type=float, default=0.001)
    p.add_argument("--lr_disc", type=float, default=0.001)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--cl_batch_size", type=int, default=1024)
    p.add_argument("--margin_hom", type=float, default=0.5)
    p.add_argument("--margin_het", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--str_dim", type=int, default=16)
    p.add_argument("--cache_dir", default="results/Amazon_clothing/exp041/cache")
    p.add_argument("--heldout_fraction", type=float, default=0.1)
    p.add_argument("--eval_freq", type=int, default=20)
    p.add_argument("--num_tasks", type=int, default=200)
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--pool_sizes", default="200,1000")
    p.add_argument("--remove_fractions", default="0.25,0.5,0.75,0.9")
    p.add_argument("--query_count", type=int, default=10)
    p.add_argument("--output_dir", default="results/Amazon_clothing/exp041")
    return p.parse_args()


def symmetric_edges(edge_index, n):
    arr = edge_index.detach().cpu().numpy()
    a = sp.csr_matrix((np.ones(arr.shape[1], dtype=np.float32), (arr[0], arr[1])), shape=(n, n))
    a = ((a + a.T) > 0).astype(np.float32).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    coo = sp.triu(a, 1).tocoo()
    return np.stack((coo.row, coo.col), axis=1).astype(np.int64)


def split_edges(pairs, frac, rng):
    order = rng.permutation(len(pairs))
    n_hold = max(1, round(frac * len(pairs)))
    return pairs[order[n_hold:]], pairs[order[:n_hold]]


def directed_tensor(pairs, device):
    both = np.concatenate((pairs, pairs[:, ::-1]), axis=0)
    return torch.as_tensor(both.T.copy(), dtype=torch.long, device=device)


def random_walk_se(pairs, n, steps):
    """GREET's diagonal T^k positional encoding; sparse, training graph only."""
    both = np.concatenate((pairs, pairs[:, ::-1]), axis=0)
    a = sp.csr_matrix((np.ones(len(both), dtype=np.float64), (both[:, 0], both[:, 1])), shape=(n, n))
    degree = np.asarray(a.sum(0)).ravel()
    invdegree = 1 / np.maximum(degree, 1)
    t = (a @ sp.diags(invdegree)).tocsr()
    m = t
    cols = []
    for idx in range(steps):
        cols.append(m.diagonal().astype(np.float32))
        if idx + 1 < steps:
            m = (m @ t).tocsr()
    return torch.from_numpy(np.stack(cols, axis=1))


class ScalableGREET(GCL):
    """Original two SGC branches and projection heads, sparse batch loss."""

    def set_sparse_knn(self, x, k):
        # No dense N x N mask (original implementation materializes two).
        self.knn_csr = kneighbors_graph(x.cpu().numpy(), k, metric="cosine").tocsr()
        self.knn_csr.setdiag(1)
        self.knn_csr.eliminate_zeros()

    def batch_nce_loss(self, z1, z2, temperature=0.2, pos_mask=None, neg_mask=None):
        n = z1.size(0)
        order = torch.randperm(n).tolist()
        losses = []
        for start in range(0, n, self.batch_size):
            ids = order[start : start + self.batch_size]
            mask_np = self.knn_csr[ids, :][:, ids].toarray() > 0
            pm = torch.as_tensor(mask_np, device=z1.device)
            nm = ~pm
            za = F.normalize(z1[ids], dim=1)
            zb = F.normalize(z2[ids], dim=1)

            # Original GREET: denominator consists of negatives only, and
            # numerator sums over selected feature kNN positive pairs.
            def directional(a, b):
                sim = (a @ b.T) / temperature
                exp_sim = sim.exp()
                denom = (exp_sim * nm).sum(1).clamp_min(1e-10)
                per_pair = sim - denom.log().unsqueeze(1)
                return -((per_pair * pm).sum(1) / pm.sum(1).clamp_min(1)).mean()

            losses.append((directional(za, zb) + directional(zb, za)) * 0.5 * (len(ids) / n))
        return sum(losses)


def diagnostic(discriminator, features, pairs, labels, device, name):
    with torch.no_grad():
        emb = discriminator.get_node_embedding(features)
        values = []
        for batch in np.array_split(pairs, max(1, (len(pairs) + 8191) // 8192)):
            idx = torch.as_tensor(batch.T.copy(), device=device)
            logits = discriminator.get_edge_weight(emb, idx)
            values.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(values)
    truth = labels[pairs[:, 0]] == labels[pairs[:, 1]]
    return dict(
        name=name,
        pairs=len(pairs),
        gt_same_rate=float(truth.mean()),
        auc=float(roc_auc_score(truth, probs)) if np.unique(truth).size == 2 else None,
        ap=float(average_precision_score(truth, probs)),
        same_mean=float(probs[truth].mean()) if truth.any() else None,
        different_mean=float(probs[~truth].mean()) if (~truth).any() else None,
        prob_min=float(probs.min()),
        prob_max=float(probs.max()),
    )


def sample_nonedges(n, train_pairs, held_pairs, count, rng):
    all_edges = np.concatenate((train_pairs, held_pairs))
    edge_set = {int(u) * n + int(v) for u, v in all_edges}
    edge_set.update(int(v) * n + int(u) for u, v in all_edges)
    out = []
    while len(out) < count:
        ps = rng.integers(n, size=(count, 2))
        out.extend((int(u), int(v)) for u, v in ps if u != v and int(u) * n + int(v) not in edge_set)
    return np.asarray(out[:count], dtype=np.int64)


def filter_diagnostic(discriminator, features, raw_feat, labels, supports, pools, rates, q, device, rng):
    results = {str(p): {str(r): [] for r in [0.0, *rates]} for p in pools}
    with torch.no_grad():
        node_emb = discriminator.get_node_embedding(features)
        for i, support in enumerate(tqdm(supports, desc="Negative Filtering", unit="support")):
            similarities = raw_feat @ raw_feat[support]
            similarities[support] = -float("inf")
            max_pool = max(pools)
            candidates = torch.topk(similarities, k=max_pool).indices
            scores = []
            for block in candidates.split(8192):
                pair = torch.stack((torch.full_like(block, support), block))
                scores.append(discriminator.get_edge_weight(node_emb, pair))
            scores = torch.cat(scores)
            for pool_size in pools:
                candidates_p = candidates[:pool_size]
                scores_p = scores[:pool_size]
                correct = torch.as_tensor(labels[candidates_p.cpu().numpy()] == labels[support], device=device)
                base_rate = correct.float().mean().item()
                for rate in [0.0, *rates]:
                    n_remove = int(round(pool_size * rate))
                    # Lowest homophily scores are the strongest negative candidates.
                    rank = torch.argsort(scores_p, descending=False)
                    removed = rank[:n_remove]
                    remain = rank[n_remove:]
                    sorted_remaining = remain[torch.argsort(similarities[candidates_p[remain]], descending=True)]
                    mid = max(0, (len(sorted_remaining) - q) // 2)
                    query = sorted_remaining[mid : mid + q]
                    value = dict(
                        candidate_precision=base_rate,
                        removed_negative_precision=(~correct[removed]).float().mean().item() if n_remove else None,
                        negative_removal_rate=((~correct[removed]).sum() / (~correct).sum().clamp_min(1)).item(),
                        positive_retention_rate=(correct[remain].sum() / correct.sum().clamp_min(1)).item(),
                        remaining_precision=correct[remain].float().mean().item(),
                        middle_query_precision=correct[query].float().mean().item(),
                        remaining_size=len(remain),
                    )
                    results[str(pool_size)][str(rate)].append(value)
            if (i + 1) % 100 == 0:
                print(f"[exp041] scored {i+1}/{len(supports)} supports", flush=True)
    return {
        pool: {
            rate: {
                key: float(np.mean([x[key] for x in vals if x[key] is not None])) if any(x[key] is not None for x in vals) else None
                for key in vals[0]
            }
            for rate, vals in tab.items()
        }
        for pool, tab in results.items()
    }


def main():
    a = args_parse()
    a.setting, a.type, a.query_generation = "unsup", "feature", "NaQ"
    torch.set_num_threads(4)
    seed_everything(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device(f"cuda:{a.device}" if torch.cuda.is_available() else "cpu")

    preprocess_start = time.perf_counter()
    print("[Preprocessing] Started", flush=True)

    data = load_data(a)
    labels = np.asarray(data.labels.cpu()).reshape(-1)
    raw = F.normalize(data.features.float().to(device), dim=1)
    n, nf = raw.shape
    pairs = symmetric_edges(data.edge_index, n)
    train_edges, held_edges = split_edges(pairs, a.heldout_fraction, rng)
    graph_edges = directed_tensor(train_edges, device)
    cache_dir = Path(a.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    graph_hash = hashlib.sha256(train_edges.tobytes()).hexdigest()[:16]
    se_file = cache_dir / f"se_n{n}_d{a.str_dim}_{graph_hash}.pt"
    t0 = time.perf_counter()
    if se_file.exists():
        structural = torch.load(se_file, map_location="cpu", weights_only=True)
        print(f"[Preprocessing] Loaded SE cache: {se_file}", flush=True)
    else:
        print("[Preprocessing] Computing exact RW structural encoding...", flush=True)
        structural = random_walk_se(train_edges, n, a.str_dim)
        torch.save(structural, se_file)
    structural = structural.to(device)
    print(f"[Preprocessing] Random Walk SE: {time.perf_counter() - t0:.2f}s", flush=True)
    inp = torch.cat((raw, structural), dim=1)
    model = ScalableGREET(
        nlayers=2, nlayers_proj=1, in_dim=nf, emb_dim=a.emb_dim, proj_dim=a.proj_dim, dropout=0.5, sparse=1, batch_size=a.cl_batch_size
    ).to(device)
    t0 = time.perf_counter()
    # Same raw features and dataset fingerprint -> same kNN cache.
    feat_hash = hashlib.sha256(raw.detach().cpu().numpy().tobytes()).hexdigest()[:16]
    knn_file = cache_dir / f"knn_n{n}_k{a.k}_{feat_hash}.npz"
    if knn_file.exists():
        model.knn_csr = sp.load_npz(knn_file).tocsr()
        print(f"[Preprocessing] Loaded kNN cache: {knn_file}", flush=True)
    else:
        print("[Preprocessing] Computing feature kNN...", flush=True)
        model.set_sparse_knn(raw.cpu(), a.k)
        sp.save_npz(knn_file, model.knn_csr)
    print(f"[Preprocessing] Feature kNN: {time.perf_counter() - t0:.2f}s", flush=True)
    disc = Edge_Discriminator(n, nf + a.str_dim, a.alpha, sparse=1).to(device)
    oc = torch.optim.Adam(model.parameters(), lr=a.lr_gcl)
    od = torch.optim.Adam(disc.parameters(), lr=a.lr_disc)

    preprocess_elapsed = time.perf_counter() - preprocess_start

    print(f"[Preprocessing] Completed in {preprocess_elapsed:.2f}s " f"({preprocess_elapsed / 60:.2f}min)", flush=True)

    logs = []
    print(f"[exp041] nodes={n} edges(train/heldout)={len(train_edges)}/{len(held_edges)} device={device}", flush=True)
    for epoch in tqdm(range(1, a.epochs + 1), desc="GREET Training", unit="epoch"):
        model.train()
        disc.eval()
        disc.requires_grad_(False)
        for _ in range(a.cl_rounds):
            adj1, adj2, _, _ = disc(inp, graph_edges)
            # Original GREET feature masks and edge dropout, per view.
            mask1 = torch.rand(nf, device=device) < 0.1
            mask2 = torch.rand(nf, device=device) < 0.5
            x1, x2 = raw.clone(), raw.clone()
            x1[:, mask1] = 0
            x2[:, mask2] = 0
            # DGL edge dropout on weights, not explicit edge removal.
            adj1 = sparse_edge_dropout(adj1, 0.5)
            adj2 = sparse_edge_dropout(adj2, 0.1)
            lc = model(x1, adj1, x2, adj2)
            oc.zero_grad()
            lc.backward()
            oc.step()
        disc.requires_grad_(True)
        model.requires_grad_(False)
        model.eval()
        disc.train()
        adj1, adj2, wl, wh = disc(inp, graph_edges)
        embedding = model.get_embedding(raw, adj1, adj2)
        edge_sim = F.cosine_similarity(embedding[graph_edges[0]], embedding[graph_edges[1]])
        pivot = torch.randint(n, size=graph_edges.shape, device=device)
        rnd_sim = F.cosine_similarity(embedding[pivot[0]], embedding[pivot[1]])
        positive = torch.ones_like(edge_sim)
        lp = F.margin_ranking_loss(edge_sim, rnd_sim, positive, margin=a.margin_hom, reduction="none")
        lh = F.margin_ranking_loss(rnd_sim, edge_sim, positive, margin=a.margin_het, reduction="none")
        lr = ((lp * F.relu(wl - 0.5)).mean() + (lh * F.relu(wh - 0.5)).mean()) / 2
        od.zero_grad()
        lr.backward()
        od.step()
        model.requires_grad_(True)
        if epoch % a.eval_freq == 0 or epoch == 1 or epoch == a.epochs:
            logs.append(dict(epoch=epoch, cl_loss=float(lc.item()), rank_loss=float(lr.item())))
            print(f"[exp041] epoch={epoch} cl={lc.item():.4f} rank={lr.item():.4f}", flush=True)
    disc.eval()
    random_pairs = sample_nonedges(n, train_edges, held_edges, len(held_edges), rng)
    result = dict(
        experiment="exp041",
        variant="GREET edge discriminator + dual-channel SGC / PyTorch sparse backend",
        config=vars(a),
        diagnostic_training_edges=diagnostic(disc, inp, train_edges, labels, device, "train_observed_edges"),
        diagnostic_heldout_edges=diagnostic(disc, inp, held_edges, labels, device, "heldout_edges_removed_from_training_graph"),
        diagnostic_random_nonedges=diagnostic(disc, inp, random_pairs, labels, device, "random_nonedges"),
        training=logs,
    )
    print("[exp041] diagnostics", result["diagnostic_heldout_edges"], result["diagnostic_random_nonedges"], flush=True)
    supports = []
    for _ in range(a.num_tasks):
        support_ids, _ = spt_sampling(data.id_train, a.n_way, 1)
        supports.extend(np.asarray(support_ids).reshape(-1).astype(int).tolist())
    supports = list(dict.fromkeys(supports))
    pool_sizes = [int(v) for v in a.pool_sizes.split(",")]
    fractions = [float(v) for v in a.remove_fractions.split(",")]
    result["num_supports"] = len(supports)
    result["pool_results"] = filter_diagnostic(disc, inp, raw, labels, supports, pool_sizes, fractions, a.query_count, device, rng)
    result["notes"] = [
        "No labels used to train discriminator, construct graph or select queries.",
        "Heldout graph edges removed before training and structural encoding.",
        "Raw homophily logits used for ranking (not stochastic Gumbel samples).",
        "Sparse kNN mini-batch contrastive objective approximates original full graph GREET.",
        "PyTorch sparse graph backend replaces DGL (same low/high propagation formulas).",
        "Middle-K diagnostic is not GFSL downstream accuracy.",
    ]
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "exp041.json").write_text(json.dumps(result, indent=2))
    torch.save(dict(discriminator=disc.state_dict(), encoder=model.state_dict(), args=vars(a)), out / "exp041.pt")
    print("[exp041] saved", str(out / "exp041.json"), flush=True)


if __name__ == "__main__":
    main()
