"""HW04m: graph-structure quantities for train-free positive-candidate measurement.

Not imported by any training code (hw/run.py, losses, neighbors, ...).
All functions are deterministic (no RNG). Graph input: scipy.sparse CSR adjacency A that is
undirected, without self-loops and without duplicate edges (see `normalize_graph`).
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

ZERO_TOL = 1e-9


# ---------------------------------------------------------------- graph
def normalize_graph(edge_index, num_nodes):
    """edge_index [2, E] (any direction, may contain loops / duplicates) -> binary symmetric CSR A
    with zero diagonal."""
    ei = np.asarray(edge_index)
    u, v = ei[0].astype(np.int64), ei[1].astype(np.int64)
    keep = u != v
    u, v = u[keep], v[keep]
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    a = sp.coo_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    a.sum_duplicates()
    a.data[:] = 1.0
    a.eliminate_zeros()
    a.sort_indices()
    return a


def degrees(a):
    return np.asarray(a.sum(axis=1)).ravel().astype(np.int64)


def undirected_edges(a):
    """(u, v) arrays with u < v, one entry per undirected edge."""
    coo = sp.triu(a, k=1).tocoo()
    return coo.row.astype(np.int64), coo.col.astype(np.int64)


# ---------------------------------------------------------------- hop sets
def hop_sets_dense(a, rows):
    """For source nodes `rows` (1-D int array): dense bool [len(rows), N] masks H1, H2, H3.

    H1 = N(s); H2 = N^2(s) \\ (H1 ∪ {s}); H3 = N^3(s) \\ (H1 ∪ H2 ∪ {s}).
    """
    r1 = a[rows]
    r2 = r1 @ a
    r3 = r2 @ a
    n = a.shape[0]
    h1 = r1.toarray() > 0
    self_mask = np.zeros((len(rows), n), dtype=bool)
    self_mask[np.arange(len(rows)), rows] = True
    h2 = (r2.toarray() > 0) & ~h1 & ~self_mask
    h3 = (r3.toarray() > 0) & ~h1 & ~h2 & ~self_mask
    return h1, h2, h3


# ---------------------------------------------------------------- features
def a2x_trainfree(adj_sparse, x):
    """Â²X exactly as in hw/trainfree.py main(): Â = data.adj, ax = Â X, a2x = Â ax (torch sparse mm)."""
    ax = torch.sparse.mm(adj_sparse, x)
    return torch.sparse.mm(adj_sparse, ax)


def l2_rows(x):
    return F.normalize(x, p=2, dim=1)


# ---------------------------------------------------------------- PPR
@torch.no_grad()
def ppr_topk(a, alpha, k=100, chunk=2048, max_iter=100, tol=1e-6, device="cpu", sources=None):
    """Personalized PageRank pi_s = alpha e_s + (1 - alpha) pi_s P, P = D^-1 A (row-stochastic).

    Power iteration per source chunk; stops at max_iter or when max |Δ| < tol.
    Returns dict: idx [S, k] (int64, -1 = none), score [S, k] (float32, self excluded,
    zero-score entries marked -1), sources [S], iters (per chunk), max_delta (per chunk),
    row_sum [S] (sum of pi_s incl. self), self_score [S].
    Isolated sources are skipped (caller passes non-isolated `sources`).
    """
    n = a.shape[0]
    deg = degrees(a).astype(np.float64)
    if sources is None:
        sources = np.nonzero(deg > 0)[0]
    inv = np.zeros_like(deg)
    inv[deg > 0] = 1.0 / deg[deg > 0]
    p = sp.diags(inv) @ a  # row-stochastic for non-isolated rows
    pt = p.T.tocoo()  # pi P = (P^T pi^T)^T
    pt_t = torch.sparse_coo_tensor(np.vstack([pt.row, pt.col]), pt.data.astype(np.float32), (n, n), device=device).coalesce()
    kk = min(k, n - 1)
    out_idx = np.full((len(sources), kk), -1, dtype=np.int64)
    out_score = np.zeros((len(sources), kk), dtype=np.float32)
    row_sum = np.zeros(len(sources), dtype=np.float64)
    self_score = np.zeros(len(sources), dtype=np.float64)
    iters, deltas = [], []
    for start in range(0, len(sources), chunk):
        src = torch.as_tensor(sources[start:start + chunk], device=device)
        b = src.numel()
        e = torch.zeros((n, b), device=device)
        e[src, torch.arange(b, device=device)] = 1.0
        pi = e.clone()  # columns = sources
        it, delta = 0, float("inf")
        while it < max_iter:
            nxt = alpha * e + (1 - alpha) * torch.sparse.mm(pt_t, pi)
            delta = float((nxt - pi).abs().max().item())
            pi = nxt
            it += 1
            if delta < tol:
                break
        iters.append(it)
        deltas.append(delta)
        pi = pi.t().contiguous()  # [b, n]
        row_sum[start:start + b] = pi.sum(dim=1).double().cpu().numpy()
        self_score[start:start + b] = pi[torch.arange(b, device=device), src].double().cpu().numpy()
        pi[torch.arange(b, device=device), src] = -1.0  # exclude self
        score, idx = pi.topk(kk, dim=1)
        score, idx = score.cpu().numpy(), idx.cpu().numpy()
        idx[score <= 0] = -1
        score[score <= 0] = 0.0
        out_idx[start:start + b] = idx
        out_score[start:start + b] = score
    return {"idx": out_idx, "score": out_score, "sources": np.asarray(sources), "iters": iters, "max_delta": deltas, "row_sum": row_sum, "self_score": self_score}


# ---------------------------------------------------------------- curvature
def edge_triangles(a, u, v, chunk=200000):
    """t = |N(u) ∩ N(v)| per edge."""
    t = np.zeros(u.size, dtype=np.int64)
    for s in range(0, u.size, chunk):
        e = slice(s, s + chunk)
        t[e] = np.asarray(a[u[e]].multiply(a[v[e]]).sum(axis=1)).ravel()
    return t


def forman_augmented(du, dv, t):
    """Augmented Forman curvature F = 4 - d_u - d_v + 3t."""
    return 4.0 - np.asarray(du, dtype=np.float64) - np.asarray(dv, dtype=np.float64) + 3.0 * np.asarray(t, dtype=np.float64)


def jost_liu(du, dv, t):
    """Jost–Liu lower bound of Ollivier–Ricci curvature:
    κ = −(1 − 1/du − 1/dv − t/min)₊ − (1 − 1/du − 1/dv − t/max)₊ + t/max."""
    du = np.asarray(du, dtype=np.float64)
    dv = np.asarray(dv, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    lo, hi = np.minimum(du, dv), np.maximum(du, dv)
    base = 1.0 - 1.0 / du - 1.0 / dv
    return -np.maximum(base - t / lo, 0.0) - np.maximum(base - t / hi, 0.0) + t / hi


def curvature(a):
    u, v = undirected_edges(a)
    deg = degrees(a)
    t = edge_triangles(a, u, v)
    du, dv = deg[u], deg[v]
    return {"u": u, "v": v, "t": t, "forman": forman_augmented(du, dv, t), "jl": jost_liu(du, dv, t)}


# ---------------------------------------------------------------- binning
def quantile_bins(values, q=5):
    """Value-based quantile bins: edges = quantiles at 1/q..(q-1)/q; bin k holds edges[k-1] < x <= edges[k]
    (bin 0: x <= edges[0]). Ties can make bins uneven or empty; callers report edges and counts."""
    values = np.asarray(values, dtype=np.float64)
    edges = np.quantile(values, [i / q for i in range(1, q)])
    bins = np.searchsorted(edges, values, side="left")
    return bins, edges


# ---------------------------------------------------------------- HW03b: per-dimension Dirichlet energy
def incidence(a):
    """Signed incidence B [E, N] over undirected edges u < v (row e: +1 at u, -1 at v)."""
    u, v = undirected_edges(a)
    e = np.arange(u.size)
    n = a.shape[0]
    return sp.csr_matrix((np.concatenate([np.ones(u.size), -np.ones(v.size)]), (np.concatenate([e, e]), np.concatenate([u, v]))), shape=(u.size, n))


def dirichlet_energy_per_dim(a, x):
    """Per column m: E_raw = 1/2 Σ_(i,j) A_ij (x_im − x_jm)² = Σ over undirected edges (x_um − x_vm)²
    (SPGCL eq. 13), via B X with the signed incidence B; float64. x: dense array or scipy sparse."""
    xs = sp.csr_matrix(x, dtype=np.float64)
    diff = incidence(a) @ xs
    return np.asarray(diff.multiply(diff).sum(axis=0)).ravel()


def laplacian_quadratic(a, f):
    """f^T L f with L = D − A (selftest reference)."""
    f = np.asarray(f, dtype=np.float64)
    lap = sp.diags(np.asarray(a.sum(axis=1)).ravel().astype(np.float64)) - a.astype(np.float64)
    return float(f @ (lap @ f))
