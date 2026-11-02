"""formulas: CLAUDE.md §0; Â and z0/z1/z2 from hl/hl_common.structural_scores; adjacency as hl/hl005_diag_distance_purity.sym_adjacency

Shared module of the ND track: neighbour weight matrices W, assign() and vote(). No function here reads ground-truth labels.
Run from the repo root with PYTHONPATH=$PWD:  python nd/nd_vote.py --selfcheck
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "choi_exps"), str(REPO_ROOT / "hl"), str(REPO_ROOT / "nd")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hl_common as H  # noqa: E402
from hl005_diag_distance_purity import sym_adjacency  # noqa: E402

TAUS = (0.05, 0.1, 0.2)
TAU_KEYS = ("0.05", "0.1", "0.2", "hard")
NBRS = {"x": ("1hop", "self", "a2"), "a2x": ("1hop", "self")}
MAIN_ARM = "vote_x_0.1_1hop"
SEED = 1212
OUT = Path("results/nd/nd01")
A2_LIMIT_BYTES = 2 * 1024 ** 3


def arm_name(f, tau_key, nbr):
    return f"vote_{f}_{tau_key}_{nbr}"


def grid():
    """The 20 arms of ND01 §1-3 in fixed order: f, then tau (0.05, 0.1, 0.2, hard), then nbr (1hop, self, a2)."""
    return [(f, t, nbr) for f in ("x", "a2x") for t in TAU_KEYS for nbr in NBRS[f]]


class Weight:
    """Row-stochastic sparse W [n, n]; keeps torch COO (`coo`) and CSR arrays for row selection."""

    def __init__(self, coo):
        self.coo = coo.coalesce()
        csr = self.coo.to_sparse_csr()
        self.crow, self.col, self.val = csr.crow_indices(), csr.col_indices(), csr.values()
        self.length = self.crow[1:] - self.crow[:-1]
        self.nnz = int(self.val.numel())

    def memory_bytes(self):
        # as COO: two int64 indices and one float32 value per entry
        return self.nnz * (2 * 8 + 4)


def _coo_from_scipy(m, device):
    m = m.tocoo()
    idx = torch.as_tensor(np.stack((m.row, m.col)), dtype=torch.long, device=device)
    return torch.sparse_coo_tensor(idx, torch.as_tensor(m.data, dtype=torch.float32, device=device), m.shape).coalesce()


def _sym_a(data):
    """Symmetrised binary A with self loops removed; same edge handling as hl_common.structural_scores."""
    edges = data.edge_index.detach().cpu().numpy()
    n = data.features.size(0)
    a = sp.csr_matrix((np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])), shape=(n, n))
    a = ((a + a.T) > 0).astype(np.float32).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    return a


def _row_normalise(m):
    d = np.asarray(m.sum(axis=1), dtype=np.float64).reshape(-1)
    return sp.diags(np.where(d > 0, 1.0 / np.maximum(d, 1e-30), 0.0)) @ m


def build_weights(data, device, with_a2=True):
    """{'1hop', 'self', 'a2'} -> Weight, plus an info dict. `a2` is dropped if it would exceed 2GB (ND01 §1-1)."""
    a = _sym_a(data)
    n = a.shape[0]
    deg = np.asarray(a.sum(axis=1)).reshape(-1)
    w = {"1hop": Weight(_coo_from_scipy(_row_normalise(a), device)),
         "self": Weight(_coo_from_scipy(_row_normalise(a + sp.eye(n, dtype=np.float32, format="csr")), device))}
    info = {"n": n, "edges_sym_no_self": int(a.nnz), "isolated_count": int((deg == 0).sum())}
    if with_a2:
        hat = sym_adjacency(data, device)
        a2 = torch.sparse.mm(hat, hat).coalesce()
        idx, val = a2.indices().cpu().numpy(), a2.values().cpu().numpy().astype(np.float64)
        del a2
        keep = val > 0
        m = sp.csr_matrix((val[keep], (idx[0][keep], idx[1][keep])), shape=(n, n))
        info["a2_nnz"] = int(m.nnz)
        if m.nnz * (2 * 8 + 4) > A2_LIMIT_BYTES:
            info["a2_dropped"] = True
        else:
            w["a2"] = Weight(_coo_from_scipy(_row_normalise(m), device))
    return w, info


def similarity(zf, P):
    """S = zf @ P.T; zf, P row-normalised."""
    return zf @ P.T


def assign_from_s(S, tau_key):
    if tau_key == "hard":
        return F.one_hot(S.argmax(dim=1), S.size(1)).to(S.dtype)
    return torch.softmax(S / float(tau_key), dim=1)


def assign(zf, P, tau, mode):
    """Q [n, N]. mode: 'soft' or 'hard'."""
    return assign_from_s(similarity(zf, P), "hard" if mode == "hard" else tau)


def vote_rows(Q, W, idx):
    """(W @ Q)[idx] via CSR row selection. A row of W that is empty (isolated node in 1hop) takes that node's Q row."""
    idx = torch.as_tensor(idx, dtype=torch.long, device=Q.device)
    ln = W.length[idx]
    total = int(ln.sum())
    out = torch.zeros(len(idx), Q.size(1), dtype=Q.dtype, device=Q.device)
    if total:
        rows = torch.arange(len(idx), device=Q.device).repeat_interleave(ln)
        ends = ln.cumsum(0)
        offs = torch.arange(total, device=Q.device) - (ends - ln).repeat_interleave(ln) + W.crow[idx].repeat_interleave(ln)
        out.index_add_(0, rows, W.val[offs].unsqueeze(1) * Q[W.col[offs]])
    empty = ln == 0
    if empty.any():
        out[empty] = Q[idx[empty]]
    return out


def vote(Q, W):
    """D [n, N] = W @ Q; empty rows of W take the node's own Q row."""
    return vote_rows(Q, W, torch.arange(Q.size(0), device=Q.device))


def selfcheck():
    args = H.base_args()
    data, device = H.load_everything(args)
    z0, z1, z2 = H.structural_scores(data, device)
    with torch.no_grad():
        w, info = build_weights(data, device)
        res = {"info": info, "W": {}}
        for k, W in w.items():
            sums = torch.sparse.sum(W.coo, dim=1).to_dense()
            nonempty = W.length > 0
            res["W"][k] = {"nnz": W.nnz, "memory_bytes_coo": W.memory_bytes(), "empty_rows": int((~nonempty).sum()),
                           "max_abs_row_sum_err": float((sums[nonempty] - 1).abs().max())}
        anchors = np.random.default_rng(SEED).choice(z0.size(0), 5, replace=False)
        res["anchors"] = anchors.tolist()
        res["checks"] = {}
        for f, z in (("x", z0), ("a2x", z2)):
            P = z[torch.as_tensor(anchors, device=device)]
            for t in TAU_KEYS:
                Q = assign(z, P, None if t == "hard" else float(t), "hard" if t == "hard" else "soft")
                for nbr in NBRS[f]:
                    if nbr not in w:
                        continue
                    D = vote(Q, w[nbr])
                    full = torch.sparse.mm(w[nbr].coo, Q)
                    empty = w[nbr].length == 0
                    full[empty] = Q[empty]
                    res["checks"][arm_name(f, t, nbr)] = {
                        "max_abs_row_sum_err": float((D.sum(1) - 1).abs().max()),
                        "max_abs_err_vs_sparse_mm": float((D - full).abs().max()),
                        "argmax_agree_D_Q": float((D.argmax(1) == Q.argmax(1)).float().mean()),
                    }
        worst = max(c["max_abs_row_sum_err"] for c in res["checks"].values())
        assert worst <= 1e-4, f"D row sum error {worst}"
        res["max_abs_D_row_sum_err"] = worst
        res["argmax_agree_main_arm"] = res["checks"][MAIN_ARM]["argmax_agree_D_Q"]
        res["verify_against_exp049"] = H.verify_against_exp049(data, args, device, z=(z0, z1, z2))
    res["gpu_max_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else None
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selfcheck.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("info", "W", "max_abs_D_row_sum_err", "argmax_agree_main_arm", "gpu_max_allocated_gb")}, indent=1))
    print("exp049 max err", res["verify_against_exp049"]["max_abs_err_mean_ax_cosine"])
    print("vs sparse.mm max err", max(c["max_abs_err_vs_sparse_mm"] for c in res["checks"].values()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    if p.parse_args().selfcheck:
        selfcheck()
