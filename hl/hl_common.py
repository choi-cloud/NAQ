"""hl_common: shared utilities for HL02-HL04; formulas follow choi_exps/exp049.py (build_structural_queries) and meta.ProtoNet.

Run from the repo root with PYTHONPATH=$PWD. choi_exps/ is put on sys.path so its functions are imported, not copied.
Functions here never read ground-truth labels unless the name says so (only `labels_np`).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "choi_exps")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data import load_data  # noqa: E402
from exp049 import build_structural_queries  # noqa: E402

N_NODES_CHECK = 20


def base_args(**overrides):
    """Namespace with the exp049 defaults that load_data / build_structural_queries / ProtoNet read."""
    args = argparse.Namespace(
        dataset="Amazon_clothing", device=0, seed=1212, epoch=2000, eval_interval=160,
        n_way=5, k_shot=1, k_shot_test=1, q_query=10, q_query_test=8, lr=1e-4, latent=256,
        l2_penalty=1e-4, num_valid_tasks=50, num_downstream_tasks=500, mine_batch_size=64,
        min_candidate_policy="relax", tau=0.3, setting="unsup", type="feature", query_generation="NaQ",
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def load_everything(args):
    """load_data -> set_device -> add_diffusion(type=feature), as in the head of exp049.main."""
    args.setting, args.type, args.query_generation = "unsup", "feature", "NaQ"
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    data.add_diffusion(args)
    return data, device


def labels_np(data):
    """Ground-truth labels as a numpy array. Only diag/oracle code may call this."""
    return torch.as_tensor(data.labels).detach().cpu().numpy().reshape(-1)


def structural_scores(data, device):
    """z0, z1, z2 = row-normalised X, ÂX, Â²X with the same Â as exp049.build_structural_queries."""
    import scipy.sparse as sp

    edges = data.edge_index.detach().cpu().numpy()
    x = data.features.detach().float().to(device)
    n = x.size(0)
    adjacency = sp.csr_matrix((np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])), shape=(n, n))
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.float32).tocsr()
    adjacency.setdiag(1)
    adjacency.eliminate_zeros()
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    scale = np.maximum(degrees, 1) ** -0.5
    adjacency = (sp.diags(scale) @ adjacency @ sp.diags(scale)).tocoo()
    indices = torch.as_tensor(np.stack((adjacency.row, adjacency.col)), dtype=torch.long, device=device)
    values = torch.as_tensor(adjacency.data, dtype=torch.float32, device=device)
    a = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    h1 = torch.sparse.mm(a, x)
    h2 = torch.sparse.mm(a, h1)
    z0, z1, z2 = (F.normalize(t, dim=1) for t in (x, h1, h2))
    return z0, z1, z2


def _dummy_tasks(nodes, args):
    """Groups `nodes` into 5-node dummy tasks; the last short group is padded with already-included nodes."""
    nodes = np.asarray(nodes, dtype=np.int64).reshape(-1)
    n_way, q = args.n_way, args.q_query
    tasks = []
    for begin in range(0, len(nodes), n_way):
        group = nodes[begin:begin + n_way]
        if len(group) < n_way:
            group = np.concatenate([group, nodes[: n_way - len(group)]])
        y = torch.arange(n_way, dtype=torch.long)
        tasks.append((group, y, np.zeros(n_way * q, dtype=np.int64), y.repeat_interleave(q)))
    return tasks


def mine_queries_for(nodes, data, args, device):
    """B_Choi queries (q_query each) for `nodes` via exp049.build_structural_queries; returns {node: details}."""
    _, diagnostics = build_structural_queries(data, _dummy_tasks(nodes, args), args, device)
    return {
        int(s): {
            "selected_ids": d["selected_ids"],
            "raw_retained_count": d["raw_retained_count"],
            "mean_original_x_rank": d["mean_original_x_rank"],
            "fallback": d["fallback"],
            "mean_ax_cosine": d["mean_ax_cosine"],
        }
        for s, d in diagnostics.items()
    }


def verify_against_exp049(data, args, device, z=None, seed=0):
    """Asserts that our z0/z1/z2 reproduce exp049's selection and mean_ax_cosine on 20 random nodes."""
    z0, z1, z2 = z if z is not None else structural_scores(data, device)
    n = z0.size(0)
    # sorted unique ids in one batch (< mine_batch_size), so the score GEMM has the same shape as in exp049
    nodes = np.unique(np.random.default_rng(seed).choice(n, N_NODES_CHECK, replace=False))
    mined = mine_queries_for(nodes, data, args, device)
    all_ids = np.arange(n)
    max_err = 0.0
    ids = torch.as_tensor(nodes, dtype=torch.long, device=device)
    with torch.no_grad():
        b0, b1, b2 = ((t[ids] @ t.T).cpu().numpy() for t in (z0, z1, z2))
        for j, s in enumerate(nodes):
            d = mined[int(s)]
            candidates = all_ids[all_ids != s]
            s0, s1, s2 = b0[j, candidates], b1[j, candidates], b2[j, candidates]
            kept = np.flatnonzero((s1 >= args.tau) & (s2 >= args.tau))
            if len(kept) < args.q_query:
                kept = np.argsort(-np.minimum(s1, s2), kind="stable")[: args.q_query]
            kept = kept[np.argsort(-s0[kept], kind="stable")]
            mid = (len(kept) - args.q_query) // 2
            ours = candidates[kept[mid:mid + args.q_query]]
            assert ours.tolist() == d["selected_ids"], f"selected_ids mismatch at node {s}"
            sel = torch.as_tensor(d["selected_ids"], device=device)
            ax = float((z1[sel] @ z1[int(s)]).mean())
            max_err = max(max_err, abs(ax - d["mean_ax_cosine"]))
    assert max_err < 1e-5, f"mean_ax_cosine mismatch: {max_err}"
    return {"nodes": nodes.tolist(), "max_abs_err_mean_ax_cosine": max_err}


def x_topL(data, nodes, L):
    """Top-L ids and X cosines per node from data.Diffusion_matrix (diagonal 0), as numpy arrays."""
    idx = torch.as_tensor(np.asarray(nodes, dtype=np.int64), device=data.Diffusion_matrix.device)
    ids, cos = [], []
    with torch.no_grad():
        for begin in range(0, len(idx), 1024):
            v, i = data.Diffusion_matrix[idx[begin:begin + 1024]].topk(k=L, dim=1)
            ids.append(i.cpu().numpy())
            cos.append(v.cpu().numpy())
    return np.concatenate(ids), np.concatenate(cos)


def accepted_mask(z1, z2, src, dst, tau):
    """True where cos_ÂX(src, dst) >= tau and cos_Â²X(src, dst) >= tau (exp049 filter)."""
    return (pair_cos(z1, src, dst) >= tau) & (pair_cos(z2, src, dst) >= tau)


def pair_cos(z, src, dst, chunk=8192):
    """Row-wise cosine of already-normalised rows z[src], z[dst], in chunks to bound GPU memory."""
    src = torch.as_tensor(np.asarray(src, dtype=np.int64), device=z.device).reshape(-1)
    dst = torch.as_tensor(np.asarray(dst, dtype=np.int64), device=z.device).reshape(-1)
    out = []
    with torch.no_grad():
        for begin in range(0, len(src), chunk):
            out.append((z[src[begin:begin + chunk]] * z[dst[begin:begin + chunk]]).sum(-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def proto_softmax(Z, id_spt, id_query, y_spt, y_query):
    """meta.ProtoNet.forward formula for k=1 without training: returns (probs[q, n], target index[q], dists[q, n])."""
    with torch.no_grad():
        spt = torch.as_tensor(np.asarray(id_spt, dtype=np.int64), device=Z.device)
        qry = torch.as_tensor(np.asarray(id_query, dtype=np.int64), device=Z.device)
        prototypes = Z[spt]
        x = Z[qry]
        dists = (x.unsqueeze(1) - prototypes.unsqueeze(0)).pow(2).sum(dim=2)
        probs = F.log_softmax(-dists, dim=1).exp()
        y_spt = list(np.asarray(y_spt).reshape(-1).tolist())
        target = torch.as_tensor([y_spt.index(i) for i in np.asarray(y_query).reshape(-1).tolist()], device=Z.device)
    return probs, target, dists


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def __call__(self):
        return round(time.perf_counter() - self.t0, 2)
