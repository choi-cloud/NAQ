"""hw shim for Choi's new-series exp003 (not in the repo): `dropped_normalized_adj` only.

Used solely so that choi_exps/zip_20261107/exp004.py (BGRL) can run unmodified:
    PYTHONPATH=hw/compat:. python -u choi_exps/zip_20261107/exp004.py ...
This is NOT Choi's implementation. Embeddings produced with it are recorded as
"BGRL (exp004 + hw shim)".

Signature (from exp004.py call sites, lines 186-187):
    view = dropped_normalized_adj(data.edge_index, n, args.edge_drop)
    online(x, adj=view)  ->  model.GNNEncoder.forward -> torch.sparse.mm(adj, x)
So: (edge_index [2, E] LongTensor, num_nodes int, drop_rate float) -> torch sparse
float32 tensor [N, N] on edge_index.device, same normalization as data.adj
(utils.normalize_adj(A + I) = D^-1/2 (A+I) D^-1/2).
"""

import torch


def dropped_normalized_adj(edge_index, num_nodes, drop_rate):
    """Drop undirected edges with probability `drop_rate` (symmetric), add self-loops,
    return D^-1/2 (A+I) D^-1/2 as a coalesced sparse tensor. Uses the global torch RNG."""
    if not 0 <= drop_rate < 1:
        raise ValueError("drop_rate must be in [0, 1)")
    device = edge_index.device
    row, col = edge_index[0].long(), edge_index[1].long()
    off_diag = row != col
    row, col = row[off_diag], col[off_diag]
    # one entry per undirected edge (u < v), duplicates removed
    lo, hi = torch.minimum(row, col), torch.maximum(row, col)
    undirected = torch.unique(lo * num_nodes + hi)
    keep = torch.rand(undirected.numel(), device=device) >= drop_rate
    kept = undirected[keep]
    u, v = kept // num_nodes, kept % num_nodes
    loops = torch.arange(num_nodes, device=device)
    rows = torch.cat([u, v, loops])
    cols = torch.cat([v, u, loops])
    degree = torch.zeros(num_nodes, device=device, dtype=torch.float32).index_add_(0, rows, torch.ones_like(rows, dtype=torch.float32))
    d_inv_sqrt = degree.pow(-0.5)
    values = d_inv_sqrt[rows] * d_inv_sqrt[cols]
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), values, (num_nodes, num_nodes)).coalesce()
