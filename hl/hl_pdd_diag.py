"""hl_pdd_diag: post-hoc query descriptors for HL06-HL10 (instructions/HL06 §1-4, §2-3). Reads GT labels (diagnostics only).

`Describer.describe` is called after the queries of an anchor are chosen; nothing here feeds back into selection.
"""
import numpy as np
import scipy.sparse as sp
import torch

KEYS = ["purity", "x_rank", "s2", "s2rank", "encb", "pair_s2", "pair_encb", "hop1", "hop2", "tp_s2", "fp_s2", "choi_pass"]


def _pair_mean(m):
    k = m.shape[0]
    return float((m.sum() - np.trace(m)) / (k * (k - 1))) if k > 1 else float("nan")


class Describer:
    def __init__(self, data, z1, z2, labels, tau, emb_b=None):
        edges = data.edge_index.detach().cpu().numpy()
        n = data.features.size(0)
        adj = sp.csr_matrix((np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])), shape=(n, n))
        adj = ((adj + adj.T) > 0).astype(np.float32).tocsr()
        adj.setdiag(0)
        adj.eliminate_zeros()
        self.adj, self.z1, self.z2, self.labels, self.tau, self.emb_b = adj, z1, z2, labels, tau, emb_b
        self._hops = {}

    def hops(self, a):
        if a not in self._hops:
            self._hops[a] = self._hops_of(a)
        return self._hops[a]

    def _hops_of(self, a):
        n1 = self.adj.indices[self.adj.indptr[a]:self.adj.indptr[a + 1]]
        n2 = np.unique(self.adj[n1].indices) if len(n1) else np.zeros(0, dtype=np.int64)
        return n1, np.union1d(n1, n2)

    def describe(self, a, chosen, ranked_row, band_ids, s2rank_row, xrank=None):
        """`ranked_row`: X-rank order ids of anchor a (self excluded, >= 3000 long); `band_ids`, `s2rank_row`: band node ids and their s2 rank.
        `xrank` (optional): full [n] X-rank lookup of anchor a, used instead of `ranked_row` when given."""
        chosen = np.asarray(chosen, dtype=np.int64)
        dev = self.z2.device
        c = torch.as_tensor(chosen, device=dev)
        with torch.no_grad():
            s2 = (self.z2[c] @ self.z2[a]).cpu().numpy()
            ax = (self.z1[c] @ self.z1[a]).cpu().numpy()
            ps2 = (self.z2[c] @ self.z2[c].T).cpu().numpy()
            if self.emb_b is not None:
                enc = (self.emb_b[c] @ self.emb_b[a]).cpu().numpy()
                penc = (self.emb_b[c] @ self.emb_b[c].T).cpu().numpy()
        if xrank is not None:
            xr = xrank[chosen].astype(float)
        else:
            pos = {int(v): i + 1 for i, v in enumerate(ranked_row)}
            xr = np.asarray([pos.get(int(v), np.nan) for v in chosen], dtype=float)  # NaN beyond the cached X-rank depth
        bpos = {int(v): int(r) for v, r in zip(band_ids, s2rank_row)}
        s2r = [bpos[int(v)] for v in chosen if int(v) in bpos]
        same = self.labels[chosen] == self.labels[a]
        n1, n2 = self.hops(a)
        out = {
            "purity": float(same.mean()), "n_same": int(same.sum()),
            "x_rank": float(np.nanmean(xr)) if not np.isnan(xr).all() else float("nan"),
            "s2": float(s2.mean()),
            "s2rank": float(np.mean(s2r)) if len(s2r) == len(chosen) else float("nan"),
            "pair_s2": _pair_mean(ps2),
            "hop1": float(np.isin(chosen, n1).mean()), "hop2": float(np.isin(chosen, n2).mean()),
            "tp_s2": float(s2[same].mean()) if same.any() else float("nan"),
            "fp_s2": float(s2[~same].mean()) if (~same).any() else float("nan"),
            "choi_pass": float(((ax >= self.tau) & (s2 >= self.tau)).mean()),
        }
        if self.emb_b is not None:
            out["encb"], out["pair_encb"] = float(enc.mean()), _pair_mean(penc)
        else:
            out["encb"], out["pair_encb"] = float("nan"), float("nan")
        return out


def aggregate(rows, q=10):
    """Means over anchors (or anchor occurrences); purity_pooled = same-GT queries / all queries."""
    out = {"purity_pooled": float(np.sum([r["n_same"] for r in rows]) / (q * len(rows))),
           "purity_anchor_mean": float(np.mean([r["purity"] for r in rows]))}
    for k in KEYS[1:]:
        v = np.asarray([r[k] for r in rows], dtype=float)
        out[f"mean_{k}" if k in ("x_rank", "s2", "s2rank", "encb") else f"{k}_frac" if k in ("hop1", "hop2", "choi_pass") else k] = \
            float(np.nanmean(v)) if not np.isnan(v).all() else float("nan")
    return out
