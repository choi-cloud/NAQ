"""hl_pdd: label-free judges and distance rules for HL06-HL10 (purity-distance split). Dataset-agnostic; never reads labels.

Definitions follow instructions/HL06_purity_distance_split.md §1. Existing judges reuse hl005 functions (sym_adjacency,
ppr_columns, encoder_embeddings, full_percentile); `comb` is the hl005 definition (mv + enc_b).
Band = X-cosine rank window (default 201-1000, self excluded, stable ties). s2 = Â²X cosine (hl_common z2). s2 rank = rank
inside the band by s2 descending (ties by X rank). Strata T1-T5 = five equal s2-rank slices of the band.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hl_common as H  # noqa: E402
from hl005_diag_distance_purity import encoder_embeddings, full_percentile, ppr_columns, sym_adjacency  # noqa: E402

OLD_JUDGES = ["x", "ax", "a2x", "ppr", "mv", "enc_b", "enc_n", "comb"]
NEW_JUDGES = ["xidf", "nbr_b", "knnppr_b", "clu_b"]
JUDGES = OLD_JUDGES + NEW_JUDGES
SCORED_RULES = ["top10", "ex20", "far50", "strat5"]
EXTRA_RULES = ["strat3", "strat10", "ex10", "ex40"]  # HL10
RANDOM_RULES = ["rand", "rand_ex20", "rand_far50", "rand_strat5"]
N_STRATA = 5
KNN = 10
KMEANS_K, KMEANS_RUNS = 200, 5
ROW_CHUNK = 2048


def band_rank(data, anchors, band, depth=3000):
    """X-rank order of each anchor's candidates (self excluded, stable), first max(depth, band hi) columns, as torch [A, d]."""
    dm = data.Diffusion_matrix
    a = torch.as_tensor(np.asarray(anchors, dtype=np.int64), device=dm.device)
    with torch.no_grad():
        rows = dm[a].clone()
        rows[torch.arange(len(a), device=dm.device), a] = -float("inf")
        return torch.sort(rows, dim=1, descending=True, stable=True).indices[:, :max(depth, band[1])]


def s2_rank(s2_band):
    """[A, m] s2 values in X-rank column order -> [A, m] rank 1..m by s2 descending (ties by X rank)."""
    order = np.argsort(-s2_band, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(1, s2_band.shape[1] + 1)[None, :].repeat(len(order), 0), axis=1)
    return rank


def _top(cols, score, k):
    """k of `cols` (X-rank ascending) with highest score; ties by X rank."""
    return cols[np.argsort(-score[cols], kind="stable")[:k]]


def strata_of(s2rank_row):
    """Five equal s2-rank slices T1-T5 of the band, as arrays of band column indices."""
    m = len(s2rank_row)
    size = m // N_STRATA
    allc = np.arange(m)
    return [allc[(s2rank_row > i * size) & (s2rank_row <= ((i + 1) * size if i < N_STRATA - 1 else m))] for i in range(N_STRATA)]


def select(rule, score_row, s2rank_row, rng, q=10):
    """Band column indices (q of them) for one anchor. `score_row`: judge score per band column (None for random rules);
    `s2rank_row`: s2 rank 1..m per band column. No labels."""
    m = len(s2rank_row)
    allc = np.arange(m)
    ex = allc[s2rank_row > m // N_STRATA]  # s2 rank 161-800 for m = 800
    far = allc[s2rank_row > m // 2]  # 401-800
    size = m // N_STRATA
    strata = [allc[(s2rank_row > i * size) & (s2rank_row <= (i + 1) * size if i < N_STRATA - 1 else s2rank_row <= m)] for i in range(N_STRATA)]
    per = q // N_STRATA
    if rule == "top10":
        return _top(allc, score_row, q)
    if rule == "ex20":
        return _top(ex, score_row, q)
    if rule == "far50":
        return _top(far, score_row, q)
    if rule == "strat5":
        return np.concatenate([_top(s, score_row, per) for s in strata])
    if rule == "strat3":  # HL10: s2 rank 1-267 / 268-534 / 535-800 -> 3 + 3 + 4
        cut = [0, round(m / 3), round(2 * m / 3), m]
        return np.concatenate([_top(allc[(s2rank_row > cut[i]) & (s2rank_row <= cut[i + 1])], score_row, k) for i, k in enumerate((3, 3, 4))])
    if rule == "strat10":  # HL10: ten s2-rank slices, top 1 each
        cut = [round(i * m / 10) for i in range(11)]
        return np.concatenate([_top(allc[(s2rank_row > cut[i]) & (s2rank_row <= cut[i + 1])], score_row, 1) for i in range(10)])
    if rule == "ex10":  # HL10: s2 rank 81-800
        return _top(allc[s2rank_row > m // 10], score_row, q)
    if rule == "ex40":  # HL10: s2 rank 321-800
        return _top(allc[s2rank_row > 2 * m // 5], score_row, q)
    if rule == "rand":
        return rng.choice(allc, q, replace=False)
    if rule == "rand_ex20":
        return rng.choice(ex, q, replace=False)
    if rule == "rand_far50":
        return rng.choice(far, q, replace=False)
    if rule == "rand_strat5":
        return np.concatenate([rng.choice(s, per, replace=False) for s in strata])
    raise ValueError(rule)


class Judges:
    """Lazily built label-free per-dataset state; `rows(name, anchors)` returns a torch [A, n] score row per anchor."""

    def __init__(self, data, device, ckpt_b, ckpt_n, latent=256, alpha=0.15, iters=30):
        self.data, self.device = data, device
        self.ckpt_b, self.ckpt_n, self.latent = ckpt_b, ckpt_n, latent
        self.alpha, self.iters = alpha, iters
        self._c = {}

    def _get(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    @property
    def z(self):
        return self._get("z", lambda: H.structural_scores(self.data, self.device))

    @property
    def a_sym(self):
        return self._get("a_sym", lambda: sym_adjacency(self.data, self.device))

    @property
    def emb_b(self):
        return self._get("emb_b", lambda: encoder_embeddings(self.data, self.ckpt_b, self.latent, self.device))

    @property
    def emb_n(self):
        return self._get("emb_n", lambda: encoder_embeddings(self.data, self.ckpt_n, self.latent, self.device))

    @property
    def xidf(self):
        def build():
            x = self.data.features.detach().float().to(self.device)
            n = x.size(0)
            df = (x > 0).sum(0).float()
            idf = torch.log((1 + n) / (1 + df)) + 1
            return F.normalize(torch.log1p(x.clamp(min=0)) * idf, dim=1)  # zero rows stay zero
        return self._get("xidf", build)

    @property
    def knn(self):
        """[n, KNN] E-cosine nearest neighbours (self excluded), enc_b embeddings, ROW_CHUNK rows at a time."""
        def build():
            e = self.emb_b
            out = []
            with torch.no_grad():
                for b in range(0, e.size(0), ROW_CHUNK):
                    s = e[b:b + ROW_CHUNK] @ e.T
                    s[torch.arange(s.size(0), device=s.device), torch.arange(b, b + s.size(0), device=s.device)] = -float("inf")
                    out.append(s.topk(KNN, dim=1).indices)
            return torch.cat(out)
        return self._get("knn", build)

    @property
    def nbr(self):
        def build():
            e = self.emb_b
            return F.normalize(e + e[self.knn].sum(1), dim=1)
        return self._get("nbr", build)

    @property
    def knn_sym(self):
        """D^-1/2 (max(K, K^T) + I) D^-1/2 of the enc_b kNN graph (hl005.sym_adjacency normalisation)."""
        def build():
            knn = self.knn.cpu().numpy()
            n = knn.shape[0]
            k = sp.csr_matrix((np.ones(knn.size, dtype=np.float32), (np.repeat(np.arange(n), KNN), knn.reshape(-1))), shape=(n, n))
            k = ((k + k.T) > 0).astype(np.float32).tocsr()
            k.setdiag(1)
            k.eliminate_zeros()
            deg = np.asarray(k.sum(axis=1)).reshape(-1)
            scale = np.maximum(deg, 1) ** -0.5
            k = (sp.diags(scale) @ k @ sp.diags(scale)).tocoo()
            idx = torch.as_tensor(np.stack((k.row, k.col)), dtype=torch.long, device=self.device)
            return torch.sparse_coo_tensor(idx, torch.as_tensor(k.data, dtype=torch.float32, device=self.device), (n, n)).coalesce()
        return self._get("knn_sym", build)

    @property
    def clusters(self):
        def build():
            from sklearn.cluster import KMeans
            e = self.emb_b.cpu().numpy()
            return torch.as_tensor(np.stack([KMeans(n_clusters=KMEANS_K, n_init=1, random_state=r).fit_predict(e) for r in range(KMEANS_RUNS)]),
                                   device=self.device)
        return self._get("clusters", build)

    def rows(self, name, anchors):
        """Full score rows [A, n] (torch) for judge `name`; mv/comb return full-set percentiles (hl005 definitions)."""
        a = torch.as_tensor(np.asarray(anchors, dtype=np.int64), device=self.device)
        _, z1, z2 = self.z
        with torch.no_grad():
            if name == "x":
                r = self.data.Diffusion_matrix[a].clone()
                r[torch.arange(len(a), device=a.device), a] = -float("inf")
                return r
            if name == "ax":
                return z1[a] @ z1.T
            if name == "a2x":
                return z2[a] @ z2.T
            if name == "ppr":
                return ppr_columns(self.a_sym, np.asarray(anchors), self.alpha, self.iters).T.contiguous()
            if name == "enc_b":
                return self.emb_b[a] @ self.emb_b.T
            if name == "enc_n":
                return self.emb_n[a] @ self.emb_n.T
            if name in ("mv", "comb"):
                allc = torch.arange(self.data.features.size(0), device=a.device).expand(len(a), -1)
                mv = sum(full_percentile(self.rows(k, anchors), a, allc) for k in ("ax", "a2x", "ppr")) / 3
                if name == "mv":
                    return mv
                return (full_percentile(mv, a, allc) + full_percentile(self.rows("enc_b", anchors), a, allc)) / 2
            if name == "xidf":
                return self.xidf[a] @ self.xidf.T
            if name == "nbr_b":
                return self.nbr[a] @ self.nbr.T
            if name == "knnppr_b":
                return ppr_columns(self.knn_sym, np.asarray(anchors), self.alpha, self.iters).T.contiguous()
            if name == "clu_b":
                c = self.clusters
                same = (c[:, a].unsqueeze(2) == c.unsqueeze(1)).float().mean(0)
                return same + 1e-3 * (self.emb_b[a] @ self.emb_b.T)
        raise ValueError(name)

    def band(self, names, anchors, band, depth=3000):
        """Returns ranked [A, d] numpy (X-rank order), band node ids [A, m], s2 [A, m], s2 rank [A, m], {judge: [A, m]}."""
        ranked = band_rank(self.data, anchors, band, depth)
        cols = ranked[:, band[0] - 1:band[1]]
        s2 = self.rows("a2x", anchors).gather(1, cols)
        scores = {k: self.rows(k, anchors).gather(1, cols).cpu().numpy() for k in names}
        s2 = s2.cpu().numpy()
        return ranked.cpu().numpy(), cols.cpu().numpy(), s2, s2_rank(s2), scores
