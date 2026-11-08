"""HW05: extra negatives for L_original (`extra_neg` block) and the L_neighbor term split (`nbr_term`).

extra_neg = {"mode": none|random|hard_raw|hard_a2x, "m": int}
- The L_original logits [Q, 5] are extended per query to [Q, 5 + m] with m extra nodes, all negatives
  (hw/losses.py). No λ, no separate term.
- Excluded candidates for every query of an episode: the 5 supports and the 50 pseudo-queries (which
  include the query itself).
- random: m distinct nodes drawn uniformly from the allowed nodes: per query, uniform scores from a
  dedicated torch.Generator (seed + EXTRA_NEG_SEED_OFFSET), excluded nodes masked, top-m. No global RNG.
- hard_raw / hard_a2x: per node, the top-HARD_TABLE_K nodes by cos (L2-normalized X or Â²X,
  hw.structure.a2x_trainfree; self excluded), computed once before training. Per query: the first m
  table entries that are not excluded; a shortfall is filled by the random rule (excluding the excluded
  set and the already chosen nodes) and counted in `fill`.
"""

import torch

from hw import structure

EXTRA_NEG_MODES = ("none", "random", "hard_raw", "hard_a2x")
EXTRA_NEG_SEED_OFFSET = 23757
HARD_TABLE_K = 100
HARD_TABLE_CHUNK = 2048
NBR_TERMS = ("infonce", "pull", "push")


def neighbor_term(neighbor_logits, positive_mask, nbr_term):
    """pull = mean_q −logit(q, s⁺) = mean_q ‖z_q − z_n(q,s⁺)‖²/T (one positive support per query);
    push = mean_q logsumexp_s logit(q, s). pull + push = InfoNCE − log|P| (|P| = 1 here)."""
    if nbr_term == "pull":
        positives = positive_mask.sum(dim=1)
        if not bool((positives == 1).all()):
            raise ValueError("nbr_term pull expects exactly one positive support per query")
        return -(neighbor_logits.masked_fill(~positive_mask, 0.0).sum(dim=1)).mean()
    if nbr_term == "push":
        return torch.logsumexp(neighbor_logits, dim=1).mean()
    raise ValueError(f"nbr_term must be one of {NBR_TERMS}, got {nbr_term!r}")


@torch.no_grad()
def hard_table(space_normalized, k=HARD_TABLE_K, chunk=HARD_TABLE_CHUNK):
    """[N, k] node ids by descending cos to each node (self excluded)."""
    n = space_normalized.shape[0]
    out = torch.empty((n, k), dtype=torch.long, device=space_normalized.device)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        cos = space_normalized[start:stop] @ space_normalized.T
        cos[torch.arange(stop - start, device=cos.device), torch.arange(start, stop, device=cos.device)] = -torch.inf
        out[start:stop] = cos.topk(k, dim=1).indices
    return out


class ExtraNegatives:
    def __init__(self, mode, m, num_nodes, device, seed, features=None, adj=None):
        if mode not in EXTRA_NEG_MODES or mode == "none":
            raise ValueError(f"extra_neg mode must be one of {EXTRA_NEG_MODES[1:]}, got {mode!r}")
        self.mode, self.m, self.num_nodes, self.device = mode, int(m), int(num_nodes), device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(seed) + EXTRA_NEG_SEED_OFFSET)
        self.table = None
        if mode.startswith("hard"):
            with torch.no_grad():
                x = features.detach().float()
                space = structure.l2_rows(x) if mode == "hard_raw" else structure.l2_rows(structure.a2x_trainfree(adj.coalesce(), x))
                self.table = hard_table(space)

    def _excluded_mask(self, support_ids, query_ids):
        mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        mask[support_ids] = True
        mask[query_ids] = True
        return mask

    def _random(self, q_count, banned):
        """banned [Q, N] bool -> [Q, m] distinct allowed ids, uniform without replacement."""
        scores = torch.rand((q_count, self.num_nodes), generator=self.generator, device=self.device)
        scores = scores.masked_fill(banned, -1.0)
        return scores.topk(self.m, dim=1).indices

    @torch.no_grad()
    def sample(self, support_ids, query_ids):
        """Returns (extra ids [Q, m], fill fraction (float, share of the Q·m entries filled at random))."""
        q_count = query_ids.numel()
        if self.m == 0:
            return torch.empty((q_count, 0), dtype=torch.long, device=self.device), 0.0
        excluded = self._excluded_mask(support_ids, query_ids)
        if self.mode == "random":
            return self._random(q_count, excluded[None, :].expand(q_count, -1)), 0.0
        cand = self.table[query_ids]  # [Q, K]
        ok = ~excluded[cand]
        rank = ok.long().cumsum(dim=1)
        take = ok & (rank <= self.m)
        found = take.sum(dim=1)  # [Q]
        chosen = torch.full((q_count, self.m), -1, dtype=torch.long, device=self.device)
        rows, cols = take.nonzero(as_tuple=True)
        chosen[rows, rank[rows, cols] - 1] = cand[rows, cols]
        short = found < self.m
        fill = 0.0
        if bool(short.any()):
            banned = excluded[None, :].expand(q_count, -1).clone()
            vr, vc = (chosen >= 0).nonzero(as_tuple=True)
            banned[vr, chosen[vr, vc]] = True
            filler = self._random(q_count, banned)  # [Q, m] distinct, disjoint from chosen
            slot = torch.arange(self.m, device=self.device)[None, :]
            need = slot >= found[:, None]
            chosen = torch.where(need, filler[:, : self.m].gather(1, (slot - found[:, None]).clamp(min=0)), chosen)
            fill = float(need.float().mean().item())
        return chosen, fill


# ---------------------------------------------------------------- HW06: far / band / cluster
"""HW06 extra_neg modes (selection-table slot, CLAUDE.md §4.4; no global RNG).
Per episode every query q excludes the 5 supports and the 50 pseudo-queries (q included), as in `random`;
one query's m negatives are distinct (uniform scores from the mode's own torch.Generator, banned masked, top-m).
- far:     additionally excludes the shortest-path ≤ 2 nodes of q (hw.structure.normalize_graph graph) and
           the raw-cos top-100 of q (= the first 100 columns of the raw-cos table below). seed + 31337.
- band:    candidates = q's raw-cos ranks band_lo..band_hi (1-based, self excluded; default 101..500). seed + 40503.
- cluster: candidates = nodes in a different k-means cluster than q (Â²X L2-normalized, sklearn KMeans
           n_clusters k (default 100), n_init 4, random_state 20261108). seed + 52361.
Tables: raw-cos top-RAW_TABLE_K (values + ids) computed on CPU in chunks, cached at cache/rawcos/<ds>_top500.pt;
k-means labels cached at cache/kmeans/<ds>_k<k>.pt. Both are deterministic, seed independent, built once per dataset.
"""

import os
from pathlib import Path

STRATEGY_MODES = ("far", "band", "cluster")
EXTRA_NEG_MODES = EXTRA_NEG_MODES + STRATEGY_MODES
STRATEGY_SEED_OFFSETS = {"far": 31337, "band": 40503, "cluster": 52361}
RAW_TABLE_K = 500
FAR_TOP = 100
BAND_DEFAULT = (101, 500)
KMEANS_DEFAULT_K = 100
KMEANS_N_INIT = 4
KMEANS_SEED = 20261108
CACHE_ROOT = Path("cache")


def _atomic_save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def raw_cos_table(dataset, features, k=RAW_TABLE_K, chunk=HARD_TABLE_CHUNK):
    """{"ids": [N, k] long, "cos": [N, k] float} raw-cos (L2-normalized X) top-k per node, self excluded,
    descending (torch.topk sorted). CPU, chunked (no N x N dense)."""
    path = CACHE_ROOT / "rawcos" / f"{dataset}_top{k}.pt"
    if path.exists():
        return torch.load(path)
    x = structure.l2_rows(features.detach().float().cpu())
    n = x.shape[0]
    ids = torch.empty((n, k), dtype=torch.long)
    vals = torch.empty((n, k), dtype=torch.float32)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        cos = x[start:stop] @ x.T
        cos[torch.arange(stop - start), torch.arange(start, stop)] = -torch.inf
        top = cos.topk(k, dim=1)
        ids[start:stop], vals[start:stop] = top.indices, top.values
    table = {"ids": ids, "cos": vals}
    _atomic_save(table, path)
    return table


@torch.no_grad()
def kmeans_labels(dataset, features, adj, k=KMEANS_DEFAULT_K):
    """[N] long cluster ids of KMeans on L2-normalized Â²X (CPU), cached."""
    path = CACHE_ROOT / "kmeans" / f"{dataset}_k{k}.pt"
    if path.exists():
        return torch.load(path)
    from sklearn.cluster import KMeans

    x = features.detach().float().cpu()
    a2x = structure.l2_rows(structure.a2x_trainfree(adj.detach().cpu().coalesce().float(), x)).numpy()
    labels = KMeans(n_clusters=int(k), n_init=KMEANS_N_INIT, random_state=KMEANS_SEED).fit_predict(a2x)
    out = torch.as_tensor(labels, dtype=torch.long)
    _atomic_save(out, path)
    return out


def adjacency_torch(edge_index, num_nodes, device):
    """Binary symmetric adjacency without self loops (hw.structure.normalize_graph) as a torch sparse CSR."""
    a = structure.normalize_graph(edge_index.detach().cpu().numpy() if torch.is_tensor(edge_index) else edge_index, num_nodes)
    return torch.sparse_csr_tensor(
        torch.as_tensor(a.indptr, dtype=torch.long), torch.as_tensor(a.indices, dtype=torch.long), torch.as_tensor(a.data, dtype=torch.float32), size=a.shape
    ).to(device)


@torch.no_grad()
def within_two_hops(a_torch, query_ids):
    """[Q, N] bool: shortest-path distance ≤ 2 from each query (the query itself included)."""
    n = a_torch.shape[0]
    onehot = torch.zeros((n, query_ids.numel()), dtype=torch.float32, device=query_ids.device)
    onehot[query_ids, torch.arange(query_ids.numel(), device=query_ids.device)] = 1.0
    h1 = a_torch @ onehot
    h2 = a_torch @ h1
    return ((onehot + h1 + h2) > 0).T


class StrategyNegatives:
    """HW06 far / band / cluster. `sample(support_ids, query_ids)` -> (ids [Q, m], fill 0.0), as ExtraNegatives."""

    def __init__(self, cfg, num_nodes, device, seed, dataset, features, adj, edge_index):
        mode = cfg["mode"]
        if mode not in STRATEGY_MODES:
            raise ValueError(f"StrategyNegatives mode must be one of {STRATEGY_MODES}, got {mode!r}")
        self.mode, self.m, self.num_nodes, self.device = mode, int(cfg["m"]), int(num_nodes), device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(seed) + STRATEGY_SEED_OFFSETS[mode])
        self.info = {}
        if mode in ("far", "band"):
            table = raw_cos_table(dataset, features)
        if mode == "far":
            self.top = table["ids"][:, :FAR_TOP].to(device)
            self.adj = adjacency_torch(edge_index, num_nodes, device)
        if mode == "band":
            lo, hi = int(cfg.get("band_lo", BAND_DEFAULT[0])), int(cfg.get("band_hi", BAND_DEFAULT[1]))
            if not (1 <= lo <= hi <= RAW_TABLE_K):
                raise ValueError(f"band_lo/band_hi must satisfy 1 <= lo <= hi <= {RAW_TABLE_K}")
            self.band = table["ids"][:, lo - 1 : hi].to(device)  # ranks lo..hi (1-based)
            self.info.update(band_lo=lo, band_hi=hi)
        if mode == "cluster":
            k = int(cfg.get("k", KMEANS_DEFAULT_K))
            self.cluster = kmeans_labels(dataset, features, adj, k).to(device)
            self.info.update(k=k)

    def _uniform(self, banned):
        """banned [Q, C] bool -> [Q, m] column indices, distinct, uniform over the allowed columns."""
        allowed = (~banned).sum(dim=1)
        if bool((allowed < self.m).any()):
            raise ValueError(f"extra_neg {self.mode}: a query has {int(allowed.min())} allowed candidates < m {self.m}")
        scores = torch.rand(banned.shape, generator=self.generator, device=self.device)
        return scores.masked_fill(banned, -1.0).topk(self.m, dim=1).indices

    @torch.no_grad()
    def sample(self, support_ids, query_ids):
        q_count = query_ids.numel()
        if self.m == 0:
            return torch.empty((q_count, 0), dtype=torch.long, device=self.device), 0.0
        excluded = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        excluded[support_ids] = True
        excluded[query_ids] = True
        if self.mode == "band":
            cand = self.band[query_ids]  # [Q, C]
            return cand.gather(1, self._uniform(excluded[cand])), 0.0
        banned = excluded[None, :].expand(q_count, -1).clone()
        if self.mode == "far":
            banned |= within_two_hops(self.adj, query_ids)
            banned.scatter_(1, self.top[query_ids], True)
        else:  # cluster
            banned |= self.cluster[None, :] == self.cluster[query_ids][:, None]
        return self._uniform(banned), 0.0
