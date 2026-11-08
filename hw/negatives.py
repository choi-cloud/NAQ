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
