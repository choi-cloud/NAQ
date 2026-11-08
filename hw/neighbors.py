"""1-hop neighbor lookup and frozen selection embeddings (raw features / BGRL).

Transcribed from choi_exps/zip_20261107/exp015.py (raw) and exp016.py (BGRL).
"""

from pathlib import Path

import torch
import torch.nn.functional as F


def build_neighbor_lookup(edge_index, num_nodes):
    """Unique 1-hop neighbors of original graph; exclude self-loops."""
    u, v = edge_index.detach().cpu().long().tolist()
    lookup = [set() for _ in range(num_nodes)]
    for a, b in zip(u, v):
        if a != b:
            lookup[a].add(b)
            lookup[b].add(a)
    return [list(neighbors) for neighbors in lookup]


def load_selection_embeddings(selector, features, dataset, num_nodes, device, bgrl_dir=None):
    """Return (cosine-normalized selection embeddings, embedding path or None)."""
    if selector == "raw":
        return F.normalize(features.detach(), p=2, dim=1), None
    if selector == "bgrl":
        embedding_path = Path(bgrl_dir) / f"{dataset}.pt"
        if not embedding_path.is_file():
            raise FileNotFoundError(f"Missing BGRL embeddings: {embedding_path}")
        checkpoint = torch.load(embedding_path, map_location="cpu", weights_only=True)
        if checkpoint.get("dataset") != dataset:
            raise ValueError(f"BGRL dataset mismatch: {embedding_path}")
        pretrained = checkpoint["embeddings"].float()
        if pretrained.ndim != 2 or pretrained.shape[0] != num_nodes:
            raise ValueError("BGRL node count does not match dataset")
        node_ids = checkpoint.get("node_ids")
        if node_ids is None or not torch.equal(node_ids.cpu(), torch.arange(num_nodes)):
            raise ValueError("BGRL embeddings must follow original graph node order")
        return F.normalize(pretrained.to(device), p=2, dim=1), str(embedding_path)
    raise ValueError(f"Unknown selector: {selector}")


def select_neighbors(selection_embeddings_normalized, support_ids, query_ids, neighbor_lookup, max_neighbors):
    """Per (query, support) Top-1 1-hop neighbor of the support by frozen selection cosine.

    Returns (selected_neighbor_ids [Q, S], has_neighbor [Q, S], sizes list).
    """
    device = support_ids.device
    # Default: look through ALL original-graph 1-hop neighbors of each support.
    # Optionally limit the candidate pool using max_neighbors > 0.
    neighbor_lists = []
    for sid in support_ids.detach().cpu().tolist():
        neighbors = neighbor_lookup[sid]
        if max_neighbors > 0 and len(neighbors) > max_neighbors:
            picked = torch.randperm(len(neighbors))[:max_neighbors].tolist()
            neighbors = [neighbors[i] for i in picked]
        neighbor_lists.append(neighbors)

    sizes = [len(nodes) for nodes in neighbor_lists]
    largest = max(1, max(sizes))
    neighbor_ids = torch.zeros((len(sizes), largest), dtype=torch.long, device=device)
    neighbor_valid = torch.zeros((len(sizes), largest), dtype=torch.bool, device=device)
    for i, nodes in enumerate(neighbor_lists):
        if nodes:
            neighbor_ids[i, : len(nodes)] = torch.as_tensor(nodes, device=device)
            neighbor_valid[i, : len(nodes)] = True

    # [Q, S, K]; exclude query itself so it cannot trivially select its own embedding.
    valid = neighbor_valid[None, :, :] & (neighbor_ids[None, :, :] != query_ids[:, None, None])
    with torch.no_grad():
        selection_sim = torch.einsum("qd,skd->qsk", selection_embeddings_normalized[query_ids], selection_embeddings_normalized[neighbor_ids])
        selection_sim = selection_sim.masked_fill(~valid, -torch.inf)
        selected_positions = selection_sim.argmax(dim=-1)  # [Q, S]
        selected_neighbor_ids = neighbor_ids.unsqueeze(0).expand(query_ids.numel(), -1, -1).gather(2, selected_positions.unsqueeze(-1)).squeeze(-1)
        has_neighbor = valid.any(dim=-1)
        # If support has no eligible 1-hop neighbor, use its original ID.
        selected_neighbor_ids = torch.where(has_neighbor, selected_neighbor_ids, support_ids[None, :].expand_as(selected_neighbor_ids))
    return selected_neighbor_ids, has_neighbor, sizes


PICKS = ("max", "contrast", "random")


def select_neighbors_pick(selection_embeddings_normalized, support_ids, query_ids, neighbor_lookup, max_neighbors, pick="max", generator=None, selection_out=None):
    """Neighbor selection with `pick` (HW03a).

    pick "max": calls `select_neighbors` unchanged (argmax cos(neighbor, query)).
    pick "contrast": argmax [cos(e_n, e_support) - cos(e_n, e_query)] in the selection space.
    pick "random": uniform over eligible neighbors, drawn from `generator` only (no global RNG).
    Eligible neighbors and the fallback (support itself) are the same as in `select_neighbors`.
    selection_out (dict or None): receives `selected` [Q, S], `has_neighbor` [Q, S] and
    `valid_count` [Q, S] (eligible neighbors per pair) for diagnostics; no effect on selection.
    """
    if pick == SUP_COS:  # HW04a: separate path; the max/contrast/random lines below are unchanged
        return select_neighbors_sup_cos(neighbor_lookup, support_ids, query_ids, selection_out)
    if pick in CONTROL_PICKS:  # HW03b: separate path; the lines below are unchanged
        return select_neighbors_control(pick, selection_embeddings_normalized.shape[0], support_ids, query_ids, generator, selection_out)
    if pick not in PICKS:
        raise ValueError(f"pick must be one of {PICKS}, got {pick!r}")
    if pick == "max" and selection_out is None:
        return select_neighbors(selection_embeddings_normalized, support_ids, query_ids, neighbor_lookup, max_neighbors)
    if pick == "max":
        selected, has_neighbor, sizes = select_neighbors(selection_embeddings_normalized, support_ids, query_ids, neighbor_lookup, max_neighbors)
        valid, _ = _eligible(support_ids, query_ids, neighbor_lookup, max_neighbors)
        selection_out.update(selected=selected, has_neighbor=has_neighbor, valid_count=valid.sum(dim=-1))
        return selected, has_neighbor, sizes
    if max_neighbors > 0:
        raise ValueError("pick contrast/random is implemented for max_neighbors == 0 only")
    with torch.no_grad():
        valid, (neighbor_ids, sizes) = _eligible(support_ids, query_ids, neighbor_lookup, max_neighbors)
        if pick == "contrast":
            e = selection_embeddings_normalized
            to_support = torch.einsum("sd,skd->sk", e[support_ids], e[neighbor_ids])  # [S, K]
            to_query = torch.einsum("qd,skd->qsk", e[query_ids], e[neighbor_ids])  # [Q, S, K]
            score = to_support.unsqueeze(0) - to_query
        else:
            if generator is None:
                raise ValueError("pick 'random' needs a dedicated torch.Generator")
            score = torch.rand(valid.shape, generator=generator, device=valid.device)
        score = score.masked_fill(~valid, -torch.inf)
        positions = score.argmax(dim=-1)  # [Q, S]
        selected = neighbor_ids.unsqueeze(0).expand(query_ids.numel(), -1, -1).gather(2, positions.unsqueeze(-1)).squeeze(-1)
        has_neighbor = valid.any(dim=-1)
        selected = torch.where(has_neighbor, selected, support_ids[None, :].expand_as(selected))
    if selection_out is not None:
        selection_out.update(selected=selected, has_neighbor=has_neighbor, valid_count=valid.sum(dim=-1))
    return selected, has_neighbor, sizes


def _eligible(support_ids, query_ids, neighbor_lookup, max_neighbors):
    """Eligibility mask [Q, S, K] as in `select_neighbors` (1-hop neighbors of each support, query
    itself excluded). For max_neighbors > 0 the subsample in `select_neighbors` draws global RNG, so
    this helper (diagnostics / contrast / random) supports max_neighbors == 0 only."""
    if max_neighbors > 0:
        raise ValueError("neighbor diagnostics are implemented for max_neighbors == 0 only")
    device = support_ids.device
    neighbor_lists = [neighbor_lookup[sid] for sid in support_ids.detach().cpu().tolist()]
    sizes = [len(nodes) for nodes in neighbor_lists]
    largest = max(1, max(sizes))
    neighbor_ids = torch.zeros((len(sizes), largest), dtype=torch.long, device=device)
    neighbor_valid = torch.zeros((len(sizes), largest), dtype=torch.bool, device=device)
    for i, nodes in enumerate(neighbor_lists):
        if nodes:
            neighbor_ids[i, : len(nodes)] = torch.as_tensor(nodes, device=device)
            neighbor_valid[i, : len(nodes)] = True
    valid = neighbor_valid[None, :, :] & (neighbor_ids[None, :, :] != query_ids[:, None, None])
    return valid, (neighbor_ids, sizes)


# ---------------------------------------------------------------- HW04a: pick "sup_cos"
SUP_COS = "sup_cos"
ALL_PICKS = PICKS + (SUP_COS,)
SUP_COS_HOPS = (1, 2)
SUP_COS_SPACES = ("a2x", "raw")
SUP_COS_TABLE_CHUNK = 1024


class SupCosCandidates:
    """Candidate source for pick "sup_cos", passed in the `neighbor_lookup` slot of
    `select_neighbors_pick` (hw/losses.py forwards it unchanged).

    hop 1: candidates = the eligible 1-hop neighbors of `_eligible` (same set as pick "max");
           argmax cos(candidate, support) in `space` (L2-normalized rows of `embeddings`).
    hop 2: candidates = nodes at shortest-path distance exactly 2 (hw.structure.hop_sets_dense on
           hw.structure.normalize_graph). `top2` [N, 2] = per node the two hop-2 nodes with the
           largest cos to it (-1 = none), computed once before training; no max_neighbors cap.
    Deterministic, no RNG.
    """

    def __init__(self, hop, space, embeddings, neighbor_lookup, top2=None, hop2_size=None):
        self.hop, self.space = int(hop), space
        self.embeddings = embeddings
        self.neighbor_lookup = neighbor_lookup
        self.top2 = top2
        self.hop2_size = hop2_size


@torch.no_grad()
def sup_cos_space(space, features, adj):
    """L2-normalized rows of X (raw) or of Â²X (a2x, hw.structure.a2x_trainfree, Â = data.adj)."""
    from hw import structure

    x = features.detach().float()
    if space == "raw":
        return structure.l2_rows(x)
    if space == "a2x":
        return structure.l2_rows(structure.a2x_trainfree(adj.coalesce(), x))
    raise ValueError(f"pick_space must be one of {SUP_COS_SPACES}, got {space!r}")


@torch.no_grad()
def hop2_top2_table(embeddings, edge_index, num_nodes, chunk=SUP_COS_TABLE_CHUNK):
    """Per node: the two distance-2 nodes with the largest cos (ties: torch.topk order), -1 if
    fewer exist; and the size of its distance-2 set."""
    import numpy as np

    from hw import structure

    a = structure.normalize_graph(edge_index.detach().cpu().numpy(), num_nodes)
    device = embeddings.device
    top2 = torch.full((num_nodes, 2), -1, dtype=torch.long, device=device)
    size = torch.zeros(num_nodes, dtype=torch.long, device=device)
    for start in range(0, num_nodes, chunk):
        rows = np.arange(start, min(num_nodes, start + chunk))
        _, h2, _ = structure.hop_sets_dense(a, rows)
        mask = torch.as_tensor(h2, device=device)
        cos = embeddings[start:start + rows.size] @ embeddings.T
        cos = cos.masked_fill(~mask, -torch.inf)
        score, idx = cos.topk(2, dim=1)
        idx[~torch.isfinite(score)] = -1
        top2[start:start + rows.size] = idx
        size[start:start + rows.size] = mask.sum(dim=1)
    return top2, size


def build_sup_cos(hop, space, features, adj, edge_index, num_nodes, neighbor_lookup):
    if int(hop) not in SUP_COS_HOPS:
        raise ValueError(f"pick_hop must be one of {SUP_COS_HOPS}, got {hop!r}")
    embeddings = sup_cos_space(space, features, adj)
    if int(hop) == 1:
        return SupCosCandidates(hop, space, embeddings, neighbor_lookup)
    top2, size = hop2_top2_table(embeddings, edge_index, num_nodes)
    return SupCosCandidates(hop, space, embeddings, neighbor_lookup, top2=top2, hop2_size=size)


@torch.no_grad()
def select_neighbors_sup_cos(candidates, support_ids, query_ids, selection_out=None):
    """pick "sup_cos". Returns (selected [Q, S], has_neighbor [Q, S], sizes list) like `select_neighbors`.

    The query of the pair is never chosen; no usable candidate -> support itself (fallback).
    hop 2: rank-1 of `top2` unless it is the query, then rank-2. valid_count = usable entries of top2.
    """
    if not isinstance(candidates, SupCosCandidates):
        raise TypeError("pick 'sup_cos' needs a SupCosCandidates object in the neighbor_lookup slot")
    e = candidates.embeddings
    q = query_ids[:, None]
    if candidates.hop == 1:
        valid, (neighbor_ids, sizes) = _eligible(support_ids, query_ids, candidates.neighbor_lookup, 0)
        to_support = torch.einsum("sd,skd->sk", e[support_ids], e[neighbor_ids])  # [S, K]
        score = to_support.unsqueeze(0).expand(query_ids.numel(), -1, -1).masked_fill(~valid, -torch.inf)
        positions = score.argmax(dim=-1)
        selected = neighbor_ids.unsqueeze(0).expand(query_ids.numel(), -1, -1).gather(2, positions.unsqueeze(-1)).squeeze(-1)
        has_neighbor = valid.any(dim=-1)
        valid_count = valid.sum(dim=-1)
    else:
        t = candidates.top2[support_ids]  # [S, 2]
        first, second = t[None, :, 0], t[None, :, 1]
        use1 = (first >= 0) & (first != q)
        use2 = (second >= 0) & (second != q)
        selected = torch.where(use1, first, second).expand(query_ids.numel(), -1)
        has_neighbor = use1 | use2
        valid_count = use1.long() + use2.long()
        sizes = candidates.hop2_size[support_ids].tolist()
    selected = torch.where(has_neighbor, selected, support_ids[None, :].expand_as(selected))
    if selection_out is not None:
        selection_out.update(selected=selected, has_neighbor=has_neighbor, valid_count=valid_count)
    return selected, has_neighbor, sizes


# ---------------------------------------------------------------- HW03b: control picks
CONTROL_PICKS = ("self", "global_random")
ALL_PICKS = ALL_PICKS + CONTROL_PICKS


@torch.no_grad()
def select_neighbors_control(pick, num_nodes, support_ids, query_ids, generator=None, selection_out=None):
    """pick "self": positive = the support itself for every (query, support) pair.
    pick "global_random": per pair one node drawn uniformly from all nodes except that support and
    that query, from `generator` only (no global RNG). Draw: r ~ U{0..N-1-m} (m = number of distinct
    excluded ids), then r is shifted past the excluded ids in ascending order.
    Returns (selected [Q, S], has_neighbor [Q, S] (all True), sizes list) like `select_neighbors`.
    """
    q_count, s_count = query_ids.numel(), support_ids.numel()
    device = support_ids.device
    sup = support_ids[None, :].expand(q_count, -1)
    if pick == "self":
        selected = sup.clone()
        sizes = [1] * s_count
        valid_count = torch.ones((q_count, s_count), dtype=torch.long, device=device)
    elif pick == "global_random":
        if generator is None:
            raise ValueError("pick 'global_random' needs a dedicated torch.Generator")
        qry = query_ids[:, None].expand(-1, s_count)
        lo, hi = torch.minimum(sup, qry), torch.maximum(sup, qry)
        distinct = lo != hi
        span = num_nodes - 1 - distinct.long()  # number of allowed nodes
        u = torch.rand((q_count, s_count), generator=generator, device=device)
        r = torch.minimum((u * span).long(), span - 1)
        r = r + (r >= lo).long()
        r = r + (distinct & (r >= hi)).long()
        selected = r
        sizes = [num_nodes - 2] * s_count
        valid_count = span
    else:
        raise ValueError(f"control pick must be one of {CONTROL_PICKS}, got {pick!r}")
    has_neighbor = torch.ones((q_count, s_count), dtype=torch.bool, device=device)
    if selection_out is not None:
        selection_out.update(selected=selected, has_neighbor=has_neighbor, valid_count=valid_count)
    return selected, has_neighbor, sizes
