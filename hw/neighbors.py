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
