"""Episode losses: original (support InfoNCE) and neighbor (selected-neighbor InfoNCE).

Transcribed from choi_exps/zip_20261107/exp015.py `episode_contrastive_loss`.
loss == "original": the neighbor term and neighbor selection are not computed.
"""

import numpy as np
import torch

from hw.neighbors import select_neighbors


def _info_nce(logits, positive_mask):
    log_pos = torch.logsumexp(logits.masked_fill(~positive_mask, -torch.inf), dim=1) - positive_mask.sum(dim=1).float().log()
    return (torch.logsumexp(logits, dim=1) - log_pos).mean()


def episode_loss(
    embeddings,
    id_spt,
    y_spt,
    id_query,
    y_query,
    temperature,
    loss_kind,
    selection_embeddings_normalized=None,
    neighbor_lookup=None,
    lambda_neighbor=0.0,
    max_neighbors=0,
):
    device = embeddings.device
    support_ids = torch.as_tensor(id_spt, dtype=torch.long, device=device).flatten()
    query_ids = torch.as_tensor(id_query, dtype=torch.long, device=device).flatten()
    support_labels = torch.as_tensor(y_spt, dtype=torch.long, device=device).flatten()
    query_labels = torch.as_tensor(y_query, dtype=torch.long, device=device).flatten()
    if support_ids.numel() != support_labels.numel():
        raise ValueError("Support IDs/labels mismatch")
    if query_ids.numel() != query_labels.numel():
        raise ValueError("Query IDs/labels mismatch")

    supports = embeddings[support_ids]  # [S, D]
    queries = embeddings[query_ids]  # [Q, D]

    # (1) Original support-vs-query contrastive objective.
    direct_dists = (queries[:, None, :] - supports[None, :, :]).square().sum(-1)
    direct_logits = -direct_dists / temperature  # [Q, S]
    positive_mask = query_labels[:, None] == support_labels[None, :]
    if not bool(positive_mask.any(dim=1).all()):
        raise ValueError("Every query needs a positive support")
    original_loss = _info_nce(direct_logits, positive_mask)

    diagnostics = {"original_loss": float(original_loss.detach().item())}
    if loss_kind == "original":
        loss = original_loss
        diagnostics.update(neighbor_loss=None, mean_support_neighbors_considered=None, fraction_valid_neighbor_selections=None)
    elif loss_kind == "original+neighbor":
        # (2) Per-query, per-support neighbor selected by frozen selection embeddings.
        selected_neighbor_ids, has_neighbor, sizes = select_neighbors(selection_embeddings_normalized, support_ids, query_ids, neighbor_lookup, max_neighbors)
        neighbor_vectors = embeddings[selected_neighbor_ids]  # [Q, S, D]
        neighbor_dists = (queries[:, None, :] - neighbor_vectors).square().sum(-1)
        neighbor_logits = -neighbor_dists / temperature
        neighbor_loss = _info_nce(neighbor_logits, positive_mask)
        loss = original_loss + lambda_neighbor * neighbor_loss
        diagnostics.update(
            neighbor_loss=float(neighbor_loss.detach().item()),
            mean_support_neighbors_considered=float(np.mean(sizes)),
            fraction_valid_neighbor_selections=float(has_neighbor.float().mean().item()),
        )
    else:
        raise ValueError(f"Unknown loss: {loss_kind}")

    classes = support_labels.unique(sorted=True)
    class_scores = torch.stack([-direct_dists[:, support_labels == cls].mean(dim=1) for cls in classes], dim=1)
    prediction = classes[class_scores.argmax(dim=1)]
    accuracy = float((prediction == query_labels).float().mean().detach().item())
    return loss, accuracy, diagnostics
