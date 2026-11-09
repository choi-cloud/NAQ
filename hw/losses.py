"""Episode losses: original (support InfoNCE) and neighbor (selected-neighbor InfoNCE).

Transcribed from choi_exps/zip_20261107/exp015.py `episode_contrastive_loss`.
loss == "original": the neighbor term and neighbor selection are not computed.
hsan is None: the code path below `episode_loss` is the HW01 path, unchanged.
hsan given: `episode_loss_hsan` (HW02) — HSAN weights on the orig / both terms (hw/weights.py).
pick (HW03a): neighbor choice, hw.neighbors.select_neighbors_pick; "max" = HW01 selection.
selection_out: optional dict filled with the selected neighbor ids (diagnostics only).
"""

import numpy as np
import torch
import torch.nn.functional as F

from hw.negatives import neighbor_term
from hw.neighbors import select_neighbors_pick
from hw.weights import apply_weights, hsan_weights


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
    hsan=None,
    space_features=None,
    pick="max",
    generator=None,
    selection_out=None,
    nbr_term="infonce",
    extra_neg=None,
    xneg_out=None,
    ep_query_neg=False,
    ep_query_pos=False,
    qep_out=None,
):
    if hsan is not None and (nbr_term != "infonce" or extra_neg is not None):  # HW05
        raise ValueError("nbr_term / extra_neg are not combined with hsan")
    if hsan is not None and (ep_query_neg or ep_query_pos):  # HW06
        raise ValueError("ep_query_neg / ep_query_pos are not combined with hsan")
    if ep_query_pos and not ep_query_neg:  # HW06
        raise ValueError("ep_query_pos is used only together with ep_query_neg")
    if hsan is not None:
        return episode_loss_hsan(
            embeddings, id_spt, y_spt, id_query, y_query, temperature, selection_embeddings_normalized, neighbor_lookup, lambda_neighbor, max_neighbors, hsan, space_features,
            pick=pick, generator=generator, selection_out=selection_out,
        )
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
    if extra_neg is not None:  # HW05: L_original with m extra negatives per query (no λ, no separate term)
        if getattr(extra_neg, "needs_episode_labels", False):  # HW07 cluster_ref anchor: pseudo labels only
            extra_neg.set_episode(support_labels, query_labels)
        extra_ids, fill = extra_neg.sample(support_ids, query_ids)  # [Q, m]
        extra_logits = -(queries[:, None, :] - embeddings[extra_ids]).square().sum(-1) / temperature
        extended_mask = torch.cat([positive_mask, torch.zeros_like(extra_logits, dtype=torch.bool)], dim=1)
        original_loss = _info_nce(torch.cat([direct_logits, extra_logits], dim=1), extended_mask)
        if xneg_out is not None:
            xneg_out.update(extra=extra_ids, fill=fill)
    if ep_query_neg:  # HW06: other-pseudo-class pseudo-queries as negatives (+ same-pseudo-class ones as positives)
        q_count = query_ids.numel()
        same = query_labels[:, None] == query_labels[None, :]
        other = ~same
        mate = same & ~torch.eye(q_count, dtype=torch.bool, device=device)
        n_neg, n_pos = int(other.sum(dim=1)[0]), int(mate.sum(dim=1)[0])
        if not (bool((other.sum(dim=1) == n_neg).all()) and bool((mate.sum(dim=1) == n_pos).all())):
            raise ValueError("ep_query_neg expects equal-size pseudo-classes")
        qq_logits = -(queries[:, None, :] - queries[None, :, :]).square().sum(-1) / temperature  # [Q, Q]
        parts = [direct_logits] + ([extra_logits] if extra_neg is not None else []) + [qq_logits[other].view(q_count, n_neg)]
        masks = [positive_mask] + ([torch.zeros_like(extra_logits, dtype=torch.bool)] if extra_neg is not None else []) + [torch.zeros((q_count, n_neg), dtype=torch.bool, device=device)]
        if ep_query_pos:
            parts.append(qq_logits[mate].view(q_count, n_pos))
            masks.append(torch.ones((q_count, n_pos), dtype=torch.bool, device=device))
        original_loss = _info_nce(torch.cat(parts, dim=1), torch.cat(masks, dim=1))
        if qep_out is not None:
            col = torch.arange(q_count, device=device)[None, :].expand(q_count, -1)
            qep_out.update(neg=query_ids[col[other].view(q_count, n_neg)], pos=query_ids[col[mate].view(q_count, n_pos)] if ep_query_pos else None)

    diagnostics = {"original_loss": float(original_loss.detach().item())}
    if loss_kind == "original":
        loss = original_loss
        diagnostics.update(neighbor_loss=None, mean_support_neighbors_considered=None, fraction_valid_neighbor_selections=None)
    elif loss_kind == "original+neighbor":
        # (2) Per-query, per-support neighbor selected by frozen selection embeddings.
        selected_neighbor_ids, has_neighbor, sizes = select_neighbors_pick(selection_embeddings_normalized, support_ids, query_ids, neighbor_lookup, max_neighbors, pick, generator, selection_out)
        neighbor_vectors = embeddings[selected_neighbor_ids]  # [Q, S, D]
        neighbor_dists = (queries[:, None, :] - neighbor_vectors).square().sum(-1)
        neighbor_logits = -neighbor_dists / temperature
        neighbor_loss = _info_nce(neighbor_logits, positive_mask)
        if nbr_term != "infonce":  # HW05: pull-only / push-only split of the line above
            neighbor_loss = neighbor_term(neighbor_logits, positive_mask, nbr_term)
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


def _cos_logits(queries, candidates, tau_cos):
    """queries [Q, D]; candidates [C, D] or [Q, C, D] -> cos / tau_cos, [Q, C]."""
    q = F.normalize(queries, p=2, dim=-1)
    c = F.normalize(candidates, p=2, dim=-1)
    if c.dim() == 2:
        return (q @ c.t()) / tau_cos
    return torch.einsum("qd,qcd->qc", q, c) / tau_cos


def episode_loss_hsan(
    embeddings,
    id_spt,
    y_spt,
    id_query,
    y_query,
    temperature,
    selection_embeddings_normalized,
    neighbor_lookup,
    lambda_neighbor,
    max_neighbors,
    hsan,
    space_features=None,
    pick="max",
    generator=None,
    selection_out=None,
):
    """L_original + lambda * L_neighbor with HSAN weights M on the selected terms.

    hsan: validated block (hw.config.validate_hsan). mode "none" = no weights, cos logits.
    space_features: cosine space for hardness when hsan.space == "raw"; None -> detached GCN.
    """
    device = embeddings.device
    support_ids = torch.as_tensor(id_spt, dtype=torch.long, device=device).flatten()
    query_ids = torch.as_tensor(id_query, dtype=torch.long, device=device).flatten()
    support_labels = torch.as_tensor(y_spt, dtype=torch.long, device=device).flatten()
    query_labels = torch.as_tensor(y_query, dtype=torch.long, device=device).flatten()
    if support_ids.numel() != support_labels.numel():
        raise ValueError("Support IDs/labels mismatch")
    if query_ids.numel() != query_labels.numel():
        raise ValueError("Query IDs/labels mismatch")

    mode = hsan["mode"]
    logit_kind = hsan["logit"]
    tau_cos = float(hsan["tau_cos"])
    weighted_terms = set() if mode == "none" else ({"orig"} if hsan["apply"] == "orig" else {"orig", "nbr"})
    space = embeddings.detach() if hsan.get("space", "gcn") == "gcn" else space_features

    supports = embeddings[support_ids]  # [S, D]
    queries = embeddings[query_ids]  # [Q, D]
    direct_dists = (queries[:, None, :] - supports[None, :, :]).square().sum(-1)
    positive_mask = query_labels[:, None] == support_labels[None, :]
    if not bool(positive_mask.any(dim=1).all()):
        raise ValueError("Every query needs a positive support")
    diagnostics = {}

    # (1) orig term: query vs episode supports.
    if logit_kind == "cos_hsan":
        direct_logits = _cos_logits(queries, supports, tau_cos)
    else:
        direct_logits = -direct_dists / temperature  # [Q, S]
    if "orig" in weighted_terms:
        weights, diag = hsan_weights(space[query_ids], space[support_ids], positive_mask, mode, hsan["beta"])
        direct_logits = apply_weights(direct_logits, weights, logit_kind)
        diagnostics.update({f"orig_{k}": v for k, v in diag.items()})
    original_loss = _info_nce(direct_logits, positive_mask)

    # (2) nbr term: query vs per-(query, class) selected neighbor.
    selected_neighbor_ids, has_neighbor, sizes = select_neighbors_pick(selection_embeddings_normalized, support_ids, query_ids, neighbor_lookup, max_neighbors, pick, generator, selection_out)
    neighbor_vectors = embeddings[selected_neighbor_ids]  # [Q, S, D]
    if logit_kind == "cos_hsan":
        neighbor_logits = _cos_logits(queries, neighbor_vectors, tau_cos)
    else:
        neighbor_dists = (queries[:, None, :] - neighbor_vectors).square().sum(-1)
        neighbor_logits = -neighbor_dists / temperature
    if "nbr" in weighted_terms:
        weights, diag = hsan_weights(space[query_ids], space[selected_neighbor_ids], positive_mask, mode, hsan["beta"])
        neighbor_logits = apply_weights(neighbor_logits, weights, logit_kind)
        diagnostics.update({f"nbr_{k}": v for k, v in diag.items()})
    neighbor_loss = _info_nce(neighbor_logits, positive_mask)
    loss = original_loss + lambda_neighbor * neighbor_loss

    classes = support_labels.unique(sorted=True)
    class_scores = torch.stack([-direct_dists[:, support_labels == cls].mean(dim=1) for cls in classes], dim=1)
    prediction = classes[class_scores.argmax(dim=1)]
    accuracy = float((prediction == query_labels).float().mean().detach().item())
    diagnostics = {
        "original_loss": float(original_loss.detach().item()),
        "neighbor_loss": float(neighbor_loss.detach().item()),
        "mean_support_neighbors_considered": float(np.mean(sizes)),
        "fraction_valid_neighbor_selections": float(has_neighbor.float().mean().item()),
        **diagnostics,
    }
    return loss, accuracy, diagnostics
