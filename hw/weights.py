"""HSAN-style hard-sample weights (Liu et al., AAAI'23, Eq. 9-10), episode pseudo-label as Q.

For one loss term with Q queries and C candidates:
    s[q,c]  = cos(h_q, h_c)                      (h = detached GCN embedding or raw feature)
    s~      = (s - min s) / (max s - min s + 1e-12)   over the whole Q x C matrix of that term
    M[q,c]  = |P[q,c] - s~[q,c]|^beta            (P = same pseudo-class; no high-confidence set)
    mode p  : M[not P] := 1  (pulling only);  mode pn: both
Application to logits l:
    euclid_gradscale: l' = M*l + (1-M)*l.detach()   (value unchanged, gradient x M)
    cos_hsan        : l' = M*l                      (l = cos(z_q, z_c) / tau_cos)
No RNG is consumed; weights are computed under no_grad.
"""

import torch
import torch.nn.functional as F

DIAG_KEYS = ("pos_weight_mean", "neg_weight_mean", "pos_weight_lt05_frac", "pos_s_mean", "neg_s_mean")


@torch.no_grad()
def hsan_weights(h_query, h_cand, positive_mask, mode, beta):
    """h_query [Q, D]; h_cand [C, D] (shared candidates) or [Q, C, D] (per-query candidates).

    Returns (M [Q, C] float, diagnostics dict).
    """
    hq = F.normalize(h_query.detach().float(), p=2, dim=-1)
    hc = F.normalize(h_cand.detach().float(), p=2, dim=-1)
    if hc.dim() == 2:
        s = hq @ hc.t()
    else:
        s = torch.einsum("qd,qcd->qc", hq, hc)
    s_norm = (s - s.min()) / (s.max() - s.min() + 1e-12)
    pos = positive_mask.float()
    weights = (pos - s_norm).abs().pow(beta)
    if mode == "p":
        weights = torch.where(positive_mask, weights, torch.ones_like(weights))
    elif mode != "pn":
        raise ValueError(f"Unknown hsan mode: {mode}")
    neg_mask = ~positive_mask
    diagnostics = {
        "pos_weight_mean": float(weights[positive_mask].mean().item()),
        "neg_weight_mean": float(weights[neg_mask].mean().item()) if bool(neg_mask.any()) else None,
        "pos_weight_lt05_frac": float((weights[positive_mask] < 0.05).float().mean().item()),
        "pos_s_mean": float(s[positive_mask].mean().item()),
        "neg_s_mean": float(s[neg_mask].mean().item()) if bool(neg_mask.any()) else None,
    }
    return weights, diagnostics


def apply_weights(logits, weights, logit_kind):
    if logit_kind == "euclid_gradscale":
        return weights * logits + (1.0 - weights) * logits.detach()
    if logit_kind == "cos_hsan":
        return weights * logits
    raise ValueError(f"Unknown hsan logit: {logit_kind}")
