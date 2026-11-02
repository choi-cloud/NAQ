"""ND01 gate B: same-class AUROC of d_v anchor-slot scores on B4 candidates (no training); candidates and ax/a2x/enc from hl/hl008_distant_query_train.b4_scores.

Labels: only grade_scores (hl_common.labels_np). Anchors, candidates and all scores are label-free.
Writes results/nd/nd01/gateB/{anchors.npy, auroc_rows.npz, summary.json}.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import nd_vote as V
from nd_vote import H
from hl005_diag_distance_purity import CKPTS, auroc_pooled, auroc_rows, encoder_embeddings, sym_adjacency
from hl008_distant_query_train import B4, b4_scores

N_EPISODES = 400
N_WAY = 5
BATCH = 256
OUT = V.OUT / "gateB"


def sample_anchors(n):
    rng = np.random.default_rng(V.SEED)
    return np.stack([rng.choice(n, N_WAY, replace=False) for _ in range(N_EPISODES)])


def base_scores(data, device, flat, z1, z2, emb):
    """Candidates [A, 800] (X rank 201-1000, self excluded) and ax / a2x / enc columns from hl008.b4_scores. No labels."""
    a_sym = sym_adjacency(data, device)
    ns = argparse.Namespace(ppr_alpha=0.15, ppr_iters=30, tau=0.3)
    ranked, cols = [], {"ax": [], "a2x": [], "enc_n": []}
    for b in range(0, len(flat), BATCH):
        r, s = b4_scores(data, device, flat[b:b + BATCH], z1, z2, a_sym, emb, ns)
        ranked.append(r[:, B4[0] - 1:B4[1]])
        for k in cols:
            cols[k].append(s[k].astype(np.float32))
    return np.concatenate(ranked), {k: np.concatenate(v) for k, v in cols.items()}


def slot_scores(anchors, cands, zs, weights, device):
    """own_soft_*, own_hard_*, vote_* columns [A, 800]: the score of candidate c at the slot of its anchor a. No labels."""
    A = anchors.size
    out = {}

    def put(name, ep, block):
        # block [5 * 800, 5] -> anchor j's 800 candidates at slot j
        block = block.reshape(N_WAY, 800, N_WAY)
        vals = torch.stack([block[j, :, j] for j in range(N_WAY)]).cpu().numpy()
        if name not in out:
            out[name] = np.empty((A, 800), dtype=np.float32)
        out[name][ep * N_WAY:(ep + 1) * N_WAY] = vals

    for ep in range(len(anchors)):
        ids = torch.as_tensor(anchors[ep], device=device)
        idx = torch.as_tensor(cands[ep * N_WAY:(ep + 1) * N_WAY].reshape(-1), device=device)
        for f, z in zs.items():
            S = V.similarity(z, z[ids])
            for t in V.TAU_KEYS:
                Q = V.assign_from_s(S, t)
                put(f"own_hard_{f}" if t == "hard" else f"own_soft_{f}_{t}", ep, Q[idx])
                for nbr in V.NBRS[f]:
                    if nbr in weights:
                        put(V.arm_name(f, t, nbr), ep, V.vote_rows(Q, weights[nbr], idx))
        if ep % 100 == 0:
            print(f"episode {ep}", flush=True)
    return out


def grade_scores(scores, cands, flat, data):
    """AUROC per score column. Reads labels (hl_common.labels_np); the only place in gate B that does."""
    labels = H.labels_np(data)
    same = labels[cands] == labels[flat][:, None]
    order = np.argsort(-scores["a2x"], axis=1, kind="stable")[:, 400:]  # far50: B4 columns at a2x rank 401-800
    same_far = np.take_along_axis(same, order, 1)
    both = (same.sum(1) > 0) & (same.sum(1) < same.shape[1])
    both_far = (same_far.sum(1) > 0) & (same_far.sum(1) < same_far.shape[1])
    summary, rows_all, rows_far = {}, {}, {}
    for k, s in scores.items():
        ra = auroc_rows(s[both], same[both])
        sf = np.take_along_axis(s, order, 1)
        rf = auroc_rows(sf[both_far], same_far[both_far])
        rows_all[k], rows_far[k] = ra, rf
        summary[k] = {"auroc_all": float(np.nanmean(ra)), "n_rows": int(both.sum()), "auroc_pooled": auroc_pooled(s, same),
                      "auroc_far50": float(np.nanmean(rf)), "n_rows_far50": int(both_far.sum())}
    base = {"same_rate": float(same.mean()), "rows_without_both": int((~both).sum()), "rows_without_both_far50": int((~both_far).sum())}
    return summary, rows_all, rows_far, both, both_far, base


def main():
    t0 = time.time()
    args = H.base_args()
    data, device = H.load_everything(args)
    z0, z1, z2 = H.structural_scores(data, device)
    zs = {"x": z0, "a2x": z2}
    OUT.mkdir(parents=True, exist_ok=True)
    anchors = sample_anchors(z0.size(0))
    np.save(OUT / "anchors.npy", anchors)
    flat = anchors.reshape(-1)
    ckpt = Path(CKPTS["enc_b"])
    has_enc = ckpt.exists()
    emb = encoder_embeddings(data, str(ckpt), args.latent, device) if has_enc else z1
    with torch.no_grad():
        weights, info = V.build_weights(data, device)
        cands, base = base_scores(data, device, flat, z1, z2, emb)
        scores = {"ax": base["ax"], "a2x": base["a2x"]}
        if has_enc:
            scores["enc_b"] = base["enc_n"]
        t1 = time.time()
        scores.update(slot_scores(anchors, cands, zs, weights, device))
        t2 = time.time()
    summary, rows_all, rows_far, both, both_far, gbase = grade_scores(scores, cands, flat, data)
    np.savez_compressed(OUT / "auroc_rows.npz", both=both, both_far=both_far,
                        **{f"all__{k}": v for k, v in rows_all.items()}, **{f"far__{k}": v for k, v in rows_far.items()})
    meta = {"summary": summary, "grade_base": gbase, "enc_b_available": has_enc, "weights": info,
            "unique_anchors": int(len(np.unique(flat))), "n_anchors": int(flat.size), "n_columns": len(scores),
            "slot_scores_s": t2 - t1, "elapsed_s": time.time() - t0,
            "gpu_max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else None}
    (OUT / "summary.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "summary"}, indent=1))


if __name__ == "__main__":
    main()
