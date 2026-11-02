"""ND02 §1 episode generation (no training); query mining from hl/hl_common.mine_queries_for, X-rank exclusion as hl/hl008_distant_query_train.b4_scores.

pseudo_k3, pseudo_k5, pseudo_choi never read labels. gt_episodes (file gt_k3.npz) is the oracle set and the only function here that does.
"""
import json
from pathlib import Path

import numpy as np
import torch

import nd_vote as V
from nd_vote import H

E, N, M = 300, 5, 10
OUT = Path("results/nd/nd02/episodes")
DEPTH = 400


def sample_anchors(n):
    rng = np.random.default_rng(V.SEED)
    return np.stack([rng.choice(n, N, replace=False) for _ in range(E)])


def pseudo_episodes(data, device, anchors, k):
    """Disjoint groups: per anchor in order, top (k-1+M) X-cosine nodes, skipping every anchor of the episode and nodes already taken."""
    support = np.empty((E, N, k), dtype=np.int64)
    query = np.empty((E, N, M), dtype=np.int64)
    with torch.no_grad():
        for e in range(E):
            a = torch.as_tensor(anchors[e], device=device)
            rows = data.Diffusion_matrix[a].clone()
            rows[torch.arange(N, device=device), a] = -float("inf")
            ranked = torch.sort(rows, dim=1, descending=True, stable=True).indices[:, :DEPTH].cpu().numpy()
            taken = set(anchors[e].tolist())
            for g in range(N):
                got = []
                for c in ranked[g].tolist():
                    if c not in taken:
                        got.append(c)
                        taken.add(c)
                        if len(got) == k - 1 + M:
                            break
                assert len(got) == k - 1 + M
                support[e, g, 0] = anchors[e, g]
                support[e, g, 1:] = got[:k - 1]
                query[e, g] = got[k - 1:]
            assert len(np.unique(np.concatenate((support[e].ravel(), query[e].ravel())))) == N * (k + M)
    return support, query


def choi_episodes(data, args, device, anchors, support):
    """Support = pseudo_k3. Query = B_Choi mining per anchor; a mined node already used in the episode is dropped without refill (-1 padded)."""
    mined = H.mine_queries_for(np.unique(anchors), data, args, device)
    query = np.full((E, N, M), -1, dtype=np.int64)
    removed = {"own_support": 0, "other_support": 0, "earlier_group_query": 0}
    fallback = 0
    for e in range(E):
        used = {}
        for g in range(N):
            for c in support[e, g].tolist():
                used[c] = (g, "s")
        for g in range(N):
            d = mined[int(anchors[e, g])]
            fallback += bool(d["fallback"])
            keep = []
            for c in d["selected_ids"]:
                c = int(c)
                if c in used:
                    owner, kind = used[c]
                    removed["own_support" if kind == "s" and owner == g else "other_support" if kind == "s" else "earlier_group_query"] += 1
                    continue
                used[c] = (g, "q")
                keep.append(c)
            query[e, g, :len(keep)] = keep
    return query, {"removed": removed, "fallback_anchor_slots": fallback, "kept_queries": int((query >= 0).sum()), "total_query_slots": E * N * M}


def gt_episodes(data, k):
    """Oracle episodes (labels read): 5 classes without replacement from all classes, k + M uniform nodes per class; classes under k + M nodes are redrawn."""
    labels = H.labels_np(data)
    rng = np.random.default_rng(V.SEED)
    classes = np.unique(labels)
    by_class = {c: np.flatnonzero(labels == c) for c in classes}
    support = np.empty((E, N, k), dtype=np.int64)
    query = np.empty((E, N, M), dtype=np.int64)
    redraws = 0
    for e in range(E):
        chosen = []
        while len(chosen) < N:
            c = rng.choice(classes)
            if c in chosen:
                continue
            if len(by_class[c]) < k + M:
                redraws += 1
                continue
            chosen.append(c)
        for g, c in enumerate(chosen):
            nodes = rng.choice(by_class[c], k + M, replace=False)
            support[e, g], query[e, g] = nodes[:k], nodes[k:]
    info = {"n_classes": int(len(classes)), "classes_under_size": int(sum(len(v) < k + M for v in by_class.values())), "redraws": redraws}
    return support, query, info


def main():
    args = H.base_args()
    data, device = H.load_everything(args)
    OUT.mkdir(parents=True, exist_ok=True)
    anchors = sample_anchors(data.features.size(0))
    info = {}
    s3, q3 = pseudo_episodes(data, device, anchors, 3)
    np.savez_compressed(OUT / "pseudo_k3.npz", anchors=anchors, support=s3, query=q3)
    s5, q5 = pseudo_episodes(data, device, anchors, 5)
    np.savez_compressed(OUT / "pseudo_k5.npz", anchors=anchors, support=s5, query=q5)
    qc, info["pseudo_choi"] = choi_episodes(data, args, device, anchors, s3)
    np.savez_compressed(OUT / "pseudo_choi.npz", anchors=anchors, support=s3, query=qc)
    sg, qg, info["gt_k3"] = gt_episodes(data, 3)
    np.savez_compressed(OUT / "gt_k3.npz", anchors=sg[:, :, 0], support=sg, query=qg)
    info["unique_anchors"] = int(len(np.unique(anchors)))
    (OUT / "generation.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
