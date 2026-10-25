"""hl007_diag_topk_purity: derived from hl/hl005_diag_distance_purity.py — same anchors, bands and scores (imported from hl005, not re-implemented); only the selection rules differ. Reads GT labels for grading only.

Rules per anchor within a band: top:k (k = 10, 20, 30), mid:m (top m by score, then the X-rank centre 10; m = 30, 60),
choi (filter-passed candidates, X-rank centre 10; anchors with < 10 passed are excluded), rand (10 uniform, numpy seed).
Outputs (results/hl/hl04/g0b/ and reports/HL04_tables/): G0b_topk.csv, G0b_summary.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hl_common as H  # noqa: E402
from hl005_diag_distance_purity import BANDS, CKPTS, encoder_embeddings, pair_scores, ppr_columns, sym_adjacency, write_csv  # noqa: E402

GRADE_BANDS = ("B3", "B4", "B5")
SCORED = ["x", "ax", "a2x", "ppr", "mv", "enc_b", "enc_n", "comb"]
RULES = ["top:10", "top:20", "top:30", "mid:30", "mid:60"]
Q = 10


def select(s, rule):
    """Boolean [A, m] selection for a scored rule; ties broken by X rank (column order, stable) as in hl005.top_k_mask."""
    kind, size = rule.split(":")
    size = int(size)
    order = np.argsort(-s, axis=1, kind="stable")[:, :size]
    if kind == "mid":
        order = np.sort(order, axis=1)  # back to X-rank order within the top m
        mid = (size - Q) // 2
        order = order[:, mid:mid + Q]
    mask = np.zeros_like(s, dtype=bool)
    np.put_along_axis(mask, order, True, axis=1)
    return mask


def select_choi(passed):
    """X-rank centre 10 of filter-passed candidates (B_Choi's middle selection inside the band); rows with < 10 passed are all False."""
    mask = np.zeros_like(passed, dtype=bool)
    for i, row in enumerate(passed):
        idx = np.flatnonzero(row)
        if len(idx) >= Q:
            mid = (len(idx) - Q) // 2
            mask[i, idx[mid:mid + Q]] = True
    return mask


def select_rand(shape, seed):
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, dtype=bool)
    for i in range(shape[0]):
        mask[i, rng.choice(shape[1], Q, replace=False)] = True
    return mask


def grade_rows(band, score, rule, mask, y, passed, lo):
    used = mask.any(1)
    xr = np.arange(mask.shape[1])[None, :] + lo  # global X rank of each column
    k = mask.sum(1)
    return {
        "band": band, "score": score, "rule": rule,
        "purity_pooled": float(y[mask].mean()),
        "purity_anchor_mean": float(((y & mask).sum(1)[used] / k[used]).mean()),
        "mean_x_rank": float(np.broadcast_to(xr, mask.shape)[mask].mean()),
        "choi_pass_frac": float(passed[mask].mean()),
        "n_anchors": int(used.sum()),
        "excluded_frac": float(1 - used.mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--n_anchors", type=int, default=2000)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--ppr_iters", type=int, default=30)
    p.add_argument("--chunk", type=int, default=500)
    p.add_argument("--anchors", default="results/hl/hl03/g0/anchors.npy")
    p.add_argument("--output_dir", default="results/hl/hl04/g0b")
    p.add_argument("--table_dir", default="reports/HL04_tables")
    a = p.parse_args()
    timer, times = H.Timer(), {}
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    args = H.base_args(device=a.device, seed=a.seed, tau=a.tau)
    data, device = H.load_everything(args)
    n = data.features.size(0)
    times["load"] = timer()
    anchors = np.load(a.anchors)
    regenerated = np.random.default_rng(a.seed).choice(n, a.n_anchors, replace=False)
    if not np.array_equal(anchors, regenerated):
        raise RuntimeError("anchors.npy differs from the hl005 draw")

    # ---- label-free scores: identical to hl005 ----
    z = H.structural_scores(data, device)
    check = H.verify_against_exp049(data, args, device, z=z)
    ppr = ppr_columns(sym_adjacency(data, device), anchors, a.alpha, a.ppr_iters)
    emb = {k: encoder_embeddings(data, path, args.latent, device) for k, path in CKPTS.items()}
    cand, scores = pair_scores(data, device, anchors, z, ppr, emb, a.tau, a.chunk)
    times["scores"] = timer()
    del ppr, emb, z
    torch.cuda.empty_cache()

    # ---- grading (GT labels read only here) ----
    labels = H.labels_np(data)
    same = labels[cand] == labels[anchors][:, None]
    rows = []
    for band in GRADE_BANDS:
        lo, hi = BANDS[band]
        cols = slice(lo - 1, hi)
        y, passed = same[:, cols], scores["choi"][:, cols] > 0.5
        for score in SCORED:
            for rule in RULES:
                rows.append(grade_rows(band, score, rule, select(scores[score][:, cols], rule), y, passed, lo))
        rows.append(grade_rows(band, "choi", "choi", select_choi(passed), y, passed, lo))
        rows.append(grade_rows(band, "-", "rand", select_rand(y.shape, a.seed), y, passed, lo))
    times["grade"] = timer()
    tables = Path(a.table_dir)
    tables.mkdir(parents=True, exist_ok=True)
    for d in (out, tables):
        write_csv(d / "G0b_topk.csv", rows)
    b4 = [r for r in rows if r["band"] == "B4" and r["rule"] in ("top:10", "mid:30")]
    passing = [(r["score"], r["rule"], r["purity_pooled"]) for r in b4 if r["purity_pooled"] >= 0.55]
    summary = {
        "n_nodes": n, "n_anchors": len(anchors), "anchors_file": a.anchors, "seed": a.seed, "tau": a.tau,
        "bands": {b: BANDS[b] for b in GRADE_BANDS}, "scores": SCORED, "rules": RULES + ["choi", "rand"], "checkpoints": CKPTS,
        "verify_against_exp049": check,
        "gate_G0b": {"B4_top10_or_mid30_purity_ge_0.55": passing},
        "seconds_cumulative": times,
        "max_cuda_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2) if device.type == "cuda" else None,
        "label_use": "GT labels read once after all scores were computed; used only for same-GT grading.",
    }
    (out / "G0b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
