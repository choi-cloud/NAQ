"""hl005_diag_distance_purity: HL03 G0 — label-free same-class scorers within X-cosine distance bands (no training). Reads GT labels for scoring only.

Scores are computed by `pair_scores` without labels; `labels_np` is called once, after all scores exist, for grading.
Outputs (results/hl/hl03/g0/): anchors.npy, G0_purity.csv, G0_coverage.csv, G0_summary.json.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hl_common as H  # noqa: E402
from model import GNNEncoder  # noqa: E402

BANDS = {"B1": (1, 10), "B2": (11, 50), "B3": (51, 200), "B4": (201, 1000), "B5": (1001, 3000)}
SCORES = ["x", "ax", "a2x", "ppr", "choi", "mv", "enc_b", "enc_n", "comb"]
CKPTS = {
    "enc_b": "results/hl/p0/Amazon_clothing/ours/tau_0.3/seed_1212/best_1shot.pt",
    "enc_n": "results/hl/p0/Amazon_clothing/naq/baseline/seed_1212/best_1shot.pt",
}
COVERAGE_Q = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def sym_adjacency(data, device):
    """Â_sym = D^-1/2 (A + I) D^-1/2 built exactly as in hl_common.structural_scores (= exp049)."""
    edges = data.edge_index.detach().cpu().numpy()
    n = data.features.size(0)
    adjacency = sp.csr_matrix((np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])), shape=(n, n))
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.float32).tocsr()
    adjacency.setdiag(1)
    adjacency.eliminate_zeros()
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    scale = np.maximum(degrees, 1) ** -0.5
    adjacency = (sp.diags(scale) @ adjacency @ sp.diags(scale)).tocoo()
    indices = torch.as_tensor(np.stack((adjacency.row, adjacency.col)), dtype=torch.long, device=device)
    values = torch.as_tensor(adjacency.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


def ppr_columns(a_sym, anchors, alpha, iters):
    """P <- (1-alpha) Â_sym P + alpha E, `iters` times from P = E; returns [n, len(anchors)] (column j = seed anchors[j])."""
    n = a_sym.size(0)
    e = torch.zeros(n, len(anchors), device=a_sym.device)
    e[torch.as_tensor(anchors, device=a_sym.device), torch.arange(len(anchors), device=a_sym.device)] = 1.0
    p = e.clone()
    for _ in range(iters):
        p = (1 - alpha) * torch.sparse.mm(a_sym, p) + alpha * e
    return p


def encoder_embeddings(data, path, latent, device):
    """Row-normalised GNNEncoder output of a meta.ProtoNet checkpoint (same config as exp049), eval mode."""
    config = [("GCN", (data.features.shape[1], 2 * latent), True), ("GCN", (2 * latent, latent), True), ("linear", (latent, 5), True)]
    net = GNNEncoder(config).to(device)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
    net.eval()
    with torch.no_grad():
        return F.normalize(net(data.features, adj=data.adj), dim=1)


def full_percentile(rows, self_col, picked):
    """Percentile (0-1, ties averaged) of rows[i, picked[i]] among all candidates of row i except column self_col[i]."""
    rows = rows.clone()
    r = torch.arange(rows.size(0), device=rows.device)
    rows[r, self_col] = -float("inf")
    srt = rows.sort(dim=1).values
    v = rows.gather(1, picked).contiguous()
    lo = torch.searchsorted(srt, v, right=False) - 1  # minus the -inf self entry
    hi = torch.searchsorted(srt, v, right=True) - 1
    return (lo + hi).float() / 2 / (rows.size(1) - 1)


def pair_scores(data, device, anchors, z, ppr, emb, tau, chunk):
    """Label-free scores for every (anchor, candidate) with X rank in [1, 3000]. Returns cand ids [A, 3000] and {score: [A, 3000]}."""
    _, z1, z2 = z
    depth = max(hi for _, hi in BANDS.values())
    out_ids, out = [], {s: [] for s in SCORES}
    with torch.no_grad():
        for begin in range(0, len(anchors), chunk):
            a = torch.as_tensor(anchors[begin:begin + chunk], device=device)
            r = torch.arange(len(a), device=device)
            xrow = data.Diffusion_matrix[a].clone()
            xrow[r, a] = -float("inf")
            order = torch.sort(xrow, dim=1, descending=True, stable=True).indices[:, :depth]  # X rank 1..depth
            rows = {
                "x": xrow,
                "ax": z1[a] @ z1.T,
                "a2x": z2[a] @ z2.T,
                "ppr": ppr[:, begin:begin + chunk].T.contiguous(),
                "enc_b": emb["enc_b"][a] @ emb["enc_b"].T,
                "enc_n": emb["enc_n"][a] @ emb["enc_n"].T,
            }
            got = {k: v.gather(1, order) for k, v in rows.items()}
            got["choi"] = ((got["ax"] >= tau) & (got["a2x"] >= tau)).float()
            pct = {k: full_percentile(rows[k], a, order) for k in ("ax", "a2x", "ppr", "enc_b")}
            # mv over full candidate set; comb = mean of the full-set percentile of mv and of enc_b
            mv_full = sum(full_percentile(rows[k], a, torch.arange(rows[k].size(1), device=device).expand(len(a), -1)) for k in ("ax", "a2x", "ppr")) / 3
            got["mv"] = (pct["ax"] + pct["a2x"] + pct["ppr"]) / 3
            got["comb"] = (full_percentile(mv_full, a, order) + pct["enc_b"]) / 2
            out_ids.append(order.cpu().numpy())
            for k in SCORES:
                out[k].append(got[k].cpu().numpy())
            del rows, got, pct, mv_full, xrow
            torch.cuda.empty_cache()
    return np.concatenate(out_ids), {k: np.concatenate(v) for k, v in out.items()}


def auroc_rows(s, y):
    """Mann-Whitney AUROC per row (ties averaged); NaN where a row lacks positives or negatives."""
    ranks = rankdata(s, axis=1)
    npos = y.sum(1)
    nneg = y.shape[1] - npos
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((ranks * y).sum(1) - npos * (npos + 1) / 2) / (npos * nneg)


def auroc_pooled(s, y):
    s, y = s.reshape(-1), y.reshape(-1).astype(bool)
    ranks = rankdata(s)
    npos, nneg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def top_k_mask(s, k):
    """Per row, the k[i] highest scores; ties broken by X rank (column order, stable)."""
    order = np.argsort(-s, axis=1, kind="stable")
    keep = np.arange(s.shape[1])[None, :] < k[:, None]
    mask = np.zeros_like(s, dtype=bool)
    np.put_along_axis(mask, order, keep, axis=1)
    return mask


def grade(scores, same, q_query):
    purity_rows, coverage_rows = [], []
    for band, (lo, hi) in BANDS.items():
        cols = slice(lo - 1, hi)
        y = same[:, cols]
        ka = scores["choi"][:, cols].sum(1).astype(np.int64)
        has = ka > 0
        both = (y.sum(1) > 0) & (y.sum(1) < y.shape[1])
        half = np.full(len(y), y.shape[1] // 2)
        coverage_rows.append({
            "band": band, "band_size": y.shape[1], "anchors": len(y),
            **{f"k_a_q{int(q * 100):02d}": float(np.quantile(ka, q)) for q in COVERAGE_Q},
            "k_a_mean": float(ka.mean()), "k_a_zero_frac": float((~has).mean()),
            "query_fill_rate": float((ka >= q_query).mean()),
            "anchors_with_pos_and_neg": int(both.sum()),
        })
        for name in SCORES:
            s = scores[name][:, cols]
            row = {"band": band, "score": name, "raw": float(y.mean())}
            if name == "choi":
                sel = s > 0.5
                row.update(AUROC_mean="", AUROC_pooled="", R2_purity="")
            else:
                row["AUROC_mean"] = float(np.nanmean(auroc_rows(s[both], y[both])))
                row["AUROC_pooled"] = auroc_pooled(s, y)
                sel = top_k_mask(s, ka)
                row["R2_purity"] = float(y[top_k_mask(s, half)].mean())
            row["R1_purity"] = float(y[sel].sum() / max(sel.sum(), 1))
            row["R1_purity_anchor_mean"] = float(((y & sel)[has].sum(1) / ka[has]).mean()) if has.any() else ""
            row["R1_selected_pairs"] = int(sel.sum())
            row["k_a_zero_frac"] = float((~has).mean())
            purity_rows.append(row)
    return purity_rows, coverage_rows


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--n_anchors", type=int, default=2000)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--ppr_iters", type=int, default=30)
    p.add_argument("--chunk", type=int, default=500)
    p.add_argument("--output_dir", default="results/hl/hl03/g0")
    p.add_argument("--table_dir", default="reports/HL03_tables")
    a = p.parse_args()
    timer, times = H.Timer(), {}
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    args = H.base_args(device=a.device, seed=a.seed, tau=a.tau)
    data, device = H.load_everything(args)
    n = data.features.size(0)
    times["load"] = timer()
    anchors = np.random.default_rng(a.seed).choice(n, a.n_anchors, replace=False)
    np.save(out / "anchors.npy", anchors)

    # ---- label-free scores ----
    z = H.structural_scores(data, device)
    check = H.verify_against_exp049(data, args, device, z=z)
    ppr = ppr_columns(sym_adjacency(data, device), anchors, a.alpha, a.ppr_iters)
    emb = {k: encoder_embeddings(data, path, args.latent, device) for k, path in CKPTS.items()}
    times["prep"] = timer()
    cand, scores = pair_scores(data, device, anchors, z, ppr, emb, a.tau, a.chunk)
    times["scores"] = timer()
    del ppr, emb, z
    torch.cuda.empty_cache()

    # ---- grading (GT labels read only here) ----
    labels = H.labels_np(data)
    same = labels[cand] == labels[anchors][:, None]
    purity, coverage = grade(scores, same, args.q_query)
    times["grade"] = timer()
    tables = Path(a.table_dir)
    tables.mkdir(parents=True, exist_ok=True)
    for d in (out, tables):
        write_csv(d / "G0_purity.csv", purity)
        write_csv(d / "G0_coverage.csv", coverage)
    b4 = {r["score"]: r for r in purity if r["band"] == "B4"}
    gate_threshold = max(0.55, b4["choi"]["R1_purity"] + 0.10)
    summary = {
        "n_nodes": n, "n_anchors": len(anchors), "seed": a.seed, "tau": a.tau, "ppr_alpha": a.alpha, "ppr_iters": a.ppr_iters,
        "bands": BANDS, "scores": SCORES, "checkpoints": CKPTS, "verify_against_exp049": check,
        "gate_B4": {"comb_R1": b4["comb"]["R1_purity"], "choi_R1": b4["choi"]["R1_purity"], "threshold": gate_threshold,
                    "comb_R1_ge_threshold": b4["comb"]["R1_purity"] >= gate_threshold},
        "seconds": times,
        "max_cuda_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2) if device.type == "cuda" else None,
        "label_use": "GT labels read once after all scores were computed; used only for same-GT grading.",
    }
    (out / "G0_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
