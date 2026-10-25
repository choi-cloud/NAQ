"""hl010_diag_pdd: HL06 §1 — judge x distance-rule purity and structure descriptors in the X-rank band (no training), plus R1 descriptors of already-trained arms. Reads GT labels for grading and oracle reproductions only.

All label-free selections (hl_pdd judges/rules, hl008 HL04 arms, B_Choi, NaQ) are made before `labels_np` is called.
Outputs (--output_dir and --table_dir): P1_judge_rule.csv, P2_auroc.csv, P3_stratum_base.csv, R1_descriptors.csv, P_summary.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hl_common as H  # noqa: E402
import hl_pdd as P  # noqa: E402
from hl005_diag_distance_purity import auroc_rows, pair_scores, ppr_columns, write_csv  # noqa: E402
from hl006_purity_oracle_train import purity_oracle_queries  # noqa: E402
from hl007_diag_topk_purity import select_choi  # noqa: E402
from hl008_distant_query_train import b4_scores, label_free_queries, p04_choidiff_queries  # noqa: E402
from hl009_structural_oracle_train import structural_oracle_queries  # noqa: E402
from hl011_pdd_train import ARM_BAND, CHOI, LABEL_FREE, choi_select, o_far50, o_topsame, rand_pick  # noqa: E402
from hl_pdd_diag import Describer, aggregate  # noqa: E402

CKPT_B = "results/hl/p0/Amazon_clothing/ours/tau_0.3/seed_1212/best_1shot.pt"
CKPT_N = "results/hl/p0/Amazon_clothing/naq/baseline/seed_1212/best_1shot.pt"
HL04_ARMS = ["comb_top10", "a2x_top10", "encn_top10", "mv_mid30", "rand10"]
HL06_ORACLE = ["o_topsame", "o_combswap", "o_far50_p03", "o_far50_p05"]
HL06_LABEL_FREE = ["encb_top10", "encb_strat5", "encb_ex20", "encb_far50", "knnppr_strat5", "knnppr_ex20"]
HL06_R1 = (["naq_x10", "bchoi"] + [f"hl04_{x}" for x in ["comb_top10", "a2x_top10", "encn_top10", "mv_mid30", "rand10"]]
           + HL06_LABEL_FREE + ["g1_p04", "g1_p10", "p04_choidiff", "hl05_near", "hl05_far"] + HL06_ORACLE)
# known test accuracy (1-shot, 5-shot) of trained arms on Amazon_clothing: CLAUDE.md §8, reports/HL04_tables/T7_g2.csv, reports/HL05_tables/T8_structural.csv
KNOWN = {
    "naq_x10": (80.65, 90.52), "bchoi": (82.71, 89.91),
    "hl04_comb_top10": (79.41, 89.69), "hl04_a2x_top10": (79.90, 90.65), "hl04_encn_top10": (79.94, 89.16),
    "hl04_mv_mid30": (80.86, 90.06), "hl04_rand10": (80.34, 88.82),
    "g1_p04": (83.92, 91.12), "g1_p10": (89.42, 94.17), "p04_choidiff": (85.24, 90.89),
    "hl05_near": (85.49, 92.60), "hl05_far": (89.46, 94.40),
}
Q = 10
B5 = (1001, 3000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Amazon_clothing")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--n_anchors", type=int, default=2000)
    p.add_argument("--anchors", default="results/hl/hl03/g0/anchors.npy")
    p.add_argument("--ckpt_b", default=CKPT_B)
    p.add_argument("--ckpt_n", default=CKPT_N)
    p.add_argument("--x_band", default="201,1000")
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--ppr_iters", type=int, default=30)
    p.add_argument("--chunk", type=int, default=250)
    p.add_argument("--r1_rows", default="hl06", help="'hl06' (HL06 §1-5 set), '' (none) or a comma list of R1 rows")
    p.add_argument("--prefix", default="", help="file-name prefix of P1-P3 (e.g. dblp_)")
    p.add_argument("--output_dir", default="results/hl/hl06/diag")
    p.add_argument("--table_dir", default="reports/HL06_tables")
    a = p.parse_args()
    timer, times = H.Timer(), {}
    out, tables = Path(a.output_dir), Path(a.table_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    band = tuple(int(x) for x in a.x_band.split(","))
    hargs = H.base_args(device=a.device, seed=a.seed, tau=a.tau, dataset=a.dataset)
    hargs.ppr_alpha, hargs.ppr_iters = a.alpha, a.ppr_iters
    data, device = H.load_everything(hargs)
    n = data.features.size(0)
    times["load"] = timer()
    if a.anchors and Path(a.anchors).exists():
        anchors = np.load(a.anchors)
        anchor_src = a.anchors
    else:
        anchors = np.random.default_rng(a.seed).choice(n, a.n_anchors, replace=False)
        anchor_src = str(out / "anchors.npy")
        np.save(anchor_src, anchors)
    A = len(anchors)
    want = HL06_R1 if a.r1_rows == "hl06" else [x for x in a.r1_rows.split(",") if x]
    need_s8 = any(w.startswith("hl04_") or w in ("p04_choidiff", "o_combswap") for w in want)

    # ---------------- label-free scores ----------------
    check = H.verify_against_exp049(data, hargs, device)
    J = P.Judges(data, device, a.ckpt_b, a.ckpt_n, hargs.latent, a.alpha, a.ppr_iters)
    _, z1, z2 = J.z
    ranked, band_ids, s2b, s2r, axb = [], [], [], [], []
    scores = {k: [] for k in P.JUDGES}
    s8 = {}
    xrank = np.empty((A, n), dtype=np.int32)
    s2_full = np.empty((A, n), dtype=np.float32)
    for b in range(0, A, a.chunk):
        ids = anchors[b:b + a.chunk]
        rk_full = P.band_rank(data, ids, band, depth=n - 1).cpu().numpy()
        for j in range(len(ids)):
            xrank[b + j, rk_full[j]] = np.arange(1, n)
            xrank[b + j, ids[j]] = 0
        r, cols, s2, s2rank, sc = J.band(P.JUDGES, ids, band)
        ranked.append(r[:, :B5[1]])
        band_ids.append(cols)
        s2b.append(s2)
        s2r.append(s2rank)
        with torch.no_grad():
            axb.append(J.rows("ax", ids).gather(1, torch.as_tensor(cols, device=device)).cpu().numpy())
            s2_full[b:b + len(ids)] = J.rows("a2x", ids).cpu().numpy()
        for k in P.JUDGES:
            scores[k].append(sc[k])
        if need_s8:
            _, s = b4_scores(data, device, ids, z1, z2, J.a_sym, J.emb_n, hargs)
            for k, v in s.items():
                s8.setdefault(k, []).append(v)
        print(f"[hl010] scored {min(b + a.chunk, A)}/{A}", flush=True)
    ranked, band_ids, s2b, s2r, axb = (np.concatenate(x) for x in (ranked, band_ids, s2b, s2r, axb))
    scores = {k: np.concatenate(v) for k, v in scores.items()}
    s8 = {k: np.concatenate(v) for k, v in s8.items()}
    times["scores"] = timer()

    # consistency with hl005.pair_scores (a2x, enc_b, ppr) on the band columns
    emb = {"enc_b": J.emb_b, "enc_n": J.emb_n}
    ppr = ppr_columns(J.a_sym, anchors, a.alpha, a.ppr_iters)
    cand, ref = pair_scores(data, device, anchors, J.z, ppr, emb, a.tau, a.chunk)
    del ppr
    lo, hi = band
    consistency = {"band_ids_equal": bool((cand[:, lo - 1:hi] == band_ids).all()) if hi <= cand.shape[1] else None}
    for k in ("a2x", "enc_b", "ppr"):
        consistency[f"max_abs_err_{k}"] = float(np.abs(ref[k][:, lo - 1:hi] - scores[k]).max()) if hi <= cand.shape[1] else None
    replaced = []
    for k in ("a2x", "enc_b", "ppr"):
        if consistency[f"max_abs_err_{k}"] is not None and consistency[f"max_abs_err_{k}"] > 1e-3:
            scores[k] = ref[k][:, lo - 1:hi]
            replaced.append(k)
    consistency["replaced_with_hl005"] = replaced
    del cand, ref
    torch.cuda.empty_cache()
    times["consistency"] = timer()
    print(f"[hl010] consistency {consistency}", flush=True)

    # ---------------- label-free selections (node ids [A, 10]) ----------------
    sel, excluded = {}, {}
    for judge in P.JUDGES:
        for rule in P.SCORED_RULES:
            sel[(judge, rule)] = np.stack([band_ids[i][P.select(rule, scores[judge][i], s2r[i], None, Q)] for i in range(A)])
    for rule in P.RANDOM_RULES:
        rng = np.random.default_rng(a.seed)
        sel[("-", rule)] = np.stack([band_ids[i][P.select(rule, None, s2r[i], rng, Q)] for i in range(A)])
    passed = (axb >= a.tau) & (s2b >= a.tau)
    cm = select_choi(passed)
    keep = cm.any(1)
    sel[("choi", "choi")] = np.stack([band_ids[i][cm[i]] if keep[i] else band_ids[i][:Q] for i in range(A)])
    excluded[("choi", "choi")] = ~keep
    r1 = {}
    if "naq_x10" in want:
        r1["naq_x10"] = ranked[:, :Q]
    if "bchoi" in want:
        mined = H.mine_queries_for(anchors, data, hargs, device)
        r1["bchoi"] = np.stack([np.asarray(mined[int(x)]["selected_ids"]) for x in anchors])
    for arm in HL04_ARMS:
        if f"hl04_{arm}" in want or (arm == "comb_top10" and "o_combswap" in want):
            rng = np.random.default_rng(a.seed)
            r1[f"hl04_{arm}"] = np.stack([band_ids[i][label_free_queries(arm, {k: v[i] for k, v in s8.items()}, rng, Q)] for i in range(A)])
    for arm in want:
        if arm in LABEL_FREE and arm not in ARM_BAND and arm != "encb_strat5_r2":
            judge, rule = LABEL_FREE[arm]
            rng = np.random.default_rng(a.seed)
            r1[arm] = np.stack([band_ids[i][P.select(rule, scores[judge][i] if judge else None, s2r[i], rng, Q)] for i in range(A)])
        elif arm in CHOI:
            rows_ = []
            for i in range(A):
                x = int(anchors[i])
                with torch.no_grad():
                    full = [(z[x] @ z.T).cpu().numpy() for z in J.z]
                rows_.append(choi_select(arm, x, full[0], full[1], full[2], a.tau, Q)[0])
            r1[arm] = np.stack(rows_)
    times["select"] = timer()

    # ---------------- GT labels from here on ----------------
    labels = H.labels_np(data)

    def per_anchor(fn):
        return np.stack([fn(i, int(anchors[i])) for i in range(A)])
    for name, p_ in (("g1_p04", 0.4), ("g1_p10", 1.0)):
        if name in want:
            r1[name] = per_anchor(lambda i, x: purity_oracle_queries(x, p_, ranked[i], labels, Q)[0])
    if "p04_choidiff" in want:
        r1["p04_choidiff"] = per_anchor(lambda i, x: p04_choidiff_queries(x, ranked[i], s8["choi"][i], labels, Q)[0])
    for name, mode in (("hl05_near", "near"), ("hl05_far", "far")):
        if name in want:
            r1[name] = per_anchor(lambda i, x: structural_oracle_queries(x, mode, ranked[i], s2_full[i], labels, Q)[0])
    if "o_topsame" in want:
        r1["o_topsame"] = per_anchor(lambda i, x: o_topsame(x, band_ids[i], s2b[i], ranked[i], s2_full[i], labels, Q)[0])
    if "o_combswap" in want:
        rng_o = np.random.default_rng(a.seed + 1)

        def swap(i, x):
            picked = r1["hl04_comb_top10"][i]
            is_diff = labels[picked] != labels[x]
            pool = band_ids[i][(labels[band_ids[i]] != labels[x]) & ~np.isin(band_ids[i], picked)]
            c = picked.copy()
            c[is_diff] = rand_pick(rng_o, pool, int(is_diff.sum()))
            return c
        r1["o_combswap"] = per_anchor(swap)
    for arm, k in (("o_far50_p03", 3), ("o_far50_p05", 5)):
        if arm in want:
            rng_o = np.random.default_rng(a.seed + 1)
            r1[arm] = per_anchor(lambda i, x: o_far50(x, k, band_ids[i], s2r[i], labels, rng_o, Q)[0])
    if "o_combswap" in want and "hl04_comb_top10" not in want:
        del r1["hl04_comb_top10"]
    r1 = {k: r1[k] for k in want if k in r1}
    missing = [k for k in want if k not in r1]
    times["oracle_select"] = timer()

    desc = Describer(data, z1, z2, labels, a.tau, emb_b=J.emb_b)

    def describe_all(chosen, mask=None):
        rows = [desc.describe(int(anchors[i]), chosen[i], None, band_ids[i], s2r[i], xrank=xrank[i]) for i in range(A) if mask is None or mask[i]]
        return aggregate(rows, Q), len(rows)

    p1 = []
    for (judge, rule), chosen in sel.items():
        mask = ~excluded[(judge, rule)] if (judge, rule) in excluded else None
        st, used = describe_all(chosen, mask)
        p1.append({"judge": judge, "rule": rule, **st, "n_anchors": used, "excluded_frac": 1 - used / A})
    times["describe_p1"] = timer()
    r1_rows = []
    for name, chosen in r1.items():
        st, used = describe_all(chosen)
        one, five = KNOWN.get(name, ("", "")) if a.dataset == "Amazon_clothing" else ("", "")
        r1_rows.append({"row": name, "test_1shot": one, "test_5shot": five, **st, "n_anchors": used})
    times["describe_r1"] = timer()

    # AUROC (all / ex20 / far50) and stratum base purity
    same = labels[band_ids] == labels[anchors][:, None]
    m = band_ids.shape[1]
    order = np.argsort(s2r, axis=1)  # columns by s2 rank 1..m
    sub = {"all": np.arange(m), "ex20": np.arange(m // P.N_STRATA, m), "far50": np.arange(m // 2, m)}
    p2 = []
    for judge in P.JUDGES:
        row = {"judge": judge}
        for key, pos in sub.items():
            cols = order[:, pos]
            s = np.take_along_axis(scores[judge], cols, 1)
            y = np.take_along_axis(same, cols, 1)
            both = (y.sum(1) > 0) & (y.sum(1) < y.shape[1])
            row[f"auroc_{key}"] = float(np.nanmean(auroc_rows(s[both], y[both])))
            row[f"n_anchors_{key}"] = int(both.sum())
        p2.append(row)
    p3 = []
    size = m // P.N_STRATA
    for t in range(P.N_STRATA):
        cols = order[:, t * size:(t + 1) * size if t < P.N_STRATA - 1 else m]
        p3.append({"stratum": f"T{t + 1}", "s2rank_lo": t * size + 1, "s2rank_hi": (t + 1) * size if t < P.N_STRATA - 1 else m,
                   "base_purity": float(np.take_along_axis(same, cols, 1).mean()),
                   "mean_s2": float(np.take_along_axis(s2b, cols, 1).mean()),
                   "mean_x_rank": float((cols + lo).mean())})
    times["grade"] = timer()

    for d in (out, tables):
        write_csv(d / f"{a.prefix}P1_judge_rule.csv", p1)
        write_csv(d / f"{a.prefix}P2_auroc.csv", p2)
        write_csv(d / f"{a.prefix}P3_stratum_base.csv", p3)
        if r1_rows:
            write_csv(d / f"{a.prefix}R1_descriptors.csv", r1_rows)
    summary = {
        "dataset": a.dataset, "n_nodes": n, "n_classes": int(len(np.unique(labels))), "n_anchors": A, "anchors": anchor_src,
        "x_band": band, "tau": a.tau, "ckpt_b": a.ckpt_b, "ckpt_n": a.ckpt_n, "verify_against_exp049": check,
        "consistency_vs_hl005": consistency, "r1_rows": list(r1), "r1_missing": missing, "seconds": times,
        "max_cuda_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2) if device.type == "cuda" else None,
        "label_use": "GT labels read once after all label-free selections; used for grading, oracle reproductions (g1, p04_choidiff, hl05, HL06 oracle arms) and descriptors.",
    }
    for d in (out, tables):
        (d / f"{a.prefix}P_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
