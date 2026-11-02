"""hl_hl13_tables: HL13 T31-T36 and pre-registration values HLP153-168 (T30 comes from hl016). Partial results give the rows that exist (n_seeds shown); missing values read 미실행. Reads report.json and csv tables; no labels."""
import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path("results/hl/hl13/curve")
HL11 = Path("results/hl/hl11/curve")
HL12 = Path("results/hl/hl12/curve")
FEAT = Path("results/hl/hl13/feat")
TABLES = Path("reports/HL13_tables")
T18 = {r["arm"]: r for r in csv.DictReader(open("reports/HL11_tables/T18_feat.csv", encoding="utf-8"))}
T22 = {r["arm"]: r for r in csv.DictReader(open("reports/HL12_tables/T22_arms.csv", encoding="utf-8"))}
S5 = (1212, 1213, 1214, 1215, 1216)
S10 = (1217, 1218, 1219, 1220, 1221)
SHOTS = (1, 5)
STAGE1 = ["sgc64_rand_cos_naq_lr1e-3", "sgc64_rand_cos_sup_lr1e-3", "sgc256_rand_cos_naq_lr1e-3", "sgc64_pcainit_cos_naq_lr1e-3", "sgc64_pcainit_cos_sup_lr1e-3", "pca1024_lin64_cos_naq_lr1e-3"]
STAGE2 = ["pca64_lin_cos_naq_lr1e-3", "pca64_lin_cos_naq_lr1e-2", "pca64_lin_cos_sup_lr1e-3", "pca64_lin_cos_sup_lr1e-2", "sgc64_rand_cos_naq_lr1e-4", "sgc64_rand_cos_naq_lr1e-2"]
POST_HOC = {"gcn_raw_naq": HL11 / "naq", "gcn_raw_ours": HL11 / "ours"}


def write(name, rows):
    if rows:
        TABLES.mkdir(parents=True, exist_ok=True)
        with (TABLES / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def docs_of(arm, base=ROOT, seeds=S5):
    out = {}
    for seed in seeds:
        p = base / arm / f"seed_{seed}" / "report.json"
        if p.exists():
            out[seed] = json.loads(p.read_text())
    return out


def posthoc_docs(arm):
    out = {}
    for seed in S5:
        p = POST_HOC[arm] / f"seed_{seed}" / "report.json"
        if p.exists():
            out[seed] = json.loads(p.read_text())
    return out


def pick(history, shot, include_zero=False):
    rows = [h for h in history if include_zero or h["epoch"] > 0]
    top = max(h[f"valid_{shot}shot_l2"] for h in rows)
    return next(h for h in rows if h[f"valid_{shot}shot_l2"] == top)


def values(doc, shot):
    """bestR_test_raw, bestL_test_l2, bestL_epoch (fractions); works for hl013 (post-hoc L) and hl014/hl015 reports."""
    ev = doc["evaluation"]
    if "L" in ev:
        return ev["R"][str(shot)]["test_raw"], ev["L"][str(shot)]["test_l2"], ev["L"][str(shot)]["best_epoch"]
    chosen = pick(doc["history"], shot)
    return ev[str(shot)]["test_accuracy"], chosen[f"test_{shot}shot_l2"], chosen["epoch"]


def row_of(arm, docs):
    """Columns of T22 for the available seeds."""
    row = {"arm": arm, "n_seeds": len(docs)}
    if not docs:
        return row
    seeds = sorted(docs)
    for shot in SHOTS:
        s, col = f"{shot}shot", f"test_{shot}shot_l2"
        vals = np.array([values(docs[x], shot) for x in seeds], dtype=np.float64)
        epochs = [h["epoch"] for h in docs[seeds[0]]["history"]]
        traj = np.array([[next(h for h in docs[x]["history"] if h["epoch"] == e)[col] for e in epochs] for x in seeds]) * 100
        mean = traj.mean(axis=0)
        zero = np.array([pick(docs[x]["history"], shot, True)[col] for x in seeds]) * 100
        row.update({
            f"epoch0_test_l2_{s}": f"{mean[0]:.4f}", f"bestR_test_raw_{s}": f"{vals[:, 0].mean() * 100:.4f}",
            f"bestL_test_l2_{s}": f"{vals[:, 1].mean() * 100:.4f}", f"bestL_test_l2_std_{s}": f"{vals[:, 1].std(ddof=0) * 100:.4f}", f"bestL_epoch_mean_{s}": f"{vals[:, 2].mean():.1f}",
            f"bestL_incl_epoch0_test_l2_{s}": f"{zero.mean():.4f}", f"trajmax_test_l2_{s}": f"{mean.max():.4f}", f"trajmax_epoch_{s}": epochs[int(mean.argmax())], f"final_test_l2_{s}": f"{mean[-1]:.4f}",
        })
    return row


def fill(rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    return [{k: r.get(k, "") for k in keys} for r in rows]


def ref_row(name, one, five, source):
    return {"arm": f"{name} ({source})", "n_seeds": "", "bestL_test_l2_1shot": one, "bestL_test_l2_5shot": five}


def agreement(docs):
    v = [d["t_struct_agreement"]["mean"] for d in docs.values() if d.get("t_struct_agreement")]
    return f"{np.mean(v):.4f}" if v else ""


def t31():
    rows = [{"stage": 1, **row_of(a, docs_of(a))} for a in STAGE1] + [{"stage": 2, **row_of(a, docs_of(a))} for a in STAGE2]
    rows.append({"stage": "ref", **ref_row("a2x_pca64_l2", T18["a2x_pca64_l2"]["test_1shot"], T18["a2x_pca64_l2"]["test_5shot"], "HL11 T18, no training")})
    rows.append({"stage": "ref", **ref_row("pca64_lin_cos_naq", T22["pca64_lin_cos_naq"]["bestL_test_l2_1shot"], T22["pca64_lin_cos_naq"]["bestL_test_l2_5shot"], "HL12 T22, lr 1e-4")})
    write("T31_projection.csv", fill(rows))


def t32():
    spec = [
        ("gcn raw", "naq", 0, "gcn_raw_naq", "T22"), ("gcn raw", "naq", 0.3, "gcn_raw_soft0.3_naq", None), ("gcn raw", "naq", 0.6, "gcn_raw_soft0.6_naq", None), ("gcn raw", "naq", 1.0, "gcn_raw_soft1.0_naq", None),
        ("gcn raw", "ours", 0, "gcn_raw_ours", "T22"), ("gcn raw", "ours", 0.3, "gcn_raw_soft0.3_ours", None),
        ("pca64_lin cos", "naq", 0, "pca64_lin_cos_naq", "T22 (lr 1e-4)"), ("pca64_lin cos", "naq", 0, "pca64_lin_cos_naq_lr1e-3", None),
        ("pca64_lin cos", "naq", 0.3, "pca64_lin_cos_soft0.3_naq_lr1e-3", None), ("pca64_lin cos", "naq", 1.0, "pca64_lin_cos_soft1.0_naq_lr1e-3", None),
    ]
    rows = []
    for enc, ep, a, arm, source in spec:
        if source:
            rows.append({"encoder_loss": enc, "episodes": ep, "A": a, "arm": arm, "source": source, "n_seeds": T22[arm]["n_seeds"], "bestL_test_l2_1shot": T22[arm]["bestL_test_l2_1shot"], "bestL_test_l2_std_1shot": T22[arm]["bestL_test_l2_std_1shot"],
                         "bestL_test_l2_5shot": T22[arm]["bestL_test_l2_5shot"], "t_struct_agreement": ""})
        else:
            docs = docs_of(arm)
            r = row_of(arm, docs)
            rows.append({"encoder_loss": enc, "episodes": ep, "A": a, "arm": arm, "source": "hl015", "n_seeds": len(docs), "bestL_test_l2_1shot": r.get("bestL_test_l2_1shot", ""), "bestL_test_l2_std_1shot": r.get("bestL_test_l2_std_1shot", ""),
                         "bestL_test_l2_5shot": r.get("bestL_test_l2_5shot", ""), "t_struct_agreement": agreement(docs)})
    write("T32_soft.csv", rows)


def t33():
    rows = []
    for ratio, arm, source in ((0, "gcn_raw_naq", "T22"), (0.1, "gcn_raw_mix_r0.1", None), (0.25, "gcn_raw_mix_r0.25", None), (0.5, "gcn_raw_mix_r0.5", None), (1, "gcn_raw_sup", "T22")):
        r = T22[arm] if source else row_of(arm, docs_of(arm))
        row = {"R": ratio, "arm": arm, "source": source or "hl015", "n_seeds": r.get("n_seeds", 0)}
        for s in ("1shot", "5shot"):
            for k in (f"bestL_test_l2_{s}", f"bestL_test_l2_std_{s}", f"trajmax_test_l2_{s}", f"bestL_epoch_mean_{s}"):
                row[k] = r.get(k, "")
        rows.append(row)
    write("T33_mix.csv", rows)


def t34():
    rows = []
    for arm in ("gcn_raw_oracle_strat5", "pca64_lin_cos_oracle_strat5_lr1e-3", "sgc64_rand_cos_oracle_strat5_lr1e-3"):
        docs = docs_of(arm)
        r = row_of(arm, docs)
        gt = [d["diagnostics"]["query_gt_match"] for d in docs.values()]
        rows.append({"arm": arm, "n_seeds": len(docs), "bestR_test_raw_1shot": r.get("bestR_test_raw_1shot", ""), "bestL_test_l2_1shot": r.get("bestL_test_l2_1shot", ""), "bestL_test_l2_std_1shot": r.get("bestL_test_l2_std_1shot", ""),
                     "bestR_test_raw_5shot": r.get("bestR_test_raw_5shot", ""), "bestL_test_l2_5shot": r.get("bestL_test_l2_5shot", ""), "query_gt_match": f"{np.mean(gt):.4f}" if gt else ""})
    rows.append({"arm": "HL07 o_grid_strat5_p10 (ref, gpusystem, raw)", "n_seeds": 5, "bestR_test_raw_1shot": "90.12", "bestL_test_l2_1shot": "", "bestL_test_l2_std_1shot": "", "bestR_test_raw_5shot": "94.74", "bestL_test_l2_5shot": "", "query_gt_match": ""})
    write("T34_oracle_l2.csv", rows)


def t35():
    rows = []
    p = FEAT / "pca_base_eval.json"
    if p.exists():
        e = json.loads(p.read_text())
        rows.append({"row": "pca64_base no training", "n_seeds": "", "n_fit_nodes": e["n_fit_nodes"], "explained_variance_ratio_sum": f"{e['explained_variance_ratio_sum']:.4f}", "valid_1shot": f"{e['valid_1shot']:.4f}", "test_1shot": f"{e['test_1shot']:.4f}", "test_1shot_ci95": f"{e['test_1shot_ci95']:.4f}", "test_5shot": f"{e['test_5shot']:.4f}", "paired_delta_1shot_vs_a2x_pca64_l2": ""})
        delta = np.load(FEAT / "pca64_base_1shot_test.npy") * 100 - np.load("results/hl/hl11/feat/a2x_pca64_l2_1shot_test.npy") * 100
        rows[-1]["paired_delta_1shot_vs_a2x_pca64_l2"] = f"{delta.mean():.4f} ± {1.96 * delta.std(ddof=0) / np.sqrt(len(delta)):.4f}"
    t = T18["a2x_pca64_l2"]
    rows.append({"row": "a2x_pca64_l2 (HL11, all nodes)", "n_seeds": "", "n_fit_nodes": 24919, "explained_variance_ratio_sum": "0.5344", "valid_1shot": t["valid_1shot"], "test_1shot": t["test_1shot"], "test_1shot_ci95": t["test_1shot_ci95"], "test_5shot": t["test_5shot"], "paired_delta_1shot_vs_a2x_pca64_l2": ""})
    arm = "pca64_lin_cos_naq_pcabase_lr1e-3"
    docs = docs_of(arm)
    r = row_of(arm, docs)
    rows.append({"row": arm + " (bestL)", "n_seeds": len(docs), "n_fit_nodes": 10711, "explained_variance_ratio_sum": "", "valid_1shot": "", "test_1shot": r.get("bestL_test_l2_1shot", ""), "test_1shot_ci95": "", "test_5shot": r.get("bestL_test_l2_5shot", ""), "paired_delta_1shot_vs_a2x_pca64_l2": ""})
    write("T35_pca_inductive.csv", rows)


def t36():
    selection = FEAT.parent / "stage9_selection.json"
    arms = ["gcn_raw_naq", "gcn_raw_ours", "pca64_lin_cos_sup"]
    if selection.exists():
        arms += [v["selected_by_bestL_test_l2"] for v in json.loads(selection.read_text()).values()]
    rows = []
    for arm in arms:
        first = posthoc_docs(arm) if arm in POST_HOC else docs_of(arm, HL12 if arm == "pca64_lin_cos_sup" else ROOT)
        second = docs_of(arm, ROOT, S10)
        row = {"arm": arm, "n_first5": len(first), "n_second5": len(second)}
        for shot in SHOTS:
            a = np.array([values(d, shot)[1] for d in first.values()]) * 100
            b = np.array([values(d, shot)[1] for d in second.values()]) * 100
            both = np.concatenate([a, b])
            s = f"{shot}shot"
            row[f"first5_{s}"] = f"{a.mean():.4f} ± {a.std(ddof=0):.4f}" if len(a) else ""
            row[f"second5_{s}"] = f"{b.mean():.4f} ± {b.std(ddof=0):.4f}" if len(b) else ""
            row[f"all_{s}"] = f"{both.mean():.4f} ± {both.std(ddof=0):.4f}" if len(both) else ""
        rows.append(row)
    write("T36_tenseed.csv", rows)


def prereg():
    def a(arm, shot=1, key="bestL_test_l2"):
        if arm in T22 and not docs_of(arm):
            return float(T22[arm][f"{key}_{shot}shot"])
        docs = docs_of(arm)
        if not docs:
            return None
        return float(row_of(arm, docs)[f"{key}_{shot}shot"])

    def n(arm):
        k = len(docs_of(arm))
        return f"(n={k})" if k not in (0, 5) else ""

    def v(arm, shot=1, key="bestL_test_l2"):
        x = a(arm, shot, key)
        return "미실행" if x is None else f"{x:.4f}{n(arm)}"

    def d(x, y, xs=None, ys=None):
        p, q = (a(x) if isinstance(x, str) else x), (a(y) if isinstance(y, str) else y)
        if p is None or q is None:
            return "미실행"
        return f"{xs or (x if isinstance(x, str) else '')} {p:.4f}{n(x) if isinstance(x, str) else ''} − {ys or (y if isinstance(y, str) else '')} {q:.4f}{n(y) if isinstance(y, str) else ''} = {p - q:+.4f}"

    pca = float(T18["a2x_pca64_l2"]["test_1shot"])
    naq12 = float(T22["pca64_lin_cos_naq"]["bestL_test_l2_1shot"])
    naq = float(T22["gcn_raw_naq"]["bestL_test_l2_1shot"])
    sgc = "sgc64_rand_cos_naq_lr1e-3"
    agree_naq = agreement({**{f"{k}{s}": x for k in ("gcn_raw_soft0.3_naq", "gcn_raw_soft0.6_naq", "gcn_raw_soft1.0_naq") for s, x in docs_of(k).items()}})
    agree_ours = agreement(docs_of("gcn_raw_soft0.3_ours"))
    lr2 = "; ".join(f"`{x}` {v(x)}" + ("" if a(x) is None else f"(− 89.05 = {a(x) - 89.05:+.4f}; epoch 0 포함 {v(x, 1, 'bestL_incl_epoch0_test_l2')})") for x in ("pca64_lin_cos_naq_lr1e-2", "pca64_lin_cos_sup_lr1e-2"))
    oracle = "gcn_raw_oracle_strat5"
    base = "미실행"
    if (FEAT / "pca_base_eval.json").exists():
        e = json.loads((FEAT / "pca_base_eval.json").read_text())
        base = f"{e['test_1shot']:.4f}; − 89.05 = {e['test_1shot'] - 89.05:+.4f}"
    norm = "미실행"
    if (TABLES / "T30_norm_diag.csv").exists():
        norm = "; ".join(f"`{r['checkpoint']}` {r['norm_induced_error']}" for r in csv.DictReader(open(TABLES / "T30_norm_diag.csv", encoding="utf-8")))
    rows = [
        ("HLP153", "**A 핵심**: `sgc64_rand_cos_naq`(lr 1e-3) ≥ 87.0", v(sgc)),
        ("HLP154", "`sgc64_rand_cos_naq` ≤ `a2x_pca64_l2` − 0.5", d(sgc, pca, ys="a2x_pca64_l2")),
        ("HLP155", "`sgc64_pcainit_cos_naq` ≥ `a2x_pca64_l2` − 0.3", d("sgc64_pcainit_cos_naq_lr1e-3", pca, ys="a2x_pca64_l2")),
        ("HLP156", "`sgc64_pcainit_cos_sup` ≤ `sgc64_pcainit_cos_naq`", d("sgc64_pcainit_cos_sup_lr1e-3", "sgc64_pcainit_cos_naq_lr1e-3")),
        ("HLP157", "`sgc64_rand_cos_sup` ≤ `sgc64_rand_cos_naq` − 2.0", d("sgc64_rand_cos_sup_lr1e-3", sgc)),
        ("HLP158", "**B**: `pca64_lin_cos_naq_lr1e-3` − HL12 `pca64_lin_cos_naq`(89.15) ∈ [−0.3, +0.5]", d("pca64_lin_cos_naq_lr1e-3", naq12, ys="HL12 pca64_lin_cos_naq")),
        ("HLP159", "`pca64_lin_cos_*_lr1e-2`가 epoch 0보다 낮아진다(≤ 89.05 − 0.5)", lr2),
        ("HLP160", "**C 핵심**: `gcn_raw_soft0.3_naq` ≥ `gcn_raw_naq`(87.72) + 0.5", d("gcn_raw_soft0.3_naq", naq, ys="gcn_raw_naq")),
        ("HLP161", "`gcn_raw_soft1.0_naq` ≥ `gcn_raw_naq` − 1.0", d("gcn_raw_soft1.0_naq", naq, ys="gcn_raw_naq")),
        ("HLP162", "t_struct 동의도 평균(`naq`) ≥ 0.5, `ours`는 그보다 낮다", f"naq {agree_naq or '미실행'}; ours {agree_ours or '미실행'}"),
        ("HLP163", "`gcn_raw_mix_r0.1` ≥ `gcn_raw_naq` − 0.3", d("gcn_raw_mix_r0.1", naq, ys="gcn_raw_naq")),
        ("HLP164", "`gcn_raw_mix_r0.5` ≤ `gcn_raw_naq` − 2.0", d("gcn_raw_mix_r0.5", naq, ys="gcn_raw_naq")),
        ("HLP165", "**D**: `gcn_raw_oracle_strat5` bestR_test_raw가 HL07 90.12와 ±0.5 안; bestL_test_l2 ≥ 91.5", "미실행" if a(oracle) is None else f"bestR_test_raw {v(oracle, 1, 'bestR_test_raw')}(− 90.12 = {a(oracle, 1, 'bestR_test_raw') - 90.12:+.4f}); bestL_test_l2 {v(oracle)}"),
        ("HLP166", "`pca64_lin_cos_oracle_strat5` ≥ 90.0", v("pca64_lin_cos_oracle_strat5_lr1e-3")),
        ("HLP167", "`pca64_base` 학습 없는 l2 ≥ 89.05 − 0.5", base),
        ("HLP168", "노름: `encn`·`encb`의 norm-induced error ≥ 0.3; `gcn_cos_*`는 ≤ 0.1", norm),
    ]
    write("prereg_HL13.csv", [dict(id=i, prediction=p, computed=c) for i, p, c in rows])


def main():
    for f in (t31, t32, t33, t34, t35, t36, prereg):
        try:
            f()
        except Exception:
            print(f"[tables] {f.__name__} failed", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
