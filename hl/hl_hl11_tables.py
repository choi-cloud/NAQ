"""hl_hl11_tables: HL11 T18, T19 (feat), T20, T20b, T21 (curve), pre-registration values HLP119-130. Reads saved per-episode accuracies and report.json; no labels."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

FEAT = Path("results/hl/hl11/feat")
CURVE = Path("results/hl/hl11/curve")
TABLES = Path("reports/HL11_tables")
SEEDS = (1212, 1213, 1214, 1215, 1216)
METHODS = ("naq", "ours")
BASE = "a2x_l2"
FEAT_ARMS = ["x_l2", "x_raw", "ax_l2", "ax_raw", "a2x_l2", "a2x_raw", "a3x_l2", "a3x_raw", "cat_ax_a2x_l2", "sgc_mean_l2",
             "a2x_pca64_l2", "a2x_pca256_l2", "a2x_pca1024_l2", "a2x_pca256_raw", "a2x_rp256_l2_s0", "a2x_rp256_l2_s1", "a2x_rp256_l2_s2",
             "encb_raw", "encb_l2", "encn_raw", "encn_l2"]
T19_ARMS = ["a2x_l2", "a2x_pca64_l2", "a2x_pca256_l2", "a2x_pca1024_l2", "a2x_pca256_raw", "a2x_rp256_l2_s0", "a2x_rp256_l2_s1", "a2x_rp256_l2_s2",
            "RP", "encb_raw", "encb_l2", "encn_raw", "encn_l2"]
CURVE_COLS = ["valid_1shot", "test_1shot", "test_1shot_l2", "valid_5shot", "test_5shot", "test_5shot_l2"]


def write(name, rows):
    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def read(name):
    with (TABLES / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def ep(arm, shot, pool):
    return np.load(FEAT / f"{arm}_{shot}shot_{pool}.npy") * 100


def ci(values):
    return 1.96 * values.std(ddof=0) / np.sqrt(len(values))


def feat_row(arm, summary, label=None, selected=""):
    row = {"arm": label or arm, "selected_arm": selected}
    for shot in (1, 5):
        test = ep(arm, shot, "test")
        row[f"valid_{shot}shot"] = f"{ep(arm, shot, 'valid').mean():.4f}"
        row[f"test_{shot}shot"] = f"{test.mean():.4f}"
        row[f"test_{shot}shot_ci95"] = f"{ci(test):.4f}"
    row["dim"] = summary[arm]["dim"]
    for shot in (1, 5):
        d = ep(arm, shot, "test") - ep(BASE, shot, "test")
        row[f"paired_delta_{shot}shot_vs_a2x_l2"] = f"{d.mean():.4f}"
        row[f"paired_ci95_{shot}shot"] = f"{ci(d):.4f}"
        row[f"paired_win_frac_{shot}shot"] = f"{(d > 0).mean():.4f}"
    return row


def feat_tables():
    summary = json.loads((FEAT / "summary.json").read_text())
    selection = json.loads((FEAT / "selection.json").read_text())
    arms = [a for a in FEAT_ARMS if a in summary]
    best = selection["feat_best"]
    rows = [feat_row(a, summary) for a in arms] + [feat_row(best, summary, "feat_best", best)]
    write("T18_feat.csv", rows)
    by = {r["arm"]: r for r in rows}
    out = []
    keys = ["test_1shot", "test_1shot_ci95", "paired_delta_1shot_vs_a2x_l2", "paired_ci95_1shot", "paired_win_frac_1shot",
            "test_5shot", "test_5shot_ci95", "paired_delta_5shot_vs_a2x_l2", "paired_ci95_5shot", "paired_win_frac_5shot"]
    rp = [by[f"a2x_rp256_l2_s{s}"] for s in range(3)]
    for arm in T19_ARMS:
        if arm == "RP":
            r = {"arm": "a2x_rp256_l2 mean ± std (3 s)", "dim": 256}
            for k in keys:
                vals = np.array([float(x[k]) for x in rp])
                r[k] = f"{vals.mean():.4f} ± {vals.std(ddof=0):.4f}"
            out.append(r)
        elif arm in by:
            out.append({"arm": arm, "dim": by[arm]["dim"], **{k: by[arm][k] for k in keys}})
    write("T19_dim_control.csv", out)


def reports():
    docs = {}
    for m in METHODS:
        for s in SEEDS:
            p = CURVE / m / f"seed_{s}" / "report.json"
            if p.exists():
                docs[(m, s)] = json.loads(p.read_text())
    return docs


def curve_tables():
    docs = reports()
    epochs = sorted({h["epoch"] for d in docs.values() for h in d["history"]})
    mean_rows, std_rows, per = [], [], {}
    for m in METHODS:
        seeds = [s for s in SEEDS if (m, s) in docs]
        for e in epochs:
            hs = [next(h for h in docs[(m, s)]["history"] if h["epoch"] == e) for s in seeds]
            a, b = {"method": m, "epoch": e, "n_seeds": len(seeds)}, {"method": m, "epoch": e, "n_seeds": len(seeds)}
            for c in CURVE_COLS:
                v = np.array([h[c] for h in hs]) * 100
                a[c], b[c] = f"{v.mean():.4f}", f"{v.std(ddof=0):.4f}"
                per.setdefault((m, c), {})[e] = v
            mean_rows.append(a)
            std_rows.append(b)
    write("T20_curve.csv", mean_rows)
    write("T20b_curve_std.csv", std_rows)

    rows = []
    for m in METHODS:
        seeds = [s for s in SEEDS if (m, s) in docs]
        r = {"row": m, "n_seeds": len(seeds)}
        for shot in (1, 5):
            col, col_l2 = f"test_{shot}shot", f"test_{shot}shot_l2"
            mean = {e: v.mean() for e, v in per[(m, col)].items()}
            mean_l2 = {e: v.mean() for e, v in per[(m, col_l2)].items()}
            top, top_l2 = max(mean, key=mean.get), max(mean_l2, key=mean_l2.get)
            bv = np.array([docs[(m, s)]["evaluation"][str(shot)]["test_accuracy"] for s in seeds]) * 100
            be = [docs[(m, s)]["evaluation"][str(shot)]["best_epoch"] for s in seeds]
            r.update({
                f"epoch0_{col}": f"{mean[0]:.4f}",
                f"epoch0_{col_l2}": f"{mean_l2[0]:.4f}",
                f"best_valid_epoch_mean_{shot}shot": f"{np.mean(be):.1f}",
                f"best_valid_{col}": f"{bv.mean():.4f}",
                f"best_valid_{col}_std": f"{bv.std(ddof=0):.4f}",
                f"traj_max_{col}": f"{mean[top]:.4f}",
                f"traj_max_epoch_{shot}shot": top,
                f"traj_max_{col}_per_seed_mean": f"{np.mean([max(per[(m, col)][e][i] for e in epochs) for i in range(len(seeds))]):.4f}",
                f"traj_final_{col}": f"{mean[epochs[-1]]:.4f}",
                f"traj_max_{col_l2}": f"{mean_l2[top_l2]:.4f}",
                f"traj_max_epoch_{shot}shot_l2": top_l2,
                f"traj_final_{col_l2}": f"{mean_l2[epochs[-1]]:.4f}",
            })
        rows.append(r)
    t18 = {x["arm"]: x for x in read("T18_feat.csv")}
    for name in (BASE, "feat_best"):
        r = dict.fromkeys(rows[0].keys(), "")
        label = name if name == BASE else f"feat_best ({t18[name]['selected_arm']})"
        r.update({"row": label, "best_valid_test_1shot": t18[name]["test_1shot"], "best_valid_test_5shot": t18[name]["test_5shot"]})
        rows.append(r)
    write("T21_curve_summary.csv", rows)


def prereg():
    t18 = {x["arm"]: x for x in read("T18_feat.csv")}
    t19 = {x["arm"]: x for x in read("T19_dim_control.csv")}
    t21 = {x["row"]: x for x in read("T21_curve_summary.csv")}
    a = lambda arm: float(t18[arm]["test_1shot"])  # noqa: E731
    base = a(BASE)
    best = t18["feat_best"]["selected_arm"]
    rp = t19["a2x_rp256_l2 mean ± std (3 s)"]["test_1shot"]
    e0 = {m: float(t21[m]["epoch0_test_1shot"]) for m in METHODS}
    naq, ours = t21["naq"], t21["ours"]
    rows = [
        ("HLP119", "`a2x_l2` test 1-shot이 ND01 값 85.10과 ±0.05 안 (재현)", f"{base:.4f}; − 85.10 = {base - 85.10:+.4f}; ND01 표 값 85.0950과 차이 {base - 85.0950:+.4f}"),
        ("HLP120", "**Q1 핵심**: `a2x_pca256_l2` test 1-shot ≥ 84.0", f"{a('a2x_pca256_l2'):.4f}"),
        ("HLP121", "`a2x_rp256_l2` 3-seed 평균 ≥ 83.0", rp),
        ("HLP122", "`encb_l2` − `encb_raw` (test 1-shot) ∈ [−0.5, +1.0]", f"{a('encb_l2'):.4f} − {a('encb_raw'):.4f} = {a('encb_l2') - a('encb_raw'):+.4f}"),
        ("HLP123", "`a2x_raw` − `a2x_l2` ≤ −1.0", f"{a('a2x_raw'):.4f} − {base:.4f} = {a('a2x_raw') - base:+.4f}"),
        ("HLP124", "**Q2**: `feat_best`는 `a2x_l2` 또는 `cat_ax_a2x_l2` 또는 `a3x_l2` 중 하나이고, test 1-shot이 `a2x_l2` + 1.5 이내", f"feat_best = `{best}`; test 1-shot {a(best):.4f}; − `a2x_l2` = {a(best) - base:+.4f}"),
        ("HLP125", "`ax_l2` < `a2x_l2` (test 1-shot)", f"{a('ax_l2'):.4f} vs {base:.4f}; 차이 {a('ax_l2') - base:+.4f}"),
        ("HLP126", "`a3x_l2` − `a2x_l2` ∈ [−1.0, +1.0]", f"{a('a3x_l2'):.4f} − {base:.4f} = {a('a3x_l2') - base:+.4f}"),
        ("HLP127", "**Q3 핵심**: epoch 0 test 1-shot(`ours`·`naq` 평균) ≥ 78.0", f"naq {e0['naq']:.4f}, ours {e0['ours']:.4f}; 평균 {(e0['naq'] + e0['ours']) / 2:.4f}"),
        ("HLP128", "`ours` test_1shot 궤적 최대값 ≤ `a2x_l2` − 1.0", f"5 seed 평균 궤적 최대 {float(ours['traj_max_test_1shot']):.4f}(epoch {ours['traj_max_epoch_1shot']}); − `a2x_l2` = {float(ours['traj_max_test_1shot']) - base:+.4f}; seed별 최대의 평균 {float(ours['traj_max_test_1shot_per_seed_mean']):.4f}(− `a2x_l2` = {float(ours['traj_max_test_1shot_per_seed_mean']) - base:+.4f})"),
        ("HLP129", "`ours` epoch 0 test 1-shot < best-valid test 1-shot", f"{e0['ours']:.4f} vs {float(ours['best_valid_test_1shot']):.4f}; 차이 {e0['ours'] - float(ours['best_valid_test_1shot']):+.4f}"),
        ("HLP130", "`naq`·`ours` 모두 test_1shot_l2 궤적 최대값 − test_1shot 궤적 최대값 ∈ [−0.5, +1.0]", "; ".join(f"{m} {float(r['traj_max_test_1shot_l2']):.4f} − {float(r['traj_max_test_1shot']):.4f} = {float(r['traj_max_test_1shot_l2']) - float(r['traj_max_test_1shot']):+.4f}" for m, r in (("naq", naq), ("ours", ours)))),
    ]
    write("prereg_HL11.csv", [dict(id=i, prediction=p, computed=c) for i, p, c in rows])


if __name__ == "__main__":
    {"feat": feat_tables, "curve": curve_tables, "prereg": prereg}[sys.argv[1]]()
