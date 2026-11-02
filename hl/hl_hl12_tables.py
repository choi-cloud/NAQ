"""hl_hl12_tables: HL12 T22, T22b, T23, T23b, T24, T25 and pre-registration values HLP131-142. Reads report.json, per-episode accuracies and HL11 tables; no labels."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("results/hl/hl12/curve")
HL11 = Path("results/hl/hl11/curve")
HL11_FEAT = Path("results/hl/hl11/feat")
HL11_T18 = Path("reports/HL11_tables/T18_feat.csv")
TABLES = Path("reports/HL12_tables")
SEEDS = (1212, 1213, 1214, 1215, 1216)
SHOTS = (1, 5)
ARMS = ["gcn_raw_naq", "gcn_raw_ours", "gcn_raw_sup", "gcn_raw_sup_strat",
        "gcn_cos_naq", "gcn_cos_ours", "gcn_cos_sup",
        "pca64_lin_cos_naq", "pca64_lin_cos_ours", "pca64_lin_cos_sup", "pca64_lin_cos_sup_strat",
        "pca256_lin_cos_sup", "pca256_mlp_cos_sup"]
SCALE_ARMS = ["pca64_lin_cos_sup_s5", "pca64_lin_cos_sup_s20"]
POST_HOC = {"gcn_raw_naq": "naq", "gcn_raw_ours": "ours"}
REFS = ["a2x_l2", "a2x_pca64_l2", "a2x_pca256_l2", "encb_l2", "encn_l2"]


def write(name, rows):
    if not rows:
        return
    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def read(path):
    with Path(path).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load(arm):
    docs = {}
    for seed in SEEDS:
        base = HL11 / POST_HOC[arm] if arm in POST_HOC else ROOT / arm
        p = base / f"seed_{seed}" / "report.json"
        if p.exists():
            docs[seed] = json.loads(p.read_text())
    return docs


def pick(history, shot, include_zero=False):
    """First epoch with the highest l2 valid (the training-loop rule); epoch 0 only when include_zero."""
    rows = [h for h in history if include_zero or h["epoch"] > 0]
    top = max(h[f"valid_{shot}shot_l2"] for h in rows)
    return next(h for h in rows if h[f"valid_{shot}shot_l2"] == top)


def rule_values(doc, arm, shot):
    """bestR_test_raw, bestR_test_l2, bestL_test_l2, bestL_epoch, in fractions."""
    history = {h["epoch"]: h for h in doc["history"]}
    if arm in POST_HOC:
        ev = doc["evaluation"][str(shot)]
        chosen = pick(doc["history"], shot)
        return ev["test_accuracy"], history[ev["best_epoch"]][f"test_{shot}shot_l2"], chosen[f"test_{shot}shot_l2"], chosen["epoch"]
    r, l = doc["evaluation"]["R"][str(shot)], doc["evaluation"]["L"][str(shot)]
    return r["test_raw"], r["test_l2"], l["test_l2"], l["best_epoch"]


def arm_row(arm, docs):
    row = {"arm": arm, "source": "HL11 hl013; L post-hoc" if arm in POST_HOC else "hl014", "n_seeds": len(docs)}
    seeds = sorted(docs)
    for shot in SHOTS:
        vals = np.array([rule_values(docs[s], arm, shot) for s in seeds], dtype=np.float64)
        col = f"test_{shot}shot_l2"
        epochs = [h["epoch"] for h in docs[seeds[0]]["history"]]
        traj = np.array([[next(h for h in docs[s]["history"] if h["epoch"] == e)[col] for e in epochs] for s in seeds]) * 100
        mean = traj.mean(axis=0)
        with_zero = np.array([pick(docs[s]["history"], shot, True)[col] for s in seeds]) * 100
        zero_chosen = sum(pick(docs[s]["history"], shot, True)["epoch"] == 0 for s in seeds)
        s = f"{shot}shot"
        row.update({
            f"epoch0_test_l2_{s}": f"{mean[0]:.4f}",
            f"bestR_test_raw_{s}": f"{vals[:, 0].mean() * 100:.4f}",
            f"bestR_test_l2_{s}": f"{vals[:, 1].mean() * 100:.4f}",
            f"bestL_test_l2_{s}": f"{vals[:, 2].mean() * 100:.4f}",
            f"bestL_test_l2_std_{s}": f"{vals[:, 2].std(ddof=0) * 100:.4f}",
            f"bestL_epoch_mean_{s}": f"{vals[:, 3].mean():.1f}",
            f"trajmax_test_l2_{s}": f"{mean.max():.4f}",
            f"trajmax_epoch_{s}": epochs[int(mean.argmax())],
            f"final_test_l2_{s}": f"{mean[-1]:.4f}",
            f"bestL_incl_epoch0_test_l2_{s}": f"{with_zero.mean():.4f}",
            f"bestL_incl_epoch0_n_chosen_{s}": zero_chosen,
        })
    return row


def available():
    return [(a, load(a)) for a in ARMS if load(a)]


def tables():
    done = available()
    rows = [arm_row(a, d) for a, d in done]
    t18 = {r["arm"]: r for r in read(HL11_T18)}
    for ref in REFS:
        row = dict.fromkeys(rows[0].keys(), "")
        row.update({"arm": ref, "source": "HL11 T18" + (" (seed 1212 ckpt)" if ref.startswith("enc") else " (no training)")})
        row["bestL_test_l2_1shot"], row["bestL_test_l2_5shot"] = t18[ref]["test_1shot"], t18[ref]["test_5shot"]
        rows.append(row)
    write("T22_arms.csv", rows)

    paired = []
    for arm, docs in done:
        if arm in POST_HOC:
            continue
        row = {"arm": arm, "n_seeds": len(docs), "baseline": "a2x_pca64_l2"}
        for shot in SHOTS:
            eps = np.mean([docs[s]["evaluation"]["L"][str(shot)]["test_l2_episodes"] for s in docs], axis=0) * 100
            d = eps - np.load(HL11_FEAT / f"a2x_pca64_l2_{shot}shot_test.npy") * 100
            row.update({f"paired_delta_{shot}shot": f"{d.mean():.4f}", f"paired_ci95_{shot}shot": f"{1.96 * d.std(ddof=0) / np.sqrt(len(d)):.4f}", f"paired_win_frac_{shot}shot": f"{(d > 0).mean():.4f}"})
        paired.append(row)
    write("T22b_paired_vs_pca64.csv", paired)

    mean_rows, std_rows = [], []
    for arm, docs in done + [(a, {1212: json.loads((ROOT / a / "seed_1212" / "report.json").read_text())}) for a in SCALE_ARMS if (ROOT / a / "seed_1212" / "report.json").exists()]:
        seeds = sorted(docs)
        for e in [h["epoch"] for h in docs[seeds[0]]["history"]]:
            hs = [next(h for h in docs[s]["history"] if h["epoch"] == e) for s in seeds]
            a, b = {"arm": arm, "epoch": e, "n_seeds": len(seeds)}, {"arm": arm, "epoch": e, "n_seeds": len(seeds)}
            for c in ("valid_1shot_l2", "test_1shot_l2", "test_5shot_l2"):
                v = np.array([h[c] for h in hs]) * 100
                a[c], b[c] = f"{v.mean():.4f}", f"{v.std(ddof=0):.4f}"
            mean_rows.append(a)
            std_rows.append(b)
    write("T23_curves_l2.csv", mean_rows)
    write("T23b_curves_l2_std.csv", std_rows)

    scale_rows = []
    for s, arm in ((5, "pca64_lin_cos_sup_s5"), (10, "pca64_lin_cos_sup"), (20, "pca64_lin_cos_sup_s20")):
        p = ROOT / arm / "seed_1212" / "report.json"
        if not p.exists():
            continue
        doc = json.loads(p.read_text())
        row = {"arm": "pca64_lin_cos_sup", "seed": 1212, "s": s}
        for shot in SHOTS:
            col = f"test_{shot}shot_l2"
            top = max(doc["history"], key=lambda h: h[col])
            row.update({f"bestL_test_l2_{shot}shot": f"{doc['evaluation']['L'][str(shot)]['test_l2'] * 100:.4f}", f"bestL_epoch_{shot}shot": doc["evaluation"]["L"][str(shot)]["best_epoch"],
                        f"trajmax_test_l2_{shot}shot": f"{top[col] * 100:.4f}", f"trajmax_epoch_{shot}shot": top["epoch"]})
        scale_rows.append(row)
    write("T24_scale.csv", scale_rows)

    episode_rows = []
    for name in ("sup", "sup_strat"):
        for seed in SEEDS:
            p = ROOT / f"episodes_{name}_seed{seed}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            row = {k: d[k] for k in ("episodes", "seed", "n_episodes", "n_base_classes", "class_size_min", "class_size_median", "class_size_max", "n_classes_sampled", "classes_sampled_min", "classes_sampled_max", "drawn_with_replacement")}
            row["query_a2x_cos_mean"] = f"{d['query_a2x_cos_mean']:.4f}"
            for i, v in enumerate(d["strata_a2x_cos_mean"], 1):
                row[f"stratum{i}_a2x_cos_mean"] = "" if v is None else f"{v:.4f}"
            row["episode_sha256_prefix"] = d["episode_sha256"][:12]
            episode_rows.append(row)
    write("T25_episodes.csv", episode_rows)


def prereg():
    t = {r["arm"]: r for r in read(TABLES / "T22_arms.csv")}
    t24 = {int(r["s"]): r for r in read(TABLES / "T24_scale.csv")}
    a = lambda arm, s=1: float(t[arm][f"bestL_test_l2_{s}shot"])  # noqa: E731
    e0 = lambda arm: float(t[arm]["epoch0_test_l2_1shot"])  # noqa: E731
    z = lambda arm: float(t[arm]["bestL_incl_epoch0_test_l2_1shot"])  # noqa: E731
    lin = ["pca64_lin_cos_naq", "pca64_lin_cos_ours", "pca64_lin_cos_sup", "pca64_lin_cos_sup_strat", "pca256_lin_cos_sup"]
    refs = {arm: 89.05 if arm.startswith("pca64") else 88.55 for arm in lin}
    s10 = float(t24[10]["bestL_test_l2_1shot"])
    rows = [
        ("HLP131", "**Q1**: `gcn_raw_sup` ≥ `gcn_raw_ours` + 1.0", f"{a('gcn_raw_sup'):.4f} − {a('gcn_raw_ours'):.4f}(post-hoc L) = {a('gcn_raw_sup') - a('gcn_raw_ours'):+.4f}"),
        ("HLP132", "`gcn_raw_sup_strat` ≥ `gcn_raw_sup` + 0.5", f"{a('gcn_raw_sup_strat'):.4f} − {a('gcn_raw_sup'):.4f} = {a('gcn_raw_sup_strat') - a('gcn_raw_sup'):+.4f}"),
        ("HLP133", "**Q3**: `gcn_cos_naq` ≥ `gcn_raw_naq`(post-hoc L) − 0.3", f"{a('gcn_cos_naq'):.4f} − {a('gcn_raw_naq'):.4f}(post-hoc L) = {a('gcn_cos_naq') - a('gcn_raw_naq'):+.4f}"),
        ("HLP134", "`gcn_cos_sup` ≥ 89.0", f"{a('gcn_cos_sup'):.4f}"),
        ("HLP135", "**Q2 핵심**: `pca64_lin_cos_sup` ≥ 89.05 + 0.5", f"{a('pca64_lin_cos_sup'):.4f}; − 89.05 = {a('pca64_lin_cos_sup') - 89.05:+.4f}. epoch 0을 선택 후보에 넣으면 {z('pca64_lin_cos_sup'):.4f}"),
        ("HLP136", "`pca64_lin_cos_naq` ≥ 89.05 − 0.5", f"{a('pca64_lin_cos_naq'):.4f}; − 89.05 = {a('pca64_lin_cos_naq') - 89.05:+.4f}. epoch 0을 선택 후보에 넣으면 {z('pca64_lin_cos_naq'):.4f}"),
        ("HLP137", "`pca64_lin_cos_ours` − `pca64_lin_cos_naq` ∈ [−0.5, +0.5]", f"{a('pca64_lin_cos_ours'):.4f} − {a('pca64_lin_cos_naq'):.4f} = {a('pca64_lin_cos_ours') - a('pca64_lin_cos_naq'):+.4f}"),
        ("HLP138", "`pca64_lin_cos_sup_strat` ≥ `pca64_lin_cos_sup` + 0.3", f"{a('pca64_lin_cos_sup_strat'):.4f} − {a('pca64_lin_cos_sup'):.4f} = {a('pca64_lin_cos_sup_strat') - a('pca64_lin_cos_sup'):+.4f}"),
        ("HLP139", "`pca256_mlp_cos_sup` ≤ `pca256_lin_cos_sup` + 0.5", f"{a('pca256_mlp_cos_sup'):.4f} − {a('pca256_lin_cos_sup'):.4f} = {a('pca256_mlp_cos_sup') - a('pca256_lin_cos_sup'):+.4f}"),
        ("HLP140", "`pca*_lin` 모든 arm의 epoch0_test_l2가 HL11 PCA 값과 ±0.05 안", "; ".join(f"`{x}` {e0(x):.4f}(− {refs[x]} = {e0(x) - refs[x]:+.4f})" for x in lin)),
        ("HLP141", "5-shot: `pca64_lin_cos_sup` ≥ 93.72", f"{a('pca64_lin_cos_sup', 5):.4f}; − 93.72 = {a('pca64_lin_cos_sup', 5) - 93.72:+.4f}"),
        ("HLP142", "scale s=5·20의 `bestL_test_l2`가 s=10과 ±1.0 안", f"seed 1212. s=10 {s10:.4f}; s=5 {float(t24[5]['bestL_test_l2_1shot']):.4f}({float(t24[5]['bestL_test_l2_1shot']) - s10:+.4f}); s=20 {float(t24[20]['bestL_test_l2_1shot']):.4f}({float(t24[20]['bestL_test_l2_1shot']) - s10:+.4f})"),
    ]
    write("prereg_HL12.csv", [dict(id=i, prediction=p, computed=c) for i, p, c in rows])


if __name__ == "__main__":
    {"tables": tables, "prereg": prereg}[sys.argv[1]]()
