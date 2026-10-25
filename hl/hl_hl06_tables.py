"""hl_hl06_tables: HL06 T10 / T10b, R1_descriptors refresh and pre-registration values (HLP057-079) from run reports and diagnostics csv.
Reusable `train_rows` / `per_seed_rows` (HL07-HL10 tables import them). Reads no labels."""
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hl_hl03_tables import mean_std, pct, read  # noqa: E402

SEEDS = (1212, 1213, 1214, 1215, 1216)
BCHOI = (82.71, 89.91)
ROOT = Path("results/hl/hl06/train/Amazon_clothing")
TABLES = Path("reports/HL06_tables")
LABEL_FREE = ["encb_top10", "encb_strat5", "encb_ex20", "encb_far50", "knnppr_strat5", "knnppr_ex20"]
ORACLE = ["o_topsame", "o_combswap", "o_far50_p03", "o_far50_p05"]
DESC = ["purity", "mean_x_rank", "mean_s2", "mean_s2rank", "pair_s2", "hop2_frac", "choi_pass_frac", "supplement_query_frac"]
STAGES = ["primary", "b5", "diff_fill", "mid_same", "swapped", "secondary", "band_diff_fill", "adjacent", "across", "b4_centre"]
# reference rows: (name in T10, R1 descriptor row, accuracy source)
REFS = [("NaQ (HL03 T0; seed 1212)", "naq_x10", ("T7", "NaQ (HL03 T0; seed 1212)")), ("B_Choi uniform (HL03 T0)", "bchoi", ("T7", "B_Choi uniform (HL03 T0)")),
        ("G1 p=0.4 (HL03 T6)", "g1_p04", ("T6", "0.4")), ("G1 p=1.0 (HL03 T6)", "g1_p10", ("T6", "1.0")),
        ("p04_choidiff (HL04 T7)", "p04_choidiff", ("T7", "p04_choidiff")), ("HL05 near (T8)", "hl05_near", ("T8", "near")),
        ("HL05 far (T8)", "hl05_far", ("T8", "far")), ("HL04 comb_top10 (T7)", "hl04_comb_top10", ("T7", "comb_top10")),
        ("HL04 encn_top10 (T7)", "hl04_encn_top10", ("T7", "encn_top10")), ("HL04 rand10 (T7)", "hl04_rand10", ("T7", "rand10"))]


def write(tables, name, rows):
    tables.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with (tables / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {tables / name}")


def reports(root, arm, seeds=SEEDS):
    out = {}
    for s in seeds:
        p = Path(root) / arm / f"seed_{s}" / "report.json"
        if p.exists():
            out[s] = json.loads(p.read_text())
    return out


def r4(x):
    return "" if x is None or x != x else round(x, 4)


def train_rows(root, arms, base, label_free_arms, seeds=SEEDS):
    """One row per arm with >= 1 completed seed: accuracy mean/std, Δ vs `base` (1-shot, 5-shot) and descriptor means."""
    rows = []
    for arm in arms:
        docs = reports(root, arm, seeds)
        if not docs:
            rows.append({"arm": arm, "label_free": arm in label_free_arms, "n_seeds": 0})
            continue
        one = [pct(d["evaluation"]["1"]["test_accuracy"]) for d in docs.values()]
        five = [pct(d["evaluation"]["5"]["test_accuracy"]) for d in docs.values()]
        q = [d["query_diagnostics"] for d in docs.values()]
        m1, s1 = mean_std(one)
        m5, s5 = mean_std(five)
        mean = lambda k: statistics.mean(x[k] for x in q) if all(k in x and x[k] == x[k] for x in q) else None  # noqa: E731
        row = {"arm": arm, "label_free": arm in label_free_arms, "n_seeds": len(one), "test_1shot_mean": m1, "test_1shot_std": s1,
               "test_5shot_mean": m5, "test_5shot_std": s5,
               "delta_1shot_vs_bchoi": round(statistics.mean(one) - base[0], 2), "delta_5shot_vs_bchoi": round(statistics.mean(five) - base[1], 2),
               "purity": r4(mean("purity_pooled")), "mean_x_rank": r4(mean("mean_x_rank")), "mean_s2": r4(mean("mean_s2")),
               "mean_s2rank": r4(mean("mean_s2rank")), "pair_s2": r4(mean("pair_s2")), "hop2_frac": r4(mean("hop2_frac")),
               "choi_pass_frac": r4(mean("choi_pass_frac")), "supplement_query_frac": r4(mean("supplement_query_frac"))}
        for k in STAGES[1:]:
            if mean(f"{k}_query_frac") is not None:
                row[f"{k}_query_frac"] = r4(mean(f"{k}_query_frac"))
        for k in ("n_passed_mean", "relax_mean", "half_fallback_mean", "dropped_fp_frac", "post_warmup_query_fp_frac"):
            if mean(k) is not None:
                row[k] = r4(mean(k))
        rows.append(row)
    return rows


def per_seed_rows(root, arms, seeds=SEEDS):
    rows = []
    for arm in arms:
        for s, d in reports(root, arm, seeds).items():
            q = d["query_diagnostics"]
            wall = Path(root) / arm / f"seed_{s}" / "wall_seconds.txt"
            gpu = Path(root) / arm / f"seed_{s}" / "gpu.txt"
            rows.append({"arm": arm, "seed": s, "test_1shot": pct(d["evaluation"]["1"]["test_accuracy"]), "test_5shot": pct(d["evaluation"]["5"]["test_accuracy"]),
                         "valid_1shot": pct(d["evaluation"]["1"]["valid_accuracy"]), "valid_5shot": pct(d["evaluation"]["5"]["valid_accuracy"]),
                         **{k: r4(q.get(k if k != "purity" else "purity_pooled")) for k in DESC},
                         "elapsed_seconds": d["elapsed_seconds"], "wall_seconds": wall.read_text().strip() if wall.exists() else "",
                         "gpu": gpu.read_text().strip() if gpu.exists() else "", "max_cuda_memory_reserved_gb": d.get("max_cuda_memory_reserved_gb")})
    return rows


def ref_acc():
    t6 = {r["p"]: r for r in read("reports/HL03_tables/T6_purity_curve.csv")}
    t7 = {r["arm"]: r for r in read("reports/HL04_tables/T7_g2.csv")}
    t8 = {r["arm"]: r for r in read("reports/HL05_tables/T8_structural.csv")}
    return {"T6": t6, "T7": t7, "T8": t8}


def t10():
    r1 = {r["row"]: r for r in read(TABLES / "R1_descriptors.csv")}
    rows = train_rows(ROOT, LABEL_FREE + ORACLE, BCHOI, LABEL_FREE)
    src = ref_acc()
    for name, r1row, (tab, key) in REFS:
        a = src[tab][key]
        d = r1[r1row]
        m1, m5 = float(a["test_1shot_mean"]), float(a["test_5shot_mean"])
        rows.append({"arm": name, "label_free": r1row in ("naq_x10", "bchoi") or r1row.startswith("hl04_"), "n_seeds": a["n_seeds"],
                     "test_1shot_mean": a["test_1shot_mean"], "test_1shot_std": a["test_1shot_std"], "test_5shot_mean": a["test_5shot_mean"],
                     "test_5shot_std": a["test_5shot_std"], "delta_1shot_vs_bchoi": round(m1 - BCHOI[0], 2), "delta_5shot_vs_bchoi": round(m5 - BCHOI[1], 2),
                     "purity": r4(float(d["purity_pooled"])), **{k: r4(float(d[k])) if d.get(k) not in ("", "nan", None) else "" for k in DESC[1:-1]},
                     "supplement_query_frac": "", "descriptor_source": f"R1_descriptors.csv:{r1row} (anchors of HL03 G0)"})
    write(TABLES, "T10_train.csv", rows)
    write(TABLES, "T10b_train_per_seed.csv", per_seed_rows(ROOT, LABEL_FREE + ORACLE))
    # refresh R1 with the new arms' accuracy
    t10r = {r["arm"]: r for r in rows}
    for name, row in r1.items():
        if name in t10r and t10r[name].get("n_seeds"):
            row["test_1shot"], row["test_5shot"] = t10r[name]["test_1shot_mean"], t10r[name]["test_5shot_mean"]
    write(TABLES, "R1_descriptors.csv", list(r1.values()))


def prereg():
    p1 = {(r["judge"], r["rule"]): r for r in read(TABLES / "P1_judge_rule.csv")}
    p2 = {r["judge"]: r for r in read(TABLES / "P2_auroc.csv")}
    r1 = {r["row"]: r for r in read(TABLES / "R1_descriptors.csv")}
    t10 = {r["arm"]: r for r in read(TABLES / "T10_train.csv")}
    pur = lambda j, rule: float(p1[(j, rule)]["purity_pooled"])  # noqa: E731
    au = lambda j, k: float(p2[j][f"auroc_{k}"])  # noqa: E731
    d = lambda row, k: float(r1[row][k])  # noqa: E731

    def acc(arm, shot=1):
        r = t10.get(arm)
        return float(r[f"test_{shot}shot_mean"]) if r and r.get(f"test_{shot}shot_mean") not in ("", None) else None

    def fmt(x):
        return "미실행" if x is None else f"{x:.2f}"

    def diff(a, b):
        return "미실행" if a is None or b is None else f"{a:.2f} − {b:.2f} = {a - b:+.2f}"

    def nseed(arm):
        r = t10.get(arm)
        return f" (n={r['n_seeds']})" if r and r.get("n_seeds") not in ("5", 5) else ""

    from hl_pdd import JUDGES, NEW_JUDGES
    g3 = [f"{j} {rule} {pur(j, rule):.4f}" for j in JUDGES for rule in ("strat5", "ex20") if pur(j, rule) >= 0.45]
    best_g3 = max(((j, rule, pur(j, rule)) for j in JUDGES for rule in ("strat5", "ex20")), key=lambda t: t[2])
    h063 = "; ".join(f"{j} {pur(j, 'strat5') - pur('enc_b', 'strat5'):+.4f}" for j in NEW_JUDGES)
    lf = [a for a in LABEL_FREE if acc(a) is not None]
    best = max(lf, key=acc) if lf else None
    rows = [
        ("HLP057", "`rand_far50` 순도 ≤ 0.10", f"{pur('-', 'rand_far50'):.4f}"),
        ("HLP058", "`rand_ex20` 순도 ≤ 0.16", f"{pur('-', 'rand_ex20'):.4f}"),
        ("HLP059", "`enc_b` `auroc_far50` ≥ 0.65", f"{au('enc_b', 'far50'):.4f}"),
        ("HLP060", "`enc_b` `strat5` 순도 ∈ [0.33, 0.45]", f"{pur('enc_b', 'strat5'):.4f}"),
        ("HLP061", "`enc_b` `far50` 순도 ∈ [0.20, 0.33]", f"{pur('enc_b', 'far50'):.4f}"),
        ("HLP062", "게이트 G3: 어느 판정기든 `strat5` 또는 `ex20` 순도 ≥ 0.45", f"통과 {g3 if g3 else '없음'}; 최고 {best_g3[0]} {best_g3[1]} {best_g3[2]:.4f}"),
        ("HLP063", "새 판정기 중 하나 이상이 `strat5` 순도에서 `enc_b`보다 +0.03 이상", f"strat5 순도 − enc_b({pur('enc_b', 'strat5'):.4f}): {h063}"),
        ("HLP064", "`xidf` `auroc_far50` ≥ `x` `auroc_far50` + 0.03", f"xidf {au('xidf', 'far50'):.4f} − x {au('x', 'far50'):.4f} = {au('xidf', 'far50') - au('x', 'far50'):+.4f}"),
        ("HLP065", "`hl04_comb_top10`의 `mean_s2` ≥ 0.70", f"{d('hl04_comb_top10', 'mean_s2'):.4f} (hl05_near {d('hl05_near', 'mean_s2'):.4f})"),
        ("HLP066", "`hl04_comb_top10`의 `pair_s2` ≥ 0.60, `bchoi`의 `pair_s2` ≤ 0.40", f"comb_top10 {d('hl04_comb_top10', 'pair_s2'):.4f}, bchoi {d('bchoi', 'pair_s2'):.4f}"),
        ("HLP067", "`hl04_comb_top10`의 `hop2_frac` ≥ 2 × `bchoi`의 `hop2_frac`",
         f"comb_top10 {d('hl04_comb_top10', 'hop2_frac'):.4f} vs 2 × bchoi {2 * d('bchoi', 'hop2_frac'):.4f} (bchoi {d('bchoi', 'hop2_frac'):.4f})"),
        ("HLP068", "`hl04_comb_top10`의 `fp_s2` ≥ `p04_choidiff`의 `fp_s2`", f"comb_top10 {d('hl04_comb_top10', 'fp_s2'):.4f} vs p04_choidiff {d('p04_choidiff', 'fp_s2'):.4f}"),
        ("HLP069", "`o_topsame` 1-shot ≤ HL05 near − 1.0 (= 84.49)", fmt(acc("o_topsame")) + nseed("o_topsame")),
        ("HLP070", "|`o_combswap` − HL04 `comb_top10`(79.41)| ≤ 1.0", diff(acc("o_combswap"), 79.41) + nseed("o_combswap")),
        ("HLP071", "`o_far50_p05` 1-shot ≥ 85.0", fmt(acc("o_far50_p05")) + nseed("o_far50_p05")),
        ("HLP072", "`o_far50_p03` 1-shot ≥ B_Choi + 0.5 (= 83.21)", fmt(acc("o_far50_p03")) + nseed("o_far50_p03")),
        ("HLP073", "`encb_strat5` − B_Choi ≥ +1.0 (1-shot)", diff(acc("encb_strat5"), BCHOI[0]) + nseed("encb_strat5")),
        ("HLP074", "`encb_ex20` − B_Choi ≥ +1.0 (1-shot)", diff(acc("encb_ex20"), BCHOI[0]) + nseed("encb_ex20")),
        ("HLP075", "`encb_far50` < `encb_strat5` (1-shot)", diff(acc("encb_far50"), acc("encb_strat5"))),
        ("HLP076", "`knnppr_strat5` ≥ `encb_strat5` (1-shot)", diff(acc("knnppr_strat5"), acc("encb_strat5"))),
        ("HLP077", "레이블 없는 6 arm 중 1-shot 최고 arm의 5-shot ≥ B_Choi(89.91)", "미실행" if best is None else f"{best} 1-shot {acc(best):.2f}, 5-shot {acc(best, 5):.2f}"),
        ("HLP078", "핵심: `encb_strat5` − `encb_top10` ≥ +2.0 (1-shot)", diff(acc("encb_strat5"), acc("encb_top10"))),
        ("HLP079", "`encb_top10`은 HL04 `encn_top10`(79.94)과 ±1.0 안", diff(acc("encb_top10"), 79.94)),
    ]
    write(TABLES, "prereg_HL06.csv", [dict(id=a, prediction=b, computed=c) for a, b, c in rows])


def extra():
    """OVERNIGHT §여분: label-free arms with seeds 1217-1221 (results/hl/hl06/train_extra) next to the 5 main seeds; 10-seed means."""
    extra_root, extra_seeds = Path("results/hl/hl06/train_extra"), (1217, 1218, 1219, 1220, 1221)
    rows = []
    for arm in LABEL_FREE:
        main, ext = reports(ROOT, arm), reports(extra_root, arm, extra_seeds)
        row = {"arm": arm}
        for shot in ("1", "5"):
            a = [pct(d["evaluation"][shot]["test_accuracy"]) for d in main.values()]
            b = [pct(d["evaluation"][shot]["test_accuracy"]) for d in ext.values()]
            for tag, v in (("seeds1212_1216", a), ("seeds1217_1221", b), ("all", a + b)):
                m, sd = mean_std(v) if v else ("", "")
                row.update({f"test_{shot}shot_{tag}_n": len(v), f"test_{shot}shot_{tag}_mean": m, f"test_{shot}shot_{tag}_std": sd})
            row[f"delta_{shot}shot_all_vs_bchoi"] = round(statistics.mean(a + b) - BCHOI[int(shot) // 5], 2) if a + b else ""
        rows.append(row)
    write(TABLES, "T10x_extra_seeds.csv", rows)
    write(TABLES, "T10x_extra_seeds_per_seed.csv", per_seed_rows(extra_root, LABEL_FREE, extra_seeds))


if __name__ == "__main__":
    {"t10": t10, "prereg": prereg, "extra": extra}[sys.argv[1]]()
