"""hl_hl05_tables: HL05 tables T8 / T9 and pre-registration values (HLP050-056) from run reports and HL03/HL04 tables. Reads no labels."""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hl_hl03_tables import mean_std, pct, read  # noqa: E402

ROOT = Path("results/hl/hl05")
TABLES = Path("reports/HL05_tables")
SEEDS = (1212, 1213, 1214, 1215, 1216)
MODES = ("near", "far")
BCHOI = (82.71, 89.91)
G1_P10 = (89.42, 94.17)
EMB_KEYS = ["dist_mean", "dist_median", "p_own_mean", "p_own_median", "p_own_ge_095_frac", "loss_mean"]


def write(name, rows):
    import csv
    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {TABLES / name}")


def reports(mode):
    out = {}
    for s in SEEDS:
        p = ROOT / mode / f"seed_{s}" / "report.json"
        if p.exists():
            out[s] = json.loads(p.read_text())
    return out


def t8():
    rows = []
    for mode in MODES:
        docs = reports(mode)
        one = [pct(d["evaluation"]["1"]["test_accuracy"]) for d in docs.values()]
        five = [pct(d["evaluation"]["5"]["test_accuracy"]) for d in docs.values()]
        q = [d["query_diagnostics"] for d in docs.values()]
        m1, s1 = mean_std(one)
        m5, s5 = mean_std(five)
        rows.append(dict(
            arm=mode, n_seeds=len(one), test_1shot_mean=m1, test_1shot_std=s1, test_5shot_mean=m5, test_5shot_std=s5,
            delta_1shot_vs_bchoi=round(statistics.mean(one) - BCHOI[0], 2), delta_5shot_vs_bchoi=round(statistics.mean(five) - BCHOI[1], 2),
            delta_1shot_vs_g1_p1=round(statistics.mean(one) - G1_P10[0], 2), delta_5shot_vs_g1_p1=round(statistics.mean(five) - G1_P10[1], 2),
            purity=round(statistics.mean(x["mean_purity"] for x in q), 4), mean_x_rank=round(statistics.mean(x["mean_x_rank"] for x in q), 1),
            mean_a2x=round(statistics.mean(x["mean_a2x"] for x in q), 4), supplement_query_frac=round(statistics.mean(x["supplement_query_frac"] for x in q), 4),
        ))
    t6 = {r["p"]: r for r in read("reports/HL03_tables/T6_purity_curve.csv")}
    t7 = {r["arm"]: r for r in read("reports/HL04_tables/T7_g2.csv")}
    g1 = t6["1.0"]

    def ref(name, m1, s1, m5, s5, purity, xr):
        return dict(arm=name, n_seeds=5, test_1shot_mean=m1, test_1shot_std=s1, test_5shot_mean=m5, test_5shot_std=s5,
                    delta_1shot_vs_bchoi=round(float(m1) - BCHOI[0], 2), delta_5shot_vs_bchoi=round(float(m5) - BCHOI[1], 2),
                    delta_1shot_vs_g1_p1=round(float(m1) - G1_P10[0], 2), delta_5shot_vs_g1_p1=round(float(m5) - G1_P10[1], 2),
                    purity=purity, mean_x_rank=xr, mean_a2x="", supplement_query_frac="")
    b = t7["B_Choi uniform (HL03 T0)"]
    rows.append(ref("B_Choi uniform (HL03 T0)", b["test_1shot_mean"], b["test_1shot_std"], b["test_5shot_mean"], b["test_5shot_std"], b["query_gt_purity"], b["mean_x_rank"]))
    rows.append(ref("G1 p=1.0 (HL03 T6)", g1["test_1shot_mean"], g1["test_1shot_std"], g1["test_5shot_mean"], g1["test_5shot_std"], g1["purity_mean"], g1["x_rank_mean"]))
    for arm in ("comb_top10", "rand10"):
        r = t7[arm]
        rows.append(ref(f"HL04 {arm}", r["test_1shot_mean"], r["test_1shot_std"], r["test_5shot_mean"], r["test_5shot_std"], r["query_gt_purity"], r["mean_x_rank"]))
    write("T8_structural.csv", rows)
    per_seed = []
    for mode in MODES:
        for s, d in reports(mode).items():
            q = d["query_diagnostics"]
            per_seed.append(dict(arm=mode, seed=s, test_1shot=pct(d["evaluation"]["1"]["test_accuracy"]), test_5shot=pct(d["evaluation"]["5"]["test_accuracy"]),
                                 valid_1shot=pct(d["evaluation"]["1"]["valid_accuracy"]), purity=q["mean_purity"], x_rank=q["mean_x_rank"], mean_a2x=q["mean_a2x"],
                                 **{k: q[k] for k in ("supplement_query_frac", "adjacent_query_frac", "across_query_frac", "b5_query_frac", "b4_centre_query_frac", "diff_fill_query_frac")},
                                 elapsed_seconds=d["elapsed_seconds"]))
    write("T8b_structural_per_seed.csv", per_seed)


def t9():
    main, allseeds = [], []
    for mode in MODES:
        for s, d in reports(mode).items():
            for ck, e in d["query_diagnostics"]["embedding"].items():
                row = dict(arm=mode, seed=s, checkpoint=ck, n_episodes=e["n_episodes"], n_queries=e["n_queries"], **{k: round(e[k], 4) for k in EMB_KEYS})
                allseeds.append(row)
                if s == 1212:
                    main.append({k: v for k, v in row.items() if k != "seed"})
    write("T9_embed_diag.csv", main)
    write("T9b_embed_diag_all_seeds.csv", allseeds)


def prereg():
    t8r = {r["arm"]: r for r in read(TABLES / "T8_structural.csv")}
    t9r = {(r["arm"], r["checkpoint"]): r for r in read(TABLES / "T9_embed_diag.csv")}
    f = lambda a, k: float(t8r[a][k])  # noqa: E731
    e = lambda a, k: float(t9r[(a, "bchoi")][k])  # noqa: E731
    rows = [
        ("HLP050", "핵심: far − near ≥ +3.0 (1-shot, 5-seed 평균)", f"{f('far', 'test_1shot_mean'):.2f} − {f('near', 'test_1shot_mean'):.2f} = {f('far', 'test_1shot_mean') - f('near', 'test_1shot_mean'):+.2f}"),
        ("HLP051", "near 1-shot ≤ B_Choi + 1.0 (= 83.71)", f"near {f('near', 'test_1shot_mean'):.2f}"),
        ("HLP052", "far 1-shot ≥ G1 p=1.0 − 1.0 (= 88.42)", f"far {f('far', 'test_1shot_mean'):.2f}"),
        ("HLP053", "far − near (5-shot) > 0", f"{f('far', 'test_5shot_mean'):.2f} − {f('near', 'test_5shot_mean'):.2f} = {f('far', 'test_5shot_mean') - f('near', 'test_5shot_mean'):+.2f}"),
        ("HLP054", "T9 (B_Choi ckpt): near 거리 중앙값 < far의 절반",
         f"near {e('near', 'dist_median'):.4f} vs far/2 {e('far', 'dist_median') / 2:.4f} (far {e('far', 'dist_median'):.4f})"),
        ("HLP055", "T9 (B_Choi ckpt): near p_own ≥ 0.95 비율 ≥ 0.5, far ≤ 0.2",
         f"near {e('near', 'p_own_ge_095_frac'):.4f}, far {e('far', 'p_own_ge_095_frac'):.4f}"),
        ("HLP056", "T8: 두 arm의 평균 X 순위 차이 ≤ 50", f"far {f('far', 'mean_x_rank'):.1f} − near {f('near', 'mean_x_rank'):.1f} = {f('far', 'mean_x_rank') - f('near', 'mean_x_rank'):+.1f}"),
    ]
    write("prereg_HL05.csv", [dict(id=a, prediction=b, computed=c) for a, b, c in rows])


if __name__ == "__main__":
    {"t8": t8, "t9": t9, "prereg": prereg}[sys.argv[1]]()
