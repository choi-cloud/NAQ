"""hl_hl04_tables: HL04 table T7 and pre-registration values (HLP036-049) from G0b csv, G2 summary and HL03 tables. Reads no labels."""
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hl_hl03_tables import mean_std, pct, read  # noqa: E402

G2 = Path("results/hl/hl04/g2")
BASE = Path("results/hl/hl03/base")
TABLES = Path("reports/HL04_tables")
HL03 = Path("reports/HL03_tables")
SEEDS = (1212, 1213, 1214, 1215, 1216)
ARMS = ["a2x_top10", "mv_top10", "mv_mid30", "encn_top10", "comb_top10", "rand10", "shuf_comb", "p04_choidiff"]
SCORED = ["x", "ax", "a2x", "ppr", "mv", "enc_b", "enc_n", "comb"]
G1_P04 = 83.92


def write(name, rows):
    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {TABLES / name}")


def bchoi():
    rows = [r for r in read(BASE / "summary.csv") if r["status"] == "complete"]
    one = [pct(r["test_accuracy"]) for r in rows if r["shot"] == "1"]
    five = [pct(r["test_accuracy"]) for r in rows if r["shot"] == "5"]
    r1 = [r for r in rows if r["shot"] == "1"]
    return one, five, statistics.mean(float(r["query_gt_match"]) for r in r1), statistics.mean(float(r["mean_original_x_rank"]) for r in r1)


def g1_curve():
    t6 = read(HL03 / "T6_purity_curve.csv")
    return [(float(r["purity_mean"]), float(r["test_1shot_mean"])) for r in t6]


def interp(curve, x):
    """Linear interpolation of the G1 1-shot curve at purity x; None outside the measured purity range."""
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return None


def t7():
    one_b, five_b, pur_b, xr_b = bchoi()
    base1, base5 = statistics.mean(one_b), statistics.mean(five_b)
    g = {(r["arm"], int(r["seed"]), int(r["shot"])): r for r in read(G2 / "summary.csv") if r["status"] == "complete"}
    curve = g1_curve()
    rows = []
    for arm in ARMS:
        one = [pct(g[(arm, s, 1)]["test_accuracy"]) for s in SEEDS if (arm, s, 1) in g]
        five = [pct(g[(arm, s, 5)]["test_accuracy"]) for s in SEEDS if (arm, s, 5) in g]
        r1 = [g[(arm, s, 1)] for s in SEEDS if (arm, s, 1) in g]
        m1, s1 = mean_std(one)
        m5, s5 = mean_std(five)
        purity = statistics.mean(float(r["purity_mean"]) for r in r1)
        g1 = interp(curve, purity)
        rows.append(dict(
            arm=arm, label_free=arm != "p04_choidiff", n_seeds=len(one), test_1shot_mean=m1, test_1shot_std=s1, test_5shot_mean=m5, test_5shot_std=s5,
            delta_1shot_vs_bchoi=round(statistics.mean(one) - base1, 2), delta_5shot_vs_bchoi=round(statistics.mean(five) - base5, 2),
            query_gt_purity=round(purity, 4), mean_x_rank=round(statistics.mean(float(r["purity_x_rank_mean"]) for r in r1), 1),
            choi_pass_frac=round(statistics.mean(float(r["choi_pass_frac"]) for r in r1), 4),
            g1_curve_1shot_at_purity="" if g1 is None else round(g1, 2),
            minus_g1_curve="" if g1 is None else round(statistics.mean(one) - g1, 2),
        ))
    naq = {int(r["shot"]): r for r in read(BASE / "summary_smoke.csv")}
    m1, s1 = mean_std(one_b)
    m5, s5 = mean_std(five_b)
    rows.append(dict(arm="B_Choi uniform (HL03 T0)", label_free=True, n_seeds=5, test_1shot_mean=m1, test_1shot_std=s1, test_5shot_mean=m5, test_5shot_std=s5,
                     delta_1shot_vs_bchoi=0.0, delta_5shot_vs_bchoi=0.0, query_gt_purity=round(pur_b, 4), mean_x_rank=round(xr_b, 1), choi_pass_frac="",
                     g1_curve_1shot_at_purity="", minus_g1_curve=""))
    rows.append(dict(arm="NaQ (HL03 T0; seed 1212)", label_free=True, n_seeds=1, test_1shot_mean=pct(naq[1]["test_accuracy"]), test_1shot_std="",
                     test_5shot_mean=pct(naq[5]["test_accuracy"]), test_5shot_std="",
                     delta_1shot_vs_bchoi=round(pct(naq[1]["test_accuracy"]) - base1, 2), delta_5shot_vs_bchoi=round(pct(naq[5]["test_accuracy"]) - base5, 2),
                     query_gt_purity=round(float(naq[1]["query_gt_match"]), 4), mean_x_rank="", choi_pass_frac="", g1_curve_1shot_at_purity="", minus_g1_curve=""))
    write("T7_g2.csv", rows)
    per_seed = []
    for arm in ARMS:
        for s in SEEDS:
            if (arm, s, 1) in g:
                r1, r5 = g[(arm, s, 1)], g[(arm, s, 5)]
                per_seed.append(dict(arm=arm, seed=s, test_1shot=pct(r1["test_accuracy"]), test_5shot=pct(r5["test_accuracy"]),
                                     valid_1shot=pct(r1["valid_accuracy"]), purity=r1["purity_mean"], x_rank=r1["purity_x_rank_mean"],
                                     choi_pass_frac=r1["choi_pass_frac"], n_diff_pass_query_frac=r1["n_diff_pass_query_frac"],
                                     n_diff_nonpass_fill_query_frac=r1["n_diff_nonpass_fill_query_frac"], wall_seconds=r1["wall_seconds"]))
    write("T7b_g2_per_seed.csv", per_seed)


def prereg():
    g0b = {(r["score"], r["rule"]): r for r in read(TABLES / "G0b_topk.csv") if r["band"] == "B4"}
    r1 = {r["score"]: float(r["R1_purity"]) for r in read(HL03 / "G0_purity.csv") if r["band"] == "B4"}
    t7r = {r["arm"]: r for r in read(TABLES / "T7_g2.csv")}
    pur = lambda s, rule: float(g0b[(s, rule)]["purity_pooled"])  # noqa: E731
    d1 = lambda a: float(t7r[a]["delta_1shot_vs_bchoi"])  # noqa: E731
    m1 = lambda a: float(t7r[a]["test_1shot_mean"])  # noqa: E731
    gate = [f"{s} {rule} {pur(s, rule):.4f}" for s in SCORED for rule in ("top:10", "mid:30") if pur(s, rule) >= 0.55]
    best_gate = max(((s, rule, pur(s, rule)) for s in SCORED for rule in ("top:10", "mid:30")), key=lambda t: t[2])
    h041 = [f"{s} {pur(s, 'top:10'):.4f}/{pur(s, 'top:30'):.4f}/{r1[s]:.4f} {pur(s, 'top:10') > pur(s, 'top:30') > r1[s]}" for s in SCORED]
    h042 = [f"{s} {pur(s, 'mid:30') - pur(s, 'top:30'):+.4f}" for s in SCORED]
    h047 = []
    for a in ARMS:
        r = t7r[a]
        if r["minus_g1_curve"] == "":
            h047.append(f"{a} 순도 {float(r['query_gt_purity']):.3f} G1 범위 밖")
        else:
            h047.append(f"{a} {float(r['test_1shot_mean']):.2f}−{float(r['g1_curve_1shot_at_purity']):.2f}={float(r['minus_g1_curve']):+.2f}")
    best = max((a for a in ARMS if a != "p04_choidiff"), key=d1)
    rows = [
        ("HLP036", "G0b B4: rand 순도 0.178 ± 0.02", f"rand {pur('-', 'rand'):.4f}"),
        ("HLP037", "G0b B4: choi 10개 순도 0.42 ± 0.04", f"choi {pur('choi', 'choi'):.4f} (필터 통과 < 10 anchor 제외 {float(g0b[('choi', 'choi')]['excluded_frac']):.4f})"),
        ("HLP038", "G0b B4: a2x top:10 순도 ≥ 0.45", f"a2x top:10 {pur('a2x', 'top:10'):.4f}"),
        ("HLP039", "G0b B4: enc_n top:10 순도 ≥ 0.50", f"enc_n top:10 {pur('enc_n', 'top:10'):.4f}"),
        ("HLP040", "게이트: B4 top:10 또는 mid:30의 어느 점수가 순도 ≥ 0.55",
         f"통과 {gate if gate else '없음'}; 최고 {best_gate[0]} {best_gate[1]} {best_gate[2]:.4f}"),
        ("HLP041", "G0b B4: 모든 점수에서 top:10 > top:30 > R1(HL03)", "top10/top30/R1: " + "; ".join(h041)),
        ("HLP042", "G0b B4: mid:30 순도는 top:30과 ±0.03 안", "mid30−top30: " + "; ".join(h042)),
        ("HLP043", "T7: rand10 − B_Choi ≤ −1.0", f"{d1('rand10'):+.2f}"),
        ("HLP044", "T7: shuf_comb는 rand10과 ±0.5 안", f"{m1('shuf_comb'):.2f} − {m1('rand10'):.2f} = {m1('shuf_comb') - m1('rand10'):+.2f}"),
        ("HLP045", "T7: encn_top10 − B_Choi ≥ +1.0 (1-shot)", f"{d1('encn_top10'):+.2f}"),
        ("HLP046", "T7: a2x_top10 − B_Choi ≥ +0.5 (1-shot)", f"{d1('a2x_top10'):+.2f}"),
        ("HLP047", "T7: 각 arm 1-shot은 G1 곡선(실제 순도 위치)과 ±1.0 안", "; ".join(h047)),
        ("HLP048", "T7: encn_top10·comb_top10의 5-shot Δ ≥ 0",
         f"encn_top10 {float(t7r['encn_top10']['delta_5shot_vs_bchoi']):+.2f}, comb_top10 {float(t7r['comb_top10']['delta_5shot_vs_bchoi']):+.2f}"),
        ("HLP049", "보조: p04_choidiff − G1 p=0.4(83.92) ≤ −0.8", f"{m1('p04_choidiff'):.2f} − {G1_P04:.2f} = {m1('p04_choidiff') - G1_P04:+.2f}"),
        ("(참고)", "T7 레이블 없는 arm 중 최고 1-shot Δ", f"{best} {d1(best):+.2f}"),
    ]
    write("prereg_HL04.csv", [dict(id=a, prediction=b, computed=c) for a, b, c in rows])


if __name__ == "__main__":
    {"t7": t7, "prereg": prereg}[sys.argv[1]]()
