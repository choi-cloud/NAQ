"""hl_hl03_tables: HL03 tables T0 / T6 and pre-registration values (HLP024-035) from run summaries and G0 csv. Reads no labels."""
import csv
import statistics
import sys
from pathlib import Path

BASE = Path("results/hl/hl03/base")
G0 = Path("results/hl/hl03/g0")
G1 = Path("results/hl/hl03/g1")
TABLES = Path("reports/HL03_tables")
SEEDS = (1212, 1213, 1214, 1215, 1216)
PS = (0.4, 0.6, 0.8, 1.0)
OLD = {"naq_1212_1shot": 80.64, "uniform_1shot": "82.70 ± 0.29", "uniform_5shot": 89.91, "uniform_1shot_mean": 82.70}


def read(path):
    with Path(path).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(name, rows):
    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {TABLES / name}")


def pct(x):
    return round(100 * float(x), 2)


def mean_std(v):
    return round(statistics.mean(v), 2), (round(statistics.stdev(v), 2) if len(v) > 1 else "")


def by_seed_shot(rows, key="seed"):
    return {(int(r[key]), int(r["shot"])): r for r in rows if r["status"] == "complete"}


def uniform_baseline():
    u = by_seed_shot(read(BASE / "summary.csv"))
    one = [pct(u[(s, 1)]["test_accuracy"]) for s in SEEDS]
    five = [pct(u[(s, 5)]["test_accuracy"]) for s in SEEDS]
    return u, one, five


def t0():
    naq = by_seed_shot(read(BASE / "summary_smoke.csv"))
    u, one, five = uniform_baseline()
    rows = [dict(arm="NaQ smoke", seed=1212, test_1shot=pct(naq[(1212, 1)]["test_accuracy"]), test_5shot=pct(naq[(1212, 5)]["test_accuracy"]),
                 valid_1shot=pct(naq[(1212, 1)]["valid_accuracy"]), wall_seconds=naq[(1212, 1)]["wall_seconds"],
                 old_server_1shot=OLD["naq_1212_1shot"], old_server_5shot="")]
    for s in SEEDS:
        rows.append(dict(arm="B_Choi uniform", seed=s, test_1shot=pct(u[(s, 1)]["test_accuracy"]), test_5shot=pct(u[(s, 5)]["test_accuracy"]),
                         valid_1shot=pct(u[(s, 1)]["valid_accuracy"]), wall_seconds=u[(s, 1)]["wall_seconds"], old_server_1shot="", old_server_5shot=""))
    m1, s1 = mean_std(one)
    m5, s5 = mean_std(five)
    walls = [float(u[(s, 1)]["wall_seconds"]) for s in SEEDS]
    rows.append(dict(arm="B_Choi uniform", seed="mean ± std", test_1shot=f"{m1} ± {s1}", test_5shot=f"{m5} ± {s5}",
                     valid_1shot=f"{mean_std([pct(u[(s, 1)]['valid_accuracy']) for s in SEEDS])[0]}", wall_seconds=round(statistics.mean(walls), 1),
                     old_server_1shot=OLD["uniform_1shot"], old_server_5shot=OLD["uniform_5shot"]))
    delta = round(statistics.mean(one) - OLD["uniform_1shot_mean"], 2)
    rows.append(dict(arm="G-S", seed="Δ = new − 82.70", test_1shot=delta, test_5shot="", valid_1shot="", wall_seconds="",
                     old_server_1shot="", old_server_5shot=""))
    write("T0_new_server_baseline.csv", rows)
    print(f"G-S delta={delta} |delta|<=1.0: {abs(delta) <= 1.0}")


def t6():
    _, one_b, five_b = uniform_baseline()
    base1, base5 = statistics.mean(one_b), statistics.mean(five_b)
    g = by_seed_shot_p(read(G1 / "summary.csv"))
    rows = []
    for p in PS:
        one = [pct(g[(p, s, 1)]["test_accuracy"]) for s in SEEDS if (p, s, 1) in g]
        five = [pct(g[(p, s, 5)]["test_accuracy"]) for s in SEEDS if (p, s, 5) in g]
        r1 = [g[(p, s, 1)] for s in SEEDS if (p, s, 1) in g]
        m1, s1 = mean_std(one)
        m5, s5 = mean_std(five)
        rows.append(dict(
            p=p, n_seeds=len(one), test_1shot_mean=m1, test_1shot_std=s1, test_5shot_mean=m5, test_5shot_std=s5,
            delta_1shot_vs_bchoi_new=round(statistics.mean(one) - base1, 2), delta_5shot_vs_bchoi_new=round(statistics.mean(five) - base5, 2),
            purity_mean=round(statistics.mean(float(r["purity_mean"]) for r in r1), 4),
            x_rank_mean=round(statistics.mean(float(r["purity_x_rank_mean"]) for r in r1), 1),
            supplement_query_frac=round(statistics.mean(float(r["supplement_query_frac"]) for r in r1), 4),
            supplement_rate_anchor=round(statistics.mean(float(r["supplement_rate_anchor"]) for r in r1), 4),
            bchoi_new_1shot=round(base1, 2), bchoi_new_5shot=round(base5, 2), bchoi_old_1shot=82.70, bchoi_old_5shot=89.91,
        ))
    write("T6_purity_curve.csv", rows)
    per_seed = []
    for p in PS:
        for s in SEEDS:
            if (p, s, 1) in g:
                r1, r5 = g[(p, s, 1)], g[(p, s, 5)]
                per_seed.append(dict(p=p, seed=s, test_1shot=pct(r1["test_accuracy"]), test_5shot=pct(r5["test_accuracy"]),
                                     valid_1shot=pct(r1["valid_accuracy"]), purity=r1["purity_mean"], x_rank=r1["purity_x_rank_mean"],
                                     supplement_query_frac=r1["supplement_query_frac"], wall_seconds=r1["wall_seconds"]))
    write("T6b_purity_per_seed.csv", per_seed)


def by_seed_shot_p(rows):
    return {(float(r["p"]), int(r["seed"]), int(r["shot"])): r for r in rows if r["status"] == "complete"}


def prereg():
    pur = {(r["band"], r["score"]): r for r in read(TABLES / "G0_purity.csv")}
    cov = {r["band"]: r for r in read(TABLES / "G0_coverage.csv")}
    t6r = {float(r["p"]): r for r in read(TABLES / "T6_purity_curve.csv")}
    _, one_b, _ = uniform_baseline()
    base1 = statistics.mean(one_b)
    f = lambda b, s, k: float(pur[(b, s)][k])  # noqa: E731
    raws = [round(f(b, "x", "raw"), 4) for b in ("B1", "B2", "B3", "B4", "B5")]
    choi_r1 = f("B4", "choi", "R1_purity")
    thr = max(0.55, choi_r1 + 0.10)
    m = [float(t6r[p]["test_1shot_mean"]) for p in PS]
    rows = [
        ("HLP024", "raw 순도는 B1 → B5로 단조 감소", f"raw B1–B5 = {' / '.join(map(str, raws))}; 단조 감소 {all(a > b for a, b in zip(raws, raws[1:]))}"),
        ("HLP025", "B4에서 choi R1 순도 > B4 raw 순도", f"choi R1 {choi_r1:.4f} vs raw {f('B4', 'x', 'raw'):.4f}"),
        ("HLP026", "B4에서 enc_b AUROC_mean ≥ 0.70", f"enc_b AUROC_mean {f('B4', 'enc_b', 'AUROC_mean'):.4f}"),
        ("HLP027", "B4에서 mv AUROC_mean − x AUROC_mean ≥ 0.05",
         f"mv {f('B4', 'mv', 'AUROC_mean'):.4f} − x {f('B4', 'x', 'AUROC_mean'):.4f} = {f('B4', 'mv', 'AUROC_mean') - f('B4', 'x', 'AUROC_mean'):.4f}"),
        ("HLP028", "B4에서 enc_n AUROC_mean < enc_b AUROC_mean", f"enc_n {f('B4', 'enc_n', 'AUROC_mean'):.4f} vs enc_b {f('B4', 'enc_b', 'AUROC_mean'):.4f}"),
        ("HLP029", "G0 게이트: B4 comb R1 ≥ max(0.55, choi R1 + 0.10)", f"comb R1 {f('B4', 'comb', 'R1_purity'):.4f} vs 기준 {thr:.4f}"),
        ("HLP030", "B4에서 choi query 충족률 ≥ 0.5", f"query_fill_rate {float(cov['B4']['query_fill_rate']):.4f}"),
        ("HLP031", "T6: 1-shot test는 p에 대해 단조 증가", f"p=0.4/0.6/0.8/1.0: {' / '.join(f'{x:.2f}' for x in m)}; 단조 증가 {all(a < b for a, b in zip(m, m[1:]))}"),
        ("HLP032", "T6: p=1.0 − p=0.4 ≥ +4.0 (1-shot)", f"{m[3]:.2f} − {m[0]:.2f} = {m[3] - m[0]:+.2f}"),
        ("HLP033", "T6: p=0.6 − B_Choi(새 서버) ≥ +1.0 (1-shot)", f"{m[1]:.2f} − {base1:.2f} = {m[1] - base1:+.2f}"),
        ("HLP034", "T6: p=0.4의 1-shot은 B_Choi(새 서버)와 ±0.8 안", f"{m[0]:.2f} − {base1:.2f} = {m[0] - base1:+.2f}"),
        ("HLP035", "T0: 새 서버 uniform 5-seed 1-shot 평균은 82.70과 ±0.5 안", f"{base1:.2f} − 82.70 = {base1 - 82.70:+.2f}"),
    ]
    write("prereg_HL03.csv", [dict(id=a, prediction=b, computed=c) for a, b, c in rows])


if __name__ == "__main__":
    {"t0": t0, "t6": t6, "prereg": prereg}[sys.argv[1]]()
