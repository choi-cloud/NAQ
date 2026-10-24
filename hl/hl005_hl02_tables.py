"""hl005_hl02_tables: builds HL02 tables T4, T5 and pre-registration values (HLP007-023) from hl002-hl004 outputs."""

import argparse
import csv
import json
import statistics
from pathlib import Path

POLICIES = ["uniform", "quintile:1", "quintile:2", "quintile:3", "quintile:4", "quintile:5",
            "gt_distinct", "hard:2", "hard:4", "hard_blind:4", "ab"]


def read_csv(path):
    with Path(path).open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write(path, rows, fields=None):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[tables] wrote {path}")


def ms(v):
    return (round(statistics.mean(v), 2), round(statistics.stdev(v), 2) if len(v) > 1 else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oracle_root", default="results/hl/hl02/oracle")
    p.add_argument("--table_dir", default="reports/HL02_tables")
    p.add_argument("--seeds", default="1212,1213,1214,1215,1216")
    p.add_argument("--p0_bchoi_mean", type=float, default=82.81)
    a = p.parse_args()
    td = Path(a.table_dir)
    seeds = [int(x) for x in a.seeds.split(",")]

    per = {}
    for pol in POLICIES:
        acc = {"t1": [], "t5": [], "v1": [], "same": [], "pi": [], "keff": [], "seeds": []}
        for seed in seeds:
            f = Path(a.oracle_root) / pol.replace(":", "_") / f"seed_{seed}" / "report.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            acc["t1"].append(100 * d["evaluation"]["1"]["test_accuracy"])
            acc["t5"].append(100 * d["evaluation"]["5"]["test_accuracy"])
            acc["v1"].append(100 * d["evaluation"]["1"]["valid_accuracy"])
            ad = d["anchor_diagnostics"]
            acc["same"].append(ad["same_gt_pair_rate"])
            acc["pi"].append(ad["anchor_pi_bchoi_mean"])
            if ad["k_eff_mean"] is not None:
                acc["keff"].append(ad["k_eff_mean"])
            acc["seeds"].append(seed)
        per[pol] = acc

    u1 = statistics.mean(per["uniform"]["t1"])
    u5 = statistics.mean(per["uniform"]["t5"])
    t4 = []
    for pol in POLICIES:
        x = per[pol]
        if not x["t1"]:
            t4.append(dict(policy=pol, n_seeds=0))
            continue
        m1, s1 = ms(x["t1"])
        m5, s5 = ms(x["t5"])
        t4.append(dict(
            policy=pol, n_seeds=len(x["t1"]), test_1shot_mean=m1, test_1shot_std=s1, test_5shot_mean=m5, test_5shot_std=s5,
            valid_1shot_mean=round(statistics.mean(x["v1"]), 2),
            delta_1shot_vs_uniform=round(statistics.mean(x["t1"]) - u1, 2),
            delta_5shot_vs_uniform=round(statistics.mean(x["t5"]) - u5, 2),
            same_gt_pair_rate=round(statistics.mean(x["same"]), 4),
            anchor_pi_bchoi_mean=round(statistics.mean(x["pi"]), 4),
            k_eff_mean=round(statistics.mean(x["keff"]), 3) if x["keff"] else "",
        ))
    write(td / "T4_oracle.csv", t4, list(t4[1]) if len(t4) > 1 else None)
    per_seed = [dict(policy=pol, seed=s, test_1shot=round(t1, 2), test_5shot=round(t5, 2), valid_1shot=round(v1, 2))
                for pol in POLICIES for s, t1, t5, v1 in zip(per[pol]["seeds"], per[pol]["t1"], per[pol]["t5"], per[pol]["v1"])]
    write(td / "T4b_oracle_per_seed.csv", per_seed)

    t5 = []
    for pol in ["uniform"] + [f"quintile:{q}" for q in range(1, 6)]:
        m, s = ms(per[pol]["t1"])
        t5.append(dict(policy=pol, test_1shot_mean=m, test_1shot_std=s, n_seeds=len(per[pol]["t1"])))
    write(td / "T5_quintile_curve.csv", t5)

    m = {pol: statistics.mean(per[pol]["t1"]) for pol in POLICIES}
    m5 = {pol: statistics.mean(per[pol]["t5"]) for pol in POLICIES}
    curve = [m[f"quintile:{q}"] for q in range(1, 6)]
    d1 = {r["stat"]: r for r in read_csv(td / "D1_node_stats.csv")}
    d2 = {r["set"]: r for r in read_csv(td / "D2_pair_stats.csv")}
    d3 = {r["stat"]: float(r["value"]) for r in read_csv(td / "D3_episode_stats.csv")}
    e1 = read_csv(td / "E1_pair_dist.csv") if (td / "E1_pair_dist.csv").exists() else []
    e2 = read_csv(td / "E2_neg_softmax.csv") if (td / "E2_neg_softmax.csv").exists() else []
    f = lambda x: f"{float(x):.4f}"  # noqa: E731
    acc_r, rej_r, uni_r = (float(d2[k]["same_gt_rate"]) for k in ("topL_accepted", "topL_rejected", "uniform_random"))
    rows = [
        ("HLP007", f"pi_bchoi mean={f(d1['pi_bchoi']['mean'])}; pi_naq mean={f(d1['pi_naq']['mean'])}"),
        ("HLP008", f"rho(raw_retained,degree)={f(d1['spearman:raw_retained~degree']['mean'])}"),
        ("HLP009", f"rho(raw_retained,pi_bchoi)={f(d1['spearman:raw_retained~pi_bchoi']['mean'])}"),
        ("HLP010", f"rejected={rej_r:.4f}; accepted={acc_r:.4f} (half={acc_r/2:.4f}); uniform={uni_r:.4f}"),
        ("HLP011", f"all_distinct={d3['anchor_gt_all_distinct']:.4f}; exactly_one_pair={d3['anchor_gt_exactly_one_pair_same']:.4f}"),
        ("HLP012", f"query_overlap_episode_rate={d3['query_sets_overlap_between_anchors']:.4f}"),
        ("HLP013", f"quintile 1..5 = {', '.join(f'{v:.2f}' for v in curve)}; max-min={max(curve)-min(curve):.2f}"),
        ("HLP014", f"q5-uniform={m['quintile:5']-m['uniform']:.2f}; uniform={m['uniform']:.2f}; "
                   f"curve q1..q5 = {', '.join(f'{v:.2f}' for v in curve)}"),
        ("HLP015", f"gt_distinct-uniform={m['gt_distinct']-m['uniform']:.2f}"),
        ("HLP016", f"hard:4-uniform={m['hard:4']-m['uniform']:.2f}"),
        ("HLP017", f"hard:4-hard_blind:4={m['hard:4']-m['hard_blind:4']:.2f}"),
        ("HLP018", f"uniform={m['uniform']:.2f}; hard:2={m['hard:2']:.2f}; hard:4={m['hard:4']:.2f}"),
        ("HLP019", f"ab={m['ab']:.2f}; max(quintile:5,hard:4)-0.3={max(m['quintile:5'], m['hard:4'])-0.3:.2f}"),
        ("HLP020", "; ".join(f"{pol}: d1={m[pol]-m['uniform']:.2f}, d5={m5[pol]-m5['uniform']:.2f}" for pol in POLICIES if pol != "uniform")),
    ]
    if e1:
        g = {(r["checkpoint"], r["set"]): r for r in e1}
        rows.append(("HLP021", f"ours ckpt: rejected median={f(g[('ours_tau0.3','rejected')]['median'])}; "
                               f"uniform median={f(g[('ours_tau0.3','uniform')]['median'])}"))
    if e2:
        g = {(r["checkpoint"], r["policy"]): r for r in e2}
        rows.append(("HLP022", f"ours ckpt uniform: p_neg_sum median={f(g[('ours_tau0.3','uniform')]['p_neg_sum_median'])}"))
        hb, un = float(g[("ours_tau0.3", "hard_blind:4")]["p_neg_sum_mean"]), float(g[("ours_tau0.3", "uniform")]["p_neg_sum_mean"])
        hb_n, un_n = float(g[("naq", "hard_blind:4")]["p_neg_sum_mean"]), float(g[("naq", "uniform")]["p_neg_sum_mean"])
        rows.append(("HLP023", f"ours ckpt: hard_blind:4 mean={hb:.4f}, uniform mean={un:.4f}, ratio={hb/un:.3f}; "
                               f"(naq ckpt ratio={hb_n/un_n:.3f})"))
    best_q = max(curve)
    rows.append(("decision_input", f"max(quintile best={best_q:.2f}, hard:4={m['hard:4']:.2f}, ab={m['ab']:.2f}) - uniform="
                                   f"{max(best_q, m['hard:4'], m['ab'])-m['uniform']:.2f}"))
    rows.append(("consistency", f"uniform 5-seed mean={m['uniform']:.2f}; P0 B_Choi 3-seed mean={a.p0_bchoi_mean}; "
                                f"diff={m['uniform']-a.p0_bchoi_mean:.2f}"))
    write(td / "prereg_HL02.csv", [dict(id=i, value=v) for i, v in rows])


if __name__ == "__main__":
    main()
