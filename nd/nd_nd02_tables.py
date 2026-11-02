"""ND02 tables T0-T5 and pre-registration values (--ndp) from results/nd/nd02; computed values only, no verdicts."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path("results/nd/nd02")
T = Path("reports/ND02_tables")
RUNS = ["main", "pseudo_k5", "pseudo_choi", "f_a2x", "unassign_q30", "nbr_a2", "score_var", "main_rewire", "main_gt"]
Q2 = ["s_own", "h", "a2x", "x", "comb"]
Q3 = ["cos_inter5", "cos_full5", "cos_x", "cos_a2x"]


def f4(x):
    return "" if x is None else f"{x:.4f}"


def write(name, rows):
    T.mkdir(parents=True, exist_ok=True)
    with open(T / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load():
    return {r: json.loads((ROOT / "summary" / f"{r}.json").read_text()) for r in RUNS if (ROOT / "summary" / f"{r}.json").exists()}


def t0():
    rows = list(csv.DictReader(open(ROOT / "episodes" / "stats.csv")))
    gen = json.loads((ROOT / "episodes" / "generation.json").read_text())
    out = []
    for name in ("pseudo_k3", "pseudo_k5", "pseudo_choi", "gt_k3"):
        r = [x for x in rows if x["episodes"] == name]
        mean = lambda k: sum(float(x[k]) for x in r) / len(r)  # noqa: E731
        out.append({"episodes": name, "n_episodes": len(r), "support_purity_mean": f4(mean("support_purity")), "query_purity_mean": f4(mean("query_purity")),
                    "anchor_gt_conflicts_mean": f4(mean("anchor_gt_conflicts")), "episodes_with_conflict_frac": f4(sum(int(x["anchor_gt_conflicts"]) > 0 for x in r) / len(r)),
                    "n_query_mean": f4(mean("n_query"))})
    write("T0_episodes.csv", out)
    return gen


def tables(d):
    write("T1_pattern.csv", [{"run": r, "defined_support_frac": f4(v["pattern"]["defined_support_frac"]), "defined_query_frac": f4(v["pattern"]["defined_query_frac"]),
                              "group_pattern_frac": f4(v["pattern"]["group_pattern_frac"]), "pi_entropy_mean": f4(v["pattern"]["pi_entropy_mean"]),
                              "pi_cos_mean": f4(v["pattern"]["pi_cos_mean"]), "h_mean": f4(v["pattern"]["h_mean"]),
                              "n_inter_mean": f4(v["pattern"]["n_inter_mean"]), "n_inter_median": f4(v["pattern"]["n_inter_median"])} for r, v in d.items()])
    write("T2_q1_identify.csv", [{"run": r, "acc_pseudo": f4(v["q1"]["acc_pseudo"]), "ci95": f4(v["q1"]["ci95"]), "n_episodes": v["q1"]["n_episodes"],
                                  "excluded_query_frac": f4(v["q1"]["excluded_query_frac"]), "gt_mismatch_pred_ne_own": f4(v["q1"]["gt_mismatch_pred_ne_own"]),
                                  "gt_mismatch_pred_eq_own": f4(v["q1"]["gt_mismatch_pred_eq_own"])} for r, v in d.items()])
    write("T3_q2_noise.csv", [{"run": r, "score": k, "auroc_rows": f4(v["q2"][k]["auroc_rows"]), "n_episodes": v["q2"][k]["n_episodes"],
                               "auroc_pooled": f4(v["q2"][k]["auroc_pooled"]), "n_queries": v["q2"][k]["n_queries"]} for r, v in d.items() for k in Q2])
    write("T4_q3_pairs.csv", [{"run": r, "score": k, "auroc_all": f4(v["q3"][k]["auroc_all"]), "auroc_episode_mean": f4(v["q3"][k]["auroc_episode_mean"]),
                               **{f"auroc_cos_x_q{q}": f4(v["q3"][k][f"auroc_q{q}"]) for q in (1, 2, 3, 4)},
                               "auroc_same_group": f4(v["q3"][k]["auroc_same_group"]), "auroc_diff_group": f4(v["q3"][k]["auroc_diff_group"]),
                               "n_pairs": v["q3"][k]["n_pairs"]} for r, v in d.items() for k in Q3])
    m = d["main"]
    rows = [{"row": f"n_inter {b}", "query_frac": f4(c["query_frac"]), "acc_pseudo": f4(c["acc_pseudo"]), "auroc_s_own": f4(c["auroc_s_own"]), "auroc_a2x": f4(c["auroc_a2x"]),
             "n_q1": c["n_q1"], "n_ep_s_own": c["n_ep_s_own"], "n_ep_a2x": c["n_ep_a2x"], "pair_cos_inter5": "", "pair_cos_full5": "", "pair_cos_x": "", "pair_cos_a2x": "", "n_pairs": ""}
            for b, c in m["coverage"].items()]
    rows.append({"row": "pairs n_inter>=3 (both nodes)", "query_frac": "", "acc_pseudo": "", "auroc_s_own": "", "auroc_a2x": "", "n_q1": "", "n_ep_s_own": "", "n_ep_a2x": "",
                 **{f"pair_{k}": f4(m["q3"][k]["auroc_ninter_ge3"]) for k in Q3}, "n_pairs": m["q3"]["cos_inter5"]["n_pairs_ninter_ge3"]})
    write("T5_coverage.csv", rows)


def ndp(d):
    m, rw, gt = d["main"], d["main_rewire"], d["main_gt"]
    a = lambda r: r["q1"]["acc_pseudo"]  # noqa: E731
    i = lambda r: r["q3"]["cos_inter5"]["auroc_all"]  # noqa: E731
    q = lambda k: m["q2"][k]["auroc_rows"]  # noqa: E731
    c = m["coverage"]
    rows = [
        ("NDP011", f"n_inter>=1 query frac = {m['pattern']['defined_query_frac']:.4f}"),
        ("NDP012", f"acc_pseudo = {a(m):.4f} (CI {m['q1']['ci95']:.4f})"),
        ("NDP013", f"{a(m):.4f} - rewire {a(rw):.4f} = {a(m) - a(rw):+.4f}"),
        ("NDP014", f"cos_inter5 auroc_all (pooled) = {i(m):.4f}; episode mean = {m['q3']['cos_inter5']['auroc_episode_mean']:.4f}"),
        ("NDP015", f"cos_inter5 auroc cos_x q4 = {m['q3']['cos_inter5']['auroc_q4']:.4f}"),
        ("NDP016", f"{i(m):.4f} - rewire {i(rw):.4f} = {i(m) - i(rw):+.4f}"),
        ("NDP017", f"s_own {q('s_own'):.4f} - h {q('h'):.4f} = {q('s_own') - q('h'):+.4f}"),
        ("NDP018", f"s_own {q('s_own'):.4f} - a2x {q('a2x'):.4f} = {q('s_own') - q('a2x'):+.4f}"),
        ("NDP019", f"comb {q('comb'):.4f} - a2x {q('a2x'):.4f} = {q('comb') - q('a2x'):+.4f}"),
        ("NDP020", f"_gt acc_pseudo = {a(gt):.4f}; _gt cos_inter5 auroc_all = {i(gt):.4f}"),
        ("NDP021", f"_gt {a(gt):.4f} - main {a(m):.4f} = {a(gt) - a(m):+.4f}"),
        ("NDP022", f"n_inter>=5 acc {c['>=5']['acc_pseudo']:.4f} - n_inter 1-2 acc {c['1-2']['acc_pseudo']:.4f} = {c['>=5']['acc_pseudo'] - c['1-2']['acc_pseudo']:+.4f}"),
    ]
    write("NDP_values.csv", [{"id": k, "computed": v} for k, v in rows])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ndp", action="store_true")
    args = p.parse_args()
    runs = load()
    t0()
    tables(runs)
    if args.ndp:
        ndp(runs)
