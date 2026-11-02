"""ND01 tables: TA1-TA3 (--gate A), TB1-TB3 (--gate B), pre-registration values (--gate ndp) from results/nd/nd01; computed values only, no verdicts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import nd_vote as V

R = V.OUT
T = Path("reports/ND01_tables")
HL06_A2X = 0.7564


def write(name, rows):
    T.mkdir(parents=True, exist_ok=True)
    with open(T / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def read(name):
    with open(T / name) as fh:
        return list(csv.DictReader(fh))


def fmt(x, d=4):
    return f"{x:.{d}f}"


def ci95(a):
    return 1.96 * a.std(ddof=0) / np.sqrt(len(a))


def gate_a():
    meta = json.loads((R / "gateA/selection.json").read_text())
    sel = meta["selection"]

    def load(arm, shot, pool):
        return 100 * np.load(R / f"gateA/{shot}shot/{arm}_{pool}.npy")

    arms = ["feat_x", "feat_a2x"]
    votes = [V.arm_name(*g) for g in V.grid() if (R / f"gateA/1shot/{V.arm_name(*g)}_test.npy").exists()]
    arms += votes
    for f in ("x", "a2x"):
        for src in sel[f]["comb_sources"]:
            arms += [f"comb_{src[len('vote_'):]}_l{lam}" for lam in (0.25, 0.5, 0.75)]
        arms.append(f"comb_{f}_sel")
    ta1 = []
    for arm in arms:
        f = "a2x" if ("_a2x_" in arm or arm.endswith("_a2x")) else "x"
        row = {"arm": arm}
        paired = {}
        for shot in (1, 5):
            test, base = load(arm, shot, "test"), load(f"feat_{f}", shot, "test")
            row[f"valid_{shot}shot"] = fmt(load(arm, shot, "valid").mean())
            row[f"test_{shot}shot"] = fmt(test.mean())
            row[f"test_{shot}shot_ci95"] = fmt(ci95(test))
            d = test - base
            paired[shot] = {f"paired_delta_{shot}shot_mean": fmt(d.mean()), f"paired_delta_{shot}shot_ci95": fmt(ci95(d)),
                            f"paired_win_frac_{shot}shot": fmt((d > 0).mean())}
        row["delta_1shot_vs_feat_same_f"] = fmt(load(arm, 1, "test").mean() - load(f"feat_{f}", 1, "test").mean())
        row.update(paired[1])
        row.update(paired[5])
        ta1.append(row)
    write("TA1_gateA.csv", ta1)
    ta2 = []
    for f, t, nbr in V.grid():
        arm = V.arm_name(f, t, nbr)
        if arm in votes:
            ta2.append({"f": f, "tau": t, "nbr": nbr, "valid_1shot": fmt(load(arm, 1, "valid").mean()), "test_1shot": fmt(load(arm, 1, "test").mean())})
    write("TA2_grid_1shot.csv", ta2)
    write("TA3_selection.csv", [{
        "f": f, "vote_best": s["vote_best"], "vote_best_valid_1shot": fmt(s["vote_best_valid_1shot"]),
        "vote_best_tied": " ".join(s["vote_best_tied"]), "comb_sel_arm": s["comb_sel_arm"], "comb_sel_lambda": s["comb_sel_lambda"],
        "comb_sel_valid_1shot": fmt(s["comb_sel_valid_1shot"]), "comb_sel_lambda_tied": " ".join(map(str, s["comb_sel_lambda_tied"])),
    } for f, s in sel.items()])


def gate_b():
    meta = json.loads((R / "gateB/summary.json").read_text())
    s = meta["summary"]
    order = [k for k in ("ax", "a2x", "enc_b") if k in s]
    for f in ("x", "a2x"):
        for t in V.TAU_KEYS:
            order.append(f"own_hard_{f}" if t == "hard" else f"own_soft_{f}_{t}")  # own row directly next to its vote rows
            order += [V.arm_name(f, t, nbr) for nbr in V.NBRS[f] if V.arm_name(f, t, nbr) in s]
    assert set(order) == set(s), set(order) ^ set(s)
    write("TB1_auroc.csv", [{"score": k, "auroc_all": fmt(s[k]["auroc_all"]), "n_rows": s[k]["n_rows"], "auroc_pooled": fmt(s[k]["auroc_pooled"]),
                             "auroc_far50": fmt(s[k]["auroc_far50"]), "n_rows_far50": s[k]["n_rows_far50"]} for k in order])
    z = np.load(R / "gateB/auroc_rows.npz")
    tb2 = []
    for other in ("a2x", "own_soft_x_0.1"):
        d = z[f"all__{V.MAIN_ARM}"] - z[f"all__{other}"]
        tb2.append({"pair": f"{V.MAIN_ARM} - {other}", "mean": fmt(d.mean()), "ci95": fmt(ci95(d)), "positive_frac": fmt((d > 0).mean()), "n_rows": len(d)})
    write("TB2_paired.csv", tb2)
    write("TB3_grid.csv", [{"f": f, "tau": t, "nbr": nbr, "auroc_all": fmt(s[V.arm_name(f, t, nbr)]["auroc_all"])}
                           for f, t, nbr in V.grid() if V.arm_name(f, t, nbr) in s])


def ndp():
    a = {r["arm"]: r for r in read("TA1_gateA.csv")}
    b = {r["score"]: r for r in read("TB1_auroc.csv")}
    t1 = lambda k: float(a[k]["test_1shot"])  # noqa: E731
    all_ = lambda k: float(b[k]["auroc_all"])  # noqa: E731
    gap = lambda k: all_(k) - float(b[k]["auroc_far50"])  # noqa: E731
    m, m1 = V.MAIN_ARM, "vote_x_hard_1hop"
    rows = [
        ("NDP001", f"comb_x_sel {t1('comb_x_sel'):.2f} - feat_x {t1('feat_x'):.2f} = {t1('comb_x_sel') - t1('feat_x'):+.2f}"),
        ("NDP002", f"paired_win_frac_1shot = {a['comb_x_sel']['paired_win_frac_1shot']}"),
        ("NDP003", f"{m} {t1(m):.2f}; feat_x - 3.0 = {t1('feat_x') - 3:.2f}; {m} - feat_x = {t1(m) - t1('feat_x'):+.2f}"),
        ("NDP004", f"vote_x_0.1_self {t1('vote_x_0.1_self'):.2f}; {m} {t1(m):.2f}; diff = {t1('vote_x_0.1_self') - t1(m):+.2f}"),
        ("NDP005", f"{m} {t1(m):.2f}; {m1} {t1(m1):.2f}; diff = {t1(m) - t1(m1):+.2f}"),
        ("NDP006", f"5-shot delta {float(a['comb_x_sel']['paired_delta_5shot_mean']):+.2f}; 1-shot delta {float(a['comb_x_sel']['paired_delta_1shot_mean']):+.2f}"),
        ("NDP007", f"{m} {all_(m):.4f} - a2x {all_('a2x'):.4f} = {all_(m) - all_('a2x'):+.4f}"),
        ("NDP008", f"{m} {all_(m):.4f} - own_soft_x_0.1 {all_('own_soft_x_0.1'):.4f} = {all_(m) - all_('own_soft_x_0.1'):+.4f}"),
        ("NDP009", f"{m} all-far50 = {gap(m):.4f}; a2x all-far50 = {gap('a2x'):.4f}"),
        ("NDP010", f"a2x {all_('a2x'):.4f}; a2x - HL06 P2 {HL06_A2X} = {all_('a2x') - HL06_A2X:+.4f}"),
    ]
    write("NDP_values.csv", [{"id": i, "computed": v} for i, v in rows])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gate", required=True, choices=["A", "B", "ndp"])
    {"A": gate_a, "B": gate_b, "ndp": ndp}[p.parse_args().gate]()
