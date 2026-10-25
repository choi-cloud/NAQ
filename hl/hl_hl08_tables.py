"""hl_hl08_tables: HL08 (dblp) / HL09 (Amazon_eletronics) tables — baseline T13/T15, training T14/T16 and pre-registration values
(HLP093-102 / HLP103-112). Usage: python hl/hl_hl08_tables.py {base,train,prereg} {hl08,hl09}. Reads no labels."""
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hl_hl03_tables import mean_std, pct, read  # noqa: E402
from hl_hl06_tables import per_seed_rows, train_rows, write  # noqa: E402

SEEDS = (1212, 1213, 1214, 1215, 1216)
CONF = {
    "hl08": {"dataset": "dblp", "prefix": "dblp_", "base": "T13_dblp_baseline.csv", "train": "T14_dblp_train.csv", "first_id": 93},
    "hl09": {"dataset": "Amazon_eletronics", "prefix": "elec_", "base": "T15_elec_baseline.csv", "train": "T16_elec_train.csv", "first_id": 103},
}
LABEL_FREE = ["encb_top10", "encb_strat5", "encb_ex20", "choi_far", "rand_strat5"]
ORACLE = ["g1_p04", "g1_p10", "near", "far"]


def paths(hl):
    c = CONF[hl]
    return c, Path(f"results/hl/{hl}"), Path(f"reports/{hl.upper()}_tables")


def base(hl):
    c, root, tables = paths(hl)
    mem = {}
    mem_file = root / "base" / "gpu_mem_samples.csv"
    rows = []
    for method, sub in (("naq", "naq/baseline"), ("ours", "ours/tau_0.3")):
        docs = {}
        for s in SEEDS:
            p = root / "base" / c["dataset"] / sub / f"seed_{s}" / "report.json"
            if p.exists():
                docs[s] = (json.loads(p.read_text()), p.parent)
        if not docs:
            rows.append({"arm": "NaQ" if method == "naq" else "B_Choi (τ=0.3)", "n_seeds": 0})
            continue
        one = [pct(d["evaluation"]["1"]["test_accuracy"]) for d, _ in docs.values()]
        five = [pct(d["evaluation"]["5"]["test_accuracy"]) for d, _ in docs.values()]
        v1 = [pct(d["evaluation"]["1"]["valid_accuracy"]) for d, _ in docs.values()]
        v5 = [pct(d["evaluation"]["5"]["valid_accuracy"]) for d, _ in docs.values()]
        wall = [float((p / "wall_seconds.txt").read_text()) for _, p in docs.values() if (p / "wall_seconds.txt").exists()]
        el = [d["elapsed_seconds"] for d, _ in docs.values()]
        qm = [d["diagnostics"]["query_gt_match"] for d, _ in docs.values()]
        m1, s1 = mean_std(one)
        m5, s5 = mean_std(five)
        rows.append({"arm": "NaQ" if method == "naq" else "B_Choi (τ=0.3)", "n_seeds": len(one), "test_1shot_mean": m1, "test_1shot_std": s1,
                     "test_5shot_mean": m5, "test_5shot_std": s5, "valid_1shot_mean": round(statistics.mean(v1), 2),
                     "valid_5shot_mean": round(statistics.mean(v5), 2), "query_gt_purity": round(statistics.mean(qm), 4),
                     "elapsed_seconds_mean": round(statistics.mean(el), 1), "wall_seconds_mean": round(statistics.mean(wall), 1) if wall else "",
                     "seeds": ",".join(str(s) for s in docs)})
    if mem_file.exists():
        peak = {}
        for r in read(mem_file):
            peak[r["gpu"]] = max(peak.get(r["gpu"], 0), int(r["used_mib"]))
        for r in rows:
            r["gpu_mem_used_peak_mib_by_gpu"] = ";".join(f"gpu{g}:{v}" for g, v in sorted(peak.items()))
    write(tables, c["base"], rows)
    per = []
    for method, sub in (("naq", "naq/baseline"), ("ours", "ours/tau_0.3")):
        for s in SEEDS:
            p = root / "base" / c["dataset"] / sub / f"seed_{s}"
            if (p / "report.json").exists():
                d = json.loads((p / "report.json").read_text())
                per.append({"arm": method, "seed": s, "test_1shot": pct(d["evaluation"]["1"]["test_accuracy"]), "test_5shot": pct(d["evaluation"]["5"]["test_accuracy"]),
                            "valid_1shot": pct(d["evaluation"]["1"]["valid_accuracy"]), "valid_5shot": pct(d["evaluation"]["5"]["valid_accuracy"]),
                            "query_gt_purity": d["diagnostics"]["query_gt_match"], "elapsed_seconds": d["elapsed_seconds"],
                            "wall_seconds": (p / "wall_seconds.txt").read_text().strip() if (p / "wall_seconds.txt").exists() else ""})
    write(tables, c["base"].replace(".csv", "_per_seed.csv"), per)


def bchoi(hl):
    c, _, tables = paths(hl)
    r = {x["arm"]: x for x in read(tables / c["base"])}
    return float(r["B_Choi (τ=0.3)"]["test_1shot_mean"]), float(r["B_Choi (τ=0.3)"]["test_5shot_mean"]), r


def train(hl):
    c, root, tables = paths(hl)
    b1, b5, base_rows = bchoi(hl)
    troot = root / "train" / c["dataset"]
    rows = train_rows(troot, LABEL_FREE + ORACLE, (b1, b5), LABEL_FREE)
    for name in ("NaQ", "B_Choi (τ=0.3)"):
        r = base_rows[name]
        rows.append({"arm": f"{name} ({c['base'].split('_')[0]})", "label_free": True, "n_seeds": r["n_seeds"], "test_1shot_mean": r["test_1shot_mean"],
                     "test_1shot_std": r["test_1shot_std"], "test_5shot_mean": r["test_5shot_mean"], "test_5shot_std": r["test_5shot_std"],
                     "delta_1shot_vs_bchoi": round(float(r["test_1shot_mean"]) - b1, 2), "delta_5shot_vs_bchoi": round(float(r["test_5shot_mean"]) - b5, 2),
                     "purity": r["query_gt_purity"]})
    write(tables, c["train"], rows)
    write(tables, c["train"].replace(".csv", "_per_seed.csv"), per_seed_rows(troot, LABEL_FREE + ORACLE))


def prereg(hl):
    c, root, tables = paths(hl)
    b1, _, base_rows = bchoi(hl)
    t = {r["arm"]: r for r in read(tables / c["train"])}
    p1 = {(r["judge"], r["rule"]): r for r in read(tables / f"{c['prefix']}P1_judge_rule.csv")}
    p2 = {r["judge"]: r for r in read(tables / f"{c['prefix']}P2_auroc.csv")}

    def acc(arm):
        r = t.get(arm)
        return float(r["test_1shot_mean"]) if r and r.get("test_1shot_mean") not in ("", None) else None

    def n(arm):
        r = t.get(arm)
        return f" (n={r['n_seeds']})" if r and str(r.get("n_seeds")) != "5" else ""

    def diff(a, b):
        x, y = acc(a), (b if isinstance(b, float) else acc(b))
        return "미실행" if x is None or y is None else f"{x:.2f} − {y:.2f} = {x - y:+.2f}" + (n(a) if isinstance(b, float) else n(a) + n(b))

    naq1 = float(base_rows["NaQ"]["test_1shot_mean"])
    i = c["first_id"]
    rows = [
        (f"HLP{i:03d}", "B_Choi − NaQ (1-shot) > 0", f"{b1:.2f} − {naq1:.2f} = {b1 - naq1:+.2f}"),
        (f"HLP{i + 1:03d}", "B4 기본 순도(`rand`) ≤ 0.18", f"{float(p1[('-', 'rand')]['purity_pooled']):.4f}"),
        (f"HLP{i + 2:03d}", "`enc_b` `auroc_all` ≥ 0.70", f"{float(p2['enc_b']['auroc_all']):.4f}"),
        (f"HLP{i + 3:03d}", "`g1_p10` − `g1_p04` ≥ +3.0 (1-shot)", diff("g1_p10", "g1_p04")),
        (f"HLP{i + 4:03d}", "`far` − `near` ≥ +2.0 (1-shot)", diff("far", "near")),
        (f"HLP{i + 5:03d}", "`encb_strat5` − `encb_top10` ≥ +1.5 (1-shot)", diff("encb_strat5", "encb_top10")),
        (f"HLP{i + 6:03d}", "`encb_strat5` − B_Choi ≥ +0.5", diff("encb_strat5", b1)),
        (f"HLP{i + 7:03d}", "`choi_far` − B_Choi ≥ 0", diff("choi_far", b1)),
        (f"HLP{i + 8:03d}", "`rand_strat5` ≤ B_Choi", diff("rand_strat5", b1)),
        (f"HLP{i + 9:03d}", "`enc_b` `strat5` 순도 ∈ [0.25, 0.45]", f"{float(p1[('enc_b', 'strat5')]['purity_pooled']):.4f}"),
    ]
    write(tables, f"prereg_{hl.upper()}.csv", [dict(id=a, prediction=b, computed=cc) for a, b, cc in rows])


if __name__ == "__main__":
    {"base": base, "train": train, "prereg": prereg}[sys.argv[1]](sys.argv[2])
