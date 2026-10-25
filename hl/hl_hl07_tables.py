"""hl_hl07_tables: HL07 T11 (purity x distance oracle grid), T12 (label-free variants) and pre-registration values (HLP080-092). Reads no labels."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hl_hl03_tables import read  # noqa: E402
from hl_hl06_tables import BCHOI, ROOT as HL06_ROOT, per_seed_rows, train_rows, write  # noqa: E402

ROOT = Path("results/hl/hl07/train/Amazon_clothing")
TABLES = Path("reports/HL07_tables")
GRID = ["o_grid_t1_p03", "o_grid_t1_p05", "o_grid_t1_p07", "o_grid_t1_p10", "o_grid_strat5_p03", "o_grid_strat5_p05", "o_grid_strat5_p07",
        "o_grid_strat5_p10", "o_grid_far50_p07", "o_grid_far50_p10"]
VARIANTS = ["rand_strat5", "choi_far", "choi_strat", "encb_strat5_wide", "encb_strat5_b45", "encb_strat5_r2", "encb_strat5_drop2"]
HL06_T10 = Path("reports/HL06_tables/T10_train.csv")


def t11():
    rows = []
    for r in train_rows(ROOT, GRID, BCHOI, []):
        _, _, region, pp = r["arm"].split("_")
        rows.append({"region": region, "p": int(pp[1:]) / 10, **r})
    for r in train_rows(HL06_ROOT, ["o_far50_p03", "o_far50_p05", "o_topsame"], BCHOI, []):
        region, p = ("far50", int(r["arm"][-2:]) / 10) if r["arm"].startswith("o_far50") else ("t1-top (s2 top same)", 1.0)
        rows.append({"region": region, "p": p, **r, "arm": f"HL06 {r['arm']}"})
    t10 = {r["arm"]: r for r in read(HL06_T10)}
    for name in ("G1 p=0.4 (HL03 T6)", "G1 p=1.0 (HL03 T6)", "HL05 near (T8)", "HL05 far (T8)"):
        r = dict(t10[name])
        rows.append({"region": "reference", "p": "", **r})
    rows.sort(key=lambda r: ({"t1": 0, "strat5": 1, "far50": 2}.get(r["region"], 3), r["p"] if r["p"] != "" else 9))
    write(TABLES, "T11_grid.csv", rows)
    write(TABLES, "T11b_grid_per_seed.csv", per_seed_rows(ROOT, GRID))


def t12():
    rows = train_rows(ROOT, VARIANTS, BCHOI, VARIANTS)
    t10 = {r["arm"]: r for r in read(HL06_T10)}
    for name in ("encb_strat5", "encb_top10", "HL04 rand10 (T7)", "B_Choi uniform (HL03 T0)"):
        r = dict(t10[name])
        r["arm"] = name if name.startswith(("HL04", "B_Choi")) else f"HL06 {name}"
        rows.append(r)
    write(TABLES, "T12_variants.csv", rows)
    write(TABLES, "T12b_variants_per_seed.csv", per_seed_rows(ROOT, VARIANTS))


def prereg():
    t11 = {r["arm"]: r for r in read(TABLES / "T11_grid.csv")}
    t12 = {r["arm"]: r for r in read(TABLES / "T12_variants.csv")}
    allr = {**t11, **t12}

    def acc(arm):
        r = allr.get(arm)
        return float(r["test_1shot_mean"]) if r and r.get("test_1shot_mean") not in ("", None) else None

    def n(arm):
        r = allr.get(arm)
        return f" (n={r['n_seeds']})" if r and str(r.get("n_seeds")) != "5" else ""

    def val(arm):
        return "미실행" if acc(arm) is None else f"{acc(arm):.2f}{n(arm)}"

    def diff(a, b, bname=None):
        x, y = acc(a), (b if isinstance(b, float) else acc(b))
        return "미실행" if x is None or y is None else f"{x:.2f} − {y:.2f} = {x - y:+.2f}"

    mono = []
    for region, arms in (("t1", [f"o_grid_t1_p{p}" for p in ("03", "05", "07", "10")]), ("strat5", [f"o_grid_strat5_p{p}" for p in ("03", "05", "07", "10")]),
                         ("far50", ["HL06 o_far50_p03", "HL06 o_far50_p05", "o_grid_far50_p07", "o_grid_far50_p10"])):
        v = [acc(a) for a in arms]
        mono.append(f"{region} " + "/".join("미실행" if x is None else f"{x:.2f}" for x in v)
                    + ("" if None in v else f" {all(b > a for a, b in zip(v, v[1:]))}"))
    d = t12.get("encb_strat5_drop2", {})
    rows = [
        ("HLP080", "`o_grid_t1_p10` 1-shot ≤ 84.5", val("o_grid_t1_p10")),
        ("HLP081", "`o_grid_strat5_p10` ≥ 88.5", val("o_grid_strat5_p10")),
        ("HLP082", "`o_grid_far50_p10` ≥ 89.0", val("o_grid_far50_p10")),
        ("HLP083", "p=0.5에서 `strat5` − `t1` ≥ +3.0", diff("o_grid_strat5_p05", "o_grid_t1_p05")),
        ("HLP084", "`o_grid_strat5_p03` ≥ B_Choi − 0.5 (= 82.21)", val("o_grid_strat5_p03")),
        ("HLP085", "세 region 모두 1-shot이 p에 대해 단조 증가", "; ".join(mono)),
        ("HLP086", "`rand_strat5` ≤ HL04 `rand10` + 1.0 (= 81.34)", val("rand_strat5")),
        ("HLP087", "HL06 `encb_strat5` − `rand_strat5` ≥ +2.0", diff("HL06 encb_strat5", "rand_strat5")),
        ("HLP088", "`choi_far` − B_Choi ≥ +0.5", diff("choi_far", BCHOI[0])),
        ("HLP089", "`choi_strat` − B_Choi ≥ +0.5", diff("choi_strat", BCHOI[0])),
        ("HLP090", "|`encb_strat5_wide` − HL06 `encb_strat5`| ≤ 1.0", diff("encb_strat5_wide", "HL06 encb_strat5")),
        ("HLP091", "`encb_strat5_r2` ≥ HL06 `encb_strat5`", diff("encb_strat5_r2", "HL06 encb_strat5")),
        ("HLP092", "`encb_strat5_drop2` ≥ HL06 `encb_strat5` − 0.3, 제외된 query의 FP 비율 ≥ 0.7",
         diff("encb_strat5_drop2", "HL06 encb_strat5") + (f"; 제외 FP 비율 {float(d['dropped_fp_frac']):.4f}" if d.get("dropped_fp_frac") not in ("", None) else "")),
    ]
    write(TABLES, "prereg_HL07.csv", [dict(id=a, prediction=b, computed=c) for a, b, c in rows])


if __name__ == "__main__":
    {"t11": t11, "t12": t12, "prereg": prereg}[sys.argv[1]]()
