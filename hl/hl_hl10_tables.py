"""hl_hl10_tables: HL10 T17 (Clothing filler variants) and pre-registration values (HLP113-118). Reads no labels."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hl_hl03_tables import read  # noqa: E402
from hl_hl06_tables import BCHOI, per_seed_rows, train_rows, write  # noqa: E402

ROOT = Path("results/hl/hl10/train/Amazon_clothing")
TABLES = Path("reports/HL10_tables")
ARMS = ["encb_strat10", "encb_strat3", "encb_ex10", "encb_ex40", "nbr_strat5", "xidf_strat5", "clu_strat5", "knnppr_far50"]
HL06_T10 = Path("reports/HL06_tables/T10_train.csv")


def t17():
    rows = train_rows(ROOT, ARMS, BCHOI, ARMS)
    t10 = {r["arm"]: r for r in read(HL06_T10)}
    for name in ("encb_strat5", "encb_ex20", "encb_far50", "encb_top10", "B_Choi uniform (HL03 T0)"):
        r = dict(t10[name])
        r["arm"] = name if name.startswith("B_Choi") else f"HL06 {name}"
        rows.append(r)
    write(TABLES, "T17_fillers.csv", rows)
    write(TABLES, "T17b_fillers_per_seed.csv", per_seed_rows(ROOT, ARMS))


def prereg():
    t = {r["arm"]: r for r in read(TABLES / "T17_fillers.csv")}

    def acc(arm):
        r = t.get(arm)
        return float(r["test_1shot_mean"]) if r and r.get("test_1shot_mean") not in ("", None) else None

    def n(arm):
        r = t.get(arm)
        return f" (n={r['n_seeds']})" if r and str(r.get("n_seeds")) not in ("5", "") else ""

    def diff(a, b):
        x, y = acc(a), acc(b)
        return "미실행" if x is None or y is None else f"{x:.2f} − {y:.2f} = {x - y:+.2f}{n(a)}"

    rows = [
        ("HLP113", "`encb_strat10`·`encb_strat3`은 HL06 `encb_strat5`와 ±1.0 안", diff("encb_strat10", "HL06 encb_strat5") + "; " + diff("encb_strat3", "HL06 encb_strat5")),
        ("HLP114", "`encb_ex10` < HL06 `encb_ex20`", diff("encb_ex10", "HL06 encb_ex20")),
        ("HLP115", "`encb_ex40` ≤ HL06 `encb_ex20` + 0.5", diff("encb_ex40", "HL06 encb_ex20")),
        ("HLP116", "`nbr_strat5` ≥ HL06 `encb_strat5` − 0.5", diff("nbr_strat5", "HL06 encb_strat5")),
        ("HLP117", "`xidf_strat5` ≤ HL06 `encb_strat5` − 1.0", diff("xidf_strat5", "HL06 encb_strat5")),
        ("HLP118", "`clu_strat5`는 HL06 `encb_strat5`와 ±1.5 안", diff("clu_strat5", "HL06 encb_strat5")),
    ]
    write(TABLES, "prereg_HL10.csv", [dict(id=a, prediction=b, computed=c) for a, b, c in rows])


if __name__ == "__main__":
    {"t17": t17, "prereg": prereg}[sys.argv[1]]()
