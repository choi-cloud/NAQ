"""HW09-c report tables -> results/hw_iislab4/HW09_tables.json.

    PYTHONPATH=. python -m hw.tables_hw09

Accuracies in % (x100); mean ± sd with ddof=0 over seeds; paired Δ = same seed, same metric.
Sources: dblp / Amazon_eletronics = h9_<arm> metrics.json (training runs of HW09);
Cora_full / Amazon_clothing = HW08 l2_<arm> metrics.json for 5-shot keys and ckpt_eval/l2_<arm>/... for 1-shot keys.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path("results/hw_iislab4")
SHORT = ("naq", "rand_m20", "rand_m50", "clus_m20")
NEW = ("dblp", "Amazon_eletronics")
OLD = ("Cora_full", "Amazon_clothing")
ALL = OLD + NEW
SEEDS = tuple(range(1212, 1222))
K5 = ("test_acc", "test_acc_l2_at_bestraw", "test_acc_l2", "test_acc_raw_at_bestl2")
K1 = tuple(k + "_1shot" for k in K5)
PAIRS = [("rand_m20", "naq"), ("rand_m50", "naq"), ("clus_m20", "naq"), ("clus_m20", "rand_m20"), ("rand_m50", "rand_m20")]
GEOM_KEYS = ("norm_mean", "norm_std", "uniformity", "eff_rank", "corr_offdiag_mean")
NORM_KEYS = ("zero_rows_all", "zero_rows_test", "test_norm_mean", "test_norm_std", "proto_norm_cv")
COLLAPSE = 25.0


def arm_name(short, ds):
    return ("h9_" if ds in NEW else "l2_") + short


def metrics(short, ds, seed):
    p = ROOT / arm_name(short, ds) / ds / f"seed{seed}" / "metrics.json"
    return json.loads(p.read_text()) if p.is_file() else None


def value(short, ds, seed, key):
    if ds in OLD and key.endswith("_1shot"):
        p = ROOT / "ckpt_eval" / arm_name(short, ds) / ds / f"seed{seed}.json"
        return 100 * json.loads(p.read_text())[key] if p.is_file() else None
    m = metrics(short, ds, seed)
    return 100 * m[key] if m and m.get("status") == "completed" and key in m else None


def ms(vals):
    x = np.asarray(vals, dtype=np.float64)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=0)), "n": int(x.size)} if x.size else None


def paired(a, b):
    keys = sorted(set(a) & set(b))
    d = np.asarray([a[k] - b[k] for k in keys])
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=0)), "pos": int((d > 0).sum()), "n": int(d.size), "per_seed": {int(k): float(a[k] - b[k]) for k in keys}}


def main():
    out = {}
    vals = {}
    for s in SHORT:
        for ds in ALL:
            for key in K5 + K1:
                vals[(s, ds, key)] = {seed: v for seed in SEEDS if (v := value(s, ds, seed, key)) is not None}
    status = {}
    for s in SHORT:
        for ds in NEW:
            st = [(metrics(s, ds, seed) or {}).get("status", "missing") for seed in SEEDS]
            status[f"{arm_name(s, ds)}/{ds}"] = {x: st.count(x) for x in set(st)}
    out["status"] = status
    out["accuracy"] = {f"{s}/{ds}/{key}": ms(list(vals[(s, ds, key)].values())) for s in SHORT for ds in ALL for key in K5 + K1}
    out["paired"] = {f"{a} - {b}/{ds}/{key}": paired(vals[(a, ds, key)], vals[(b, ds, key)]) for key in K5 + K1 for ds in ALL for a, b in PAIRS}
    tf = {}
    for ds in ALL:
        for shot, suffix in ((5, ""), (1, "_1shot")):
            t = json.loads((ROOT / "trainfree" / f"HW03a_trainfree_{ds}{suffix}.json").read_text())
            tf[f"{ds}/{shot}"] = {k: 100 * v["test"]["accuracy_mean"] for k, v in t["variants"].items()}
    out["trainfree_test"] = tf
    ep, geo, g7 = {}, {}, {}
    for s in SHORT:
        for ds in NEW:
            g = [m for seed in SEEDS if (m := metrics(s, ds, seed)) and m.get("status") == "completed"]
            if not g:
                continue
            raw_ep, l2_ep = [m["best_epoch_500"] for m in g], [m["best_epoch_500_l2"] for m in g]
            ep[f"{s}/{ds}"] = {"raw": sorted(raw_ep), "l2": sorted(l2_ep), "raw_mean": float(np.mean(raw_ep)), "l2_mean": float(np.mean(l2_ep)),
                               "same_epoch_l2": int(sum(m["same_epoch_l2"] for m in g)), "n": len(g),
                               "raw_at_4000": int(sum(e == 4000 for e in raw_ep)), "l2_at_4000": int(sum(e == 4000 for e in l2_ep))}
            geo[f"{s}/{ds}"] = {block: {k: float(np.mean([m[block][k] for m in g])) for k in keys}
                                for block, keys in (("geom", GEOM_KEYS), ("geom_l2", GEOM_KEYS), ("norm_diag", NORM_KEYS), ("norm_diag_l2", NORM_KEYS))}
            g7[f"{s}/{ds}"] = {"completed": len(g), "nan": sum(1 for seed in SEEDS if (metrics(s, ds, seed) or {}).get("status") == "nan"),
                               "collapsed_raw": sum(100 * m["best"]["validation"]["accuracy_mean"] <= COLLAPSE for m in g),
                               "collapsed_l2": sum(100 * m["best_l2"]["validation"]["accuracy_mean"] <= COLLAPSE for m in g),
                               "zero_rows_test_pos_runs": sum(m["norm_diag"]["zero_rows_test"] > 0 or m["norm_diag_l2"]["zero_rows_test"] > 0 for m in g),
                               "zero_rows_all_pos_runs": sum(m["norm_diag"]["zero_rows_all"] > 0 or m["norm_diag_l2"]["zero_rows_all"] > 0 for m in g)}
    out["best_epoch"], out["geometry_norm"], out["G7"] = ep, geo, g7
    done = [m for s in SHORT for ds in NEW for seed in SEEDS if (m := metrics(s, ds, seed)) and m.get("status") == "completed"]
    out["ckpt"] = {"files": sum(1 + (m["ckpt"]["best_l2"] is not None) for m in done), "bytes": int(sum(m["ckpt"]["bytes"] for m in done))}
    out["elapsed_seconds"] = {"sum": float(sum(m["elapsed_seconds"] for m in done)), "min": float(min(m["elapsed_seconds"] for m in done)), "max": float(max(m["elapsed_seconds"] for m in done))} if done else None
    P, A = out["paired"], out["accuracy"]
    hwp = {
        "HWP088": {ds: {k: P[f"rand_m20 - naq/{ds}/test_acc_l2"][k] for k in ("mean", "pos", "n")} for ds in NEW},
        "HWP089": {ds: {k: P[f"clus_m20 - naq/{ds}/test_acc_l2"][k] for k in ("mean", "pos", "n")} for ds in NEW},
        "HWP090": {ds: P[f"clus_m20 - rand_m20/{ds}/test_acc_l2"]["mean"] for ds in NEW},
        "HWP091": {ds: {"naq_l2": A[f"naq/{ds}/test_acc_l2"]["mean"], "a2x_l2": tf[f"{ds}/5"]["A2X_l2"]} for ds in NEW},
        "HWP092": {ds: {"best": max(((s, A[f"{s}/{ds}/test_acc_l2"]["mean"]) for s in SHORT), key=lambda t: t[1]), "a2x_l2": tf[f"{ds}/5"]["A2X_l2"]} for ds in NEW},
        "HWP093": {ds: {k: P[f"clus_m20 - naq/{ds}/test_acc_l2_1shot"][k] for k in ("mean", "pos", "n")} for ds in ALL},
        "HWP094": {ds: {"l2_1s": P[f"clus_m20 - naq/{ds}/test_acc_l2_1shot"]["mean"], "l2": P[f"clus_m20 - naq/{ds}/test_acc_l2"]["mean"]} for ds in ALL},
        "HWP095": {ds: P[f"rand_m50 - rand_m20/{ds}/test_acc_l2"]["mean"] for ds in NEW},
    }
    out["HWP"] = hwp
    (ROOT / "HW09_tables.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "ckpt": out["ckpt"], "elapsed": out["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
