"""HW08-c report tables -> results/hw_iislab4/HW08_tables.json (+ printed markdown).

    PYTHONPATH=. python -m hw.tables_hw08

Accuracies in % (x100). mean ± std use ddof=0 over the seeds. Paired Δ: same seed, same metric.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path("results/hw_iislab4")
ARMS = ("l2_naq", "l2_rand_m20", "l2_rand_m50", "l2_rand_m200", "l2_clus_m20")
E4K = {a: "e4k_" + a[3:] for a in ARMS}
DATASETS = ("Cora_full", "Amazon_clothing")
SEEDS = tuple(range(1212, 1222))
METRICS = ("test_acc", "test_acc_l2_at_bestraw", "test_acc_l2", "test_acc_raw_at_bestl2")
PAIRS = [("l2_rand_m20", "l2_naq"), ("l2_rand_m50", "l2_naq"), ("l2_rand_m200", "l2_naq"), ("l2_clus_m20", "l2_naq"),
         ("l2_clus_m20", "l2_rand_m20"), ("l2_rand_m200", "l2_rand_m20")]
GEOM_KEYS = ("norm_mean", "norm_std", "uniformity", "eff_rank", "corr_offdiag_mean")
NORM_KEYS = ("zero_rows_all", "zero_rows_test", "test_norm_mean", "test_norm_std", "proto_norm_cv")
COLLAPSE = 25.0


def load(arm, ds, seed):
    path = ROOT / arm / ds / f"seed{seed}" / "metrics.json"
    return json.loads(path.read_text()) if path.is_file() else None


def ms(values):
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=0)), "n": int(x.size)} if x.size else None


def paired(a, b):
    keys = sorted(set(a) & set(b))
    d = np.asarray([a[k] - b[k] for k in keys], dtype=np.float64)
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=0)), "pos": int((d > 0).sum()), "n": int(d.size),
            "abs_mean": float(np.abs(d).mean()), "abs_max": float(np.abs(d).max()), "per_seed": {int(k): float(v) for k, v in zip(keys, d)}}


def main():
    runs = {(a, ds, s): load(a, ds, s) for a in ARMS for ds in DATASETS for s in SEEDS}
    done = {k: v for k, v in runs.items() if v is not None and v.get("status") == "completed"}
    out = {"completed": len(done), "total": len(runs), "status_counts": {}}
    for v in runs.values():
        st = "missing" if v is None else v.get("status")
        out["status_counts"][st] = out["status_counts"].get(st, 0) + 1

    val = {}  # (arm, ds, metric) -> {seed: value %}
    for (a, ds, s), m in done.items():
        for key in METRICS:
            val.setdefault((a, ds, key), {})[s] = 100 * (m["best"]["test"]["accuracy_mean"] if key == "test_acc" else m[key])

    acc = {}
    for a in ARMS:
        for ds in DATASETS:
            group = [runs[(a, ds, s)] for s in SEEDS]
            row = {key: ms(list(val.get((a, ds, key), {}).values())) for key in METRICS}
            row["completed"] = sum(1 for g in group if g is not None and g.get("status") == "completed")
            row["nan"] = sum(1 for g in group if g is not None and g.get("status") == "nan")
            row["collapsed_raw"] = sum(1 for g in group if g and g.get("status") == "completed" and 100 * g["best"]["validation"]["accuracy_mean"] <= COLLAPSE)
            row["collapsed_l2"] = sum(1 for g in group if g and g.get("status") == "completed" and 100 * g["best_l2"]["validation"]["accuracy_mean"] <= COLLAPSE)
            row["zero_rows_test_pos_runs"] = sum(1 for g in group if g and g.get("status") == "completed"
                                                 and (g["norm_diag"]["zero_rows_test"] > 0 or g["norm_diag_l2"]["zero_rows_test"] > 0))
            acc[f"{a}/{ds}"] = row
    out["accuracy"] = acc

    tf = {}
    for ds in DATASETS:
        t = json.loads((ROOT / "trainfree" / f"HW03a_trainfree_{ds}.json").read_text())
        tf[ds] = {k: 100 * v["test"]["accuracy_mean"] for k, v in t["variants"].items()}
    out["trainfree_test"] = tf

    out["paired"] = {f"{a} - {b}/{ds}/{key}": paired(val[(a, ds, key)], val[(b, ds, key)])
                     for key in METRICS for ds in DATASETS for a, b in PAIRS}

    evaldiff = {}
    for a in ARMS:
        for ds in DATASETS:
            evaldiff[f"{a}/{ds}"] = {
                "l2_at_bestraw - raw (same checkpoint)": paired(val[(a, ds, "test_acc_l2_at_bestraw")], val[(a, ds, "test_acc")]),
                "l2 - l2_at_bestraw (selection)": paired(val[(a, ds, "test_acc_l2")], val[(a, ds, "test_acc_l2_at_bestraw")]),
            }
    out["eval_diff"] = evaldiff

    rep = {}
    for a in ARMS:
        for ds in DATASETS:
            ref = {s: 100 * m["best"]["test"]["accuracy_mean"] for s in SEEDS if (m := load(E4K[a], ds, s)) and m.get("status") == "completed"}
            rep[f"{a} - {E4K[a]}/{ds}"] = paired(val[(a, ds, "test_acc")], ref)
    out["replication_raw"] = rep

    ep = {}
    for a in ARMS:
        for ds in DATASETS:
            g = [done[(a, ds, s)] for s in SEEDS if (a, ds, s) in done]
            raw_ep = [m["best_epoch_500"] for m in g]
            l2_ep = [m["best_epoch_500_l2"] for m in g]
            ep[f"{a}/{ds}"] = {"raw": sorted(raw_ep), "l2": sorted(l2_ep), "raw_mean": float(np.mean(raw_ep)), "l2_mean": float(np.mean(l2_ep)),
                               "same_epoch_l2": int(sum(m["same_epoch_l2"] for m in g)), "n": len(g),
                               "raw_at_4000": int(sum(e == 4000 for e in raw_ep)), "l2_at_4000": int(sum(e == 4000 for e in l2_ep))}
    out["best_epoch"] = ep

    geo = {}
    for a in ARMS:
        for ds in DATASETS:
            g = [done[(a, ds, s)] for s in SEEDS if (a, ds, s) in done]
            row = {}
            for block, keys in (("geom", GEOM_KEYS), ("geom_l2", GEOM_KEYS), ("norm_diag", NORM_KEYS), ("norm_diag_l2", NORM_KEYS)):
                row[block] = {k: float(np.mean([m[block][k] for m in g])) for k in keys}
            row["proto_norm_cv_per_seed"] = {int(m["seed"]): m["norm_diag"]["proto_norm_cv"] for m in g}
            geo[f"{a}/{ds}"] = row
    out["geometry_norm"] = geo

    out["ckpt"] = {"files": sum(1 + (m["ckpt"]["best_l2"] is not None) for m in done.values()),
                   "best_l2_files": sum(m["ckpt"]["best_l2"] is not None for m in done.values()),
                   "bytes": int(sum(m["ckpt"]["bytes"] for m in done.values()))}
    out["elapsed_seconds"] = {"sum": float(sum(m["elapsed_seconds"] for m in done.values())),
                              "min": float(min(m["elapsed_seconds"] for m in done.values())),
                              "max": float(max(m["elapsed_seconds"] for m in done.values()))}

    # HWP079-087 values (judgement by a human)
    P = out["paired"]
    hwp = {}
    hwp["HWP079"] = {k: {"abs_mean": v["abs_mean"]} for k, v in rep.items()}
    hwp["HWP080"] = {ds: evaldiff[f"l2_naq/{ds}"]["l2_at_bestraw - raw (same checkpoint)"]["mean"] for ds in DATASETS}
    hwp["HWP081"] = {ds: {k: P[f"l2_rand_m20 - l2_naq/{ds}/test_acc_l2"][k] for k in ("mean", "pos", "n")} for ds in DATASETS}
    hwp["HWP082"] = {ds: {"l2": P[f"l2_rand_m20 - l2_naq/{ds}/test_acc_l2"]["mean"], "raw": P[f"l2_rand_m20 - l2_naq/{ds}/test_acc"]["mean"]} for ds in DATASETS}
    hwp["HWP083"] = P["l2_rand_m200 - l2_rand_m20/Cora_full/test_acc_l2"]["mean"]
    hwp["HWP084"] = P["l2_clus_m20 - l2_rand_m20/Cora_full/test_acc_l2"]["mean"]
    hwp["HWP085"] = {ds: max(((a, acc[f"{a}/{ds}"]["test_acc_l2"]["mean"]) for a in ARMS), key=lambda t: t[1]) for ds in DATASETS}
    hwp["HWP086"] = {k: abs(v["l2 - l2_at_bestraw (selection)"]["mean"]) for k, v in evaldiff.items()}
    hwp["HWP086_note"] = "value = |mean over seeds of (test_acc_l2 - test_acc_l2_at_bestraw)|; per-seed |Δ| mean in eval_diff[...].abs_mean"
    hwp["HWP087"] = {ds: {"l2_rand_m20": geo[f"l2_rand_m20/{ds}"]["norm_diag"]["proto_norm_cv"], "l2_naq": geo[f"l2_naq/{ds}"]["norm_diag"]["proto_norm_cv"]} for ds in DATASETS}
    out["HWP"] = hwp

    (ROOT / "HW08_tables.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("completed", "total", "status_counts", "ckpt", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
