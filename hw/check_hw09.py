"""HW09-a §1.7 checks -> results/hw_iislab4/HW09_checks.json.

    PYTHONPATH=. python -m hw.check_hw09 --start <work-start HEAD>

G-data  results/hw_iislab4/HW09_data.json (hw.data_hw09): files present, loader read, caches present,
        1-shot test classes == 5-shot test classes, 500 test tasks, valid500 500 tasks
G-cfg   merged h9_* config minus {test_shots, arm, note} == merged l2_* config minus {arm, note}; test_shots == [5, 1]
G-ckpt  ckpt_eval/<l2_*>/<ds>/seed<S>.json 5-shot values vs HW08 metrics.json, 4 keys x 80 runs, |Δ| <= 1e-4
G-tf    hw.trainfree default (5-shot) output: Â²X L2 80.12 / 92.715, raw 67.57 / 83.025, |Δ| <= 0.01 (%)
regress naq_base, nbr_raw seed 1212 under _regress/HW09 vs results/hw_iislab4, |Δ| <= 0.30 (%); key set == _regress/HW08
invariant git diff <start> for losses/negatives/neighbors/episodes/evaluate empty; deleted lines of run.py / trainfree.py
smoke   /tmp/hw_smoke/hw09_smoke (episodes 8): 1-shot keys, ckpt files; per-run time and GPU memory;
        /tmp/hw_smoke/hw09_timing_{a,b} (episodes 320, eval_every 160 / 32) -> 4000-episode estimate and the 150-minute rule
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

ROOT = Path("results/hw_iislab4")
HW08_DATASETS = ("Cora_full", "Amazon_clothing")
NEW_DATASETS = ("dblp", "Amazon_eletronics")
SEEDS = tuple(range(1212, 1222))
PAIRS = {"h9_naq": "l2_naq", "h9_rand_m20": "l2_rand_m20", "h9_rand_m50": "l2_rand_m50", "h9_clus_m20": "l2_clus_m20"}
KEYS = ("test_acc", "test_acc_l2_at_bestraw", "test_acc_l2", "test_acc_raw_at_bestl2")
KEYS_1 = tuple(k + "_1shot" for k in KEYS)
TF_EXPECTED = {"Cora_full": {"A2X_l2": 80.12, "A2X_raw": 67.57}, "Amazon_clothing": {"A2X_l2": 92.715, "A2X_raw": 83.025}}
SMOKE = Path("/tmp/hw_smoke/hw09_smoke")
FORCED = Path("/tmp/hw_smoke/hw09_forced")
TIMING_A = Path("/tmp/hw_smoke/hw09_timing_a")
TIMING_B = Path("/tmp/hw_smoke/hw09_timing_b")
TIME_LIMIT_MIN = 150.0


def g_data():
    d = json.loads((ROOT / "HW09_data.json").read_text())
    out, ok = {}, True
    for ds, row in d.items():
        c = row["caches"]
        checks = {
            "files_present": all(v is not None for v in row["files"].values()) and len(row["files"]) > 0,
            "loader_read": row["nodes"] > 0 and row["feature_dim"] > 0,
            "cache_5shot": c["5way_5shot"] is not None and c["5way_5shot"]["test_tasks"] == 500 and c["5way_5shot"]["valid_tasks"] == 50,
            "cache_1shot": c["5way_1shot"] is not None and c["5way_1shot"]["test_tasks"] == 500 and c["5way_1shot"]["k_shot"] == 1,
            "test_classes_1shot_eq_5shot": bool(c.get("test_classes_1shot_eq_5shot")),
            "valid500": c["valid500"] is not None and c["valid500"]["tasks"] == 500,
        }
        out[ds] = {**checks, "pass": all(checks.values())}
        ok &= out[ds]["pass"]
    return ok, out


def g_cfg():
    from hw.config import load_config
    out, ok = {}, True
    for new, old in PAIRS.items():
        a = {k: v for k, v in load_config(new).items() if k not in ("test_shots", "arm", "note")}
        b = {k: v for k, v in load_config(old).items() if k not in ("arm", "note")}
        diff = sorted(k for k in set(a) | set(b) if a.get(k, "<missing>") != b.get(k, "<missing>"))
        same = not diff and load_config(new).get("test_shots") == [5, 1]
        out[new] = {"ref": old, "equal": same, "diff_keys": diff}
        ok &= same
    return ok, out


def g_ckpt():
    out, ok, worst, n = {}, True, 0.0, 0
    for arm in PAIRS.values():
        for ds in HW08_DATASETS:
            for s in SEEDS:
                p = ROOT / "ckpt_eval" / arm / ds / f"seed{s}.json"
                m = json.loads((ROOT / arm / ds / f"seed{s}" / "metrics.json").read_text())
                if not p.is_file():
                    out[f"{arm}/{ds}/{s}"] = "missing"
                    ok = False
                    continue
                e = json.loads(p.read_text())
                diffs = {k: abs(e[k] - m[k]) for k in KEYS}
                worst = max(worst, max(diffs.values()))
                n += 1
                if max(diffs.values()) > 1e-4 or any(k not in e for k in KEYS_1):
                    out[f"{arm}/{ds}/{s}"] = diffs
                    ok = False
    return ok, {"runs_checked": n, "max_abs_diff": worst, "failures": out}


def g_tf():
    out, ok = {}, True
    for ds in HW08_DATASETS:
        tf = json.loads((ROOT / "trainfree" / f"HW03a_trainfree_{ds}.json").read_text())
        row = {}
        for key, expected in TF_EXPECTED[ds].items():
            value = 100 * tf["variants"][key]["test"]["accuracy_mean"]
            row[key] = {"value": value, "expected": expected, "abs_diff": abs(value - expected), "pass": abs(value - expected) <= 0.01}
            ok &= row[key]["pass"]
        out[ds] = row
    return ok, out


def regress():
    out, ok = {}, True
    for arm in ("naq_base", "nbr_raw"):
        for ds in HW08_DATASETS:
            new = json.loads((ROOT / "_regress" / "HW09" / arm / ds / "seed1212" / "metrics.json").read_text())
            old = json.loads((ROOT / arm / ds / "seed1212" / "metrics.json").read_text())
            prev = json.loads((ROOT / "_regress" / "HW08" / arm / ds / "seed1212" / "metrics.json").read_text())
            a, b = 100 * new["best"]["test"]["accuracy_mean"], 100 * old["best"]["test"]["accuracy_mean"]
            keys_same = set(new) == set(prev) and set(new["epochs"][0]) == set(prev["epochs"][0])
            out[f"{arm}/{ds}"] = {"new": a, "existing": b, "delta": a - b, "status": new["status"], "best_epoch_new": new["best"]["epoch"], "best_epoch_existing": old["best"]["epoch"],
                                  "metrics_keys_same_as_HW08_regress": keys_same, "pass": new["status"] == "completed" and abs(a - b) <= 0.30 and keys_same}
            ok &= out[f"{arm}/{ds}"]["pass"]
    return ok, out


def git(*cmd):
    return subprocess.run(["git", *cmd], capture_output=True, text=True, check=True).stdout


def invariant(start):
    frozen = ["hw/losses.py", "hw/negatives.py", "hw/neighbors.py", "hw/episodes.py", "hw/evaluate.py"]
    diff = git("diff", start, "--", *frozen)
    deleted = {}
    for path in ("hw/run.py", "hw/trainfree.py", "hw/valid500.py"):
        lines = git("diff", start, "--", path).splitlines()
        deleted[path] = [l[1:] for l in lines if l.startswith("-") and not l.startswith("---")]
    ok = diff == ""
    return ok, {"start_head": start, "frozen_files": frozen, "frozen_diff_empty": ok, "deleted_lines": deleted, "deleted_line_counts": {k: len(v) for k, v in deleted.items()}}


def smoke():
    out, ok = {}, True
    for arm in ("h9_naq", "h9_clus_m20"):
        for ds in NEW_DATASETS:
            m = json.loads((SMOKE / arm / ds / "seed1212" / "metrics.json").read_text())
            t = json.loads((SMOKE / f"timing_{arm}_{ds}.json").read_text())
            missing = [k for k in KEYS + KEYS_1 if k not in m]
            ck = m.get("ckpt", {})
            files = {p: Path(p).stat().st_size for p in (ck.get("best_raw"), ck.get("best_l2")) if p}
            row = {"status": m["status"], "missing": missing, "same_epoch_l2": m.get("same_epoch_l2"), "files": files, "bytes_match": sum(files.values()) == ck.get("bytes"),
                   "test_tasks_1shot": m.get("test_tasks_1shot"), "elapsed_seconds": m.get("elapsed_seconds"), "max_mem_alloc_MiB": t["max_mem_alloc_MiB"], "max_mem_reserved_MiB": t["max_mem_reserved_MiB"],
                   "1shot_values": {k: m.get(k) for k in KEYS_1}}
            row["pass"] = m["status"] == "completed" and not missing and row["bytes_match"] and m.get("test_tasks_1shot") == 500
            out[f"{arm}/{ds}"] = row
            ok &= row["pass"]
    # forced different-epoch run (l2 validation of the first evaluation reported as 1.0 in-process; smoke only),
    # re-evaluated from its saved checkpoints with hw.eval_ckpt
    m = json.loads((FORCED / "h9_naq" / "dblp" / "seed1212" / "metrics.json").read_text())
    e = json.loads((FORCED / "ckpt_eval" / "h9_naq" / "dblp" / "seed1212.json").read_text())
    unforced = json.loads((SMOKE / "h9_naq" / "dblp" / "seed1212" / "metrics.json").read_text())
    diffs = {k: abs(e[k] - m[k]) for k in KEYS + KEYS_1}
    row = {"status": m["status"], "best_epoch_500": m["best_epoch_500"], "best_epoch_500_l2": m["best_epoch_500_l2"], "same_epoch_l2": m["same_epoch_l2"], "ckpt": m["ckpt"],
           "reload_abs_diff": diffs, "raw_equal_unforced": m["test_acc"] == unforced["test_acc"] and m["test_acc_1shot"] == unforced["test_acc_1shot"]}
    row["pass"] = m["status"] == "completed" and not m["same_epoch_l2"] and m["ckpt"]["best_l2"] is not None and max(diffs.values()) <= 1e-6 and row["raw_equal_unforced"]
    out["forced_different_epoch_h9_naq/dblp"] = row
    ok &= row["pass"]
    return ok, out


def time_rule():
    """Linear model t = c + e * evaluations + p * episodes, per (arm, dataset), from three runs:
    smoke (8 episodes, eval_every 1 -> 8 evaluations), timing_a (320, eval_every 160 -> 2), timing_b (320, eval_every 32 -> 10).
    e = (t_b - t_a) / (n_b - n_a); c = t_smoke - 8 e - 8 p; p = (t_a - c - 2 e) / 320 (solved jointly). 4000 episodes -> 25 evaluations."""
    est = {}
    for arm in ("h9_naq", "h9_clus_m20"):
        for ds in NEW_DATASETS:
            runs = {}
            for tag, root in (("smoke", SMOKE), ("a", TIMING_A), ("b", TIMING_B)):
                m = json.loads((root / arm / ds / "seed1212" / "metrics.json").read_text())
                runs[tag] = (m["elapsed_seconds"], len(m["epochs"]), m["settings"]["epoch"])
            (ts, ns, es), (ta, na, ea), (tb, nb, eb) = runs["smoke"], runs["a"], runs["b"]
            e = (tb - ta) / (nb - na)
            p = ((ta - na * e) - (ts - ns * e)) / (ea - es)
            c = ts - ns * e - es * p
            total = c + 25 * e + 4000 * p
            est[f"{arm}/{ds}"] = {"runs": runs, "per_eval_s": e, "per_episode_s": p, "fixed_s": c, "est_4000_s": total}
    per_ds = {ds: sum(v["est_4000_s"] for k, v in est.items() if k.endswith(ds)) / 2 for ds in NEW_DATASETS}
    runs = {ds: 4 * len(SEEDS) for ds in NEW_DATASETS}  # 4 arms x 10 seeds per dataset
    total_min = sum(per_ds[ds] * runs[ds] for ds in NEW_DATASETS) / 8 / 60
    return {"per_run": est, "mean_est_4000_s_per_ds": per_ds, "runs": runs, "gpus": 8, "est_total_min": total_min, "limit_min": TIME_LIMIT_MIN, "reduce_seeds": total_min > TIME_LIMIT_MIN}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    cli = parser.parse_args()
    r = {}
    r["G-data"] = dict(zip(("pass", "datasets"), g_data()))
    r["G-cfg"] = dict(zip(("pass", "arms"), g_cfg()))
    r["G-ckpt"] = dict(zip(("pass", "detail"), g_ckpt()))
    r["G-tf"] = dict(zip(("pass", "datasets"), g_tf()))
    r["regress"] = dict(zip(("pass", "runs"), regress()))
    r["invariant"] = dict(zip(("pass", "detail"), invariant(cli.start)))
    r["smoke"] = dict(zip(("pass", "runs"), smoke()))
    r["time_rule"] = time_rule()
    r["all_pass"] = all(r[k]["pass"] for k in ("G-data", "G-cfg", "G-ckpt", "G-tf", "regress", "invariant", "smoke"))
    (ROOT / "HW09_checks.json").write_text(json.dumps(r, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: (v["pass"] if isinstance(v, dict) and "pass" in v else v) for k, v in r.items() if k != "time_rule"}, indent=1))
    print(json.dumps({k: r["time_rule"][k] for k in ("mean_est_4000_s_per_ds", "est_total_min", "reduce_seeds")}))


if __name__ == "__main__":
    main()
