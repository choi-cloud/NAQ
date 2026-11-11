"""HW08-a §1.6 checks -> results/hw_iislab4/HW08_checks.json (also §0 state_dict size records).

    PYTHONPATH=. python -m hw.check_hw08 --start <work-start HEAD>

G-cfg  merged l2_* config minus {eval_l2, save_best, arm, note} == merged e4k_* config minus {arm, note}
G-l2   evaluate_l2(unnormalized Â²X) test == hw/trainfree.py "Â²X, L2" test, |Δ| <= 1e-4
G-tf   iislab4 trainfree output: Â²X L2 80.12 / 92.72, Â²X raw 67.57 / 83.03, |Δ| <= 0.01 (%)
regress naq_base, nbr_raw seed 1212 under _regress/HW08 vs results/hw_iislab4, |Δ| <= 0.30 (%)
invariant git diff <start> for the training-path modules; deleted-line counts for run.py / evaluate.py
smoke  /tmp/hw_smoke runs: new keys, ckpt records, files; checkpoint reload reproduces test_acc
"""

import argparse
import io
import json
import subprocess
from pathlib import Path

import torch

from data import load_data
from meta import ProtoNet
from utils import seed_everything

from hw.config import build_args, load_config
from hw.episodes import build_eval_pools
from hw.evaluate import evaluate, evaluate_l2

ROOT = Path("results/hw_iislab4")
DATASETS = ("Cora_full", "Amazon_clothing")
PAIRS = {"l2_naq": "e4k_naq", "l2_rand_m20": "e4k_rand_m20", "l2_rand_m50": "e4k_rand_m50", "l2_rand_m200": "e4k_rand_m200", "l2_clus_m20": "e4k_clus_m20"}
TF_EXPECTED = {"Cora_full": {"A2X_l2": 80.12, "A2X_raw": 67.57}, "Amazon_clothing": {"A2X_l2": 92.72, "A2X_raw": 83.03}}
NEW_KEYS = ("test_acc_l2_at_bestraw", "test_acc_l2", "test_acc_raw_at_bestl2", "best_l2", "best_epoch_500_l2", "same_epoch_l2", "geom_l2", "norm_diag", "norm_diag_l2", "ckpt")
NORM_KEYS = ("zero_rows_all", "zero_rows_test", "test_norm_mean", "test_norm_std", "proto_norm_cv")
SMOKE = Path("/tmp/hw_smoke")


def g_cfg():
    out, ok = {}, True
    for new, old in PAIRS.items():
        a = {k: v for k, v in load_config(new).items() if k not in ("eval_l2", "save_best", "arm", "note")}
        b = {k: v for k, v in load_config(old).items() if k not in ("arm", "note")}
        diff = sorted(k for k in set(a) | set(b) if a.get(k, "<missing>") != b.get(k, "<missing>"))
        same = not diff and load_config(new).get("eval_l2") is True and load_config(new).get("save_best") is True
        out[new] = {"ref": old, "equal": same, "diff_keys": diff}
        ok &= same
    return ok, out


@torch.no_grad()
def g_l2_and_sizes():
    out, sizes, ok = {}, {}, True
    for ds in DATASETS:
        cfg = load_config("naq_base")
        args = build_args(cfg, ds)
        seed_everything(1212)
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        data = load_data(args)
        data.set_device(str(device))
        _, test_pool = build_eval_pools(args, data)
        x = data.features.float()
        a2x = torch.sparse.mm(data.adj, torch.sparse.mm(data.adj, x))  # same expression as hw/trainfree.py:92-93
        mine = 100 * evaluate_l2(a2x, test_pool, args.n_way, device)["accuracy_mean"]
        tf = json.loads((ROOT / "trainfree" / f"HW03a_trainfree_{ds}.json").read_text())
        ref = 100 * tf["variants"]["A2X_l2"]["test"]["accuracy_mean"]
        out[ds] = {"evaluate_l2_a2x_test": mine, "trainfree_a2x_l2_test": ref, "abs_diff": abs(mine - ref), "pass": abs(mine - ref) <= 1e-4}
        ok &= out[ds]["pass"]
        # §0 record: encoder state_dict size (same construction as hw/run.py)
        d = x.shape[1]
        config = [("GCN", (d, 2 * args.latent), True), ("GCN", (2 * args.latent, args.latent), True), ("linear", (args.latent, args.n_way), True)]
        model = ProtoNet(args, config, data)
        state = model.network.state_dict()
        buffer = io.BytesIO()
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in state.items()}, "epoch": 0}, buffer)
        sizes[ds] = {"feature_dim": int(d), "latent": int(args.latent), "params": int(sum(v.numel() for v in state.values())), "tensors": len(state), "bytes_saved": buffer.getbuffer().nbytes}
    return ok, out, sizes


def g_tf():
    out, ok = {}, True
    for ds in DATASETS:
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
        for ds in DATASETS:
            new = json.loads((ROOT / "_regress" / "HW08" / arm / ds / "seed1212" / "metrics.json").read_text())
            old = json.loads((ROOT / arm / ds / "seed1212" / "metrics.json").read_text())
            a, b = 100 * new["best"]["test"]["accuracy_mean"], 100 * old["best"]["test"]["accuracy_mean"]
            key = f"{arm}/{ds}"
            prev = json.loads((ROOT / "_regress" / "HW07" / arm / ds / "seed1212" / "metrics.json").read_text())  # default-config record format
            keys_same = set(new) == set(prev) and set(new["epochs"][0]) == set(prev["epochs"][0])
            out[key] = {"new": a, "existing": b, "delta": a - b, "best_epoch_new": new["best"]["epoch"], "best_epoch_existing": old["best"]["epoch"], "status": new["status"],
                        "metrics_keys_same_as_HW07_regress": keys_same, "pass": new["status"] == "completed" and abs(a - b) <= 0.30 and keys_same}
            ok &= out[key]["pass"]
    return ok, out


def git(*cmd):
    return subprocess.run(["git", *cmd], capture_output=True, text=True, check=True).stdout


def invariant(start):
    frozen = ["hw/losses.py", "hw/negatives.py", "hw/neighbors.py", "hw/episodes.py"]
    diff = git("diff", start, "--", *frozen)
    deleted = {}
    for path in ("hw/run.py", "hw/evaluate.py"):
        lines = git("diff", start, "--", path).splitlines()
        deleted[path] = [l[1:] for l in lines if l.startswith("-") and not l.startswith("---")]
    ok = diff == ""
    return ok, {"start_head": start, "frozen_files": frozen, "frozen_diff_empty": ok, "deleted_lines": deleted,
                "deleted_line_counts": {k: len(v) for k, v in deleted.items()}}


@torch.no_grad()
def smoke():
    out, ok = {}, True
    for arm in ("l2_naq", "l2_rand_m20", "l2_clus_m20"):
        m = json.loads((SMOKE / arm / "Cora_full" / "seed1212" / "metrics.json").read_text())
        missing = [k for k in NEW_KEYS if k not in m]
        missing += [f"epochs.{k}" for k in ("val_acc_500_l2", "val_acc_50_l2") if any(k not in e for e in m["epochs"])]
        missing += [f"{d}.{k}" for d in ("norm_diag", "norm_diag_l2") for k in NORM_KEYS if k not in m.get(d, {})]
        ck = m.get("ckpt", {})
        files = {p: Path(p).stat().st_size for p in (ck.get("best_raw"), ck.get("best_l2")) if p}
        row = {"status": m["status"], "missing": missing, "ckpt": ck, "files": files,
               "bytes_match": sum(files.values()) == ck.get("bytes"), "same_epoch_l2": m.get("same_epoch_l2"),
               "best_l2_file_rule": (ck.get("best_l2") is None) == bool(m.get("same_epoch_l2"))}
        row["pass"] = m["status"] == "completed" and not missing and row["bytes_match"] and row["best_l2_file_rule"]
        out[arm] = row
        ok &= row["pass"]
    # forced different-epoch run (evaluate_l2 on validation wrapped in-process so best_l2 = epoch 1; smoke only)
    m = json.loads((SMOKE / "forced" / "l2_naq" / "Cora_full" / "seed1212" / "metrics.json").read_text())
    ck = m["ckpt"]
    files = {p: Path(p).stat().st_size for p in (ck["best_raw"], ck["best_l2"]) if p}
    # reload both checkpoints and re-evaluate test (checks the saved state_dicts)
    cfg = load_config("l2_naq")
    args = build_args(cfg, "Cora_full")
    seed_everything(1212)
    device = torch.device("cuda:0")
    data = load_data(args)
    data.set_device(str(device))
    data.add_diffusion(args)
    _, test_pool = build_eval_pools(args, data)
    d = data.features.shape[1]
    config = [("GCN", (d, 2 * args.latent), True), ("GCN", (2 * args.latent, args.latent), True), ("linear", (args.latent, args.n_way), True)]
    model = ProtoNet(args, config, data).to(device)
    reload = {}
    for name, key, fn in (("best_raw", "test_acc", evaluate), ("best_l2", "test_acc_l2", evaluate_l2)):
        blob = torch.load(ck[name], map_location=device)
        model.network.load_state_dict(blob["state_dict"])
        model.eval()
        value = fn(model.network(data.features, adj=data.adj), test_pool, args.n_way, device)["accuracy_mean"]
        reload[name] = {"epoch": blob["epoch"], "recomputed": value, "recorded": m[key], "abs_diff": abs(value - m[key])}
    row = {"status": m["status"], "best_epoch_500": m["best_epoch_500"], "best_epoch_500_l2": m["best_epoch_500_l2"], "same_epoch_l2": m["same_epoch_l2"],
           "ckpt": ck, "files": files, "bytes_match": sum(files.values()) == ck["bytes"], "geom_l2_present": "geom_l2" in m, "reload": reload,
           "raw_test_equals_unforced": m["test_acc"] == json.loads((SMOKE / "l2_naq" / "Cora_full" / "seed1212" / "metrics.json").read_text())["test_acc"]}
    row["pass"] = (m["status"] == "completed" and not m["same_epoch_l2"] and len(files) == 2 and row["bytes_match"] and row["geom_l2_present"]
                   and all(r["abs_diff"] <= 1e-6 for r in reload.values()) and row["raw_test_equals_unforced"])
    out["forced_different_epoch_l2_naq"] = row
    ok &= row["pass"]
    return ok, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    cli = parser.parse_args()
    report = {}
    report["G-cfg"] = dict(zip(("pass", "arms"), g_cfg()))
    ok, rows, sizes = g_l2_and_sizes()
    report["G-l2"] = {"pass": ok, "datasets": rows}
    report["G-tf"] = dict(zip(("pass", "datasets"), g_tf()))
    report["regress"] = dict(zip(("pass", "runs"), regress()))
    report["invariant"] = dict(zip(("pass", "detail"), invariant(cli.start)))
    report["smoke"] = dict(zip(("pass", "runs"), smoke()))
    report["state_dict_size"] = sizes
    report["all_pass"] = all(report[k]["pass"] for k in ("G-cfg", "G-l2", "G-tf", "regress", "invariant", "smoke"))
    (ROOT / "HW08_checks.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: (v["pass"] if isinstance(v, dict) and "pass" in v else v) for k, v in report.items()}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
