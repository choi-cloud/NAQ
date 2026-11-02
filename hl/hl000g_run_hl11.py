"""hl000g_run_hl11: copied from hl/hl000_run.py; runs hl013_curve sequentially (one at a time), restartable. No episode generation (save/ is immutable). Stops when GPU free < 10GB (CLAUDE.md §7(d))."""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["shot", "method", "tau", "seed", "status", "best_epoch", "valid_accuracy", "test_accuracy", "test_ci95_half_width",
          "query_gt_match", "original_gt_match", "elapsed_seconds", "wall_seconds", "gpu_peak_allocated_mib", "checkpoint", "error"]
ARMS = [("naq", None), ("ours", 0.3)]


def gpu_free_mib():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits", "-i", "0"], text=True)
    total, used = (int(x) for x in out.strip().split(","))
    return total - used


def save_csv(root, seeds):
    rows = []
    for method, tau in ARMS:
        for seed in seeds:
            out = root / method / f"seed_{seed}"
            report, fail, wall = out / "report.json", out / "failure.txt", out / "wall_seconds.txt"
            doc = json.loads(report.read_text()) if report.exists() else None
            for shot in (1, 5):
                row = dict.fromkeys(FIELDS, "")
                row.update(shot=shot, method=method, tau="" if tau is None else tau, seed=seed)
                if doc:
                    ev, dg = doc["evaluation"][str(shot)], doc["diagnostics"]
                    row.update(status="complete", best_epoch=ev["best_epoch"], valid_accuracy=ev["valid_accuracy"], test_accuracy=ev["test_accuracy"],
                               test_ci95_half_width=ev["test_ci95_half_width"], query_gt_match=dg["query_gt_match"], original_gt_match=dg["original_gt_match"],
                               elapsed_seconds=doc["elapsed_seconds"], wall_seconds=wall.read_text().strip() if wall.exists() else "",
                               gpu_peak_allocated_mib=doc.get("gpu_peak_allocated_mib", ""), checkpoint=ev["checkpoint"])
                elif fail.exists():
                    row.update(status="failed", error=fail.read_text().strip())
                else:
                    row["status"] = "pending"
                rows.append(row)
    tmp = root / "summary.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(root / "summary.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="1212,1213,1214,1215,1216")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--output_root", default="results/hl/hl11/curve")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    root = Path(args.output_root)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for method, tau in ARMS:
        for seed in seeds:
            tag = f"{method}/seed_{seed}"
            out = root / method / f"seed_{seed}"
            out.mkdir(parents=True, exist_ok=True)
            report, failure = out / "report.json", out / "failure.txt"
            if report.exists():
                print(f"[runner] SKIP complete {tag}", flush=True)
                continue
            if failure.exists():
                failure.unlink()
            free = gpu_free_mib()
            if free < 10240:
                print(f"[runner] STOP GPU free {free} MiB < 10GB before {tag}", flush=True)
                save_csv(root, seeds)
                sys.exit(3)
            cmd = [sys.executable, "-u", "hl/hl013_curve.py", "--method", method, "--device", str(args.device), "--seed", str(seed), "--output_root", str(root)]
            if tau is not None:
                cmd += ["--tau", str(tau)]
            log_path = logs / f"{method}_seed{seed}.log"
            print(f"[runner] START {tag} (GPU free {free} MiB)", flush=True)
            begin = time.perf_counter()
            with log_path.open("w") as file:
                process = subprocess.run(cmd, stdout=file, stderr=subprocess.STDOUT, env=env)
            wall = time.perf_counter() - begin
            if process.returncode or not report.exists():
                failure.write_text(f"exit={process.returncode}; log={log_path}", encoding="utf-8")
                print(f"[runner] FAILED {tag}: {log_path}", flush=True)
            else:
                (out / "wall_seconds.txt").write_text(f"{wall:.1f}\n", encoding="utf-8")
                print(f"[runner] DONE {tag} ({wall:.1f}s)", flush=True)
            save_csv(root, seeds)
    save_csv(root, seeds)
    print("[runner] Finished", flush=True)


if __name__ == "__main__":
    main()
