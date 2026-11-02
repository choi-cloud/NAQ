"""hl000h_run_hl12: copied from hl/hl000g_run_hl11.py; runs hl014_train_variants sequentially (one at a time), restartable, by group. Stops when GPU free < 10GB (CLAUDE.md §7(d))."""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEEDS = [1212, 1213, 1214, 1215, 1216]
FIELDS = ["arm", "group", "seed", "status", "epoch0_test_1shot_l2", "R_epoch_1shot", "R_test_raw_1shot", "R_test_l2_1shot", "L_epoch_1shot", "L_test_l2_1shot",
          "R_epoch_5shot", "R_test_raw_5shot", "R_test_l2_5shot", "L_epoch_5shot", "L_test_l2_5shot", "query_gt_match", "elapsed_seconds", "wall_seconds", "gpu_peak_allocated_mib", "error"]
# (group, encoder, loss, episodes, scale, seeds); A -> C -> B -> D is the queue order of HL12 §2.
QUEUE = [
    ("A", "gcn", "raw", "sup", 10.0, SEEDS), ("A", "gcn", "raw", "sup_strat", 10.0, SEEDS),
    ("C", "pca64_lin", "cos", "naq", 10.0, SEEDS), ("C", "pca64_lin", "cos", "ours", 10.0, SEEDS), ("C", "pca64_lin", "cos", "sup", 10.0, SEEDS), ("C", "pca64_lin", "cos", "sup_strat", 10.0, SEEDS),
    ("C", "pca256_lin", "cos", "sup", 10.0, SEEDS), ("C", "pca256_mlp", "cos", "sup", 10.0, SEEDS),
    ("B", "gcn", "cos", "naq", 10.0, SEEDS), ("B", "gcn", "cos", "ours", 10.0, SEEDS), ("B", "gcn", "cos", "sup", 10.0, SEEDS),
    ("D", "pca64_lin", "cos", "sup", 5.0, [1212]), ("D", "pca64_lin", "cos", "sup", 20.0, [1212]),
]


def name_of(encoder, loss, episodes, scale):
    name = f"{encoder}_{loss}_{episodes}"
    return name if scale == 10.0 else f"{name}_s{scale:g}"


def gpu_free_mib():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits", "-i", "0"], text=True)
    total, used = (int(x) for x in out.strip().split(","))
    return total - used


def save_csv(root):
    rows = []
    for group, encoder, loss, episodes, scale, seeds in QUEUE:
        arm = name_of(encoder, loss, episodes, scale)
        for seed in seeds:
            out = root / arm / f"seed_{seed}"
            report, fail, wall = out / "report.json", out / "failure.txt", out / "wall_seconds.txt"
            row = dict.fromkeys(FIELDS, "")
            row.update(arm=arm, group=group, seed=seed)
            if report.exists():
                doc = json.loads(report.read_text())
                ev = doc["evaluation"]
                row.update(status="complete", epoch0_test_1shot_l2=doc["history"][0]["test_1shot_l2"], query_gt_match=doc["diagnostics"]["query_gt_match"],
                           elapsed_seconds=doc["elapsed_seconds"], wall_seconds=wall.read_text().strip() if wall.exists() else "", gpu_peak_allocated_mib=doc["gpu_peak_allocated_mib"])
                for shot in ("1", "5"):
                    row[f"R_epoch_{shot}shot"] = ev["R"][shot]["best_epoch"]
                    row[f"R_test_raw_{shot}shot"] = ev["R"][shot]["test_raw"]
                    row[f"R_test_l2_{shot}shot"] = ev["R"][shot]["test_l2"]
                    row[f"L_epoch_{shot}shot"] = ev["L"][shot]["best_epoch"]
                    row[f"L_test_l2_{shot}shot"] = ev["L"][shot]["test_l2"]
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
    p.add_argument("--groups", default="A,C,B,D")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--output_root", default="results/hl/hl12/curve")
    args = p.parse_args()
    groups = args.groups.split(",")
    root = Path(args.output_root)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for group in groups:
        for g, encoder, loss, episodes, scale, seeds in QUEUE:
            if g != group:
                continue
            arm = name_of(encoder, loss, episodes, scale)
            for seed in seeds:
                tag = f"{arm}/seed_{seed}"
                out = root / arm / f"seed_{seed}"
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
                    save_csv(root)
                    sys.exit(3)
                cmd = [sys.executable, "-u", "hl/hl014_train_variants.py", "--episodes", episodes, "--encoder", encoder, "--loss", loss, "--scale", str(scale),
                       "--device", str(args.device), "--seed", str(seed), "--output_root", str(root)]
                log_path = logs / f"{arm}_seed{seed}.log"
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
                save_csv(root)
    save_csv(root)
    print("[runner] Finished", flush=True)


if __name__ == "__main__":
    main()
