"""hl000f_run_hl06: copied from hl/hl000e_run_hl05.py; runs a training script over an arm x seed grid on several GPUs (restartable).

GPU slots: --gpus 2,3 --jobs_per_gpu 3 -> each job sees one physical GPU via CUDA_VISIBLE_DEVICES and gets --device 0.
Queue is seed-major (all arms of the first seed, then the next seed). --deadline HH:MM (KST): no new run starts after it.
A run exceeding --run_timeout_sec is killed and its failure.txt says timeout. summary.csv has one row per run x shot with a `gpu` column.
"""

import argparse
import csv
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KST = dt.timezone(dt.timedelta(hours=9))
HEAD = ["dataset", "shot", "arm", "seed", "gpu", "status", "best_epoch", "valid_accuracy", "test_accuracy", "test_ci95_half_width",
        "query_gt_match", "mean_original_x_rank", "fallback_occurrence_rate"]
TAIL = ["elapsed_seconds", "wall_seconds", "checkpoint", "error"]


def run_dir(root, arm, seed):
    return root / arm / f"seed_{seed}"


def now_kst():
    return dt.datetime.now(KST)


def parse_deadline(text, start):
    """Next HH:MM (KST) at or after `start` - 12 h, so a 01:30 deadline set at 23:50 means the following night."""
    if not text:
        return None
    h, m = (int(x) for x in text.split(":"))
    d = start.replace(hour=h, minute=m, second=0, microsecond=0)
    if d < start - dt.timedelta(hours=12):
        d += dt.timedelta(days=1)
    return d


def save_csv(root, dataset, arms, seeds):
    rows, extra = [], []
    for arm in arms:
        for seed in seeds:
            out = run_dir(root, arm, seed)
            report_path, fail_path, wall_path, gpu_path = out / "report.json", out / "failure.txt", out / "wall_seconds.txt", out / "gpu.txt"
            doc = json.loads(report_path.read_text()) if report_path.exists() else None
            for shot in (1, 5):
                row = {"dataset": dataset, "shot": shot, "arm": arm, "seed": seed,
                       "gpu": gpu_path.read_text().strip() if gpu_path.exists() else ""}
                if doc:
                    ev = doc["evaluation"][str(shot)]
                    dg = doc["diagnostics"]
                    row.update(
                        status="complete", best_epoch=ev["best_epoch"], valid_accuracy=ev["valid_accuracy"], test_accuracy=ev["test_accuracy"],
                        test_ci95_half_width=ev["test_ci95_half_width"], query_gt_match=dg["query_gt_match"],
                        mean_original_x_rank=dg["original_x_rank"] if dg["original_x_rank"] is not None else "",
                        fallback_occurrence_rate=dg["fallback_occurrence_rate"], elapsed_seconds=doc["elapsed_seconds"],
                        wall_seconds=wall_path.read_text().strip() if wall_path.exists() else "", checkpoint=ev["checkpoint"],
                    )
                    for k, v in doc["query_diagnostics"].items():
                        if isinstance(v, (int, float)):
                            row[f"q_{k}"] = v
                            if f"q_{k}" not in extra:
                                extra.append(f"q_{k}")
                elif fail_path.exists():
                    row.update(status="failed", error=fail_path.read_text().strip())
                else:
                    row["status"] = "pending"
                rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / "summary.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HEAD + extra + TAIL, restval="")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(root / "summary.csv")
    return sum(row["status"] == "complete" for row in rows) // 2, len(rows) // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="hl/hl011_pdd_train.py")
    ap.add_argument("--dataset", default="Amazon_clothing")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seeds", default="1212,1213,1214,1215,1216")
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--epoch", type=int, default=2000)
    ap.add_argument("--gpus", default="2,3")
    ap.add_argument("--jobs_per_gpu", type=int, default=3)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--extra", default="", help="arguments passed verbatim to the training script")
    ap.add_argument("--run_timeout_sec", type=int, default=1200)
    ap.add_argument("--deadline", default="", help="HH:MM KST; no new run starts after it")
    args = ap.parse_args()
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if len(arms) != len(set(arms)) or len(seeds) != len(set(seeds)):
        raise ValueError("Repeated arms/seeds")
    if not set(gpus) <= {"2", "3"}:
        raise ValueError("Only GPUs 2 and 3 may be used (CLAUDE.md §2, rev 7)")
    if not 1 <= args.jobs_per_gpu <= 3:
        raise ValueError("At most 3 concurrent runs per GPU (CLAUDE.md §2)")
    start = now_kst()
    deadline = parse_deadline(args.deadline, start)
    root = Path(args.output_root)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for shot in (1, 5):
            folder = Path("save/fixed_episodes") / args.dataset / f"5way_{shot}shot_8query_seed{seed}"
            if not ((folder / "valid_episodes.json").exists() and (folder / "test_episodes.json").exists()):
                raise RuntimeError(f"Missing fixed episodes: {folder} (generate with choi_exps/exp026.py first)")
    queue = []
    for seed in seeds:  # seed-major so partial results cover all arms early
        for arm in arms:
            if (run_dir(root, arm, seed) / "report.json").exists():
                print(f"[runner] SKIP complete {arm}/seed_{seed}", flush=True)
                continue
            queue.append((arm, seed))
    slots = [g for g in gpus for _ in range(args.jobs_per_gpu)]
    print(f"[runner] {len(queue)} trainings to run, slots={slots}, deadline={deadline}; output={root}", flush=True)
    save_csv(root, args.dataset, arms, seeds)
    running = []  # (arm, seed, process, begin, file, gpu)

    def finish(item, timed_out=False):
        arm, seed, process, begin, file, gpu = item
        process.wait()
        file.close()
        out = run_dir(root, arm, seed)
        wall = time.perf_counter() - begin
        tag = f"{arm}/seed_{seed}"
        if timed_out:
            (out / "failure.txt").write_text(f"timeout after {wall:.0f}s (> {args.run_timeout_sec}s); log={file.name}", encoding="utf-8")
            print(f"[runner] TIMEOUT {tag}", flush=True)
        elif process.returncode or not (out / "report.json").exists():
            (out / "failure.txt").write_text(f"exit={process.returncode}; log={file.name}", encoding="utf-8")
            print(f"[runner] FAILED {tag}: {file.name}", flush=True)
        else:
            (out / "wall_seconds.txt").write_text(f"{wall:.1f}\n", encoding="utf-8")
            print(f"[runner] DONE {tag} gpu={gpu} ({wall:.1f}s)", flush=True)
        complete, total = save_csv(root, args.dataset, arms, seeds)
        print(f"[runner] progress {complete}/{total} runs", flush=True)

    while queue or running:
        if deadline is not None and queue and now_kst() >= deadline:
            print(f"[runner] DEADLINE {deadline:%H:%M} reached; {len(queue)} runs not started", flush=True)
            queue.clear()
        busy = [r[5] for r in running]
        free = list(slots)
        for g in busy:
            free.remove(g)
        while queue and free:
            gpu = free.pop(0)
            arm, seed = queue.pop(0)
            out = run_dir(root, arm, seed)
            out.mkdir(parents=True, exist_ok=True)
            if (out / "failure.txt").exists():
                (out / "failure.txt").unlink()
            (out / "gpu.txt").write_text(gpu, encoding="utf-8")
            cmd = [sys.executable, "-u", args.script, "--dataset", args.dataset, "--tau", str(args.tau), "--device", "0",
                   "--seed", str(seed), "--epoch", str(args.epoch), "--output_root", str(root), "--arm", arm] + shlex.split(args.extra)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            env["CUDA_VISIBLE_DEVICES"] = gpu
            log_path = logs / f"{arm}_seed{seed}.log"
            print(f"[runner] START {arm}/seed_{seed} gpu={gpu} {now_kst():%H:%M:%S}", flush=True)
            file = log_path.open("w")
            process = subprocess.Popen(cmd, stdout=file, stderr=subprocess.STDOUT, env=env)
            running.append((arm, seed, process, time.perf_counter(), file, gpu))
        time.sleep(2)
        for item in list(running):
            if item[2].poll() is not None:
                running.remove(item)
                finish(item)
            elif time.perf_counter() - item[3] > args.run_timeout_sec:
                item[2].kill()
                running.remove(item)
                finish(item, timed_out=True)
    complete, total = save_csv(root, args.dataset, arms, seeds)
    print(f"[runner] Finished {complete}/{total} runs at {now_kst():%H:%M}. CSV: {root/'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
