"""hl000_run: runner derived from choi_exps/run_exp049.py; paths fixed for repo-root execution."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = ["Cora_full", "dblp", "Amazon_clothing", "Amazon_eletronics"]
FIELDS = [
    "dataset",
    "shot",
    "method",
    "tau",
    "seed",
    "status",
    "best_epoch",
    "valid_accuracy",
    "test_accuracy",
    "test_ci95_half_width",
    "query_gt_match",
    "original_gt_match",
    "mean_original_x_rank",
    "coverage_without_fallback",
    "fallback_occurrence_rate",
    "elapsed_seconds",
    "wall_seconds",
    "checkpoint",
    "error",
]


def parse_arms(text):
    arms = []
    for item in [x.strip() for x in text.split(",") if x.strip()]:
        if item == "naq":
            arms.append(("naq", None))
        elif item.startswith("ours:"):
            tau = float(item.split(":", 1)[1])
            if not -1 <= tau <= 1:
                raise ValueError(f"Invalid threshold: {item}")
            arms.append(("ours", tau))
        else:
            raise ValueError(f"Unknown arm: {item}; use 'naq' or 'ours:<tau>'")
    if not arms or len(arms) != len(set(arms)):
        raise ValueError("Empty/repeated arms")
    return arms


def run_dir(root, dataset, method, tau, seed):
    name = f"tau_{tau:g}" if tau is not None else "baseline"
    return name, root / dataset / method / name / f"seed_{seed}"


def save_csv(root, datasets, arms, seeds):
    rows = []
    for dataset in datasets:
        for method, tau in arms:
            for seed in seeds:
                _, out = run_dir(root, dataset, method, tau, seed)
                report_path = out / "report.json"
                fail_path = out / "failure.txt"
                wall_path = out / "wall_seconds.txt"
                doc = json.loads(report_path.read_text()) if report_path.exists() else None
                for shot in (1, 5):
                    row = dict.fromkeys(FIELDS, "")
                    row.update(dataset=dataset, shot=shot, method=method, tau="" if tau is None else tau, seed=seed)
                    if doc:
                        ev = doc["evaluation"][str(shot)]
                        dg = doc["diagnostics"]
                        row.update(
                            status="complete",
                            best_epoch=ev["best_epoch"],
                            valid_accuracy=ev["valid_accuracy"],
                            test_accuracy=ev["test_accuracy"],
                            test_ci95_half_width=ev["test_ci95_half_width"],
                            query_gt_match=dg["query_gt_match"],
                            original_gt_match=dg["original_gt_match"],
                            mean_original_x_rank=dg["original_x_rank"] if dg["original_x_rank"] is not None else "",
                            coverage_without_fallback=dg["coverage_without_fallback"],
                            fallback_occurrence_rate=dg["fallback_occurrence_rate"],
                            elapsed_seconds=doc["elapsed_seconds"],
                            wall_seconds=wall_path.read_text().strip() if wall_path.exists() else "",
                            checkpoint=ev["checkpoint"],
                        )
                    elif fail_path.exists():
                        row.update(status="failed", error=fail_path.read_text().strip())
                    else:
                        row["status"] = "pending"
                    rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "summary.csv"
    tmp = root / "summary.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    return sum(row["status"] == "complete" for row in rows), len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="Amazon_clothing")
    p.add_argument("--arms", default="naq")
    p.add_argument("--seeds", default="1212")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--epoch", type=int, default=2000)
    p.add_argument("--output_root", default="results/hl/p0")
    args = p.parse_args()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    if not datasets or any(x not in DATASETS for x in datasets):
        raise ValueError("Unknown dataset; permitted names: " + ", ".join(DATASETS))
    arms = parse_arms(args.arms)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Empty/repeated seeds")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    root = Path(args.output_root)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    n_runs = len(datasets) * len(arms) * len(seeds)
    print(f"[runner] {n_runs} trainings, {n_runs*2} result rows; output={root}", flush=True)
    for dataset in datasets:
        for seed in seeds:
            for shot in (1, 5):
                folder = Path("save/fixed_episodes") / dataset / f"5way_{shot}shot_8query_seed{seed}"
                valid_path, test_path = folder / "valid_episodes.json", folder / "test_episodes.json"
                if valid_path.exists() and test_path.exists():
                    print(f"[runner] existing fixed episodes {dataset} {shot}-shot seed {seed}", flush=True)
                    continue
                if valid_path.exists() != test_path.exists():
                    raise RuntimeError(f"Only one episode split exists: {folder}; manual inspection required")
                print(f"[runner] generating fixed episodes {dataset} {shot}-shot seed {seed}", flush=True)
                cmd = [
                    sys.executable,
                    "-u",
                    "choi_exps/exp026.py",
                    "--dataset",
                    dataset,
                    "--n_way",
                    "5",
                    "--k_shot_test",
                    str(shot),
                    "--q_query_test",
                    "8",
                    "--num_valid_tasks",
                    "50",
                    "--num_downstream_tasks",
                    "500",
                    "--seed",
                    str(seed),
                    "--device",
                    str(args.device),
                    "--output_dir",
                    str(folder),
                    "--json_path",
                    str(root / f"episodes_{dataset}_{shot}shot_seed{seed}.json"),
                ]
                log = logs / f"episodes_{dataset}_{shot}shot_seed{seed}.log"
                begin = time.perf_counter()
                with log.open("w") as file:
                    process = subprocess.run(cmd, stdout=file, stderr=subprocess.STDOUT, env=env)
                if process.returncode:
                    raise RuntimeError(f"Episode generation failed ({dataset}, {shot}-shot, seed {seed}). See {log}")
                print(
                    f"[runner] generated fixed episodes {dataset} {shot}-shot seed {seed} "
                    f"({time.perf_counter()-begin:.1f}s)",
                    flush=True,
                )
    for dataset in datasets:
        for method, tau in arms:
            for seed in seeds:
                name, out = run_dir(root, dataset, method, tau, seed)
                tag = f"{dataset}/{method}/{name}/seed_{seed}"
                out.mkdir(parents=True, exist_ok=True)
                report = out / "report.json"
                if report.exists():
                    print(f"[runner] SKIP complete {tag}", flush=True)
                    save_csv(root, datasets, arms, seeds)
                    continue
                failure = out / "failure.txt"
                if failure.exists():
                    failure.unlink()
                cmd = [
                    sys.executable,
                    "-u",
                    "choi_exps/exp049.py",
                    "--dataset",
                    dataset,
                    "--method",
                    method,
                    "--device",
                    str(args.device),
                    "--seed",
                    str(seed),
                    "--epoch",
                    str(args.epoch),
                    "--output_root",
                    str(root),
                ]
                if tau is not None:
                    cmd += ["--tau", str(tau)]
                log_path = logs / f"{dataset}_{method}_{name}_seed{seed}.log"
                print(f"[runner] START {tag}", flush=True)
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
                complete, total = save_csv(root, datasets, arms, seeds)
                print(f"[runner] progress {complete}/{total} result rows", flush=True)
    complete, total = save_csv(root, datasets, arms, seeds)
    print(f"[runner] Finished {complete}/{total} rows. CSV: {root/'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
