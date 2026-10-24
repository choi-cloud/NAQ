"""Sequential, restartable benchmark runner; creates frozen episodes once."""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

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
    "checkpoint",
    "error",
]


def save_csv(root, datasets, taus, seed):
    rows = []
    for dataset in datasets:
        for method, candidates in [("naq", [None]), ("ours", taus)]:
            for tau in candidates:
                name = f"tau_{tau:g}" if tau is not None else "baseline"
                out = root / dataset / method / name / f"seed_{seed}"
                report_path = out / "report.json"
                fail_path = out / "failure.txt"
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
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--taus", default="0.1,0.2,0.3,0.4,0.5,0.6")
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--epoch", type=int, default=2000)
    p.add_argument("--output_root", default="results/gfsl_benchmark_exp049")
    args = p.parse_args()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    if not datasets or any(x not in DATASETS for x in datasets):
        raise ValueError("Unknown dataset; permitted names: " + ", ".join(DATASETS))
    taus = [float(x) for x in args.taus.split(",")]
    if len(taus) != len(set(taus)) or not all(-1 <= x <= 1 for x in taus):
        raise ValueError("Invalid/repeated threshold")
    root = Path(args.output_root)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    n_runs = len(datasets) * (1 + len(taus))
    print(f"[runner] {n_runs} trainings, {n_runs*2} result rows; output={root}", flush=True)
    for dataset in datasets:
        for shot in (1, 5):
            folder = Path("save/fixed_episodes") / dataset / f"5way_{shot}shot_8query_seed{args.seed}"
            valid_path, test_path = folder / "valid_episodes.json", folder / "test_episodes.json"
            if valid_path.exists() and test_path.exists():
                print(f"[runner] existing fixed episodes {dataset} {shot}-shot", flush=True)
                continue
            if valid_path.exists() != test_path.exists():
                raise RuntimeError(f"Only one episode split exists: {folder}; manual inspection required")
            print(f"[runner] generating fixed episodes {dataset} {shot}-shot", flush=True)
            cmd = [
                sys.executable,
                "-u",
                "exp026.py",
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
                str(args.seed),
                "--device",
                str(args.device),
                "--output_dir",
                str(folder),
                "--json_path",
                str(root / f"episodes_{dataset}_{shot}shot.json"),
            ]
            log = logs / f"episodes_{dataset}_{shot}shot.log"
            with log.open("w") as file:
                process = subprocess.run(cmd, stdout=file, stderr=subprocess.STDOUT)
            if process.returncode:
                raise RuntimeError(f"Episode generation failed ({dataset}, {shot}-shot). See {log}")
    for dataset in datasets:
        for method, candidates in [("naq", [None]), ("ours", taus)]:
            for tau in candidates:
                name = f"tau_{tau:g}" if tau is not None else "baseline"
                out = root / dataset / method / name / f"seed_{args.seed}"
                out.mkdir(parents=True, exist_ok=True)
                report = out / "report.json"
                if report.exists():
                    print(f"[runner] SKIP complete {dataset}/{method}/{name}", flush=True)
                    save_csv(root, datasets, taus, args.seed)
                    continue
                failure = out / "failure.txt"
                if failure.exists():
                    failure.unlink()
                cmd = [
                    sys.executable,
                    "-u",
                    "exp049.py",
                    "--dataset",
                    dataset,
                    "--method",
                    method,
                    "--device",
                    str(args.device),
                    "--seed",
                    str(args.seed),
                    "--epoch",
                    str(args.epoch),
                    "--output_root",
                    str(root),
                ]
                if tau is not None:
                    cmd += ["--tau", str(tau)]
                log_path = logs / f"{dataset}_{method}_{name}.log"
                print(f"[runner] START {dataset}/{method}/{name}", flush=True)
                begin = time.perf_counter()
                with log_path.open("w") as file:
                    process = subprocess.run(cmd, stdout=file, stderr=subprocess.STDOUT)
                if process.returncode or not report.exists():
                    failure.write_text(f"exit={process.returncode}; log={log_path}", encoding="utf-8")
                    print(f"[runner] FAILED {dataset}/{method}/{name}: {log_path}", flush=True)
                else:
                    print(f"[runner] DONE {dataset}/{method}/{name} ({time.perf_counter()-begin:.1f}s)", flush=True)
                complete, total = save_csv(root, datasets, taus, args.seed)
                print(f"[runner] progress {complete}/{total} result rows", flush=True)
    complete, total = save_csv(root, datasets, taus, args.seed)
    print(f"[runner] Finished {complete}/{total} rows. CSV: {root/'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
