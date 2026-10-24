"""hl000b_run_oracle: copied from hl/hl000_run.py; runs hl/hl003_oracle_train.py over a policy x seed grid (restartable, up to --jobs concurrent)."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POLICIES = "uniform,quintile:1,quintile:2,quintile:3,quintile:4,quintile:5,gt_distinct,hard:2,hard:4,hard_blind:4,ab"
FIELDS = [
    "dataset",
    "shot",
    "policy",
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
    "k_eff_mean",
    "same_gt_pair_rate",
    "episodes_with_same_gt_pair",
    "anchor_pair_mean_x_cosine",
    "anchor_pair_mean_a2x_cosine",
    "uniform_fallback_rate",
    "anchor_pi_bchoi_mean",
    "unique_anchors",
    "elapsed_seconds",
    "wall_seconds",
    "checkpoint",
    "error",
]
ANCHOR_KEYS = FIELDS[FIELDS.index("k_eff_mean"):FIELDS.index("unique_anchors") + 1]


def run_dir(root, policy, seed):
    return root / policy.replace(":", "_") / f"seed_{seed}"


def policy_cli(policy):
    """'hard:4' -> ['--policy', 'hard', '--k', '4']; quintile:q is passed as is."""
    name, _, k = policy.partition(":")
    if name in ("hard", "hard_blind"):
        return ["--policy", name, "--k", k]
    return ["--policy", policy]


def save_csv(root, dataset, tau, policies, seeds):
    rows = []
    for policy in policies:
        for seed in seeds:
            out = run_dir(root, policy, seed)
            report_path, fail_path, wall_path = out / "report.json", out / "failure.txt", out / "wall_seconds.txt"
            doc = json.loads(report_path.read_text()) if report_path.exists() else None
            for shot in (1, 5):
                row = dict.fromkeys(FIELDS, "")
                row.update(dataset=dataset, shot=shot, policy=policy, tau=tau, seed=seed)
                if doc:
                    ev = doc["evaluation"][str(shot)]
                    dg = doc["diagnostics"]
                    ad = doc["anchor_diagnostics"]
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
                        **{k: ("" if ad[k] is None else ad[k]) for k in ANCHOR_KEYS},
                    )
                elif fail_path.exists():
                    row.update(status="failed", error=fail_path.read_text().strip())
                else:
                    row["status"] = "pending"
                rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / "summary.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(root / "summary.csv")
    return sum(row["status"] == "complete" for row in rows), len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Amazon_clothing")
    p.add_argument("--policies", default=POLICIES)
    p.add_argument("--seeds", default="1212,1213,1214,1215,1216")
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--epoch", type=int, default=2000)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--output_root", default="results/hl/hl02/oracle")
    p.add_argument("--node_table", default="results/hl/hl02/diag/node_table_tau0.3.csv")
    p.add_argument("--quintile_csv", default="results/hl/hl02/diag/pi_bchoi_quintiles.csv")
    args = p.parse_args()
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if len(policies) != len(set(policies)) or len(seeds) != len(set(seeds)):
        raise ValueError("Repeated policies/seeds")
    if not 1 <= args.jobs <= 2:
        raise ValueError("At most 2 concurrent runs (CLAUDE.md §2)")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    root = Path(args.output_root)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for shot in (1, 5):
            folder = Path("save/fixed_episodes") / args.dataset / f"5way_{shot}shot_8query_seed{seed}"
            if not ((folder / "valid_episodes.json").exists() and (folder / "test_episodes.json").exists()):
                raise RuntimeError(f"Missing fixed episodes: {folder} (generate with choi_exps/exp026.py first)")
    queue = []
    for seed in seeds:  # seed-major so partial results cover all policies early
        for policy in policies:
            out = run_dir(root, policy, seed)
            if (out / "report.json").exists():
                print(f"[runner] SKIP complete {policy}/seed_{seed}", flush=True)
                continue
            queue.append((policy, seed))
    print(f"[runner] {len(queue)} trainings to run, jobs={args.jobs}; output={root}", flush=True)
    save_csv(root, args.dataset, args.tau, policies, seeds)
    running = []

    def finish(item):
        policy, seed, process, begin, file = item
        process.wait()
        file.close()
        out = run_dir(root, policy, seed)
        wall = time.perf_counter() - begin
        tag = f"{policy}/seed_{seed}"
        if process.returncode or not (out / "report.json").exists():
            (out / "failure.txt").write_text(f"exit={process.returncode}; log={file.name}", encoding="utf-8")
            print(f"[runner] FAILED {tag}: {file.name}", flush=True)
        else:
            (out / "wall_seconds.txt").write_text(f"{wall:.1f}\n", encoding="utf-8")
            print(f"[runner] DONE {tag} ({wall:.1f}s)", flush=True)
        complete, total = save_csv(root, args.dataset, args.tau, policies, seeds)
        print(f"[runner] progress {complete}/{total} result rows", flush=True)

    while queue or running:
        while queue and len(running) < args.jobs:
            policy, seed = queue.pop(0)
            out = run_dir(root, policy, seed)
            out.mkdir(parents=True, exist_ok=True)
            if (out / "failure.txt").exists():
                (out / "failure.txt").unlink()
            cmd = [
                sys.executable, "-u", "hl/hl003_oracle_train.py",
                "--dataset", args.dataset, "--tau", str(args.tau), "--device", str(args.device),
                "--seed", str(seed), "--epoch", str(args.epoch), "--output_root", str(root),
                "--node_table", args.node_table, "--quintile_csv", args.quintile_csv,
            ] + policy_cli(policy)
            log_path = logs / f"{policy.replace(':', '_')}_seed{seed}.log"
            print(f"[runner] START {policy}/seed_{seed}", flush=True)
            file = log_path.open("w")
            process = subprocess.Popen(cmd, stdout=file, stderr=subprocess.STDOUT, env=env)
            running.append((policy, seed, process, time.perf_counter(), file))
        time.sleep(2)
        for item in [r for r in running if r[2].poll() is not None]:
            running.remove(item)
            finish(item)
    complete, total = save_csv(root, args.dataset, args.tau, policies, seeds)
    print(f"[runner] Finished {complete}/{total} rows. CSV: {root/'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
