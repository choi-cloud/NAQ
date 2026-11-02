"""hl000j_run_hl13: copied from hl/hl000h_run_hl12.py; runs hl015_train_hl13 by stage with a gcn slot and a linear slot, --deadline (KST), 10-minute wall limit per run. Failed runs are not retried (CLAUDE.md §7-1)."""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KST = timezone(timedelta(hours=9))
S5 = [1212, 1213, 1214, 1215, 1216]
S10 = [1217, 1218, 1219, 1220, 1221]
LAST_START = datetime(2026, 11, 3, 8, 15, tzinfo=KST)
WALL_LIMIT = 600
GPU_MIN_IDLE = 10240
GPU_MIN_SECOND = 8600  # largest measured run peak (8430 MiB, gcn soft ours) + margin


def spec(encoder, loss, episodes, lr=1e-4, pca_fit="all", seeds=S5):
    return dict(encoder=encoder, loss=loss, episodes=episodes, lr=lr, pca_fit=pca_fit, seeds=seeds)


STAGES = {
    1: [spec("sgc64_rand", "cos", "naq", 1e-3), spec("sgc64_rand", "cos", "sup", 1e-3), spec("sgc256_rand", "cos", "naq", 1e-3),
        spec("sgc64_pcainit", "cos", "naq", 1e-3), spec("sgc64_pcainit", "cos", "sup", 1e-3), spec("pca1024_lin64", "cos", "naq", 1e-3)],
    2: [spec("pca64_lin", "cos", "naq", 1e-3), spec("pca64_lin", "cos", "naq", 1e-2), spec("pca64_lin", "cos", "sup", 1e-3), spec("pca64_lin", "cos", "sup", 1e-2),
        spec("sgc64_rand", "cos", "naq", 1e-4), spec("sgc64_rand", "cos", "naq", 1e-2)],
    3: [spec("gcn", "raw_soft0.3", "naq"), spec("gcn", "raw_soft0.6", "naq"), spec("gcn", "raw_soft1.0", "naq"), spec("gcn", "raw_soft0.3", "ours"),
        spec("pca64_lin", "cos_soft0.3", "naq", 1e-3), spec("pca64_lin", "cos_soft1.0", "naq", 1e-3)],
    4: [spec("gcn", "raw", "mix_r0.1"), spec("gcn", "raw", "mix_r0.25"), spec("gcn", "raw", "mix_r0.5")],
    5: [spec("gcn", "raw", "oracle_strat5"), spec("pca64_lin", "cos", "oracle_strat5", 1e-3), spec("sgc64_rand", "cos", "oracle_strat5", 1e-3)],
    6: [spec("pca64_lin", "cos", "naq", 1e-3, "base")],
    7: [spec("gcn", "raw", "naq", seeds=S10), spec("gcn", "raw", "ours", seeds=S10), spec("pca64_lin", "cos", "sup", seeds=S10)],
}
FIELDS = ["stage", "arm", "seed", "status", "epoch0_test_1shot_l2", "R_test_raw_1shot", "L_epoch_1shot", "L_test_l2_1shot", "L_epoch_5shot", "L_test_l2_5shot",
          "query_gt_match", "elapsed_seconds", "wall_seconds", "gpu_peak_allocated_mib", "error"]
LR_NAME = {1e-4: "1e-4", 1e-3: "1e-3", 1e-2: "1e-2"}


def arm_of(s):
    name = f"{s['encoder']}_{s['loss']}_{s['episodes']}"
    if s["pca_fit"] == "base":
        name += "_pcabase"
    if s["lr"] != 1e-4 or s["encoder"].startswith("sgc"):
        name += f"_lr{LR_NAME[s['lr']]}"
    return name


def now():
    return datetime.now(KST)


def gpu_free_mib():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits", "-i", "0"], text=True)
    total, used = (int(x) for x in out.strip().split(","))
    return total - used


def wait_idle_gpu(deadline):
    """With no run of ours active: < 10GB free is re-checked at 30-minute intervals; three in a row -> stop (CLAUDE.md §7-1 ②)."""
    for attempt in range(3):
        free = gpu_free_mib()
        if free >= GPU_MIN_IDLE:
            return
        print(f"[runner] GPU free {free} MiB < 10GB (check {attempt + 1}/3) {now():%H:%M}", flush=True)
        if attempt < 2:
            time.sleep(1800)
    print("[runner] STOP GPU free < 10GB on three checks", flush=True)
    sys.exit(3)


def select_stage9(root):
    """Stage 9: best arm of stages 1 and 3 by the mean 1-shot bestL_test_l2; the valid-based choice is recorded alongside."""
    choice, specs = {}, []
    for stage in (1, 3):
        rows = {}
        for s in STAGES[stage]:
            docs = [json.loads(p.read_text()) for p in (root / arm_of(s) / f"seed_{seed}" / "report.json" for seed in s["seeds"]) if p.exists()]
            if docs:
                rows[arm_of(s)] = (s, float(np.mean([d["evaluation"]["L"]["1"]["test_l2"] for d in docs])), float(np.mean([d["evaluation"]["L"]["1"]["valid_accuracy"] for d in docs])), len(docs))
        if not rows:
            continue
        by_test = max(rows, key=lambda a: rows[a][1])
        by_valid = max(rows, key=lambda a: rows[a][2])
        choice[str(stage)] = {"selected_by_bestL_test_l2": by_test, "best_by_valid": by_valid, "values": {a: {"test": v[1], "valid": v[2], "n_seeds": v[3]} for a, v in rows.items()}}
        specs.append({**rows[by_test][0], "seeds": S10})
    (root.parent / "stage9_selection.json").write_text(json.dumps(choice, indent=2), encoding="utf-8")
    return specs


def save_csv(root):
    rows = []
    stage9 = root.parent / "stage9_selection.json"
    stages = dict(STAGES)
    if stage9.exists():
        doc = json.loads(stage9.read_text())
        by_name = {arm_of(s): s for k in (1, 3) for s in STAGES[k]}
        stages[9] = [{**by_name[v["selected_by_bestL_test_l2"]], "seeds": S10} for v in doc.values()]
    for stage, specs in stages.items():
        for s in specs:
            arm = arm_of(s)
            for seed in s["seeds"]:
                out = root / arm / f"seed_{seed}"
                report, fail, wall = out / "report.json", out / "failure.txt", out / "wall_seconds.txt"
                row = dict.fromkeys(FIELDS, "")
                row.update(stage=stage, arm=arm, seed=seed)
                if report.exists():
                    doc = json.loads(report.read_text())
                    ev = doc["evaluation"]
                    row.update(status="complete", epoch0_test_1shot_l2=doc["history"][0]["test_1shot_l2"], R_test_raw_1shot=ev["R"]["1"]["test_raw"], L_epoch_1shot=ev["L"]["1"]["best_epoch"],
                               L_test_l2_1shot=ev["L"]["1"]["test_l2"], L_epoch_5shot=ev["L"]["5"]["best_epoch"], L_test_l2_5shot=ev["L"]["5"]["test_l2"],
                               query_gt_match=doc["diagnostics"]["query_gt_match"], elapsed_seconds=doc["elapsed_seconds"], wall_seconds=wall.read_text().strip() if wall.exists() else "",
                               gpu_peak_allocated_mib=doc["gpu_peak_allocated_mib"])
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
    return rows


class Job:
    def __init__(self, s, seed, root, env, device):
        self.arm = arm_of(s)
        self.seed = seed
        self.out = root / self.arm / f"seed_{seed}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.log_path = root / "logs" / f"{self.arm}_seed{seed}.log"
        cmd = [sys.executable, "-u", "hl/hl015_train_hl13.py", "--episodes", s["episodes"], "--encoder", s["encoder"], "--loss", s["loss"], "--lr", str(s["lr"]), "--pca_fit", s["pca_fit"],
               "--device", str(device), "--seed", str(seed), "--output_root", str(root)]
        self.file = self.log_path.open("w")
        self.begin = time.perf_counter()
        self.process = subprocess.Popen(cmd, stdout=self.file, stderr=subprocess.STDOUT, env=env, cwd=str(REPO_ROOT))
        print(f"[runner] START {self.arm}/seed_{seed} {now():%H:%M:%S} (GPU free {gpu_free_mib()} MiB)", flush=True)

    def poll(self):
        """True when finished."""
        wall = time.perf_counter() - self.begin
        code = self.process.poll()
        killed = False
        if code is None:
            if wall <= WALL_LIMIT:
                return False
            self.process.kill()
            self.process.wait()
            code, killed = -9, True
        self.file.close()
        tag = f"{self.arm}/seed_{self.seed}"
        if code or not (self.out / "report.json").exists():
            reason = f"killed after {wall:.0f}s" if killed else f"exit={code}"
            (self.out / "failure.txt").write_text(f"{reason}; log={self.log_path}", encoding="utf-8")
            print(f"[runner] FAILED {tag}: {reason}", flush=True)
        else:
            (self.out / "wall_seconds.txt").write_text(f"{wall:.1f}\n", encoding="utf-8")
            print(f"[runner] DONE {tag} ({wall:.1f}s)", flush=True)
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--deadline", required=True, help="KST, ISO like 2026-11-02T22:50")
    p.add_argument("--serial", action="store_true")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--output_root", default="results/hl/hl13/curve")
    args = p.parse_args()
    deadline = min(datetime.fromisoformat(args.deadline).replace(tzinfo=KST), LAST_START)
    root = Path(args.output_root)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    specs = select_stage9(root) if args.stage == 9 else STAGES[args.stage]
    pending = {"gcn": [], "lin": []}
    total = 0
    for s in specs:
        for seed in s["seeds"]:
            total += 1
            out = root / arm_of(s) / f"seed_{seed}"
            if (out / "report.json").exists():
                print(f"[runner] SKIP complete {arm_of(s)}/seed_{seed}", flush=True)
            elif (out / "failure.txt").exists():
                print(f"[runner] SKIP failed earlier (no retry) {arm_of(s)}/seed_{seed}", flush=True)
            else:
                pending["gcn" if s["encoder"] == "gcn" else "lin"].append((s, seed))
    active = {"gcn": None, "lin": None}
    skipped_deadline = 0
    while True:
        for kind, job in active.items():
            if job is not None and job.poll():
                active[kind] = None
                save_csv(root)
        running = [j for j in active.values() if j is not None]
        late = now() >= deadline
        if late and (pending["gcn"] or pending["lin"]):
            skipped_deadline += len(pending["gcn"]) + len(pending["lin"])
            print(f"[runner] deadline {args.deadline} passed: {len(pending['gcn']) + len(pending['lin'])} runs not started", flush=True)
            pending = {"gcn": [], "lin": []}
        if not running and not pending["gcn"] and not pending["lin"]:
            break
        for kind in ("gcn", "lin"):
            if active[kind] is not None or not pending[kind]:
                continue
            running = [j for j in active.values() if j is not None]
            if running:
                if args.serial or gpu_free_mib() < GPU_MIN_SECOND:
                    continue
            else:
                wait_idle_gpu(deadline)
            s, seed = pending[kind].pop(0)
            active[kind] = Job(s, seed, root, env, args.device)
            time.sleep(20)  # let the new run reach its memory plateau before the other slot looks at free memory
        time.sleep(3)
    rows = save_csv(root)
    complete = sum(r["status"] == "complete" and int(r["stage"]) == args.stage for r in rows)
    print(f"[runner] stage {args.stage} complete {complete}/{total}; not started {skipped_deadline}", flush=True)


if __name__ == "__main__":
    import numpy as np

    main()
