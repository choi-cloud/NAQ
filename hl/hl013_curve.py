"""hl013_curve: copied from choi_exps/exp049.py: main; adds epoch 0 evaluation, test trajectory and row-normalised (_l2) evaluation. Training, selection and final test unchanged.

Epoch 0 is recorded in history but is not a checkpoint candidate (best starts empty at the first training eval, as in exp049).
Trajectory evaluation only runs network.eval() forward passes, with no RNG draw in this script; the task pool matched the p0 exp049 run, while train_loss differed from the 3rd decimal (reports/HL11_no_training_baseline.md ④).
Labels: query_diagnostics (exp049, diagnostics only) and the y_support / y_query of fixed valid/test episodes. Query selection and training read no labels.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hl_common  # noqa: E402,F401  (puts choi_exps on sys.path)
from data import load_data  # noqa: E402
from exp027 import episode_accuracy, evaluate  # noqa: E402
from exp049 import build_structural_queries, load_splits, query_diagnostics  # noqa: E402
from meta import ProtoNet  # noqa: E402
from task_generator import proto_task_generator  # noqa: E402
from utils import seed_everything  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Amazon_clothing")
    p.add_argument("--method", choices=["naq", "ours"], required=True)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--epoch", type=int, default=2000)
    p.add_argument("--eval_interval", type=int, default=160)
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=1)
    p.add_argument("--k_shot_test", type=int, default=1)
    p.add_argument("--q_query", type=int, default=10)
    p.add_argument("--q_query_test", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--latent", type=int, default=256)
    p.add_argument("--l2_penalty", type=float, default=1e-4)
    p.add_argument("--num_valid_tasks", type=int, default=50)
    p.add_argument("--num_downstream_tasks", type=int, default=500)
    p.add_argument("--mine_batch_size", type=int, default=64)
    p.add_argument("--min_candidate_policy", choices=["relax"], default="relax")
    p.add_argument("--output_root", default="results/hl/hl11/curve")
    return p.parse_args()


def pool_mean(z, episodes, args, shot):
    return float(np.mean([episode_accuracy(z, task, args.n_way, shot) for task in episodes]))


def trajectory_columns(meta, data, splits, args):
    """test_{shot}shot and *_l2 columns; valid_{shot}shot (raw) stays with exp027.evaluate in the caller."""
    meta.network.eval()
    row = {}
    with torch.no_grad():
        raw = meta.network(data.features, adj=data.adj)
        l2 = F.normalize(raw, dim=1)
        for shot in (1, 5):
            valid, test = splits[shot][0], splits[shot][1]
            row[f"test_{shot}shot"] = pool_mean(raw, test, args, shot)
            row[f"valid_{shot}shot_l2"] = pool_mean(l2, valid, args, shot)
            row[f"test_{shot}shot_l2"] = pool_mean(l2, test, args, shot)
    return row


def main():
    args = parse_args()
    if args.method == "ours" and (args.tau is None or not -1 <= args.tau <= 1):
        raise ValueError("Ours requires --tau in [-1,1]")
    if args.method == "naq" and args.tau is not None:
        raise ValueError("NAQ must not receive --tau")
    if args.k_shot != 1 or args.k_shot_test != 1 or args.n_way != 5:
        raise ValueError("Training remains 5-way 1-shot; evaluation separately tests 1 and 5 shot")
    if args.epoch <= 0 or args.eval_interval <= 0:
        raise ValueError("Invalid training schedule")
    args.setting, args.type, args.query_generation = "unsup", "feature", "NaQ"
    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    splits = load_splits(args)
    out = Path(args.output_root) / args.method / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    if report_path.exists():
        raise FileExistsError(f"Existing result cannot be overwritten: {report_path}")

    start = time.perf_counter()
    print(f"[hl013] dataset={args.dataset} method={args.method} tau={args.tau} loading", flush=True)
    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    data.add_diffusion(args)
    config = [
        ("GCN", (data.features.shape[1], 2 * args.latent), True),
        ("GCN", (2 * args.latent, args.latent), True),
        ("linear", (args.latent, args.n_way), True),
    ]
    meta = ProtoNet(args, config, data).to(device)
    original = [proto_task_generator(args, data) for _ in range(args.epoch)]
    if args.method == "ours":
        train_pool, details = build_structural_queries(data, original, args, device)
    else:
        train_pool, details = original, None
    if len(train_pool) != len(original):
        raise RuntimeError("Training episode count changed")
    diag = query_diagnostics(original, train_pool, data, details, args.q_query)
    print(f"[hl013] query diagnostics: {diag}", flush=True)
    del original

    best = {shot: {"accuracy": -1.0, "epoch": None} for shot in (1, 5)}
    history = []

    # Epoch 0: random initialisation, recorded only.
    row = {"epoch": 0, "train_loss": None, "train_acc": None}
    for shot in (1, 5):
        args.k_shot_test = shot
        row[f"valid_{shot}shot"] = evaluate(meta, data, splits[shot][0], args)["accuracy"]
    row.update(trajectory_columns(meta, data, splits, args))
    history.append(row)
    print(f"[hl013] epoch=0 valid1={row['valid_1shot']:.4%} test1={row['test_1shot']:.4%} test1_l2={row['test_1shot_l2']:.4%}", flush=True)

    for epoch, (id_spt, y_spt, id_query, y_query) in enumerate(train_pool, 1):
        args.k_shot_test = 1
        meta.network.train()
        loss, train_acc = meta(id_spt, y_spt.to(device), id_query, y_query.to(device))
        if epoch % args.eval_interval != 0 and epoch != args.epoch:
            continue
        row = {"epoch": epoch, "train_loss": float(loss.detach()), "train_acc": float(train_acc)}
        for shot in (1, 5):
            args.k_shot_test = shot
            valid = evaluate(meta, data, splits[shot][0], args)
            row[f"valid_{shot}shot"] = valid["accuracy"]
            if valid["accuracy"] > best[shot]["accuracy"]:
                best[shot] = {"accuracy": valid["accuracy"], "epoch": epoch}
                torch.save({"epoch": epoch, "model": meta.network.state_dict(), "config": vars(args).copy()}, out / f"best_{shot}shot.pt")
        row.update(trajectory_columns(meta, data, splits, args))
        history.append(row)
        print(f"[hl013] epoch={epoch} valid1={row['valid_1shot']:.4%} valid5={row['valid_5shot']:.4%} test1={row['test_1shot']:.4%} test1_l2={row['test_1shot_l2']:.4%}", flush=True)

    evaluations = {}
    for shot in (1, 5):
        state = torch.load(out / f"best_{shot}shot.pt", map_location="cpu", weights_only=True)
        meta.network.load_state_dict(state["model"])
        args.k_shot_test = shot
        test = evaluate(meta, data, splits[shot][1], args, with_episodes=True)
        evaluations[str(shot)] = {
            "best_epoch": best[shot]["epoch"],
            "valid_accuracy": best[shot]["accuracy"],
            "test_accuracy": test["accuracy"],
            "test_ci95_half_width": test["ci95_half_width"],
            "test_episode_accuracies": test["episode_accuracies"],
            "valid_episode_file": splits[shot][2],
            "test_episode_file": splits[shot][3],
            "checkpoint": str(out / f"best_{shot}shot.pt"),
        }
        print(f"[hl013] {shot}-shot test={test['accuracy']:.4%} ± {test['ci95_half_width']:.4%}", flush=True)
    report = {
        "experiment": "hl013 (exp049 + curve)",
        "args": vars(args).copy(),
        "dataset": args.dataset,
        "method": args.method,
        "tau": args.tau,
        "seed": args.seed,
        "diagnostics": diag,
        "evaluation": evaluations,
        "history": history,
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "gpu_peak_allocated_mib": int(torch.cuda.max_memory_allocated() // 2**20) if device.type == "cuda" else None,
    }
    tmp = out / "report.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(report_path)
    print(f"[hl013] DONE {report_path}", flush=True)


if __name__ == "__main__":
    main()
