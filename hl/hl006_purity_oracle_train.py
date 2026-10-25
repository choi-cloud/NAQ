"""hl006_purity_oracle_train: copied from hl/hl003_oracle_train.py (itself from choi_exps/exp049.py); anchors stay uniform, only query selection is replaced by purity_oracle_queries. Reads GT labels for oracle query selection.

Queries: for each anchor, 10 queries from its X-rank band B4 (201-1000), round(10p) GT-same and the rest GT-diff, each
taken from the centre of its X-rank-ordered list; GT-same shortfall is filled from B5 (1001-3000) GT-same in X-rank order,
then from B4 GT-diff. Training, evaluation and report.json follow hl003/exp049 unchanged; report.json additionally carries
`purity_p` and `query_diagnostics` (see instructions/HL03_distance_purity.md §2).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hl_common as H  # noqa: E402
from data import load_data  # noqa: E402
from exp027 import evaluate  # noqa: E402
from exp049 import load_splits, query_diagnostics  # noqa: E402
from hl003_oracle_train import anchor_diagnostics, read_column  # noqa: E402
from meta import ProtoNet  # noqa: E402
from task_generator import proto_task_generator  # noqa: E402
from utils import seed_everything  # noqa: E402

B4 = (201, 1000)
B5 = (1001, 3000)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--tau", type=float, default=0.3)
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
    p.add_argument("--output_root", default="results/hl/hl03/g1")
    # hl006 additions
    p.add_argument("--p", type=float, required=True)
    p.add_argument("--node_table", default="results/hl/hl02/diag/node_table_tau0.3.csv")
    return p.parse_args()


def centre(ids, k):
    """k items from the centre of an ordered list (exp049 middle-K convention)."""
    mid = (len(ids) - k) // 2
    return ids[mid:mid + k]


def purity_oracle_queries(anchor, p, ranked, labels, q):
    """`ranked`: candidate ids of `anchor` in X-rank order (rank 1 first, self excluded). Oracle: uses GT `labels`."""
    b4 = ranked[B4[0] - 1:B4[1]]
    same_b4 = b4[labels[b4] == labels[anchor]]
    diff_b4 = b4[labels[b4] != labels[anchor]]
    n_same = int(round(q * p))
    same = centre(same_b4, min(n_same, len(same_b4)))
    from_b5 = np.zeros(0, dtype=np.int64)
    if len(same) < n_same:
        b5 = ranked[B5[0] - 1:B5[1]]
        from_b5 = b5[labels[b5] == labels[anchor]][: n_same - len(same)]
    n_diff = q - len(same) - len(from_b5)
    chosen = np.concatenate([same, from_b5, centre(diff_b4, n_diff)])
    return chosen, {
        "n_same_target": n_same,
        "n_same_b4": int(len(same)),
        "n_same_b5": int(len(from_b5)),
        "n_diff_fill": int(n_same - len(same) - len(from_b5)),
    }


def build_purity_queries(data, tasks, p, labels, args, device):
    """Replaces exp049.build_structural_queries: same per-anchor cache and episode layout; returns (tasks, details)."""
    unique = np.unique(np.concatenate([np.asarray(t[0], dtype=np.int64).reshape(-1) for t in tasks]))
    depth = B5[1]
    cache, details = {}, {}
    dm = data.Diffusion_matrix
    for begin in range(0, len(unique), args.mine_batch_size):
        ids_np = unique[begin:begin + args.mine_batch_size]
        ids = torch.as_tensor(ids_np, dtype=torch.long, device=dm.device)
        with torch.no_grad():
            rows = dm[ids].clone()
            rows[torch.arange(len(ids), device=dm.device), ids] = -float("inf")
            order = torch.sort(rows, dim=1, descending=True, stable=True).indices.cpu().numpy()
        for j, a in enumerate(ids_np):
            rank = np.empty(order.shape[1], dtype=np.int64)
            rank[order[j]] = np.arange(1, order.shape[1] + 1)
            chosen, st = purity_oracle_queries(int(a), p, order[j][:depth], labels, args.q_query)
            cache[int(a)] = chosen
            details[int(a)] = {
                **st,
                "fallback": bool(st["n_same_b5"] or st["n_diff_fill"]),
                "mean_original_x_rank": float(rank[chosen].mean()),
                "purity": float((labels[chosen] == labels[a]).mean()),
            }
    modified = []
    for supports, ys, queries, yq in tasks:
        supports = np.asarray(supports, dtype=np.int64).reshape(-1)
        selected = np.concatenate([cache[int(s)] for s in supports])
        if len(selected) != len(np.asarray(queries).reshape(-1)):
            raise ValueError("Query count mismatch")
        modified.append((supports, ys, selected, yq))
    return modified, details


def purity_diagnostics(tasks, details, q):
    """Per anchor occurrence (as exp049.query_diagnostics counts them)."""
    rows = [details[int(s)] for t in tasks for s in np.asarray(t[0]).reshape(-1)]
    return {
        "mean_purity": float(np.mean([r["purity"] for r in rows])),
        "mean_x_rank": float(np.mean([r["mean_original_x_rank"] for r in rows])),
        "supplement_rate_anchor": float(np.mean([r["fallback"] for r in rows])),
        "b5_same_query_frac": float(np.sum([r["n_same_b5"] for r in rows]) / (q * len(rows))),
        "diff_fill_query_frac": float(np.sum([r["n_diff_fill"] for r in rows]) / (q * len(rows))),
        "supplement_query_frac": float(np.sum([r["n_same_b5"] + r["n_diff_fill"] for r in rows]) / (q * len(rows))),
    }


def main():
    args = parse_args()
    if not 0 <= args.p <= 1:
        raise ValueError("Requires --p in [0,1]")
    if args.k_shot != 1 or args.k_shot_test != 1 or args.n_way != 5:
        raise ValueError("Training remains 5-way 1-shot; evaluation separately tests 1 and 5 shot")
    if args.epoch <= 0 or args.eval_interval <= 0:
        raise ValueError("Invalid training schedule")
    args.policy = "uniform"
    args.method = "ours"
    args.setting, args.type, args.query_generation = "unsup", "feature", "NaQ"
    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    splits = load_splits(args)
    out = Path(args.output_root) / f"p_{args.p:g}" / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    if report_path.exists():
        raise FileExistsError(f"Existing result cannot be overwritten: {report_path}")

    start = time.perf_counter()
    print(f"[hl006] dataset={args.dataset} p={args.p} loading", flush=True)
    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    data.add_diffusion(args)
    config = [
        ("GCN", (data.features.shape[1], 2 * args.latent), True),
        ("GCN", (2 * args.latent, args.latent), True),
        ("linear", (args.latent, args.n_way), True)
    ]
    meta = ProtoNet(args, config, data).to(device)

    # anchors: hl003 `uniform` path unchanged (same python RNG state)
    labels = H.labels_np(data)
    pi_bchoi = read_column(args.node_table, "pi_bchoi", float)
    z2 = H.structural_scores(data, device)[2]
    original = [proto_task_generator(args, data) for _ in range(args.epoch)]
    stats = [{"k_eff": None, "fallback": False} for _ in original]
    anchor_diag = anchor_diagnostics(original, stats, labels, pi_bchoi, data, z2, args.n_way)
    print(f"[hl006] anchor diagnostics: {anchor_diag}", flush=True)
    del z2
    torch.cuda.empty_cache()

    # ---- the only change vs hl003 uniform: query selection (oracle, GT labels) ----
    train_pool, details = build_purity_queries(data, original, args.p, labels, args, device)
    if len(train_pool) != len(original):
        raise RuntimeError("Training episode count changed")
    diag = query_diagnostics(original, train_pool, data, details, args.q_query)
    qdiag = purity_diagnostics(original, details, args.q_query)
    print(f"[hl006] query diagnostics: {diag} {qdiag}", flush=True)
    del original
    # -------------------------------------------------------------------------------

    best = {shot: {"accuracy": -1.0, "epoch": None} for shot in (1, 5)}
    history = []
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
        history.append(row)
        print(f"[hl006] epoch={epoch} valid1={row['valid_1shot']:.4%} valid5={row['valid_5shot']:.4%}", flush=True)

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
            "checkpoint": str(out / f"best_{shot}shot.pt")
        }
        print(f"[hl006] {shot}-shot test={test['accuracy']:.4%} ± {test['ci95_half_width']:.4%}", flush=True)
    report = {
        "experiment": "hl006",
        "args": vars(args).copy(),
        "dataset": args.dataset,
        "method": "ours",
        "policy": "uniform",
        "purity_p": args.p,
        "tau": args.tau,
        "seed": args.seed,
        "diagnostics": diag,
        "query_diagnostics": qdiag,
        "anchor_diagnostics": anchor_diag,
        "evaluation": evaluations,
        "history": history,
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "max_cuda_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2) if device.type == "cuda" else None,
        "note": "hl006: hl003 uniform anchors with purity-at-distance oracle queries; GT labels used only for query selection and diagnostics."
    }
    tmp = out / "report.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(report_path)
    print(f"[hl006] DONE {report_path}", flush=True)


if __name__ == "__main__":
    main()
