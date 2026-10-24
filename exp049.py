"""exp049: one NAQ-style training per dataset/method/tau, evaluate both shots.

Needs the existing NAQ repository, exp026.py and exp027.py.
Ground truth is read only for diagnostics; never for pseudo-query selection.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from data import load_data
from exp026 import load_fixed_episodes
from exp027 import evaluate
from meta import ProtoNet
from task_generator import proto_task_generator
from utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
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
    p.add_argument("--output_root", default="results/gfsl_benchmark_exp049")
    return p.parse_args()


def episode_paths(args, shot):
    d = Path("save/fixed_episodes") / args.dataset / (
        f"{args.n_way}way_{shot}shot_{args.q_query_test}query_seed{args.seed}"
    )
    return d / "valid_episodes.json", d / "test_episodes.json"


def load_splits(args):
    result = {}
    for shot in (1, 5):
        valid_path, test_path = episode_paths(args, shot)
        valid = load_fixed_episodes(
            valid_path, dataset=args.dataset, n_way=args.n_way,
            k_shot=shot, q_query=args.q_query_test
        )
        test = load_fixed_episodes(
            test_path, dataset=args.dataset, n_way=args.n_way,
            k_shot=shot, q_query=args.q_query_test
        )
        if len(valid) != args.num_valid_tasks or len(test) != args.num_downstream_tasks:
            raise ValueError(f"Episode count mismatch dataset={args.dataset} shot={shot}")
        result[shot] = (valid, test, str(valid_path), str(test_path))
    return result


def query_diagnostics(original, modified, data, details, q):
    labels = torch.as_tensor(data.labels).detach().cpu().numpy().reshape(-1)
    original_hits = []
    selected_hits = []
    ranks = []
    fallbacks = []
    for old, new in zip(original, modified):
        supports = np.asarray(old[0]).reshape(-1).astype(int)
        old_queries = np.asarray(old[2]).reshape(-1).astype(int)
        new_queries = np.asarray(new[2]).reshape(-1).astype(int)
        expected = np.repeat(labels[supports], q)
        original_hits.extend((labels[old_queries] == expected).tolist())
        selected_hits.extend((labels[new_queries] == expected).tolist())
        if details is not None:
            for s in supports:
                d = details[int(s)]
                ranks.append(d["mean_original_x_rank"])
                fallbacks.append(bool(d["fallback"]))
    return {
        "original_gt_match": float(np.mean(original_hits)),
        "query_gt_match": float(np.mean(selected_hits)),
        "original_x_rank": float(np.mean(ranks)) if ranks else None,
        "coverage_without_fallback": float(np.mean([not d["fallback"] for d in details.values()])) if details else 1.0,
        "fallback_occurrence_rate": float(np.mean(fallbacks)) if fallbacks else 0.0
    }


def build_structural_queries(data, tasks, args, device):
    """Select queries using ONLY features/edges; labels never accessed here."""
    import scipy.sparse as sp
    import torch.nn.functional as F

    t0 = time.perf_counter()
    print("[exp048] Precomputing structural representations", flush=True)
    edges = data.edge_index.detach().cpu().numpy()
    x = data.features.detach().float().to(device)
    n = x.size(0)
    adjacency = sp.csr_matrix(
        (np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])),
        shape=(n, n)
    )
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.float32).tocsr()
    adjacency.setdiag(1)
    adjacency.eliminate_zeros()
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    scale = np.maximum(degrees, 1) ** -0.5
    adjacency = (sp.diags(scale) @ adjacency @ sp.diags(scale)).tocoo()
    indices = torch.as_tensor(
        np.stack((adjacency.row, adjacency.col)), dtype=torch.long, device=device
    )
    values = torch.as_tensor(adjacency.data, dtype=torch.float32, device=device)
    a = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    h1 = torch.sparse.mm(a, x)
    h2 = torch.sparse.mm(a, h1)
    z0, z1, z2 = (F.normalize(t, dim=1) for t in (x, h1, h2))
    all_supports = np.concatenate([
        np.asarray(t[0], dtype=np.int64).reshape(-1) for t in tasks
    ])
    unique = np.unique(all_supports)
    print(f"[exp048] Mining {len(unique)} unique supports", flush=True)
    if not -1 <= args.tau <= 1:
        raise ValueError("tau must be in [-1, 1]")
    cache = {}
    diagnostics = {}
    all_ids = np.arange(n)
    for begin in range(0, len(unique), args.mine_batch_size):
        ids_np = unique[begin:begin + args.mine_batch_size]
        ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)
        with torch.no_grad():
            scores0 = (z0[ids] @ z0.T).cpu().numpy()
            scores1 = (z1[ids] @ z1.T).cpu().numpy()
            scores2 = (z2[ids] @ z2.T).cpu().numpy()
        for j, support in enumerate(ids_np):
            candidates = all_ids[all_ids != support]
            s0, s1, s2 = (scores0[j, candidates], scores1[j, candidates], scores2[j, candidates])
            # exp047: keep iff BOTH cosine similarities meet tau.
            kept = np.flatnonzero((s1 >= args.tau) & (s2 >= args.tau))
            raw_retained = len(kept)
            fallback = raw_retained < args.q_query
            if fallback:
                if args.min_candidate_policy == "skip":
                    # Unusable for fixed-size NAQ episodes; caller explicitly selected skip.
                    cache[int(support)] = None
                    diagnostics[int(support)] = {
                        "eligible": False,
                        "raw_retained_count": raw_retained,
                        "fallback": False
                    }
                    continue
                # Adaptive tau relaxation: add strongest excluded pairs by
                # min(cos_AX, cos_A2X) until q_query candidates exist.
                joint = np.minimum(s1, s2)
                order = np.argsort(-joint, kind="stable")
                kept = order[:args.q_query]
            kept_by_x = kept[np.argsort(-s0[kept], kind="stable")]
            mid = (len(kept_by_x) - args.q_query) // 2
            chosen = kept_by_x[mid:mid + args.q_query]
            cache[int(support)] = candidates[chosen].copy()
            # Global feature ranks for comparability to exp047.
            x_order = np.argsort(-s0, kind="stable")
            x_rank = np.empty(len(x_order), dtype=np.int32)
            x_rank[x_order] = np.arange(1, len(x_order) + 1)
            diagnostics[int(support)] = {
                "eligible": not fallback,
                "fallback": bool(fallback),
                "raw_retained_count": int(raw_retained),
                "mean_original_x_rank": float(x_rank[chosen].mean()),
                "mean_x_cosine": float(s0[chosen].mean()),
                "mean_ax_cosine": float(s1[chosen].mean()),
                "mean_a2x_cosine": float(s2[chosen].mean()),
                "selected_ids": candidates[chosen].tolist(),
                "retained_count": int(len(kept))
            }
        if begin == 0 or begin + args.mine_batch_size >= len(unique) or (begin // args.mine_batch_size + 1) % 10 == 0:
            print(f"[exp048] Mined {min(begin + len(ids_np), len(unique))}/{len(unique)} supports", flush=True)
    del z0, z1, z2, a, h1, h2
    print(f"[exp048] Mining done in {time.perf_counter() - t0:.1f}s", flush=True)
    modified = []
    skipped = 0
    for task in tasks:
        supports, ys, queries, yq = task
        supports = np.asarray(supports, dtype=np.int64).reshape(-1)
        if any(cache[int(s)] is None for s in supports):
            skipped += 1
            continue
        selected = np.concatenate([cache[int(s)] for s in supports])
        if len(selected) != len(np.asarray(queries).reshape(-1)):
            raise ValueError("Query count mismatch")
        modified.append((supports, ys, selected, yq))
    if skipped:
        print(f"[exp048] skipped {skipped}/{len(tasks)} episodes due to insufficient candidates", flush=True)
    return modified, diagnostics

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
    tau_name = f"tau_{args.tau:g}" if args.method == "ours" else "baseline"
    out = Path(args.output_root) / args.dataset / args.method / tau_name / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    if report_path.exists():
        raise FileExistsError(f"Existing result cannot be overwritten: {report_path}")

    start = time.perf_counter()
    print(f"[exp049] dataset={args.dataset} method={args.method} tau={args.tau} loading", flush=True)
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
    original = [proto_task_generator(args, data) for _ in range(args.epoch)]
    if args.method == "ours":
        train_pool, details = build_structural_queries(data, original, args, device)
    else:
        train_pool, details = original, None
    if len(train_pool) != len(original):
        raise RuntimeError("Training episode count changed")
    diag = query_diagnostics(original, train_pool, data, details, args.q_query)
    print(f"[exp049] query diagnostics: {diag}", flush=True)
    del original

    # Both shot settings share one training trajectory. Each selects its own
    # checkpoint using its own FIXED validation episodes (not test episodes).
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
        print(f"[exp049] epoch={epoch} valid1={row['valid_1shot']:.4%} valid5={row['valid_5shot']:.4%}", flush=True)

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
        print(f"[exp049] {shot}-shot test={test['accuracy']:.4%} ± {test['ci95_half_width']:.4%}", flush=True)
    report = {
        "experiment": "exp049",
        "args": vars(args).copy(),
        "dataset": args.dataset,
        "method": args.method,
        "tau": args.tau,
        "seed": args.seed,
        "diagnostics": diag,
        "evaluation": evaluations,
        "history": history,
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "note": "One fixed 1-shot pseudo-training per setting; validation selects separate checkpoints for 1- and 5-shot. Labels never enter query mining."
    }
    tmp = out / "report.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(report_path)
    print(f"[exp049] DONE {report_path}", flush=True)


if __name__ == "__main__":
    main()
