"""exp048: AX/A2X threshold filtering + X Middle-K NAQ training.

Changes ONLY training pseudo-query IDs from original NAQ. Supports, pseudo labels,
ProtoNet training, fixed validation/test episodes remain as exp027/exp038.
Class labels are accessed ONLY after sampling for diagnostic reporting.
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

SELECTION_MODE = "structural_threshold_then_middle"


def parse_args():
    parser = argparse.ArgumentParser(description="exp048: structure-first filtered Middle-K NAQ")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", default="Amazon_clothing")
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--k_shot_test", type=int, default=1)
    parser.add_argument("--q_query", type=int, default=10)
    parser.add_argument("--q_query_test", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent", type=int, default=256)
    parser.add_argument("--l2_penalty", type=float, default=1e-4)
    parser.add_argument("--query_generation", default="NaQ")
    parser.add_argument("--type", default="feature")
    parser.add_argument("--num_valid_tasks", type=int, default=50)
    parser.add_argument("--num_downstream_tasks", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--episode_dir", default=None)
    parser.add_argument("--json_path", default="results/Amazon_clothing/exp048/exp048.json")
    parser.add_argument("--sample_episodes", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--min_candidate_policy", choices=["relax", "skip"], default="relax")
    parser.add_argument("--mine_batch_size", type=int, default=64)
    return parser.parse_args()


def build_structural_queries(data, tasks, args, device):
    """Select queries using ONLY features/edges; labels never accessed here."""
    import scipy.sparse as sp
    import torch.nn.functional as F

    t0 = time.perf_counter()
    print("[exp048] Precomputing structural representations", flush=True)
    edges = data.edge_index.detach().cpu().numpy()
    x = data.features.detach().float().to(device)
    n = x.size(0)
    adjacency = sp.csr_matrix((np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])), shape=(n, n))
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.float32).tocsr()
    adjacency.setdiag(1)
    adjacency.eliminate_zeros()
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    scale = np.maximum(degrees, 1) ** -0.5
    adjacency = (sp.diags(scale) @ adjacency @ sp.diags(scale)).tocoo()
    indices = torch.as_tensor(np.stack((adjacency.row, adjacency.col)), dtype=torch.long, device=device)
    values = torch.as_tensor(adjacency.data, dtype=torch.float32, device=device)
    a = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    h1 = torch.sparse.mm(a, x)
    h2 = torch.sparse.mm(a, h1)
    z0, z1, z2 = (F.normalize(t, dim=1) for t in (x, h1, h2))
    all_supports = np.concatenate([np.asarray(t[0], dtype=np.int64).reshape(-1) for t in tasks])
    unique = np.unique(all_supports)
    print(f"[exp048] Mining {len(unique)} unique supports", flush=True)
    if not -1 <= args.tau <= 1:
        raise ValueError("tau must be in [-1, 1]")
    cache = {}
    diagnostics = {}
    all_ids = np.arange(n)
    for begin in range(0, len(unique), args.mine_batch_size):
        ids_np = unique[begin : begin + args.mine_batch_size]
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
                    diagnostics[int(support)] = {"eligible": False, "raw_retained_count": raw_retained, "fallback": False}
                    continue
                # Adaptive tau relaxation: add strongest excluded pairs by
                # min(cos_AX, cos_A2X) until q_query candidates exist.
                joint = np.minimum(s1, s2)
                order = np.argsort(-joint, kind="stable")
                kept = order[: args.q_query]
            kept_by_x = kept[np.argsort(-s0[kept], kind="stable")]

            selection = "middle"

            if selection == "top":
                chosen = kept_by_x[: args.q_query]
            elif selection == "middle":
                mid = (len(kept_by_x) - args.q_query) // 2
                chosen = kept_by_x[mid : mid + args.q_query]
            elif selection == "bottom":
                chosen = kept_by_x[-args.q_query :]
            else:
                raise ValueError(f"Unknown selection: {selection}")
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
                "retained_count": int(len(kept)),
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
    args.setting = "unsup"
    if args.query_generation != "NaQ" or args.type != "feature":
        raise ValueError("exp044 is restricted to original NAQ-Feat")
    if args.k_shot != 1 or args.k_shot_test != 1:
        raise ValueError("exp044 currently supports only 1-shot")
    if args.epoch <= 0 or args.eval_interval <= 0 or args.q_query <= 0 or not -1 <= args.tau <= 1:
        raise ValueError("Invalid episode/training parameters")
    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    episode_dir = Path(
        args.episode_dir
        or (f"save/fixed_episodes/{args.dataset}/" f"{args.n_way}way_{args.k_shot_test}shot_{args.q_query_test}query_seed{args.seed}")
    )
    valid_path = episode_dir / "valid_episodes.json"
    test_path = episode_dir / "test_episodes.json"
    valid_episodes = load_fixed_episodes(valid_path, dataset=args.dataset, n_way=args.n_way, k_shot=args.k_shot_test, q_query=args.q_query_test)
    test_episodes = load_fixed_episodes(test_path, dataset=args.dataset, n_way=args.n_way, k_shot=args.k_shot_test, q_query=args.q_query_test)
    if len(valid_episodes) != args.num_valid_tasks or len(test_episodes) != args.num_downstream_tasks:
        raise ValueError("Fixed evaluation episode counts do not match")

    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    data.add_diffusion(args)
    diffusion = data.Diffusion_matrix
    if diffusion.layout != torch.strided:
        # Original NAQ sparse topk isn't used for full rank. Dense row
        # conversion in oracle_refine_episode works for sparse matrices.
        diffusion = diffusion.to_dense()
    config = [
        ("GCN", (data.features.shape[1], 2 * args.latent), True),
        ("GCN", (2 * args.latent, args.latent), True),
        ("linear", (args.latent, args.n_way), True),
    ]
    meta = ProtoNet(args, config, data).to(device)

    # Generate original NAQ episodes first. Pseudo-labels remain unchanged.
    original_pool = [proto_task_generator(args, data) for _ in range(args.epoch)]
    train_pool, diagnostics = build_structural_queries(data, original_pool, args, device)
    if len(train_pool) != len(original_pool):
        raise ValueError("Skip policy changes training episode count. Use --min_candidate_policy relax for fixed NAQ training.")
    output_dir = Path(f"results/{args.dataset}/exp048/tau_{args.tau:g}")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    args.json_path = str(output_dir / "exp048.json")
    best_valid = float("-inf")
    best_epoch = None
    trajectory = []
    # Label diagnostics are computed only AFTER every training query has been chosen.
    labels = torch.as_tensor(data.labels).detach().cpu().numpy().reshape(-1)
    original_gt = []
    middle_gt = []
    all_rank = []
    all_x_cos = []
    fallback_supports = {s for s, d in diagnostics.items() if d["fallback"]}
    queried_supports = []
    for original_task, modified_task in zip(original_pool, train_pool):
        supports = np.asarray(original_task[0], dtype=np.int64).reshape(-1)
        original_queries = np.asarray(original_task[2], dtype=np.int64).reshape(-1)
        middle_queries = np.asarray(modified_task[2], dtype=np.int64).reshape(-1)
        expected = np.repeat(labels[supports], args.q_query)
        original_gt.extend((labels[original_queries] == expected).tolist())
        middle_gt.extend((labels[middle_queries] == expected).tolist())
        for support in supports:
            queried_supports.append(int(support))
            detail = diagnostics[int(support)]
            all_rank.append(detail["mean_original_x_rank"])
            all_x_cos.append(detail["mean_x_cosine"])
    del original_pool
    original_match = float(np.mean(original_gt))
    middle_match = float(np.mean(middle_gt))
    print(
        f"[exp048] tau={args.tau:g} | "
        f"NAQ GT={original_match:.2%} | Middle GT={middle_match:.2%} | "
        f"Original X rank={np.mean(all_rank):.1f} | X cosine={np.mean(all_x_cos):.4f}",
        flush=True,
    )
    start = time.time()
    for epoch, (id_spt, y_spt, id_query, y_query) in enumerate(train_pool, 1):
        meta.network.train()
        loss, train_acc = meta(id_spt, y_spt.to(device), id_query, y_query.to(device))
        if epoch % args.eval_interval != 0 and epoch != args.epoch:
            continue
        valid = evaluate(meta, data, valid_episodes, args)
        row = {"epoch": epoch, "train_loss": float(loss.detach()), "train_accuracy": float(train_acc), "valid": valid}
        trajectory.append(row)
        if valid["accuracy"] > best_valid:
            best_valid = valid["accuracy"]
            best_epoch = epoch
            torch.save({"epoch": epoch, "model": meta.network.state_dict(), "config": vars(args)}, best_path)
        print(
            f"[exp048][Epoch {epoch}] loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} valid={valid['accuracy']:.4f} "
            f"best={best_valid:.4f}@{best_epoch}",
            flush=True,
        )

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    meta.network.load_state_dict(checkpoint["model"])
    test = evaluate(meta, data, test_episodes, args, with_episodes=True)
    report = {
        "experiment": "exp048",
        "description": "AX/A2X threshold (AND) on all nodes, then X-middle-10 NAQ training",
        "config": vars(args).copy(),
        "protocol": {
            "label_usage": "GT ONLY for post-sampling diagnostic; never used to select train queries",
            "support_sampling": "unchanged original NAQ",
            "query_ranking": "exp047 cosine_AX>=tau AND cosine_A2X>=tau, then original X cosine",
            "query_selection": "middle-q of X-sorted survivors after structural rejection",
            "ranking_scope": "all_nodes excluding support",
            "filtering": "threshold AND; relax only if fewer than q candidates",
            "fallback_policy": args.min_candidate_policy,
            "pseudo_labels": "unchanged original NAQ labels",
            "train": "original meta.ProtoNet.forward unchanged",
            "evaluation": "RAW Euclidean ProtoNet on untouched exp026 fixed episodes",
            "selection": "best fixed validation accuracy",
            "valid_file": str(valid_path),
            "test_file": str(test_path),
        },
        "query_diagnostics": {
            "num_queries": len(middle_gt),
            "original_gt_match_rate": original_match,
            "middle_gt_match_rate": middle_match,
            "mean_original_x_rank": float(np.mean(all_rank)),
            "eligible_unique_supports": len(diagnostics) - len(fallback_supports),
            "total_unique_supports": len(diagnostics),
            "coverage_without_fallback": (len(diagnostics) - len(fallback_supports)) / len(diagnostics),
            "fallback_query_occurrence_rate": sum(s in fallback_supports for s in queried_supports) / len(queried_supports),
            "raw_retained_count_mean": float(np.mean([d["raw_retained_count"] for d in diagnostics.values()])),
            "mean_selected_x_cosine": float(np.mean(all_x_cos)),
            "sample_supports": list(diagnostics.items())[: args.sample_episodes * args.n_way],
        },
        "trajectory": trajectory,
        "best": {
            "epoch": best_epoch,
            "valid_accuracy": best_valid,
            "test_accuracy": test["accuracy"],
            "test_ci95_half_width": test["ci95_half_width"],
            "checkpoint": str(best_path),
        },
        "test": test,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.json_path).open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
    summary = (
        f"exp048 — Structural Rejection + X Middle-K\n"
        f"Threshold: tau={args.tau:g} (AX AND A2X)\n"
        f"Coverage without fallback: {(len(diagnostics)-len(fallback_supports))/len(diagnostics):.2%}\n"
        f"Original GT match: {original_match:.2%}\n"
        f"Middle GT match: {middle_match:.2%}\n"
        f"Mean original X rank: {np.mean(all_rank):.1f}\n"
        f"Best epoch: {best_epoch}\nValid: {best_valid:.4%}\n"
        f"Test: {test['accuracy']:.4%} ± {test['ci95_half_width']:.4%} (95% CI)\n"
        f"Checkpoint: {best_path}\n"
    )
    (output_dir / "result.txt").write_text(summary, encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
