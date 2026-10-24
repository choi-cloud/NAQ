"""exp044: Label-free NAQ-Feat Middle-K query sampling.

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

SELECTION_MODE = "middle"


def parse_args():
    parser = argparse.ArgumentParser(description="exp044: label-free feature Middle-K NAQ")
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
    parser.add_argument("--json_path", default="exp044.json")
    parser.add_argument("--sample_episodes", type=int, default=3)
    parser.add_argument("--pool_size", type=int, default=0, help="0: entire graph minus anchor; N>0: Top-N feature neighbors before Middle-K")
    return parser.parse_args()


def middle_refine_episode(task, diffusion, q_query, pool_size):
    """Sort candidate nodes by original NAQ similarity, take central q_query.

    NO ground truth is used. The pseudo-labels of original NAQ are preserved.
    pool_size=0 means all nodes except anchor. Positive pool_size means first
    Top-N neighbors; then select the central q_query from the Top-N list.
    """
    supports, pseudo_s, original_queries, pseudo_q = task
    supports = np.asarray(supports, dtype=np.int64).reshape(-1)
    original_queries = np.asarray(original_queries, dtype=np.int64).reshape(-1)
    if len(original_queries) != len(supports) * q_query:
        raise ValueError("Expected query blocks ordered by support")
    n = diffusion.shape[1]
    candidate_count = n - 1 if pool_size == 0 else min(pool_size, n - 1)
    if candidate_count < q_query:
        raise ValueError("Candidate pool smaller than q_query")
    blocks = []
    details = []
    for i, support in enumerate(supports):
        support = int(support)
        score = diffusion[support]
        if score.layout != torch.strided:
            score = score.to_dense()
        score = score.reshape(-1).clone()
        score[support] = -float("inf")
        # Descending feature similarity; self is guaranteed absent.
        ranking = torch.argsort(score, descending=True, stable=True)[:candidate_count]
        start = (candidate_count - q_query) // 2
        chosen = ranking[start : start + q_query]
        selected = chosen.cpu().numpy().astype(np.int64)
        original = original_queries[i * q_query : (i + 1) * q_query]
        blocks.append(selected)
        details.append(
            {
                "support_node": support,
                "pool_size": candidate_count,
                "start_rank_zero_based": start,
                "end_rank_exclusive": start + q_query,
                "overlap_original_query_count": int(np.isin(selected, original).sum()),
                "selected_mean_similarity": float(score[chosen].mean().item()),
                "original_query_nodes": original.tolist(),
                "middle_query_nodes": selected.tolist(),
            }
        )
    return (supports, pseudo_s, np.concatenate(blocks), pseudo_q), details


def main():
    args = parse_args()
    args.setting = "unsup"
    if args.query_generation != "NaQ" or args.type != "feature":
        raise ValueError("exp044 is restricted to original NAQ-Feat")
    if args.k_shot != 1 or args.k_shot_test != 1:
        raise ValueError("exp044 currently supports only 1-shot")
    if args.epoch <= 0 or args.eval_interval <= 0 or args.q_query <= 0 or args.pool_size < 0:
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

    # Generate original NAQ episodes first, then change ONLY query IDs.
    # No GT labels are read in sampling or training.
    original_pool = [proto_task_generator(args, data) for _ in range(args.epoch)]
    train_pool = []
    diagnostics = []
    for task in original_pool:
        corrected, detail = middle_refine_episode(task, diffusion, args.q_query, args.pool_size)
        train_pool.append(corrected)
        diagnostics.extend(detail)
    output_dir = Path(f"results/{args.dataset}/exp044")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    best_valid = float("-inf")
    best_epoch = None
    trajectory = []
    # Label diagnostics are computed only AFTER every training query has been chosen.
    labels = torch.as_tensor(data.labels).detach().cpu().numpy().reshape(-1)
    original_gt = []
    middle_gt = []
    for original_task, modified_task in zip(original_pool, train_pool):
        original_supports = np.asarray(original_task[0], dtype=np.int64).reshape(-1)
        original_queries = np.asarray(original_task[2], dtype=np.int64).reshape(-1)
        middle_queries = np.asarray(modified_task[2], dtype=np.int64).reshape(-1)
        expected_labels = np.repeat(labels[original_supports], args.q_query)
        original_gt.extend((labels[original_queries] == expected_labels).tolist())
        middle_gt.extend((labels[middle_queries] == expected_labels).tolist())
    del original_pool
    original_match = float(np.mean(original_gt))
    middle_match = float(np.mean(middle_gt))
    print(
        f"[exp044] Candidate scope: {'all_nodes' if args.pool_size == 0 else f'top_{args.pool_size}'} "
        f"original_GT_match={original_match:.2%} middle_GT_match={middle_match:.2%}",
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
            f"[exp044][Epoch {epoch}] loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} valid={valid['accuracy']:.4f} "
            f"best={best_valid:.4f}@{best_epoch}",
            flush=True,
        )

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    meta.network.load_state_dict(checkpoint["model"])
    test = evaluate(meta, data, test_episodes, args, with_episodes=True)
    report = {
        "experiment": "exp044",
        "description": "Label-free NAQ-Feat Middle-K pseudo-query sampling; original ProtoNet training",
        "config": vars(args).copy(),
        "protocol": {
            "label_usage": "GT ONLY for post-sampling diagnostic; never used to select train queries",
            "support_sampling": "unchanged original NAQ",
            "query_ranking": "original NAQ feature Diffusion_matrix scores",
            "query_selection": "contiguous middle-q ranks, descending cosine similarity",
            "ranking_scope": "all_nodes" if args.pool_size == 0 else f"top_{args.pool_size}",
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
            "mean_original_overlap_per_support": float(np.mean([x["overlap_original_query_count"] for x in diagnostics])),
            "mean_selected_similarity": float(np.mean([x["selected_mean_similarity"] for x in diagnostics])),
            "sample_supports": diagnostics[: args.sample_episodes * args.n_way],
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
    with Path(args.json_path).open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
    summary = (
        f"exp044 — Label-free NAQ-Feat Middle-K\n"
        f"Candidate scope: {'all_nodes' if args.pool_size == 0 else f'top_{args.pool_size}'}\n"
        f"Original GT match: {original_match:.2%}\n"
        f"Middle GT match: {middle_match:.2%}\n"
        f"Best epoch: {best_epoch}\nValid: {best_valid:.4%}\n"
        f"Test: {test['accuracy']:.4%} ± {test['ci95_half_width']:.4%} (95% CI)\n"
        f"Checkpoint: {best_path}\n"
    )
    (output_dir / "result.txt").write_text(summary, encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
