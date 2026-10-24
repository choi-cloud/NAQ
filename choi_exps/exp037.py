"""exp037: Oracle false-positive-free NAQ-Feat query selection.

Research-only upper bound: training node GT labels are read by the oracle query
selector. It is NOT an unsupervised method. All other learning and evaluation
code is exp027's original NAQ ProtoNet baseline.
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
    parser = argparse.ArgumentParser(description="exp037: oracle NAQ-Feat without false-positive queries")
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
    parser.add_argument("--json_path", default="exp037.json")
    parser.add_argument("--sample_episodes", type=int, default=3)
    return parser.parse_args()


def oracle_refine_episode(task, diffusion, labels, q_query):
    """Keep support/pseudo-class ordering; take top-q same-GT feature neighbors.

    No GT is used to sample supports. Each replacement follows the *same* NAQ
    similarity ranking, continuing below the original top-q when necessary.
    """
    supports, pseudo_s, original_queries, pseudo_q = task
    supports = np.asarray(supports, dtype=np.int64).reshape(-1)
    original_queries = np.asarray(original_queries, dtype=np.int64).reshape(-1)
    n = diffusion.shape[1]
    if len(original_queries) != len(supports) * q_query:
        raise ValueError("Expected query blocks ordered by support, as in original NAQ")
    selected_blocks = []
    support_diagnostics = []
    label_index = torch.as_tensor(labels, device=diffusion.device, dtype=torch.long).reshape(-1)
    for j, s in enumerate(supports):
        s = int(s)
        # Original NAQ uses Diffusion_matrix. Rank by exactly that score.
        score = diffusion[s]
        if score.layout != torch.strided:
            score = score.to_dense()
        score = score.reshape(-1)
        desired_label = int(label_index[s])
        original = original_queries[j * q_query : (j + 1) * q_query]
        original_correct = int((labels[original] == desired_label).sum())
        # Scan the prefix of the original similarity order until q valid nodes
        # are found. Start at q; geometrically expand to avoid full sorting.
        k = min(n, max(q_query + 1, 32))
        while True:
            nearest = torch.topk(score, k=k, sorted=True).indices
            valid = (label_index[nearest] == desired_label) & (nearest != s)
            matching = nearest[valid]
            if matching.numel() >= q_query:
                break
            if k == n:
                raise RuntimeError(
                    f"Support {s} (GT {desired_label}) has fewer than {q_query} " "other same-class nodes in the NAQ candidate universe"
                )
            k = min(n, k * 2)
        selected = matching[:q_query].detach().cpu().numpy().astype(np.int64)
        selected_blocks.append(selected)
        original_set = set(original.tolist())
        retained = sum(int(node in original_set) for node in selected)
        support_diagnostics.append(
            {
                "support_node": s,
                "original_same_gt_count": original_correct,
                "original_false_positive_count": int(q_query - original_correct),
                "retained_original_query_count": int(retained),
                "replaced_count": int(q_query - retained),
                "rank_prefix_searched": int(k),
                "original_query_nodes": original.tolist(),
                "oracle_query_nodes": selected.tolist(),
            }
        )
    queries = np.concatenate(selected_blocks)
    if not np.array_equal(labels[queries], np.repeat(labels[supports], q_query)):
        raise AssertionError("Oracle retained a GT-mismatched query")
    return (supports, pseudo_s, queries, pseudo_q), support_diagnostics


def main():
    args = parse_args()
    args.setting = "unsup"
    if args.query_generation != "NaQ" or args.type != "feature":
        raise ValueError("exp037 is restricted to original NAQ-Feat")
    if args.k_shot != 1 or args.k_shot_test != 1:
        raise ValueError("exp037 currently supports only 1-shot")
    if args.epoch <= 0 or args.eval_interval <= 0 or args.q_query <= 0:
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
    labels = torch.as_tensor(data.labels).detach().cpu().numpy().reshape(-1)
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

    # Generate original NAQ episodes first: identical support and pseudo labels
    # to exp027 at this seed; then oracle-fix ONLY the training query IDs.
    original_pool = [proto_task_generator(args, data) for _ in range(args.epoch)]
    train_pool = []
    diagnostics = []
    for task in original_pool:
        corrected, detail = oracle_refine_episode(task, diffusion, labels, args.q_query)
        train_pool.append(corrected)
        diagnostics.extend(detail)
    del original_pool
    output_dir = Path(f"results/{args.dataset}/exp037")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    best_valid = float("-inf")
    best_epoch = None
    trajectory = []
    total_original_fp = sum(x["original_false_positive_count"] for x in diagnostics)
    total_original = len(diagnostics) * args.q_query
    print(
        f"[exp037] Original NAQ false-positive rate: {total_original_fp}/{total_original} "
        f"({total_original_fp / total_original:.2%}); oracle false-positive rate: 0%",
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
            f"[exp037][Epoch {epoch}] loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} valid={valid['accuracy']:.4f} "
            f"best={best_valid:.4f}@{best_epoch}",
            flush=True,
        )

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    meta.network.load_state_dict(checkpoint["model"])
    test = evaluate(meta, data, test_episodes, args, with_episodes=True)
    report = {
        "experiment": "exp037",
        "description": "Oracle NAQ-Feat query replacement: same-support-GT top-q; original ProtoNet model",
        "config": vars(args).copy(),
        "protocol": {
            "oracle_warning": "GT training labels are used to filter NAQ queries: NOT unsupervised",
            "support_sampling": "unchanged original NAQ",
            "query_ranking": "unchanged original NAQ feature diffusion-matrix scores",
            "query_filter": "select first q distinct non-support nodes matching support GT label",
            "pseudo_labels": "unchanged NAQ pseudo-labels",
            "train": "original meta.ProtoNet.forward unchanged",
            "evaluation": "RAW Euclidean ProtoNet on untouched exp026 episodes",
            "selection": "best fixed validation accuracy",
            "valid_file": str(valid_path),
            "test_file": str(test_path),
        },
        "oracle_diagnostics": {
            "original_support_query_gt_match": 1 - total_original_fp / total_original,
            "original_false_positive_fraction": total_original_fp / total_original,
            "oracle_support_query_gt_match": 1.0,
            "original_total_queries": int(total_original),
            "original_false_positive_count": int(total_original_fp),
            "mean_replaced_per_support": float(np.mean([x["replaced_count"] for x in diagnostics])),
            "mean_rank_prefix_searched": float(np.mean([x["rank_prefix_searched"] for x in diagnostics])),
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
        f"exp037 — Oracle NAQ (NOT an unsupervised method)\n"
        f"Original false positives: {total_original_fp}/{total_original} "
        f"({total_original_fp / total_original:.2%})\n"
        f"Oracle false positives: 0%\n"
        f"Best epoch: {best_epoch}\nValid: {best_valid:.4%}\n"
        f"Test: {test['accuracy']:.4%} ± {test['ci95_half_width']:.4%} (95% CI)\n"
        f"Checkpoint: {best_path}\n"
    )
    (output_dir / "result.txt").write_text(summary, encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
