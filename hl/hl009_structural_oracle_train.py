"""hl009_structural_oracle_train: copied from hl/hl006_purity_oracle_train.py; anchors stay uniform, only query selection is replaced by structural_oracle_queries (--mode near|far). Reads GT labels for oracle query selection and diagnostics.

Queries: 10 GT-same nodes from the anchor's X-rank band B4 (201-1000), stratified by X-rank sub-bands
S1=[201,400] 3 / S2=[401,600] 3 / S3=[601,800] 2 / S4=[801,1000] 2; inside a sub-band the candidates are ordered by
Â²X cosine (z2 of hl_common.structural_scores = exp049) — `near` takes the highest, `far` the lowest.
Shortfall: adjacent sub-band (S1<->S2, S3<->S4), then across (S1,S2 <- S3; S3,S4 <- S2), then B5 (1001-3000) GT-same by the
same rule, then unused B4 GT-same from the X-rank centre, then (if still short) B4 GT-diff from the X-rank centre.
The per-stage counts depend only on GT-same counts, so near and far use the same allocation for a given anchor.
Before training, embedding diagnostics on the first 500 episodes with two fixed checkpoints (instructions/HL05 §2).
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
import hl_common as H  # noqa: E402
from data import load_data  # noqa: E402
from exp027 import evaluate  # noqa: E402
from exp049 import load_splits, query_diagnostics  # noqa: E402
from hl003_oracle_train import anchor_diagnostics, read_column  # noqa: E402
from hl004_embed_check import embed  # noqa: E402
from hl006_purity_oracle_train import B4, B5, centre  # noqa: E402
from meta import ProtoNet  # noqa: E402
from task_generator import proto_task_generator  # noqa: E402
from utils import seed_everything  # noqa: E402

SUB = [(201, 400, 3), (401, 600, 3), (601, 800, 2), (801, 1000, 2)]
ADJACENT = {0: [1], 1: [0], 2: [3], 3: [2]}
ACROSS = {0: [2], 1: [2], 2: [1], 3: [1]}
STAGES = ["primary", "adjacent", "across", "b5", "b4_centre", "diff_fill"]
CKPTS = {
    "bchoi": "results/hl/p0/Amazon_clothing/ours/tau_0.3/seed_1212/best_1shot.pt",
    "naq": "results/hl/p0/Amazon_clothing/naq/baseline/seed_1212/best_1shot.pt",
}
N_DIAG_EPISODES = 500


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
    p.add_argument("--output_root", default="results/hl/hl05")
    # hl009 additions
    p.add_argument("--mode", required=True, choices=["near", "far"])
    p.add_argument("--node_table", default="results/hl/hl02/diag/node_table_tau0.3.csv")
    return p.parse_args()


def structural_oracle_queries(anchor, mode, ranked, a2x, labels, q):
    """`ranked`: candidate ids in X-rank order (rank 1 first, self excluded, length >= 3000); `a2x`: Â²X cosine to the anchor
    for every node. Oracle: uses GT `labels`. Returns chosen ids and per-stage counts."""
    sign = -1.0 if mode == "near" else 1.0  # near: descending Â²X, far: ascending

    def pool(ids):
        ids = ids[labels[ids] == labels[anchor]]
        return list(ids[np.argsort(sign * a2x[ids], kind="stable")])  # ties by X rank

    subs = [pool(ranked[lo - 1:hi]) for lo, hi, _ in SUB]
    need = [k for _, _, k in SUB]
    chosen, counts = [], dict.fromkeys(STAGES, 0)
    for i in range(4):
        take = subs[i][:need[i]]
        subs[i] = subs[i][len(take):]
        need[i] -= len(take)
        chosen += take
        counts["primary"] += len(take)
    for stage, partners in (("adjacent", ADJACENT), ("across", ACROSS)):
        for i in range(4):
            for j in partners[i]:
                take = subs[j][:need[i]]
                subs[j] = subs[j][len(take):]
                need[i] -= len(take)
                chosen += take
                counts[stage] += len(take)
    short = q - len(chosen)
    if short:
        take = pool(ranked[B5[0] - 1:B5[1]])[:short]
        chosen += take
        counts["b5"] = len(take)
        short -= len(take)
    if short:  # unused B4 GT-same, X-rank centre (only remaining sub-band nodes are unused)
        left = np.asarray(sorted(x for s in subs for x in s), dtype=np.int64)
        left = np.asarray([x for x in ranked[B4[0] - 1:B4[1]] if x in set(left.tolist())], dtype=np.int64)
        take = list(centre(left, min(short, len(left))))
        chosen += take
        counts["b4_centre"] = len(take)
        short -= len(take)
    if short:
        b4 = ranked[B4[0] - 1:B4[1]]
        take = list(centre(b4[labels[b4] != labels[anchor]], short))
        chosen += take
        counts["diff_fill"] = len(take)
    return np.asarray(chosen, dtype=np.int64), counts


def build_structural_oracle(data, tasks, mode, labels, args, device):
    """Replaces exp049.build_structural_queries: same per-anchor cache and episode layout; returns (tasks, details)."""
    unique = np.unique(np.concatenate([np.asarray(t[0], dtype=np.int64).reshape(-1) for t in tasks]))
    z2 = H.structural_scores(data, device)[2]
    dm = data.Diffusion_matrix
    cache, details = {}, {}
    for begin in range(0, len(unique), args.mine_batch_size):
        ids_np = unique[begin:begin + args.mine_batch_size]
        ids = torch.as_tensor(ids_np, dtype=torch.long, device=dm.device)
        with torch.no_grad():
            rows = dm[ids].clone()
            rows[torch.arange(len(ids), device=dm.device), ids] = -float("inf")
            order = torch.sort(rows, dim=1, descending=True, stable=True).indices.cpu().numpy()
            a2x_rows = (z2[ids] @ z2.T).cpu().numpy()
        for j, a in enumerate(ids_np):
            rank = np.empty(order.shape[1], dtype=np.int64)
            rank[order[j]] = np.arange(1, order.shape[1] + 1)
            chosen, counts = structural_oracle_queries(int(a), mode, order[j][:B5[1]], a2x_rows[j], labels, args.q_query)
            cache[int(a)] = chosen
            details[int(a)] = {
                **counts,
                "fallback": bool(args.q_query - counts["primary"]),
                "mean_original_x_rank": float(rank[chosen].mean()),
                "mean_a2x": float(a2x_rows[j][chosen].mean()),
                "purity": float((labels[chosen] == labels[a]).mean()),
            }
    del z2
    torch.cuda.empty_cache()
    modified = []
    for supports, ys, queries, yq in tasks:
        supports = np.asarray(supports, dtype=np.int64).reshape(-1)
        selected = np.concatenate([cache[int(s)] for s in supports])
        if len(selected) != len(np.asarray(queries).reshape(-1)):
            raise ValueError("Query count mismatch")
        modified.append((supports, ys, selected, yq))
    return modified, details


def structural_diagnostics(tasks, details, q):
    """Per anchor occurrence (as exp049.query_diagnostics counts them)."""
    rows = [details[int(s)] for t in tasks for s in np.asarray(t[0]).reshape(-1)]
    out = {
        "mean_purity": float(np.mean([r["purity"] for r in rows])),
        "mean_x_rank": float(np.mean([r["mean_original_x_rank"] for r in rows])),
        "mean_a2x": float(np.mean([r["mean_a2x"] for r in rows])),
        "supplement_rate_anchor": float(np.mean([r["fallback"] for r in rows])),
    }
    for st in STAGES:
        out[f"{st}_query_frac"] = float(np.sum([r[st] for r in rows]) / (q * len(rows)))
    out["supplement_query_frac"] = 1.0 - out["primary_query_frac"]
    return out


def embedding_diagnostics(data, args, tasks):
    """Fixed-checkpoint anchor-query distance and ProtoNet p(own anchor) on the given episodes (no training)."""
    out = {}
    for name, path in CKPTS.items():
        Z, ck_epoch = embed(data, args, path)
        Zn = F.normalize(Z, dim=1)
        dist, p_own, loss = [], [], []
        for id_spt, y_spt, id_query, y_query in tasks:
            id_spt = np.asarray(id_spt).reshape(-1)
            own_anchor = np.repeat(id_spt, args.q_query)  # queries are laid out anchor by anchor (exp049)
            dist.append(1.0 - H.pair_cos(Zn, own_anchor, id_query))
            probs, target, _ = H.proto_softmax(Z, id_spt, id_query, y_spt, y_query)
            own = probs.gather(1, target[:, None]).squeeze(1).clamp_min(1e-30)
            p_own.append(own.cpu().numpy())
            loss.append((-own.log()).cpu().numpy())
        dist, p_own, loss = (np.concatenate(v) for v in (dist, p_own, loss))
        out[name] = {
            "checkpoint": path, "ckpt_epoch": ck_epoch, "n_episodes": len(tasks), "n_queries": int(len(p_own)),
            "dist_mean": float(dist.mean()), "dist_median": float(np.median(dist)),
            "p_own_mean": float(p_own.mean()), "p_own_median": float(np.median(p_own)),
            "p_own_ge_095_frac": float((p_own >= 0.95).mean()), "loss_mean": float(loss.mean()),
        }
        del Z, Zn
        torch.cuda.empty_cache()
    return out


def main():
    args = parse_args()
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
    out = Path(args.output_root) / args.mode / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    if report_path.exists():
        raise FileExistsError(f"Existing result cannot be overwritten: {report_path}")

    start = time.perf_counter()
    print(f"[hl009] dataset={args.dataset} mode={args.mode} loading", flush=True)
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
    print(f"[hl009] anchor diagnostics: {anchor_diag}", flush=True)
    del z2
    torch.cuda.empty_cache()

    # ---- the only change vs hl006: query selection (oracle, GT labels) ----
    train_pool, details = build_structural_oracle(data, original, args.mode, labels, args, device)
    if len(train_pool) != len(original):
        raise RuntimeError("Training episode count changed")
    diag = query_diagnostics(original, train_pool, data, details, args.q_query)
    qdiag = structural_diagnostics(original, details, args.q_query)
    qdiag["embedding"] = embedding_diagnostics(data, args, train_pool[:N_DIAG_EPISODES])
    print(f"[hl009] query diagnostics: {diag} {json.dumps(qdiag)}", flush=True)
    del original
    # ------------------------------------------------------------------------

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
        print(f"[hl009] epoch={epoch} valid1={row['valid_1shot']:.4%} valid5={row['valid_5shot']:.4%}", flush=True)

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
        print(f"[hl009] {shot}-shot test={test['accuracy']:.4%} ± {test['ci95_half_width']:.4%}", flush=True)
    report = {
        "experiment": "hl009",
        "args": vars(args).copy(),
        "dataset": args.dataset,
        "method": "ours",
        "policy": "uniform",
        "mode": args.mode,
        "tau": args.tau,
        "seed": args.seed,
        "diagnostics": diag,
        "query_diagnostics": qdiag,
        "anchor_diagnostics": anchor_diag,
        "evaluation": evaluations,
        "history": history,
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "max_cuda_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2) if device.type == "cuda" else None,
        "note": "hl009: hl003 uniform anchors with B4 GT-same queries stratified by X sub-band, near/far by Â²X; GT labels used only for query selection and diagnostics."
    }
    tmp = out / "report.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(report_path)
    print(f"[hl009] DONE {report_path}", flush=True)


if __name__ == "__main__":
    main()
