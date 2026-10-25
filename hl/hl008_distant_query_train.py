"""hl008_distant_query_train: copied from hl/hl006_purity_oracle_train.py; anchors stay uniform (hl003 code path), only query selection is replaced by --arm. Label-free arms never receive labels; GT labels are read for post-hoc diagnostics and the oracle arm p04_choidiff only.

Queries: 10 per anchor from its X-rank band B4 (201-1000), chosen once per anchor node and cached (as exp049).
Label-free arms (scores as in hl005, except comb = mv + enc_n with the fixed NaQ seed-1212 teacher):
  a2x_top10, mv_top10, mv_mid30, encn_top10, comb_top10, rand10, shuf_comb   (instructions/HL04_topk_purity.md §2)
Oracle arm: p04_choidiff = hl006 p=0.4, but the 6 GT-diff queries come from B4 nodes that pass the exp049 filter.
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
from hl005_diag_distance_purity import encoder_embeddings, full_percentile, ppr_columns, sym_adjacency  # noqa: E402
from hl006_purity_oracle_train import B4, B5, centre  # noqa: E402
from meta import ProtoNet  # noqa: E402
from task_generator import proto_task_generator  # noqa: E402
from utils import seed_everything  # noqa: E402

LABEL_FREE = ["a2x_top10", "mv_top10", "mv_mid30", "encn_top10", "comb_top10", "rand10", "shuf_comb"]
ORACLE = ["p04_choidiff"]
TEACHER = "results/hl/p0/Amazon_clothing/naq/baseline/seed_1212/best_1shot.pt"
SCORE_BATCH = 256


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
    p.add_argument("--output_root", default="results/hl/hl04/g2")
    # hl008 additions
    p.add_argument("--arm", required=True, choices=LABEL_FREE + ORACLE)
    p.add_argument("--teacher", default=TEACHER)
    p.add_argument("--ppr_alpha", type=float, default=0.15)
    p.add_argument("--ppr_iters", type=int, default=30)
    p.add_argument("--node_table", default="results/hl/hl02/diag/node_table_tau0.3.csv")
    return p.parse_args()


def b4_scores(data, device, anchors, z1, z2, a_sym, emb, args):
    """Label-free B4 candidates and scores for a batch of anchors (hl005 definitions; comb uses enc_n).
    Returns ranked [A, 3000] (X-rank order, self excluded) and {score: [A, 800]} for the B4 columns."""
    a = torch.as_tensor(anchors, device=device)
    r = torch.arange(len(a), device=device)
    with torch.no_grad():
        xrow = data.Diffusion_matrix[a].clone()
        xrow[r, a] = -float("inf")
        ranked = torch.sort(xrow, dim=1, descending=True, stable=True).indices[:, :B5[1]]
        b4 = ranked[:, B4[0] - 1:B4[1]]
        rows = {
            "ax": z1[a] @ z1.T,
            "a2x": z2[a] @ z2.T,
            "ppr": ppr_columns(a_sym, anchors, args.ppr_alpha, args.ppr_iters).T.contiguous(),
            "enc_n": emb[a] @ emb.T,
        }
        allcols = torch.arange(xrow.size(1), device=device).expand(len(a), -1)
        mv_full = sum(full_percentile(rows[k], a, allcols) for k in ("ax", "a2x", "ppr")) / 3
        got = {k: rows[k].gather(1, b4) for k in ("ax", "a2x", "enc_n")}
        got["mv"] = mv_full.gather(1, b4)
        got["comb"] = (full_percentile(mv_full, a, b4) + full_percentile(rows["enc_n"], a, b4)) / 2
        got["choi"] = (got["ax"] >= args.tau) & (got["a2x"] >= args.tau)
    return ranked.cpu().numpy(), {k: v.cpu().numpy() for k, v in got.items()}


def top(s, k):
    """Column indices of the k highest scores; ties broken by X rank (column order, stable)."""
    return np.argsort(-s, kind="stable")[:k]


def label_free_queries(arm, s, rng, q):
    """B4 column indices for one anchor. `s`: {score: [800]} of that anchor. No labels."""
    if arm == "a2x_top10":
        return top(s["a2x"], q)
    if arm == "mv_top10":
        return top(s["mv"], q)
    if arm == "mv_mid30":
        return centre(np.sort(top(s["mv"], 3 * q)), q)
    if arm == "encn_top10":
        return top(s["enc_n"], q)
    if arm == "comb_top10":
        return top(s["comb"], q)
    if arm == "rand10":
        return rng.choice(len(s["comb"]), q, replace=False)
    if arm == "shuf_comb":
        return top(rng.permutation(s["comb"]), q)
    raise ValueError(arm)


def p04_choidiff_queries(anchor, ranked, passed_b4, labels, q):
    """Oracle: hl006 p=0.4 same-part (B4 GT-same centre, B5 fill, diff fill); GT-diff part from filter-passed B4 GT-diff,
    X-rank centre; if fewer than needed, the rest from the centre of the non-passed B4 GT-diff list. Returns B4/B5 ids."""
    b4 = ranked[B4[0] - 1:B4[1]]
    is_same = labels[b4] == labels[anchor]
    n_same = int(round(q * 0.4))
    same = centre(b4[is_same], min(n_same, int(is_same.sum())))
    from_b5 = np.zeros(0, dtype=np.int64)
    if len(same) < n_same:
        b5 = ranked[B5[0] - 1:B5[1]]
        from_b5 = b5[labels[b5] == labels[anchor]][: n_same - len(same)]
    n_diff = q - len(same) - len(from_b5)
    diff_pass = b4[~is_same & passed_b4]
    take = centre(diff_pass, min(n_diff, len(diff_pass)))
    rest = centre(b4[~is_same & ~passed_b4], n_diff - len(take))
    chosen = np.concatenate([same, from_b5, take, rest])
    return chosen, {"n_same_b5": int(len(from_b5)), "n_diff_fill_same": int(n_same - len(same) - len(from_b5)),
                    "n_diff_pass": int(len(take)), "n_diff_nonpass_fill": int(len(rest))}


def build_distant_queries(data, tasks, args, device, labels):
    """Replaces exp049.build_structural_queries; `labels` is used only by the oracle arm and for post-hoc diagnostics."""
    unique = np.unique(np.concatenate([np.asarray(t[0], dtype=np.int64).reshape(-1) for t in tasks]))
    _, z1, z2 = H.structural_scores(data, device)
    a_sym = sym_adjacency(data, device)
    emb = encoder_embeddings(data, args.teacher, args.latent, device)
    rng = np.random.default_rng(args.seed)
    cache, details = {}, {}
    for begin in range(0, len(unique), SCORE_BATCH):
        ids = unique[begin:begin + SCORE_BATCH]
        ranked, s = b4_scores(data, device, ids, z1, z2, a_sym, emb, args)
        for j, a in enumerate(ids):
            a = int(a)
            b4 = ranked[j][B4[0] - 1:B4[1]]
            extra = {}
            if args.arm in ORACLE:
                chosen, extra = p04_choidiff_queries(a, ranked[j], s["choi"][j], labels, args.q_query)
            else:
                chosen = b4[label_free_queries(args.arm, {k: v[j] for k, v in s.items()}, rng, args.q_query)]
            rank = np.empty(data.features.size(0), dtype=np.int64)
            rank[ranked[j]] = np.arange(1, len(ranked[j]) + 1)
            passed = set(b4[s["choi"][j]].tolist())
            cache[a] = chosen
            details[a] = {
                **extra,
                "fallback": bool(extra.get("n_same_b5") or extra.get("n_diff_fill_same") or extra.get("n_diff_nonpass_fill")),
                "mean_original_x_rank": float(rank[chosen].mean()),
                "choi_pass_frac": float(np.mean([c in passed for c in chosen.tolist()])),
                "purity": float((labels[chosen] == labels[a]).mean()),  # post-hoc diagnostic
            }
        if begin == 0 or begin + SCORE_BATCH >= len(unique):
            print(f"[hl008] scored {min(begin + SCORE_BATCH, len(unique))}/{len(unique)} anchors", flush=True)
    del z1, z2, a_sym, emb
    torch.cuda.empty_cache()
    modified = []
    for supports, ys, queries, yq in tasks:
        supports = np.asarray(supports, dtype=np.int64).reshape(-1)
        selected = np.concatenate([cache[int(x)] for x in supports])
        if len(selected) != len(np.asarray(queries).reshape(-1)):
            raise ValueError("Query count mismatch")
        modified.append((supports, ys, selected, yq))
    return modified, details


def distant_diagnostics(tasks, details, q):
    """Per anchor occurrence (as exp049.query_diagnostics counts them)."""
    rows = [details[int(s)] for t in tasks for s in np.asarray(t[0]).reshape(-1)]
    out = {
        "mean_purity": float(np.mean([r["purity"] for r in rows])),
        "mean_x_rank": float(np.mean([r["mean_original_x_rank"] for r in rows])),
        "choi_pass_frac": float(np.mean([r["choi_pass_frac"] for r in rows])),
        "supplement_rate_anchor": float(np.mean([r["fallback"] for r in rows])),
    }
    for k in ("n_same_b5", "n_diff_fill_same", "n_diff_pass", "n_diff_nonpass_fill"):
        if k in rows[0]:
            out[f"{k}_query_frac"] = float(np.sum([r[k] for r in rows]) / (q * len(rows)))
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
    out = Path(args.output_root) / args.arm / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    if report_path.exists():
        raise FileExistsError(f"Existing result cannot be overwritten: {report_path}")

    start = time.perf_counter()
    print(f"[hl008] dataset={args.dataset} arm={args.arm} loading", flush=True)
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
    print(f"[hl008] anchor diagnostics: {anchor_diag}", flush=True)
    del z2
    torch.cuda.empty_cache()

    # ---- the only change vs hl003 uniform: query selection ----
    t_query = time.perf_counter()
    train_pool, details = build_distant_queries(data, original, args, device, labels)
    if len(train_pool) != len(original):
        raise RuntimeError("Training episode count changed")
    diag = query_diagnostics(original, train_pool, data, details, args.q_query)
    qdiag = distant_diagnostics(original, details, args.q_query)
    qdiag["query_build_seconds"] = round(time.perf_counter() - t_query, 2)
    print(f"[hl008] query diagnostics: {diag} {qdiag}", flush=True)
    del original
    # ------------------------------------------------------------

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
        print(f"[hl008] epoch={epoch} valid1={row['valid_1shot']:.4%} valid5={row['valid_5shot']:.4%}", flush=True)

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
        print(f"[hl008] {shot}-shot test={test['accuracy']:.4%} ± {test['ci95_half_width']:.4%}", flush=True)
    report = {
        "experiment": "hl008",
        "args": vars(args).copy(),
        "dataset": args.dataset,
        "method": "ours",
        "policy": "uniform",
        "arm": args.arm,
        "label_free": args.arm in LABEL_FREE,
        "tau": args.tau,
        "seed": args.seed,
        "diagnostics": diag,
        "query_diagnostics": qdiag,
        "anchor_diagnostics": anchor_diag,
        "evaluation": evaluations,
        "history": history,
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "max_cuda_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2) if device.type == "cuda" else None,
        "note": "hl008: hl003 uniform anchors with B4 distant queries; label-free arms select without labels, labels used only for post-hoc diagnostics and the oracle arm."
    }
    tmp = out / "report.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(report_path)
    print(f"[hl008] DONE {report_path}", flush=True)


if __name__ == "__main__":
    main()
