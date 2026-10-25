"""exp057: GT-oracle query sampling with original NAQ random supports.

Rank same-GT-class candidates by the NAQ feature-similarity matrix, and
select top, middle, or bottom 10. GT labels are NEVER training targets.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import load_data
from meta import ProtoNet
from task_generator import downstream_task_generator, spt_sampling
from utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--dataset", default="Cora_full")
    p.add_argument("--epoch", type=int, default=2000)
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=1)
    p.add_argument("--q_query", type=int, default=10)
    p.add_argument("--q_query_test", type=int, default=8)
    p.add_argument("--num_valid_tasks", type=int, default=50)
    p.add_argument("--num_downstream_tasks", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--latent", type=int, default=256)
    p.add_argument("--l2_penalty", type=float, default=1e-4)
    p.add_argument("--eval_interval", type=int, default=160)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--shots", type=int, nargs="+", default=[1, 5])
    return p.parse_args()


def save_json(path, report):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def evaluate(model, data, episodes, n_way, shot):
    model.network.eval()
    scores = []
    margins = []
    with torch.no_grad():
        z = model.network(data.features, adj=data.adj)
        device = z.device
        for sid, ys, qid, yq in episodes:
            sid = torch.as_tensor(sid, dtype=torch.long, device=device).reshape(-1)
            ys = torch.as_tensor(ys, dtype=torch.long, device=device).reshape(-1)
            qid = torch.as_tensor(qid, dtype=torch.long, device=device).reshape(-1)
            yq = torch.as_tensor(yq, dtype=torch.long, device=device).reshape(-1)
            classes = torch.unique(ys, sorted=True)
            if sid.numel() != n_way * shot or classes.numel() != n_way:
                raise ValueError("Unexpected downstream episode")
            proto = torch.stack([z[sid[ys == cls]].mean(0) for cls in classes])
            dist = (z[qid, None, :] - proto[None, :, :]).square().sum(-1)
            scores.append(float((classes[dist.argmin(1)] == yq).float().mean().item()))
            nearest = torch.topk(dist, k=2, largest=False, dim=1).values
            margins.append(float((nearest[:, 1] - nearest[:, 0]).mean().item()))
    arr = np.asarray(scores, dtype=float)
    return {
        "accuracy": float(arr.mean()),
        "ci95_half_width": float(1.96 * arr.std(ddof=0) / np.sqrt(len(arr))),
        "num_episodes": int(len(arr)),
        "mean_nearest_distance_margin": float(np.mean(margins)),
    }


def main():
    args = parse_args()
    if args.dataset != "Cora_full" or args.n_way != 5 or args.k_shot != 1:
        raise ValueError("exp057 requires Cora_full, 5-way, 1-shot training")
    if min(args.epoch, args.eval_interval, args.q_query) < 1:
        raise ValueError("Invalid epoch, interval or query count")
    args.setting = "unsup"
    args.query_generation = "NaQ"
    args.type = "feature"
    args.k_shot_test = args.shots[0]
    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    root = Path(__file__).resolve().parent
    out = root / "exp057.json"
    start = time.perf_counter()
    report = {
        "experiment": "exp057",
        "status": "initializing",
        "args": vars(args).copy(),
        "protocol": {
            "anchors": "Original NAQ random spt_sampling; same episode anchor sequence for all strategies, GT collisions allowed",
            "queries": "For each anchor, filter all nodes to matching GT class, exclude ALL support nodes, sort by original NAQ-Feat diffusion-matrix similarity, choose top/middle/bottom K",
            "middle": "Contiguous K-sized slice centered at the median of sorted eligible candidates",
            "bottom": "K lowest-similarity eligible candidates",
            "labels": "Episode-local original pseudo-labels; GT only filters queries",
            "evaluation": "Squared Euclidean nearest prototype, each shot selects own best validation checkpoint, test once",
            "warning": "GT label oracle; not an unsupervised method",
        },
        "results": {},
    }
    save_json(out, report)
    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    data.add_diffusion(args)
    labels = np.asarray(torch.as_tensor(data.labels).detach().cpu()).reshape(-1)
    classes = np.unique(labels)
    by_class = {int(c): np.flatnonzero(labels == c) for c in classes}
    # This is the original NAQ support sampling call; no oracle replacement.
    supports = [spt_sampling(data.id_train, args.n_way, args.k_shot) for _ in range(args.epoch)]
    downstream = {}
    for shot in args.shots:
        args.k_shot_test = shot
        downstream[str(shot)] = downstream_task_generator(args, data, args.n_way, shot)
    # Similarity is fetched from the SAME matrix used by NAQ-Feat query_generation_NaQ.
    matrix = data.Diffusion_matrix
    if matrix.layout == torch.sparse_coo:
        raise ValueError("exp057 currently expects dense NAQ-Feat similarity matrix")
    rank_cache = {}
    for sid, _ in supports:
        for anchor in np.asarray(sid).reshape(-1):
            anchor = int(anchor)
            if anchor in rank_cache:
                continue
            candidates = by_class[int(labels[anchor])]
            candidates_t = torch.as_tensor(candidates, dtype=torch.long, device=matrix.device)
            scores = matrix[anchor, candidates_t].detach().cpu().numpy()
            # Stable descending ranking with node ID as tie-breaker.
            order = np.lexsort((candidates, -scores))
            rank_cache[anchor] = candidates[order]
    # Generate same pseudo-supports but method-specific oracle query IDs.
    oracle_pools = {name: [] for name in ("top", "middle", "bottom")}
    diagnostics = {name: [] for name in oracle_pools}
    for sid, ys in supports:
        sid = np.asarray(sid, dtype=np.int64).reshape(-1)
        selected = set(sid.tolist())
        per_strategy = {name: [] for name in oracle_pools}
        per_rank = {name: [] for name in oracle_pools}
        for anchor in sid:
            eligible = np.asarray([x for x in rank_cache[int(anchor)] if x not in selected], dtype=np.int64)
            count = len(eligible)
            if count < args.q_query:
                raise ValueError(f"Anchor {anchor}: only {count} same-class non-support nodes")
            k = args.q_query
            starts = {"top": 0, "middle": (count - k) // 2, "bottom": count - k}
            for strategy, pos in starts.items():
                per_strategy[strategy].extend(eligible[pos : pos + k].tolist())
                per_rank[strategy].append((pos + (k - 1) / 2) / max(count - 1, 1))
        for strategy in oracle_pools:
            queries = np.asarray(per_strategy[strategy], dtype=np.int64)
            yq = torch.repeat_interleave(ys, args.q_query)
            oracle_pools[strategy].append((sid, ys, queries, yq))
            diagnostics[strategy].append(
                {
                    "gt_match": float((labels[queries] == np.repeat(labels[sid], args.q_query)).mean()),
                    "anchor_distinct": float(np.unique(labels[sid]).size == args.n_way),
                    "query_duplicate_fraction": float((len(queries) - len(np.unique(queries))) / len(queries)),
                    "support_query_overlap_fraction": float(np.isin(queries, sid).mean()),
                    "mean_rank_quantile": float(np.mean(per_rank[strategy])),
                }
            )
    report["shared_support_episodes"] = len(supports)
    report["status"] = "training"
    save_json(out, report)

    config = [
        ("GCN", (data.features.shape[1], 2 * args.latent), True),
        ("GCN", (2 * args.latent, args.latent), True),
        ("linear", (args.latent, args.n_way), True),
    ]
    for strategy in ("top", "middle", "bottom"):
        # Same weight initialization for all three strategies.
        seed_everything(args.seed + 5700)
        model = ProtoNet(args, config, data).to(device)
        best_valid = {str(s): -1.0 for s in args.shots}
        paths = {str(s): root / f"exp057_{strategy}_{s}shot_best.pt" for s in args.shots}
        result = {
            "status": "training",
            "sampling_diagnostics_all": {key: float(np.mean([r[key] for r in diagnostics[strategy]])) for key in diagnostics[strategy][0]},
            "history": [],
            "best_by_shot": {},
            "test_by_shot": {},
        }
        report["results"][strategy] = result
        save_json(out, report)
        losses = []
        train_accs = []
        for epoch, (sid, ys, qid, yq) in enumerate(oracle_pools[strategy], 1):
            model.network.train()
            enc = model.network(data.features, adj=data.adj)
            sid_t = torch.as_tensor(sid, dtype=torch.long, device=device)
            qid_t = torch.as_tensor(qid, dtype=torch.long, device=device)
            ys_t = torch.as_tensor(ys, dtype=torch.long, device=device)
            yq_t = torch.as_tensor(yq, dtype=torch.long, device=device)
            dist = (enc[qid_t, None, :] - enc[sid_t][None, :, :]).square().sum(-1)
            target = (yq_t[:, None] == ys_t[None, :]).long().argmax(1)
            log_prob = F.log_softmax(-dist, dim=1)
            loss = F.nll_loss(log_prob, target)
            model.meta_optimizer.zero_grad()
            loss.backward()
            model.meta_optimizer.step()
            losses.append(float(loss.detach().item()))
            train_accs.append(float((log_prob.argmax(1) == target).float().mean().item()))
            if epoch % args.eval_interval != 0 and epoch != args.epoch:
                continue
            validations = {}
            for shot in args.shots:
                key = str(shot)
                val = evaluate(model, data, downstream[key][0], args.n_way, shot)
                validations[key] = val
                if val["accuracy"] > best_valid[key]:
                    best_valid[key] = val["accuracy"]
                    torch.save({"epoch": epoch, "model": model.network.state_dict()}, paths[key])
                    result["best_by_shot"][key] = {"epoch": epoch, "validation": val, "checkpoint": str(paths[key])}
            result["history"].append(
                {
                    "epoch": epoch,
                    "train_loss_mean": float(np.mean(losses)),
                    "train_accuracy_mean": float(np.mean(train_accs)),
                    "validation_by_shot": validations,
                }
            )
            losses.clear()
            train_accs.clear()
            save_json(out, report)
            print(
                f"[exp057:{strategy}] epoch={epoch} valid=" + " ".join(f"{shot}shot:{v['accuracy']:.2%}" for shot, v in validations.items()),
                flush=True,
            )
        for shot in args.shots:
            key = str(shot)
            state = torch.load(paths[key], map_location=device, weights_only=True)
            model.network.load_state_dict(state["model"])
            result["test_by_shot"][key] = evaluate(model, data, downstream[key][1], args.n_way, shot)
            test = result["test_by_shot"][key]
            print(
                f"[exp057:{strategy}] DONE {key}-shot best_epoch={result['best_by_shot'][key]['epoch']} "
                f"test={test['accuracy']:.4%} ±{test['ci95_half_width']:.4%}",
                flush=True,
            )
        result["status"] = "completed"
        save_json(out, report)
    report["status"] = "completed"
    report["elapsed_seconds"] = round(time.perf_counter() - start, 2)
    save_json(out, report)
    print(f"[exp057] COMPLETE json={out}", flush=True)


if __name__ == "__main__":
    main()
