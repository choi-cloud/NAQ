"""exp065: exp063 homophily filtering followed by raw-feature cosine Middle-K.

Keep the original NAQ support sampler, meta-learner, training and evaluation
unchanged. GT is used only to report diagnostics.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from data import load_data
from exp061 import GraphPairModel, sparse_adjacency, torch_adj
from exp057 import evaluate
from meta import ProtoNet
from task_generator import downstream_task_generator, spt_sampling
from utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", default="Cora_full")
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--q_query", type=int, default=10)
    parser.add_argument("--q_query_test", type=int, default=8)
    parser.add_argument("--num_valid_tasks", type=int, default=50)
    parser.add_argument("--num_downstream_tasks", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent", type=int, default=256)
    parser.add_argument("--l2_penalty", type=float, default=1e-4)
    parser.add_argument("--eval_interval", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--pretrain_checkpoint", default="exp063.pt")
    parser.add_argument("--pair_batch", type=int, default=8192)
    parser.add_argument("--retain_fraction", type=float, default=0.05)
    return parser.parse_args()


def save_json(path, report):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def load_frozen_pair_model(data, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pre = checkpoint["args"]
    if pre["dataset"] != "Cora_full":
        raise ValueError("Pretraining checkpoint dataset mismatch")
    x = data.features.detach().cpu().numpy().astype(np.float32, copy=False)
    n = x.shape[0]
    ax = np.asarray(sparse_adjacency(data, n) @ x, dtype=np.float32)
    views = []
    for mat in (x, ax):
        dim = min(int(pre["svd_dim"]), min(mat.shape) - 1)
        reduced = TruncatedSVD(n_components=dim, n_iter=5, random_state=int(pre["seed"])).fit_transform(mat)
        views.append(normalize(reduced.astype(np.float32), norm="l2", copy=True))
    features = torch.as_tensor(np.concatenate(views, axis=1), device=device)
    model = GraphPairModel(features.shape[1], int(pre["hidden"]), int(pre["embed_dim"]), float(pre["dropout"])).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    with torch.no_grad():
        z = model.encode(features, torch_adj(data, device))
    return model, z, x, pre


def rank_model_candidates(model, z, anchors, k, batch, device):
    """Exact top-K over all nodes, excluding anchor; cache sorted candidate IDs.

    Per-anchor chunk scoring avoids allocating a full N-by-N score matrix.
    """
    n = z.shape[0]
    if k >= n:
        raise ValueError("candidate_cache must be smaller than node count")
    result = {}
    with torch.no_grad():
        for counter, anchor in enumerate(anchors, 1):
            anchor = int(anchor)
            best_scores = torch.empty(0, device=device)
            best_ids = torch.empty(0, dtype=torch.long, device=device)
            for begin in range(0, n, batch):
                ids = torch.arange(begin, min(begin + batch, n), device=device)
                pairs = torch.stack((torch.full_like(ids, anchor), ids), dim=1)
                score = model.pair_logits(z, pairs)
                score = score.masked_fill(ids == anchor, -float("inf"))
                values, positions = torch.topk(score, min(k, len(ids)))
                ids = ids[positions]
                best_scores = torch.cat((best_scores, values))
                best_ids = torch.cat((best_ids, ids))
                top = torch.topk(best_scores, min(k, len(best_scores))).indices
                best_scores, best_ids = best_scores[top], best_ids[top]
            result[anchor] = best_ids.detach().cpu().numpy().astype(np.int64)
            if counter % 500 == 0 or counter == len(anchors):
                print(f"[exp065] Query cache {counter}/{len(anchors)} anchors", flush=True)
    return result


def main():
    args = parse_args()
    if args.dataset != "Cora_full" or args.n_way != 5 or args.k_shot != 1:
        raise ValueError("exp064 expects Cora_full 5-way 1-shot training")
    if not 0 < args.retain_fraction <= 1:
        raise ValueError("retain_fraction must be in (0, 1]")
    if args.pair_batch < 1:
        raise ValueError("pair_batch must be positive")
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
    ckpt_path = Path(args.pretrain_checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing pretraining checkpoint: {ckpt_path}")
    out = root / "exp065.json"
    start = time.perf_counter()
    report = {
        "experiment": "exp065",
        "status": "initializing",
        "args": vars(args).copy(),
        "protocol": {
            "support": "Original NAQ random spt_sampling; GT collisions allowed",
            "query": "Keep top retain_fraction of all nodes by frozen exp063 logits, then choose middle-K by descending raw-feature cosine after excluding all supports",
            "query_labels": "Original NAQ episode-local support pseudo labels repeated for 10 queries",
            "meta_model": "Original NAQ ProtoNet GCN, squared-Euclidean pseudo-label NLL",
            "downstream": "Original downstream_task_generator, separate 1-/5-shot best validation checkpoints",
            "gt": "Diagnostics and downstream evaluation only; no GT in training or query selection",
            "comparison": "exp051 NAQ-Feat and exp064 model Top-K; same meta-training protocol",
        },
        "history": [],
        "best_by_shot": {},
        "test_by_shot": {},
    }
    save_json(out, report)
    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))

    # Draw supports and downstream tasks BEFORE any SVD/pair-model operations,
    # keeping NAQ's sampling random stream independent of feature preprocessing.
    supports = [spt_sampling(data.id_train, args.n_way, args.k_shot) for _ in range(args.epoch)]
    downstream = {}
    for shot in args.shots:
        args.k_shot_test = shot
        downstream[str(shot)] = downstream_task_generator(args, data, args.n_way, shot)
    labels = torch.as_tensor(data.labels).detach().cpu().numpy().reshape(-1)

    print(f"[exp065] Rebuilding frozen exp063 embeddings from {ckpt_path}", flush=True)
    pair_model, z, x, pre_args = load_frozen_pair_model(data, ckpt_path, device)
    report["pretraining_args"] = pre_args
    anchors = np.unique(np.concatenate([np.asarray(sid).reshape(-1) for sid, _ in supports]))
    n = x.shape[0]
    keep = min(n - 1, max(args.q_query + args.n_way, int(np.ceil((n - 1) * args.retain_fraction))))
    print(f"[exp065] Filtering {len(anchors)} anchors to top {keep}/{n - 1} by exp063 score", flush=True)
    cache = rank_model_candidates(pair_model, z, anchors, keep, args.pair_batch, device)
    x_normalized = sp.csr_matrix(x)
    from sklearn.preprocessing import normalize

    x_normalized = normalize(x_normalized, norm="l2", axis=1, copy=False)
    cosine_cache = {}
    for anchor in anchors:
        anchor = int(anchor)
        ids = cache[anchor]
        sims = (x_normalized[ids] @ x_normalized[anchor].T).toarray().reshape(-1)
        order = np.lexsort((ids, -sims))
        cosine_cache[anchor] = ids[order]
    del z, pair_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pool = []
    correct = 0
    total = 0
    duplicate = []
    overlap = []
    for sid, ys in supports:
        sid = np.asarray(sid, dtype=np.int64).reshape(-1)
        excluded = set(sid.tolist())
        query = []
        for anchor in sid:
            allowed = [int(j) for j in cosine_cache[int(anchor)] if int(j) not in excluded]
            if len(allowed) < args.q_query:
                raise RuntimeError(f"Too few eligible candidates for anchor={anchor}; increase --retain_fraction")
            start_idx = (len(allowed) - args.q_query) // 2
            query.extend(allowed[start_idx : start_idx + args.q_query])
        qid = np.asarray(query, dtype=np.int64)
        yq = torch.repeat_interleave(torch.as_tensor(ys), args.q_query)
        pool.append((sid, ys, qid, yq))
        correct += int((labels[qid] == np.repeat(labels[sid], args.q_query)).sum())
        total += len(qid)
        duplicate.append((len(qid) - len(np.unique(qid))) / len(qid))
        overlap.append(float(np.isin(qid, sid).mean()))
    report["query_diagnostics"] = {
        "retain_fraction": args.retain_fraction,
        "num_retained_per_anchor_before_support_exclusion": keep,
        "selection": "Middle contiguous 10 in cosine-descending retained pool",
        "gt_match_rate": correct / total,
        "total_queries": total,
        "duplicate_fraction_mean": float(np.mean(duplicate)),
        "support_query_overlap_fraction": float(np.mean(overlap)),
    }
    print(f"[exp065] Query GT Match={correct / total:.4%}", flush=True)
    report["status"] = "training"
    save_json(out, report)

    # Reinitialize NAQ exactly like exp057, with NO pretraining weights transferred.
    seed_everything(args.seed + 5700)
    config = [
        ("GCN", (data.features.shape[1], 2 * args.latent), True),
        ("GCN", (2 * args.latent, args.latent), True),
        ("linear", (args.latent, args.n_way), True),
    ]
    model = ProtoNet(args, config, data).to(device)
    best_valid = {str(shot): -1.0 for shot in args.shots}
    paths = {str(shot): root / f"exp065_{shot}shot_best.pt" for shot in args.shots}
    losses = []
    train_acc = []
    for epoch, (sid, ys, qid, yq) in enumerate(pool, 1):
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
        train_acc.append(float((log_prob.argmax(1) == target).float().mean().item()))
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
                report["best_by_shot"][key] = {"epoch": epoch, "validation": val, "checkpoint": str(paths[key])}
        report["history"].append(
            {
                "epoch": epoch,
                "train_loss_mean": float(np.mean(losses)),
                "train_accuracy_mean": float(np.mean(train_acc)),
                "validation_by_shot": validations,
            }
        )
        losses.clear()
        train_acc.clear()
        save_json(out, report)
        print("[exp065] epoch=" + str(epoch) + " valid=" + " ".join(f"{shot}shot:{v['accuracy']:.2%}" for shot, v in validations.items()), flush=True)

    for shot in args.shots:
        key = str(shot)
        state = torch.load(paths[key], map_location=device, weights_only=True)
        model.network.load_state_dict(state["model"])
        report["test_by_shot"][key] = evaluate(model, data, downstream[key][1], args.n_way, shot)
        result = report["test_by_shot"][key]
        print(
            f"[exp065] {key}-shot best={report['best_by_shot'][key]['epoch']} " f"test={result['accuracy']:.4%} ±{result['ci95_half_width']:.4%}",
            flush=True,
        )
    report["status"] = "completed"
    report["elapsed_seconds"] = round(time.perf_counter() - start, 2)
    save_json(out, report)
    print(f"[exp065] COMPLETE: {out}", flush=True)


if __name__ == "__main__":
    main()
