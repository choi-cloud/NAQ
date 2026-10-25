"""exp071: Train NAQ ProtoNet for 30 GPS-Sum-filtered query sampling variants.

Two filter philosophies: retain GPS top p%, or remove GPS bottom p%.
Within survivors, choose raw-feature cosine top/middle/bottom 10.
Each condition independently trains NAQ ProtoNet, independently selects best
validation checkpoint for each shot, then tests once. GT never trains the model.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

from data import load_data
from exp057 import evaluate
from exp069 import Encoder, build_graph, torch_graph
from meta import ProtoNet
from task_generator import downstream_task_generator, spt_sampling
from utils import seed_everything


def arguments():
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
    p.add_argument("--percentages", type=float, nargs="+", default=[1, 5, 10, 20, 50])
    p.add_argument("--gps_checkpoint", default="exp069_sum.pt")
    p.add_argument("--output", default="exp071.json")
    p.add_argument("--save_checkpoints", action="store_true")
    p.add_argument("--resume", action="store_true", help="Skip completed conditions in output JSON")
    return p.parse_args()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def ordered(ids, scores):
    return ids[np.lexsort((ids, -scores[ids]))]


def select_raw(ids, raw_scores, k, position):
    ranking = ordered(ids, raw_scores)
    if position == "top":
        return ranking[:k]
    if position == "middle":
        begin = (len(ranking) - k) // 2
        return ranking[begin : begin + k]
    if position == "bottom":
        return ranking[-k:][::-1]
    raise ValueError(position)


def frozen_embeddings(data, args, device, root):
    ckpt_path = Path(args.gps_checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if state.get("mode") != "sum":
        raise ValueError("Checkpoint must be exp069 GPS-Sum")
    x = data.features.detach().cpu().numpy().astype(np.float32)
    if x.shape[1] != int(state["input_dim"]):
        raise ValueError("GPS checkpoint input dimension mismatch")
    model = Encoder(int(state["input_dim"]), int(state["hidden"]), int(state["embedding_dim"]), dropout=0).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    graph = build_graph(data, len(x))
    adjacency = torch_graph(graph, device, np.random.default_rng(args.seed), drop=0)
    with torch.no_grad():
        z, _ = model(torch.as_tensor(x, device=device), adjacency)
        z = F.normalize(z, dim=1).cpu().numpy().astype(np.float32)
    return z, normalize(sp.csr_matrix(x), norm="l2", axis=1)


def make_episodes(data, supports, args, z, x_unit, progress):
    """Construct all 30 episode pools once. No GT accessed here."""
    percentages = sorted(set(args.percentages))
    keys = [
        (mode, percentage, position)
        for mode in ("retain_top", "remove_bottom")
        for percentage in percentages
        for position in ("top", "middle", "bottom")
    ]
    pools = {key: [] for key in keys}
    n = z.shape[0]
    all_ids = np.arange(n, dtype=np.int64)
    for sid, ys in tqdm(supports, desc="Construct query episodes", unit="episode", disable=not progress):
        sid = np.asarray(sid, dtype=np.int64).reshape(-1)
        eligible = all_ids[~np.isin(all_ids, sid)]
        per_condition = {key: [] for key in keys}
        for anchor in sid:
            gps_score = z @ z[int(anchor)]
            gps_rank = ordered(eligible, gps_score)
            feat_score = (x_unit.getrow(int(anchor)) @ x_unit.T).toarray().reshape(-1)
            for percentage in percentages:
                amount = int(math.ceil(len(eligible) * percentage / 100))
                candidates = {"retain_top": gps_rank[: max(args.q_query, amount)], "remove_bottom": gps_rank[: len(eligible) - amount]}
                for mode, selected_pool in candidates.items():
                    if len(selected_pool) < args.q_query:
                        raise ValueError("Too few candidates for Top/Middle/Bottom sampling")
                    for position in ("top", "middle", "bottom"):
                        key = (mode, percentage, position)
                        per_condition[key].extend(select_raw(selected_pool, feat_score, args.q_query, position).tolist())
        for key in keys:
            qid = np.asarray(per_condition[key], dtype=np.int64)
            yq = torch.repeat_interleave(torch.as_tensor(ys), args.q_query)
            pools[key].append((sid, ys, qid, yq))
    return pools


def train_condition(key, episodes, data, downstream, args, device, root, report, path):
    mode, percentage, position = key
    tag = f"{mode}_{percentage:g}_{position}"
    seed_everything(args.seed + 5700)
    config = [
        ("GCN", (data.features.shape[1], 2 * args.latent), True),
        ("GCN", (2 * args.latent, args.latent), True),
        ("linear", (args.latent, args.n_way), True),
    ]
    model = ProtoNet(args, config, data).to(device)
    best = {str(shot): -1.0 for shot in args.shots}
    best_weights = {}
    history = []
    elapsed_start = time.perf_counter()
    progress = tqdm(episodes, total=len(episodes), desc=tag, unit="epoch", dynamic_ncols=True)
    for epoch, (sid, ys, qid, yq) in enumerate(progress, 1):
        model.network.train()
        loss, train_accuracy = model(sid, ys.to(device), qid, yq.to(device))
        if epoch % args.eval_interval == 0 or epoch == args.epoch:
            row = {"epoch": epoch, "train_loss": float(loss.detach().item()), "train_accuracy": float(train_accuracy), "validation": {}}
            for shot in args.shots:
                label = str(shot)
                validation = evaluate(model, data, downstream[label][0], args.n_way, shot)
                row["validation"][label] = validation
                if validation["accuracy"] > best[label]:
                    best[label] = validation["accuracy"]
                    best_weights[label] = {
                        "epoch": epoch,
                        "validation": validation,
                        "state": {k: v.detach().cpu().clone() for k, v in model.network.state_dict().items()},
                    }
            history.append(row)
            progress.set_postfix(loss=f"{row['train_loss']:.3f}", val1=f"{best.get('1', -1):.3f}", val5=f"{best.get('5', -1):.3f}")
    progress.close()
    tests = {}
    best_info = {}
    for shot in args.shots:
        label = str(shot)
        chosen = best_weights[label]
        model.network.load_state_dict(chosen["state"])
        tests[label] = evaluate(model, data, downstream[label][1], args.n_way, shot)
        best_info[label] = {"epoch": chosen["epoch"], "validation": chosen["validation"]}
        if args.save_checkpoints:
            torch.save({"epoch": chosen["epoch"], "model": chosen["state"], "condition": tag}, root / f"exp071_{tag}_{shot}shot_best.pt")
    result = {
        "mode": mode,
        "percentage": percentage,
        "position": position,
        "best_by_shot": best_info,
        "test_by_shot": tests,
        "history": history,
        "elapsed_seconds": round(time.perf_counter() - elapsed_start, 2),
    }
    report["results"][tag] = result
    report["completed_conditions"] = len(report["results"])
    write_json(path, report)
    print(f"[exp071] DONE {tag} " + " ".join(f"{shot}shot={tests[str(shot)]['accuracy']:.2%}" for shot in args.shots), flush=True)
    return result


def main():
    args = arguments()
    if args.dataset != "Cora_full" or args.n_way != 5 or args.k_shot != 1:
        raise ValueError("Expected Cora_full 5-way 1-shot unsupervised NAQ training")
    if min(args.epoch, args.eval_interval, args.q_query) < 1:
        raise ValueError("Invalid epoch/interval/query size")
    if any(not 0 < p < 100 for p in args.percentages):
        raise ValueError("percentages must be in (0,100)")
    args.setting = "unsup"
    args.query_generation = "NaQ"
    args.type = "feature"
    args.k_shot_test = args.shots[0]
    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    root = Path(__file__).resolve().parent
    path = Path(args.output)
    if not path.is_absolute():
        path = root / path
    report = {
        "experiment": "exp071",
        "status": "initializing",
        "args": vars(args).copy(),
        "protocol": "Same NAQ supports and downstream episodes for 30 conditions; GPS-Sum filters candidate nodes; feature cosine selects top/middle/bottom; independent ProtoNet per condition; GT used in downstream validation/testing only",
        "results": {},
        "completed_conditions": 0,
    }
    if args.resume and path.is_file():
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["args"]["seed"] != args.seed or report["args"]["epoch"] != args.epoch:
            raise ValueError("Cannot resume with changed seed or epochs")
    else:
        write_json(path, report)
    start = time.perf_counter()
    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    supports = [spt_sampling(data.id_train, args.n_way, args.k_shot) for _ in range(args.epoch)]
    downstream = {}
    for shot in args.shots:
        args.k_shot_test = shot
        downstream[str(shot)] = downstream_task_generator(args, data, args.n_way, shot)
    print("[exp071] Loading frozen GPS-Sum embeddings", flush=True)
    z, x_unit = frozen_embeddings(data, args, device, root)
    print("[exp071] Preparing query episodes for all conditions", flush=True)
    pools = make_episodes(data, supports, args, z, x_unit, progress=True)
    del z, x_unit
    for key, episodes in tqdm(pools.items(), total=len(pools), desc="All conditions", unit="condition"):
        tag = f"{key[0]}_{key[1]:g}_{key[2]}"
        if tag in report["results"]:
            print(f"[exp071] SKIP completed {tag}", flush=True)
            continue
        train_condition(key, episodes, data, downstream, args, device, root, report, path)
    report["status"] = "completed"
    report["elapsed_seconds"] = round(time.perf_counter() - start, 2)
    write_json(path, report)
    print(f"[exp071] COMPLETE {len(report['results'])} conditions -> {path}", flush=True)


if __name__ == "__main__":
    main()
