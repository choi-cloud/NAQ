"""hl004_embed_check: HL02 §4 pre-B checks on trained checkpoints (no training). Reads GT labels (hard:4 oracle anchors only)."""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hl_common as H  # noqa: E402
from hl003_oracle_train import AnchorSampler, make_episode, rejected_lists_for  # noqa: E402
from exp049 import build_structural_queries  # noqa: E402
from model import GNNEncoder  # noqa: E402
from task_generator import spt_sampling  # noqa: E402
from utils import seed_everything  # noqa: E402

CKPTS = {
    "ours_tau0.3": "results/hl/p0/Amazon_clothing/ours/tau_0.3/seed_1212/best_1shot.pt",
    "naq": "results/hl/p0/Amazon_clothing/naq/baseline/seed_1212/best_1shot.pt",
}
POLICIES = ["uniform", "hard_blind:4", "hard:4"]


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[hl004] wrote {path}", flush=True)


def embed(data, args, path):
    config = [
        ("GCN", (data.features.shape[1], 2 * args.latent), True),
        ("GCN", (2 * args.latent, args.latent), True),
        ("linear", (args.latent, args.n_way), True),
    ]
    net = GNNEncoder(config).to(data.features.device)
    state = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(state["model"])
    net.eval()
    with torch.no_grad():
        return net(data.features, adj=data.adj), int(state["epoch"])


def summarize(v):
    v = np.asarray(v, dtype=np.float64)
    return dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)),
                p25=float(np.percentile(v, 25)), p75=float(np.percentile(v, 75)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--L", type=int, default=50)
    p.add_argument("--n_anchors", type=int, default=2000)
    p.add_argument("--n_episodes", type=int, default=500)
    p.add_argument("--node_table", default="results/hl/hl02/diag/node_table_tau0.3.csv")
    p.add_argument("--table_dir", default="reports/HL02_tables")
    a = p.parse_args()
    args = H.base_args(seed=a.seed, tau=a.tau)
    seed_everything(args.seed)
    data, device = H.load_everything(args)
    labels = H.labels_np(data)
    n = len(labels)

    # 4-1 pair sets (shared by both checkpoints)
    rng = np.random.default_rng(args.seed)
    anchors = rng.choice(n, a.n_anchors, replace=False)
    z0, z1, z2 = H.structural_scores(data, device)
    del z0
    top_ids, _ = H.x_topL(data, anchors, a.L)
    acc = H.accepted_mask(z1, z2, np.repeat(anchors, a.L), top_ids.reshape(-1), args.tau).reshape(len(anchors), a.L)
    del z1, z2
    torch.cuda.empty_cache()
    rand = np.stack([rng.choice(np.delete(np.arange(n), s), a.L, replace=False) for s in anchors])
    src_top = np.repeat(anchors, a.L).reshape(len(anchors), a.L)
    sets = {
        "rejected": (src_top[~acc], top_ids[~acc]),
        "accepted": (src_top[acc], top_ids[acc]),
        "uniform": (np.repeat(anchors, a.L), rand.reshape(-1)),
    }

    # 4-2 episodes per policy (shared by both checkpoints)
    rejected, _ = rejected_lists_for(data, device, np.arange(n), a.L, args.tau)
    torch.cuda.empty_cache()
    episodes = {}
    for policy in POLICIES:
        random.seed(args.seed)
        if policy == "uniform":
            tasks = []
            for _ in range(a.n_episodes):
                id_spt, y_spt = spt_sampling(data.id_train, args.n_way, 1)
                tasks.append(make_episode(np.asarray(id_spt, dtype=np.int64), y_spt, args.q_query))
        else:
            sampler = AnchorSampler(policy, args.n_way, list(data.id_train), labels, None, rejected)
            tasks = [make_episode(*sampler.sample()[:2], args.q_query) for _ in range(a.n_episodes)]
        episodes[policy], _ = build_structural_queries(data, tasks, args, device)
        torch.cuda.empty_cache()

    e1, e2 = [], []
    for ck_name, ck_path in CKPTS.items():
        Z, ck_epoch = embed(data, args, ck_path)
        Zn = F.normalize(Z, dim=1)
        for set_name, (s, d) in sets.items():
            dist = 1.0 - H.pair_cos(Zn, s, d)
            e1.append(dict(checkpoint=ck_name, ckpt_epoch=ck_epoch, set=set_name, **summarize(dist)))
        for policy, tasks in episodes.items():
            p_own, p_neg_sum, p_neg_max, margin = [], [], [], []
            for id_spt, y_spt, id_query, y_query in tasks:
                probs, target, dists = H.proto_softmax(Z, id_spt, id_query, y_spt, y_query)
                own = probs.gather(1, target[:, None]).squeeze(1)
                mask = torch.ones_like(probs, dtype=torch.bool)
                mask.scatter_(1, target[:, None], False)
                neg_p = probs.masked_fill(~mask, 0.0)
                d_own = dists.gather(1, target[:, None]).squeeze(1)
                d_neg = dists.masked_fill(~mask, float("inf")).min(1).values
                p_own.append(own.cpu().numpy())
                p_neg_sum.append(neg_p.sum(1).cpu().numpy())
                p_neg_max.append(neg_p.max(1).values.cpu().numpy())
                margin.append((d_neg - d_own).cpu().numpy())
            p_own, p_neg_sum, p_neg_max, margin = (np.concatenate(v) for v in (p_own, p_neg_sum, p_neg_max, margin))
            e2.append(dict(
                checkpoint=ck_name, policy=policy, n_queries=len(p_own),
                p_own_mean=float(p_own.mean()), p_own_median=float(np.median(p_own)),
                p_neg_sum_mean=float(p_neg_sum.mean()), p_neg_sum_median=float(np.median(p_neg_sum)),
                p_neg_max_mean=float(p_neg_max.mean()), p_neg_max_median=float(np.median(p_neg_max)),
                margin_mean=float(margin.mean()), margin_median=float(np.median(margin)),
                **{"frac_p_neg_sum_lt_0.05": float((p_neg_sum < 0.05).mean())},
            ))
    Path(a.table_dir).mkdir(parents=True, exist_ok=True)
    write_csv(Path(a.table_dir) / "E1_pair_dist.csv", e1)
    write_csv(Path(a.table_dir) / "E2_neg_softmax.csv", e2)
    print("[hl004] DONE", flush=True)


if __name__ == "__main__":
    main()
