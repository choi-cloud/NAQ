"""hl003_oracle_train: copied from choi_exps/exp049.py (parse_args + main); only the anchor generation is replaced by --policy. Reads GT labels for oracle anchor selection.

Queries are always B_Choi (exp049.build_structural_queries, tau given by --tau). Training, evaluation and
report.json follow exp049 unchanged; report.json additionally carries `policy` and `anchor_diagnostics`.
Policies: uniform | quintile:q | gt_distinct | hard:k | hard_blind:k | ab   (see instructions/HL02_diag_oracle.md §3)
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hl_common as H  # noqa: E402
from data import load_data  # noqa: E402
from exp027 import evaluate  # noqa: E402
from exp049 import build_structural_queries, load_splits, query_diagnostics  # noqa: E402
from meta import ProtoNet  # noqa: E402
from task_generator import proto_task_generator  # noqa: E402
from utils import seed_everything  # noqa: E402

MAX_RESAMPLE = 20


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
    p.add_argument("--output_root", default="results/hl/hl02/oracle")
    # hl003 additions
    p.add_argument("--policy", required=True)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--L", type=int, default=50)
    p.add_argument("--node_table", default="results/hl/hl02/diag/node_table_tau0.3.csv")
    p.add_argument("--quintile_csv", default="results/hl/hl02/diag/pi_bchoi_quintiles.csv")
    return p.parse_args()


def policy_name(policy, k):
    """'hard' + k=4 -> 'hard:4'; 'quintile:3' stays; directory name replaces ':' with '_'."""
    if policy in ("hard", "hard_blind") and k is not None:
        policy = f"{policy}:{k}"
    return policy, policy.replace(":", "_")


def read_column(path, column, cast):
    with Path(path).open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    out = np.empty(len(rows), dtype=np.float64 if cast is float else np.int64)
    for r in rows:
        out[int(r["node"])] = cast(r[column])
    return out


class AnchorSampler:
    """Oracle anchor policies. `labels` are GT labels (oracle use only)."""

    def __init__(self, policy, n_way, all_nodes, labels, quintile, rejected_lists):
        self.policy, self.n_way = policy, n_way
        self.all_nodes, self.labels, self.quintile = all_nodes, labels, quintile
        self.rejected = rejected_lists  # node -> X-cosine-ordered top-L ids that fail the filter
        self.pools = {q: [int(i) for i in np.flatnonzero(quintile == q)] for q in range(1, 6)} if quintile is not None else {}

    def _uniform_distinct_gt(self, chosen, stats):
        """One uniform node whose GT differs from all chosen; after MAX_RESAMPLE failures, a plain uniform node."""
        used = {self.labels[c] for c in chosen}
        for _ in range(MAX_RESAMPLE + 1):
            c = random.choice(self.all_nodes)
            if c not in chosen and self.labels[c] not in used:
                return c
        stats["fallback"] = True
        while True:
            c = random.choice(self.all_nodes)
            if c not in chosen:
                return c

    def _uniform_any(self, chosen):
        while True:
            c = random.choice(self.all_nodes)
            if c not in chosen:
                return c

    def _hard(self, a1, k, stats, blind):
        chosen = [a1]
        for c in self.rejected[a1]:
            if len(chosen) - 1 >= k:
                break
            c = int(c)
            if c in chosen:
                continue
            if not blind and self.labels[c] in {self.labels[x] for x in chosen}:
                continue
            chosen.append(c)
        stats["k_eff"] = len(chosen) - 1
        while len(chosen) < self.n_way:
            chosen.append(self._uniform_any(chosen) if blind else self._uniform_distinct_gt(chosen, stats))
        return chosen

    def sample(self):
        stats = {"k_eff": None, "fallback": False}
        p = self.policy
        if p.startswith("quintile:"):
            chosen = random.sample(self.pools[int(p.split(":")[1])], self.n_way)
        elif p == "gt_distinct":
            for _ in range(MAX_RESAMPLE + 1):
                chosen = random.sample(self.all_nodes, self.n_way)
                if len({self.labels[c] for c in chosen}) == self.n_way:
                    break
            else:
                stats["fallback"] = True
        elif p.startswith("hard:"):
            chosen = self._hard(random.choice(self.all_nodes), int(p.split(":")[1]), stats, blind=False)
        elif p.startswith("hard_blind:"):
            chosen = self._hard(random.choice(self.all_nodes), int(p.split(":")[1]), stats, blind=True)
        elif p == "ab":
            chosen = self._hard(random.choice(self.pools[5]), 4, stats, blind=False)
        else:
            raise ValueError(f"Unknown policy {p}")
        # pseudo-labels as in task_generator.spt_sampling (k_shot = 1)
        y = list(range(self.n_way))
        random.shuffle(y)
        return np.asarray(chosen, dtype=np.int64), torch.LongTensor(y), stats


def make_episode(id_spt, y_spt, q):
    return id_spt, y_spt, np.zeros(len(id_spt) * q, dtype=np.int64), y_spt.repeat_interleave(q)


def rejected_lists_for(data, device, nodes, L, tau):
    """X top-L ids of each node (data.Diffusion_matrix order) that fail the exp049 filter, in X-cosine order."""
    z0, z1, z2 = H.structural_scores(data, device)
    del z0
    top_ids, _ = H.x_topL(data, nodes, L)
    acc = H.accepted_mask(z1, z2, np.repeat(nodes, L), top_ids.reshape(-1), tau).reshape(len(nodes), L)
    out = {int(s): top_ids[j][~acc[j]] for j, s in enumerate(nodes)}
    return out, z2


def anchor_diagnostics(tasks, stats, labels, pi_bchoi, data, z2, n_way):
    iu = np.triu_indices(n_way, 1)
    anchors = np.stack([np.asarray(t[0]).reshape(-1) for t in tasks])
    src, dst = anchors[:, iu[0]].reshape(-1), anchors[:, iu[1]].reshape(-1)
    same = labels[src] == labels[dst]
    dm = data.Diffusion_matrix
    with torch.no_grad():
        xcos = dm[torch.as_tensor(src, device=dm.device), torch.as_tensor(dst, device=dm.device)].cpu().numpy()
    a2 = H.pair_cos(z2, src, dst)
    k_eff = [s["k_eff"] for s in stats if s["k_eff"] is not None]
    return {
        "k_eff_mean": float(np.mean(k_eff)) if k_eff else None,
        "same_gt_pair_rate": float(same.mean()),
        "episodes_with_same_gt_pair": float(same.reshape(len(tasks), -1).any(1).mean()),
        "anchor_pair_mean_x_cosine": float(xcos.mean()),
        "anchor_pair_mean_a2x_cosine": float(a2.mean()),
        "uniform_fallback_rate": float(np.mean([s["fallback"] for s in stats])),
        "anchor_pi_bchoi_mean": float(pi_bchoi[anchors.reshape(-1)].mean()),
        "unique_anchors": int(len(np.unique(anchors))),
        "max_resample": MAX_RESAMPLE,
    }


def main():
    args = parse_args()
    if not -1 <= args.tau <= 1:
        raise ValueError("Requires --tau in [-1,1]")
    if args.k_shot != 1 or args.k_shot_test != 1 or args.n_way != 5:
        raise ValueError("Training remains 5-way 1-shot; evaluation separately tests 1 and 5 shot")
    if args.epoch <= 0 or args.eval_interval <= 0:
        raise ValueError("Invalid training schedule")
    args.policy, policy_dir = policy_name(args.policy, args.k)
    args.method = "ours"
    args.setting, args.type, args.query_generation = "unsup", "feature", "NaQ"
    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    splits = load_splits(args)
    out = Path(args.output_root) / policy_dir / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    if report_path.exists():
        raise FileExistsError(f"Existing result cannot be overwritten: {report_path}")

    start = time.perf_counter()
    print(f"[hl003] dataset={args.dataset} policy={args.policy} tau={args.tau} loading", flush=True)
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

    # ---- the only change vs exp049: anchor generation (python RNG state here equals exp049's) ----
    labels = H.labels_np(data)
    pi_bchoi = read_column(args.node_table, "pi_bchoi", float)
    all_nodes = list(data.id_train)
    needs_hard = args.policy.startswith(("hard", "ab"))
    rejected, z2 = rejected_lists_for(data, device, np.arange(len(labels)), args.L, args.tau) if needs_hard else (None, None)
    if z2 is None:
        z2 = H.structural_scores(data, device)[2]
    if args.policy == "uniform":
        original = [proto_task_generator(args, data) for _ in range(args.epoch)]
        stats = [{"k_eff": None, "fallback": False} for _ in original]
    else:
        quintile = read_column(args.quintile_csv, "quintile", int) if args.policy.startswith(("quintile", "ab")) else None
        sampler = AnchorSampler(args.policy, args.n_way, all_nodes, labels, quintile, rejected)
        original, stats = [], []
        for _ in range(args.epoch):
            id_spt, y_spt, st = sampler.sample()
            original.append(make_episode(id_spt, y_spt, args.q_query))
            stats.append(st)
    anchor_diag = anchor_diagnostics(original, stats, labels, pi_bchoi, data, z2, args.n_way)
    print(f"[hl003] anchor diagnostics: {anchor_diag}", flush=True)
    del z2, rejected
    torch.cuda.empty_cache()
    # ----------------------------------------------------------------------------------------------

    train_pool, details = build_structural_queries(data, original, args, device)
    if len(train_pool) != len(original):
        raise RuntimeError("Training episode count changed")
    diag = query_diagnostics(original, train_pool, data, details, args.q_query)
    print(f"[hl003] query diagnostics: {diag}", flush=True)
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
        print(f"[hl003] epoch={epoch} valid1={row['valid_1shot']:.4%} valid5={row['valid_5shot']:.4%}", flush=True)

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
        print(f"[hl003] {shot}-shot test={test['accuracy']:.4%} ± {test['ci95_half_width']:.4%}", flush=True)
    report = {
        "experiment": "hl003",
        "args": vars(args).copy(),
        "dataset": args.dataset,
        "method": "ours",
        "policy": args.policy,
        "tau": args.tau,
        "seed": args.seed,
        "diagnostics": diag,
        "anchor_diagnostics": anchor_diag,
        "evaluation": evaluations,
        "history": history,
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "max_cuda_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2) if device.type == "cuda" else None,
        "note": "hl003: exp049 main with oracle anchor policy; GT labels used only for anchor selection and diagnostics. Queries never use labels."
    }
    tmp = out / "report.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(report_path)
    print(f"[hl003] DONE {report_path}", flush=True)


if __name__ == "__main__":
    main()
