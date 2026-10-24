"""exp020: checkpoint-only semantic attention audit for exp019.

No training or label-based selection. Uses exp019 GCN, PPR neighborhoods and
projection-free query-to-context attention. Labels are used only after weights
are computed, for diagnostics. Run from repository root with exp019.py nearby.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import load_data
from exp019 import AnchorProtoNet, PPRNeighborhoods, RawFeatureAnchorSampler
from model import GNNEncoder
from task_generator import downstream_task_generator
from utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="exp020: exp019 checkpoint attention semantic audit")
    p.add_argument("--ckpt", default="results/Amazon_clothing/exp019/best.pt")
    p.add_argument("--dataset", default="Amazon_clothing")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--attn_tau", type=float, default=None, help="Override checkpoint tau (default from CKPT; legacy uses sqrt(latent))")
    p.add_argument("--ppr_topk", type=int, default=None)
    p.add_argument("--PPR_alpha", type=float, default=None)
    p.add_argument("--ppr_steps", type=int, default=None)
    p.add_argument("--num_pseudo_episodes", type=int, default=200)
    p.add_argument("--num_test_tasks", type=int, default=500)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--json_path", default="exp020.json")
    return p.parse_args()


class Audit:
    def __init__(self, labels):
        self.labels = labels
        self.groups = {}

    def record(self, group, context_ids, query_ids, weights, support_id):
        """Record per-query metrics. Called after attention weights computed."""
        context_labels = self.labels[np.asarray(context_ids)]
        query_labels = self.labels[np.asarray(query_ids)]
        support_label = int(self.labels[int(support_id)])
        w = weights.detach().cpu().numpy()
        matches = query_labels[:, None] == context_labels[None, :]
        has_match = matches.any(axis=1)
        baseline = matches.mean(axis=1)
        mass = (w * matches).sum(axis=1)
        selected = np.argmax(w, axis=1)
        top1_match = matches[np.arange(len(query_ids)), selected].astype(float)
        per_query = self.groups.setdefault(group, [])
        entropy = -(w * np.log(np.maximum(w, 1e-12))).sum(axis=1)
        n = len(context_ids)
        for i in range(len(query_ids)):
            per_query.append(
                {
                    "coverage": float(has_match[i]),
                    "matched_mass": float(mass[i]),
                    "uniform_mass": float(baseline[i]),
                    "mass_gain": float(mass[i] - baseline[i]),
                    "top1_semantic_match": float(top1_match[i]),
                    "top1_uniform_match_chance": float(baseline[i]),
                    "normalized_entropy": float(entropy[i] / math.log(n)) if n > 1 else 0.0,
                    "top1_weight": float(w[i].max()),
                    "same_as_support": float(query_labels[i] == support_label),
                    "context_size": n,
                    "matched_neighbor_count": int(matches[i].sum()),
                    "center_is_match": float(query_labels[i] == context_labels[0]),
                    "selected_is_center": float(selected[i] == 0),
                    "neighbor_only_coverage": float(matches[i, 1:].any()) if n > 1 else 0.0,
                    "neighbor_only_mass": float((w[i, 1:] * matches[i, 1:]).sum()),
                    "neighbor_only_uniform_mass": float(matches[i, 1:].mean()) if n > 1 else 0.0,
                }
            )

    @staticmethod
    def summarize_rows(rows):
        if not rows:
            return {"count": 0}
        result = {"count": len(rows)}
        for key in rows[0]:
            values = np.asarray([r[key] for r in rows], dtype=float)
            result[key] = {"mean": float(values.mean()), "std": float(values.std())}
        return result

    def report(self):
        result = {}
        for group, rows in self.groups.items():
            result[group] = {
                "all": self.summarize_rows(rows),
                "coverage_present": self.summarize_rows([r for r in rows if r["coverage"] > 0]),
                "neighbor_coverage_present": self.summarize_rows([r for r in rows if r["neighbor_only_coverage"] > 0]),
                "false_support_query": self.summarize_rows([r for r in rows if r["same_as_support"] == 0]),
                "false_support_query_coverage_present": self.summarize_rows([r for r in rows if r["same_as_support"] == 0 and r["coverage"] > 0]),
            }
        return result


def audit_task(audit, h, labels, sampler, support_ids, query_ids, true_target=None, kind="pseudo", tau=0.1, k_shot=1):
    s = np.asarray(support_ids, dtype=np.int64).reshape(-1, k_shot)
    q = np.asarray(query_ids, dtype=np.int64)
    qh = h[torch.as_tensor(q, device=h.device, dtype=torch.long)]
    for c, support_group in enumerate(s):
        ids, seen = [], set()
        for support in support_group:
            for v in sampler.get(int(support)):
                if int(v) not in seen:
                    ids.append(int(v))
                    seen.add(int(v))
        context = h[torch.as_tensor(ids, dtype=torch.long, device=h.device)]
        scores = qh @ context.T / tau
        weights = F.softmax(scores, dim=-1)
        # All candidates show if the model selects the query semantic class in
        # arbitrary support contexts. Positive candidate analysis is separate.
        audit.record(kind + "_all_candidates", ids, q, weights, support_group[0])
        if true_target is None:
            own = np.arange(len(q)) // (len(q) // len(s)) == c
        else:
            own = np.asarray(true_target) == c
        if own.any():
            audit.record(kind + "_own_support", ids, q[own], weights[torch.as_tensor(own, device=h.device)], support_group[0])
        if true_target is not None and (~own).any():
            audit.record(kind + "_other_support", ids, q[~own], weights[torch.as_tensor(~own, device=h.device)], support_group[0])


def main():
    args = parse_args()
    start = time.time()
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if args.num_pseudo_episodes < 0 or args.num_test_tasks < 0:
        raise ValueError("Episode counts must be nonnegative")
    torch.set_num_threads(4)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError("Expected exp019 checkpoint with 'model' and 'args' keys")
    saved = ckpt.get("args", {})
    dataset = saved.get("dataset", args.dataset)
    if dataset != args.dataset:
        raise ValueError(f"Dataset mismatch: CKPT={dataset}, requested={args.dataset}")
    latent = int(saved.get("latent", 256))
    k_shot = int(saved.get("k_shot", 1))
    k_shot_test = int(saved.get("k_shot_test", 1))
    tau = args.attn_tau if args.attn_tau is not None else saved.get("attn_tau", math.sqrt(latent))
    if tau <= 0:
        raise ValueError("attn_tau must be positive")
    topk = args.ppr_topk if args.ppr_topk is not None else saved.get("ppr_topk", 16)
    alpha = args.PPR_alpha if args.PPR_alpha is not None else saved.get("PPR_alpha", 0.1)
    steps = args.ppr_steps if args.ppr_steps is not None else saved.get("ppr_steps", 20)
    seed_everything(args.seed)
    load_args = argparse.Namespace(dataset=dataset, setting="unsup")

    load_args.device = args.device
    data = load_data(load_args)

    if device.type == "cuda":
        torch.cuda.set_device(device)
        data.set_device(str(device))
    config = [("GCN", (data.features.shape[1], 2 * latent), True), ("GCN", (2 * latent, latent), True)]
    encoder = GNNEncoder(config).to(device)
    # exp019 AnchorProtoNet wraps encoder under 'encoder.'; no other learnable layers.
    state = {key.removeprefix("encoder."): val for key, val in ckpt["model"].items() if key.startswith("encoder.")}
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    with torch.no_grad():
        h = encoder(data.features, adj=data.adj)
    labels = data.labels.detach().cpu().numpy().reshape(-1)
    sampler = PPRNeighborhoods(data, topk, alpha, steps)
    audit = Audit(labels)
    # Reuse exp019's exact unlabeled sampler. Labels are examined only by Audit.record().
    pseudo_sampler = RawFeatureAnchorSampler(
        data.features, data.id_train, int(saved.get("n_way", 5)), k_shot, int(saved.get("q_query", 10)), int(saved.get("pool_size", 1000))
    )
    with torch.no_grad():
        for _ in range(args.num_pseudo_episodes):
            s, _, q, _ = pseudo_sampler.sample()
            audit_task(audit, h, labels, sampler, s.cpu().numpy(), q.cpu().numpy(), kind="pseudo", tau=tau, k_shot=k_shot)
        if args.num_test_tasks:
            task_args = argparse.Namespace(
                dataset=dataset,
                num_valid_tasks=int(saved.get("num_valid_tasks", 50)),
                num_downstream_tasks=int(saved.get("num_downstream_tasks", 500)),
            )
            _, test_pool = downstream_task_generator(task_args, data, int(saved.get("n_way", 5)), k_shot_test)
            for support_ids, support_labels, query_ids, query_labels in test_pool[: args.num_test_tasks]:
                s_labels = torch.as_tensor(support_labels).cpu().tolist()
                q_labels = torch.as_tensor(query_labels).cpu().tolist()
                target = np.asarray([s_labels.index(x) for x in q_labels], dtype=int)
                audit_task(audit, h, labels, sampler, support_ids, query_ids, true_target=target, kind="test", tau=tau, k_shot=k_shot_test)

    result = {
        "experiment": "exp020",
        "checkpoint": args.ckpt,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_attn_tau": saved.get("attn_tau", None),
        "effective_attn_tau": float(tau),
        "config": vars(args),
        "ppr": {"topk": topk, "alpha": alpha, "steps": steps},
        "num_pseudo_episodes": args.num_pseudo_episodes,
        "num_test_tasks_requested": args.num_test_tasks,
        "metrics": audit.report(),
        "notes": [
            "No training or gradient updates; labels only used after computing attention weights.",
            "Matched mass is compared against uniform mass within the SAME sampled context.",
            "Condition on coverage_present to avoid zeros when the query class is absent.",
            "own_support uses pseudo-way assignments for pseudo tasks and true class for test tasks.",
            "False-support-query groups are meaningful mainly for pseudo episodes.",
            "Original support center is included in context; neighbor-only statistics exclude its contribution.",
            "This reports attention selection, not whether the residual improves accuracy.",
        ],
        "elapsed_seconds": round(time.time() - start, 2),
    }
    out = os.path.abspath(args.json_path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
    for name, metrics in result["metrics"].items():
        block = metrics["coverage_present"]
        if block["count"]:
            print(
                f"{name}: n={block['count']} matched={block['matched_mass']['mean']:.4f} "
                f"uniform={block['uniform_mass']['mean']:.4f} "
                f"gain={block['mass_gain']['mean']:+.4f} "
                f"top1={block['top1_semantic_match']['mean']:.4f}"
            )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
