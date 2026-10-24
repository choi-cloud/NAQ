"""exp040: GREET-inspired *diagnostic* of conservative negative filtering.

This is NOT the official GREET model: it implements a lightweight standalone
edge-pair MLP trained with a GREET-inspired random-pivot ranking objective.
No GT is used for training, threshold choice, or candidate selection.
GT is read only for reporting same-class retrieval / filtering quality.
Run in NAQ repo root (requires data.py, task_generator.py, utils.py).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import load_data
from task_generator import proto_task_generator
from utils import seed_everything


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Amazon_clothing")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch_edges", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--num_tasks", type=int, default=200)
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=1)
    p.add_argument("--q_query", type=int, default=10)
    p.add_argument("--pool_sizes", default="200,1000,all")
    p.add_argument("--remove_fractions", default="0.25,0.5,0.75,0.9")
    p.add_argument("--sample_episodes", type=int, default=3)
    p.add_argument("--output_dir", default="results/Amazon_clothing/exp040")
    p.add_argument("--json_path", default="exp040.json")
    return p.parse_args()


def structural_encoding(edge_index, n):
    """Topology-only local RW descriptors, with P^2 return probability.

    GREET uses [P_ii, P^2_ii, ...] positional encodings. Here we use degree,
    exact 2-step return, neighbor mean degree, and isolated-node indicator.
    This is a lower-cost approximation, NOT the paper's full encoding.
    """
    edges = edge_index.detach().cpu().numpy()
    a = sp.csr_matrix((np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])), shape=(n, n))
    a = ((a + a.T) > 0).astype(np.float32).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    degree = np.asarray(a.sum(axis=1)).reshape(-1)
    invdeg = 1.0 / np.maximum(degree, 1.0)
    neighbor_invdeg = np.asarray(a @ invdeg).reshape(-1)
    p2 = neighbor_invdeg * invdeg
    neighbor_degree = np.asarray(a @ degree).reshape(-1) * invdeg
    structural = np.stack((np.log1p(degree), p2, np.log1p(neighbor_degree), (degree == 0).astype(np.float32)), axis=1).astype(np.float32)
    mean, std = structural.mean(axis=0), structural.std(axis=0)
    structural = (structural - mean) / np.maximum(std, 1e-5)
    unique = sp.triu(a, k=1).tocoo()
    edge_pairs = np.column_stack((unique.row, unique.col)).astype(np.int64)
    return structural, edge_pairs


class PairDiscriminator(nn.Module):
    def __init__(self, x_dim, hidden):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(x_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.pair = nn.Sequential(nn.Linear(hidden * 3 + 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def embed(self, x):
        return self.node(x)

    def logits(self, z, pairs, cosine):
        a = z[pairs[:, 0]]
        b = z[pairs[:, 1]]
        cos = cosine[pairs[:, 0], pairs[:, 1]].unsqueeze(1)
        features = torch.cat((a + b, (a - b).abs(), a * b, cos, (a * b).sum(dim=1, keepdim=True) / a.shape[1]), dim=1)
        return self.pair(features).squeeze(-1)

    def anchor_logits(self, z, anchor, candidates, candidate_cosine, chunk=16384):
        result = []
        for ids in candidates.split(chunk):
            a = z[anchor].expand(len(ids), -1)
            b = z[ids]
            c = candidate_cosine[ids].unsqueeze(1)
            inp = torch.cat((a + b, (a - b).abs(), a * b, c, (a * b).sum(dim=1, keepdim=True) / a.shape[1]), dim=1)
            result.append(self.pair(inp).squeeze(-1))
        return torch.cat(result)


def format_letters(labs):
    key = {value: (chr(65 + i) if i < 26 else f"C{i}") for i, value in enumerate(dict.fromkeys(map(int, labs)))}
    return key


def main():
    a = arguments()
    a.setting = "unsup"
    a.type = "feature"
    a.query_generation = "NaQ"
    torch.set_num_threads(4)
    seed_everything(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device(f"cuda:{a.device}" if torch.cuda.is_available() else "cpu")
    data = load_data(a)
    data.add_diffusion(a)
    labels = torch.as_tensor(data.labels).detach().cpu().numpy().reshape(-1)
    n = len(labels)
    x = F.normalize(data.features.detach().to(device).float(), dim=-1)
    structure, edges = structural_encoding(data.edge_index, n)
    if len(edges) == 0:
        raise ValueError("No edges to train discriminator")
    st = torch.from_numpy(structure).to(device)
    features = torch.cat((x, st), dim=1)
    # Features are fixed so a pair-independent pretrained node projection is cheap.
    model = PairDiscriminator(features.shape[1], a.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    # For a lightweight method we use a *fixed raw-feature cosine pivot*.
    # GREET's full model instead alternates discriminator and dual-channel encoders.
    order = rng.permutation(len(edges))
    train_edges = edges[order[: int(len(order) * 0.8)]]
    held_edges = edges[order[int(len(order) * 0.8) :]]
    print(f"[exp040] nodes={n} undirected edges={len(edges)} train={len(train_edges)} held={len(held_edges)}", flush=True)
    trace = []
    for step in range(1, a.steps + 1):
        idx = rng.integers(len(train_edges), size=a.batch_edges)
        ep = torch.from_numpy(train_edges[idx]).to(device)
        piv = torch.from_numpy(rng.integers(n, size=(a.batch_edges, 2), dtype=np.int64)).to(device)
        with torch.no_grad():
            score = (x[ep[:, 0]] * x[ep[:, 1]]).sum(1)
            pivot = (x[piv[:, 0]] * x[piv[:, 1]]).sum(1)
            # If similarity exceeds a random baseline by margin, treat as
            # homophily-leaning. If below by margin, heterophily-leaning.
            # Ambiguous pairs are ignored; no GT labels in this step.
            reliable_pos = score > pivot + a.margin
            reliable_neg = score < pivot - a.margin
        # Encode only nodes in the sampled edge batch during optimization.
        unique_nodes, local_pair_idx = torch.unique(ep.reshape(-1), sorted=True, return_inverse=True)
        local_ep = local_pair_idx.reshape(-1, 2)
        z = model.embed(features[unique_nodes])

        # Compute only sampled pair cosine (avoid N x N dense memory).
        def pair_logits(pair):
            vec_a, vec_b = z[pair[:, 0]], z[pair[:, 1]]
            rawcos = (x[pair[:, 0]] * x[pair[:, 1]]).sum(1, keepdim=True)
            inp = torch.cat(
                (vec_a + vec_b, (vec_a - vec_b).abs(), vec_a * vec_b, rawcos, (vec_a * vec_b).sum(1, keepdim=True) / vec_a.shape[1]), dim=1
            )
            return model.pair(inp).squeeze(-1)

        lp = pair_logits(local_ep)
        targets = reliable_pos.float()
        mask = reliable_pos | reliable_neg
        if mask.any():
            loss = F.binary_cross_entropy_with_logits(lp[mask], targets[mask])
        else:
            loss = lp.sum() * 0
        # Feature-pivot bootstrap objective is not ground-truth edge supervision.
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == a.steps:
            trace.append({"step": step, "loss": float(loss.item()), "reliable_edge_fraction": float(mask.float().mean().item())})
            print(f"[exp040] step={step} loss={loss.item():.4f} reliable={mask.float().mean().item():.2%}", flush=True)

    model.eval()
    with torch.no_grad():
        z = model.embed(features)

        def predict_pairs(pairs):
            ps = torch.as_tensor(pairs, device=device, dtype=torch.long)
            out = []
            for chunk in ps.split(8192):
                v, w = z[chunk[:, 0]], z[chunk[:, 1]]
                sim = (x[chunk[:, 0]] * x[chunk[:, 1]]).sum(1, keepdim=True)
                inp = torch.cat((v + w, (v - w).abs(), v * w, sim, (v * w).sum(1, keepdim=True) / v.shape[1]), dim=1)
                out.append(torch.sigmoid(model.pair(inp).squeeze(-1)).cpu().numpy())
            return np.concatenate(out)

        def paired_diagnostic(name, pairs):
            probs = predict_pairs(pairs)
            truth = labels[pairs[:, 0]] == labels[pairs[:, 1]]
            from sklearn.metrics import roc_auc_score, average_precision_score

            return {
                "name": name,
                "pairs": len(pairs),
                "gt_same_rate": float(truth.mean()),
                "same_class_auc": float(roc_auc_score(truth, probs)) if len(np.unique(truth)) == 2 else None,
                "same_class_ap": float(average_precision_score(truth, probs)),
                "predicted_homophily_same_mean": float(probs[truth].mean()) if truth.any() else None,
                "predicted_homophily_different_mean": float(probs[~truth].mean()) if (~truth).any() else None,
            }

        random_pairs = rng.integers(n, size=(min(50000, len(held_edges)), 2), dtype=np.int64)
        edge_result = paired_diagnostic("heldout_observed_edges", held_edges)
        random_result = paired_diagnostic("random_pairs_including_nonedges", random_pairs)
        print("[exp040] GT diagnostics (labels NOT used in training):", edge_result, random_result, flush=True)
        # NAQ sampling supports identical to original protocol. Evaluation is
        # descriptive only; do not use GT to tune filters.
        supports = []
        for _ in range(a.num_tasks):
            supp, _, _, _ = proto_task_generator(a, data)
            supports.extend(np.asarray(supp).reshape(-1).astype(int).tolist())
        supports = list(dict.fromkeys(supports))
        pool_sizes = [n - 1 if v.strip() == "all" else int(v) for v in a.pool_sizes.split(",")]
        remove_fractions = [float(v) for v in a.remove_fractions.split(",")]
        results = {str(k): {str(f): [] for f in [0.0, *remove_fractions]} for k in pool_sizes}
        samples = []
        for si, support in enumerate(supports):
            feat_scores = x @ x[support]
            feat_scores[support] = -float("inf")
            ranked = torch.argsort(feat_scores, descending=True).cpu().numpy()
            anchor_gt = labels[support]
            for size in pool_sizes:
                pool = ranked[:size]
                pair = np.column_stack((np.full(len(pool), support, dtype=np.int64), pool))
                hom = predict_pairs(pair)
                # Filter most confident negatives (lowest predicted homophily).
                # Fixed pruning fractions, no GT threshold tuning.
                for fraction in [0.0, *remove_fractions]:
                    retain = max(a.q_query, int(round(len(pool) * (1 - fraction))))
                    selected = np.argsort(-hom, kind="stable")[:retain]
                    remain = pool[selected]
                    # Order retained candidates by RAW FEATURE similarity
                    # (not by learned score) then take its median q_query.
                    remain = remain[np.argsort(-feat_scores[torch.as_tensor(remain, device=device)].cpu().numpy(), kind="stable")]
                    begin = (len(remain) - a.q_query) // 2
                    middle = remain[begin : begin + a.q_query]
                    gt_pos = labels[pool] == anchor_gt
                    kept_pos = labels[remain] == anchor_gt
                    removed = np.ones(len(pool), dtype=bool)
                    removed[selected] = False
                    stats = {
                        "candidate_precision": float(gt_pos.mean()),
                        "negative_removal_rate": float(((~gt_pos) & removed).sum() / max((~gt_pos).sum(), 1)),
                        "removed_negative_precision": float(((~gt_pos) & removed).sum() / max(removed.sum(), 1)),
                        "positive_retention_rate": float((gt_pos & ~removed).sum() / max(gt_pos.sum(), 1)),
                        "remaining_precision": float(kept_pos.mean()),
                        "middle_query_precision": float((labels[middle] == anchor_gt).mean()),
                        "remaining_size": len(remain),
                    }
                    results[str(size)][str(fraction)].append(stats)
                    if len(samples) < a.sample_episodes and size == pool_sizes[0] and fraction in (0.0, 0.5):
                        letters = format_letters([anchor_gt, *labels[middle].tolist()])
                        samples.append(
                            {
                                "support": support,
                                "support_gt": letters[int(anchor_gt)],
                                "pool_size": size,
                                "removed_fraction": fraction,
                                "query_nodes": middle.tolist(),
                                "query_gt": [letters[int(v)] for v in labels[middle]],
                                "homophily_probs": [
                                    round(float(v), 3) for v in predict_pairs(np.column_stack((np.full(len(middle), support), middle)))
                                ],
                                "middle_precision": stats["middle_query_precision"],
                            }
                        )
            if (si + 1) % 100 == 0:
                print(f"[exp040] analyzed supports {si + 1}/{len(supports)}", flush=True)

    summary = {}
    for size, row in results.items():
        summary[size] = {}
        for fraction, values in row.items():
            summary[size][fraction] = {key: float(np.mean([v[key] for v in values])) for key in values[0]}
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "exp040",
        "method": "GREET-inspired feature-pivot edge discriminator (NOT official GREET)",
        "config": vars(a),
        "edge_diagnostic": edge_result,
        "random_pair_diagnostic": random_result,
        "training": trace,
        "sampled_supports": len(supports),
        "pool_results": summary,
        "examples": samples,
        "notes": [
            "GT used ONLY for diagnostics. No GT labels used to train/select/query.",
            "Discriminator is trained only on real edges, with relative raw-feature pivot pseudo-targets.",
            "Not GREET reproduction: no alternating dual-channel encoder and only cheap RW structural descriptors.",
            "Random pair diagnostics include pairs that may be observed edges, though mostly nonedges on sparse graphs.",
            "Full-pool middle is NOT comparable to oracle same-GT middle because candidate set is mixed-class.",
            "Pruning fractions are fixed ex ante and NOT selected using GT.",
        ],
    }
    dest = Path(a.json_path)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "node_dim": features.shape[1], "hidden": a.hidden, "config": vars(a)}, output / "discriminator.pt")
    lines = [
        f"exp040 GREET-inspired diagnostic: {a.dataset}",
        f"Unsupervised training edges: {len(train_edges)}; sampled NAQ supports: {len(supports)}",
        f"Heldout-edge GT AUC: {edge_result['same_class_auc']}",
        f"Random-pair GT AUC: {random_result['same_class_auc']}",
    ]
    for pool, row in summary.items():
        lines.append(f"\nPool feature-top-{pool}:")
        for fraction, st in row.items():
            lines.append(
                f"  removed={float(fraction):.0%} | removed-negative precision={st['removed_negative_precision']:.2%} | "
                f"positive retention={st['positive_retention_rate']:.2%} | "
                f"remaining precision={st['remaining_precision']:.2%} | "
                f"middle-K GT match={st['middle_query_precision']:.2%}"
            )
    (output / "result.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print("Saved:", dest, output / "result.txt", flush=True)


if __name__ == "__main__":
    main()
