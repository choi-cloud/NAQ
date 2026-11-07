"""HW03a-s3: train-free neighbor purity of the 6 selectors {raw, bgrl} x {max, contrast, random}.

    python -m hw.purity --dataset <ds> [--seed 1212] [--episodes 2000]

The training episode pool is built along the hw/run.py path (seed -> load_data -> set_device ->
add_diffusion -> (selection embeddings, no RNG) -> ProtoNet creation (consumes RNG, model never
trained) -> neighbor lookup -> train_pool), so it is the pool a seed-<S> run trains on.
Selection uses hw.neighbors.select_neighbors_pick (HW03a-a); per-episode measurement uses
hw.run.neighbor_diagnostics. `random` uses its own torch.Generator seeded seed + 7919, drawn
once per episode in episode order, as in hw/run.py.

measurement-only label use: GT labels are read only for these diagnostics; nothing is trained.
Output: results/hw/purity/HW03a_purity_<ds>.{json,md} (root: $HW_RESULTS_ROOT, default results/hw).
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import load_data
from meta import ProtoNet
from utils import seed_everything

from hw.config import build_args, load_config
from hw.episodes import build_train_pool
from hw.neighbors import _eligible, build_neighbor_lookup, load_selection_embeddings, select_neighbors_pick
from hw.run import RANDOM_PICK_SEED_OFFSET, neighbor_diagnostics

RESULTS_ROOT = Path(os.environ.get("HW_RESULTS_ROOT", "results/hw"))
SPACES = ("raw", "bgrl")
PICKS = ("max", "contrast", "random")
SEL_KEYS = ("nbr_eq_support", "nbr_eq_query", "pos_nbr_cos_raw")


def parse_cli():
    parser = argparse.ArgumentParser(description="train-free neighbor purity (HW03a-s3)")
    parser.add_argument("--dataset", required=True, choices=["Cora_full", "Amazon_clothing"])
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--arm", default="nbr_bgrl", help="arm whose config gives protocol + bgrl_dir")
    return parser.parse_args()


@torch.no_grad()
def support_homophily(labels, id_spt, y_spt, id_query, y_query, neighbor_lookup):
    """(same-GT count, eligible-neighbor count) over own-class (query, support) pairs, before selection.

    measurement-only label use."""
    device = labels.device
    support_ids = torch.as_tensor(id_spt, dtype=torch.long, device=device).flatten()
    query_ids = torch.as_tensor(id_query, dtype=torch.long, device=device).flatten()
    sp = torch.as_tensor(y_spt, dtype=torch.long, device=device).flatten()
    qp = torch.as_tensor(y_query, dtype=torch.long, device=device).flatten()
    valid, (neighbor_ids, _) = _eligible(support_ids, query_ids, neighbor_lookup, 0)  # [Q,S,K], [S,K]
    qi, si = (qp[:, None] == sp[None, :]).nonzero(as_tuple=True)
    v = valid[qi, si]  # [P, K]
    same = labels[neighbor_ids[si]] == labels[support_ids[si]][:, None]  # measurement-only label use
    return int((same & v).sum().item()), int(v.sum().item())


def main():
    cli = parse_cli()
    started = time.time()
    cfg = load_config(cli.arm)
    cfg["seed"] = cli.seed
    cfg["epochs"] = cli.episodes
    args = build_args(cfg, cli.dataset)
    torch.set_num_threads(4)
    seed_everything(cli.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    data.add_diffusion(args)
    features = data.features
    num_nodes, d = features.shape
    selection = {
        "raw": load_selection_embeddings("raw", features, cli.dataset, num_nodes, device)[0],
        "bgrl": load_selection_embeddings("bgrl", features, cli.dataset, num_nodes, device, cfg.get("bgrl_dir"))[0],
    }
    raw_normalized = F.normalize(features.detach(), p=2, dim=1)
    config = [("GCN", (d, 2 * args.latent), True), ("GCN", (2 * args.latent, args.latent), True), ("linear", (args.latent, args.n_way), True)]
    ProtoNet(args, config, data).to(device)  # created only to keep the RNG stream identical to hw/run.py
    neighbor_lookup = build_neighbor_lookup(data.edge_index, num_nodes)
    train_pool = build_train_pool(args, data)
    labels = data.labels

    generators = {}
    for space in SPACES:
        g = torch.Generator(device=device)
        g.manual_seed(cli.seed + RANDOM_PICK_SEED_OFFSET)
        generators[space] = g

    per_selector = {(s, p): {k: [] for k in SEL_KEYS} for s in SPACES for p in PICKS}
    shared = {"query_eq_support": [], "fallback_frac": [], "multi_nbr_frac": []}
    homo_same = homo_total = 0
    for id_spt, y_spt, id_query, y_query in train_pool:
        support_ids = torch.as_tensor(id_spt, dtype=torch.long, device=device).flatten()
        query_ids = torch.as_tensor(id_query, dtype=torch.long, device=device).flatten()
        same, total = support_homophily(labels, id_spt, y_spt, id_query, y_query, neighbor_lookup)
        homo_same += same
        homo_total += total
        for space in SPACES:
            for pick in PICKS:
                out = {}
                select_neighbors_pick(selection[space], support_ids, query_ids, neighbor_lookup, 0, pick, generators[space] if pick == "random" else None, out)
                diag = neighbor_diagnostics(labels, raw_normalized, id_spt, y_spt, id_query, y_query, out)
                for k in SEL_KEYS:
                    per_selector[(space, pick)][k].append(diag[k])
                if space == "raw" and pick == "max":
                    for k in shared:
                        shared[k].append(diag[k])

    # every episode has the same number of own-class pairs (n_way * q_query), so mean of means = pooled mean
    result = {
        "dataset": cli.dataset,
        "seed": cli.seed,
        "episodes": len(train_pool),
        "pairs_per_episode": args.n_way * args.q_query,
        "arm_config": cli.arm,
        "bgrl_path": str(Path(cfg.get("bgrl_dir")) / f"{cli.dataset}.pt"),
        "reference": {k: float(np.mean(v)) for k, v in shared.items()},
        "selectors": {f"{s}_{p}": {k: float(np.mean(v)) for k, v in per_selector[(s, p)].items()} for s in SPACES for p in PICKS},
        "elapsed_seconds": None,
        "note": "measurement-only label use; no training",
    }
    result["reference"]["support_1hop_homophily"] = homo_same / max(1, homo_total)
    result["elapsed_seconds"] = round(time.time() - started, 2)

    out_dir = RESULTS_ROOT / "purity"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"HW03a_purity_{cli.dataset}.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    ref = result["reference"]
    lines = [
        f"# HW03a neighbor purity — {cli.dataset} (seed {cli.seed}, {len(train_pool)} episodes × {result['pairs_per_episode']} own-class pairs, no training)",
        "",
        "| 참고 행 | 값 |",
        "|---|---|",
        f"| query_eq_support | {ref['query_eq_support']:.4f} |",
        f"| support_1hop_homophily | {ref['support_1hop_homophily']:.4f} |",
        f"| multi_nbr_frac | {ref['multi_nbr_frac']:.4f} |",
        f"| fallback_frac | {ref['fallback_frac']:.4f} |",
        "",
        "| 선택기 (space_pick) | nbr_eq_support | nbr_eq_query | pos_nbr_cos_raw |",
        "|---|---|---|---|",
    ]
    for name, vals in result["selectors"].items():
        lines.append(f"| {name} | {vals['nbr_eq_support']:.4f} | {vals['nbr_eq_query']:.4f} | {vals['pos_nbr_cos_raw']:.4f} |")
    (out_dir / f"HW03a_purity_{cli.dataset}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"elapsed {result['elapsed_seconds']} s")


if __name__ == "__main__":
    main()
