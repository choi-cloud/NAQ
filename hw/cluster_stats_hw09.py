"""HW09-b §2-3: k-means (k 100) cluster statistics and NaQ-Feat top-10 pseudo-query purity, before training.

    PYTHONPATH=. python -m hw.cluster_stats_hw09 [--datasets dblp Amazon_eletronics]

measurement-only label use: every class ratio below reads GT labels; nothing is trained or selected.
- cluster: cache/kmeans/<ds>_k100.pt (hw.negatives.kmeans_labels: Â²X L2, KMeans k 100, n_init 4, random_state 20261108).
  Same definitions as hw/neg_tables.py (HW06 G-table): size min / median / max; same-class ratio within the node's
  cluster (other members, node macro); same-class ratio among nodes of the other clusters (node macro).
- random expected FN rate Σ_c p_c² (p_c = class share of all nodes).
- NaQ-Feat top-10 purity: task_generator.query_generation_NaQ (top-10 of data.Diffusion_matrix = raw-feature cos,
  diagonal 0) applied to every node; ratio of the 10 pseudo-queries with the node's class, node macro (`all_nodes`);
  and `query_eq_support` over the seed-1212 training pool of h9_naq (4000 episodes, hw/run.py RNG order:
  seed -> load_data -> set_device -> add_diffusion -> ProtoNet -> train_pool), pooled over (query, own support) pairs.
Output: results/measure/HW09_cluster_stats.{json,md}. Cora_full / Amazon_clothing cluster rows are also copied from
results/hw_iislab4/HW06_tables.json (source column).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import load_data
from meta import ProtoNet
from task_generator import query_generation_NaQ
from utils import seed_everything

from hw.config import build_args, load_config
from hw.episodes import build_train_pool
from hw.negatives import KMEANS_DEFAULT_K, kmeans_labels

ALL = ("Cora_full", "Amazon_clothing", "dblp", "Amazon_eletronics")
OUT = Path("results/measure/HW09_cluster_stats")


@torch.no_grad()
def stats(dataset, device):
    cfg = load_config("h9_naq")
    cfg["epochs"] = int(cfg["episodes"])
    args = build_args(cfg, dataset)
    seed_everything(1212)
    data = load_data(args)
    data.set_device(str(device))
    data.add_diffusion(args)
    labels = data.labels.detach().cpu().long()
    n = labels.numel()
    clus = kmeans_labels(dataset, data.features, data.adj, KMEANS_DEFAULT_K).cpu()
    k = KMEANS_DEFAULT_K
    sizes = torch.bincount(clus, minlength=k)
    n_class = int(labels.max()) + 1
    cont = torch.zeros((k, n_class), dtype=torch.long)
    cont.index_put_((clus, labels), torch.ones(n, dtype=torch.long), accumulate=True)  # measurement-only label use
    class_sizes = cont.sum(dim=0)
    n_kc = cont[clus, labels].float()
    size_k = sizes[clus].float()
    same_in = torch.where(size_k > 1, (n_kc - 1) / (size_k - 1).clamp_min(1), torch.zeros_like(n_kc))
    same_out = (class_sizes[labels].float() - n_kc) / (n - size_k)
    p = class_sizes.float() / n
    out = {"dataset": dataset, "num_nodes": n, "classes": n_class, "cluster": {
        "k": k, "size_min": int(sizes.min()), "size_median": float(sizes.float().median()), "size_max": int(sizes.max()),
        "same_class_within_cluster_node_macro": float(same_in.mean()),
        "same_class_other_clusters_node_macro": float(same_out.mean())},
        "random_expected_fn_sum_pc2": float((p * p).sum())}
    # NaQ-Feat top-10 over all nodes
    q = args.q_query
    same_frac = []
    for start in range(0, n, 2048):
        ids = np.arange(start, min(n, start + 2048))
        id_query, _ = query_generation_NaQ(ids, torch.zeros(len(ids), dtype=torch.long), q, data, True)
        lab_q = labels[torch.as_tensor(id_query)].view(len(ids), q)
        same_frac.append((lab_q == labels[torch.as_tensor(ids)][:, None]).float().mean(dim=1))
    out["naq_top10_purity_all_nodes_node_macro"] = float(torch.cat(same_frac).mean())
    # seed-1212 h9_naq training pool (4000 episodes)
    d = data.features.shape[1]
    config = [("GCN", (d, 2 * args.latent), True), ("GCN", (2 * args.latent, args.latent), True), ("linear", (args.latent, args.n_way), True)]
    ProtoNet(args, config, data)  # RNG consumption as in hw/run.py; never used
    pool = build_train_pool(args, data)
    same, total = 0, 0
    for id_spt, y_spt, id_query, y_query in pool:
        sl = labels[torch.as_tensor(id_spt)]
        ys = torch.as_tensor(y_spt).long()
        yq = torch.as_tensor(y_query).long()
        lab_of_pseudo = torch.empty(int(ys.max()) + 1, dtype=torch.long)
        lab_of_pseudo[ys] = sl  # 1 support per pseudo-class (k_shot 1)
        same += int((labels[torch.as_tensor(id_query)] == lab_of_pseudo[yq]).sum())
        total += int(yq.numel())
    out["query_eq_support_seed1212_pool"] = same / total
    out["pool_episodes"] = len(pool)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(ALL))
    parser.add_argument("--device", type=int, default=0)
    cli = parser.parse_args()
    device = torch.device(f"cuda:{cli.device}")
    torch.cuda.set_device(device)
    res = {ds: stats(ds, device) for ds in cli.datasets}
    hw06 = json.loads(Path("results/hw_iislab4/HW06_tables.json").read_text())
    res["HW06_G_table"] = {ds: {"cluster": hw06[ds]["cluster"], "random_expected_fn_sum_pc2": hw06[ds]["random_expected_fn_sum_pc2"]} for ds in ("Cora_full", "Amazon_clothing")}
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    dss = cli.datasets
    rows = [
        ("노드 수", lambda r: f"{r['num_nodes']}"),
        ("cluster (k 100): 크기 최소 / 중앙값 / 최대", lambda r: f"{r['cluster']['size_min']} / {r['cluster']['size_median']:.0f} / {r['cluster']['size_max']}"),
        ("cluster: 같은 군집 내 같은 클래스 비율 (노드 macro)", lambda r: f"{r['cluster']['same_class_within_cluster_node_macro']:.4f}"),
        ("cluster: 다른 군집 노드 중 같은 클래스 비율 (노드 macro)", lambda r: f"{r['cluster']['same_class_other_clusters_node_macro']:.4f}"),
        ("random 기대 FN rate Σ p_c²", lambda r: f"{r['random_expected_fn_sum_pc2']:.4f}"),
        ("NaQ-Feat top-10 순도, 전체 노드 (노드 macro)", lambda r: f"{r['naq_top10_purity_all_nodes_node_macro']:.4f}"),
        ("NaQ-Feat top-10 순도, seed 1212 학습 pool 4000 episode (`query_eq_support`, 쌍 비율)", lambda r: f"{r['query_eq_support_seed1212_pool']:.4f}"),
    ]
    lines = ["# HW09 cluster statistics (before training; measurement-only label use)", "",
             "k-means: Â²X L2 정규화, k 100, n_init 4, random_state 20261108 (`cache/kmeans/<ds>_k100.pt`). 정의는 `hw/cluster_stats_hw09.py`.", "",
             "| 항목 | " + " | ".join(dss) + " |", "|---|" + "---|" * len(dss)]
    for name, f in rows:
        lines.append(f"| {name} | " + " | ".join(f(res[ds]) for ds in dss) + " |")
    lines += ["", "HW06 G-table (`results/hw_iislab4/HW06_tables.json`) 값:", "", "| 항목 | Cora_full | Amazon_clothing |", "|---|---|---|"]
    for name, f in rows[1:5]:
        lines.append(f"| {name} | " + " | ".join(f(res["HW06_G_table"][ds]) for ds in ("Cora_full", "Amazon_clothing")) + " |")
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
