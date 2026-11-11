"""HW09 §0 G-data and §1.1/§1.4 cache records -> results/hw_iislab4/HW09_data.json.

    PYTHONPATH=. python -m hw.data_hw09 [--build-tables]

Per dataset (Cora_full, Amazon_clothing, dblp, Amazon_eletronics): data/ files and sizes, loader string,
nodes, undirected edges (unique pairs u<v of edge_index, self loops excluded), feature dim, classes
(all / train / valid / test as data.py splits them under seed 1212), and the md5 / task counts / class sets
of save/downstream_tasks/<ds>/5way_{5,1}shot.pkl and save/valid500/<ds>.pkl.
--build-tables also builds (if missing) cache/kmeans/<ds>_k100.pt and cache/rawcos/<ds>_top500.pt with the
unchanged hw/negatives.py functions (k 100, n_init 4, random_state 20261108).
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from data import load_data
from hw.config import build_args, load_config
from hw.negatives import kmeans_labels, raw_cos_table
from utils import seed_everything

DATASETS = ("Cora_full", "Amazon_clothing", "dblp", "Amazon_eletronics")
OUT = Path("results/hw_iislab4/HW09_data.json")


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def files(ds):
    if ds == "Cora_full":
        paths = sorted(p for p in Path("data/corafull").rglob("*") if p.is_file())
    else:
        paths = [Path(f"data/{ds}_{s}") for s in ("network", "train.mat", "test.mat")]
    return {str(p): (p.stat().st_size if p.exists() else None) for p in paths}


def pool_classes(pool, labels):
    # measurement-only label use: class sets of cached tasks
    return sorted({int(c) for t in pool for c in labels[np.concatenate([np.asarray(t[0]), np.asarray(t[2])])]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-tables", action="store_true")
    cli = parser.parse_args()
    out = {}
    for ds in DATASETS:
        row = {"loader_dataset_string": ds, "files": files(ds)}
        args = build_args(load_config("naq_base"), ds)
        seed_everything(1212)
        data = load_data(args)
        ei = data.edge_index
        u, v = ei[0], ei[1]
        mask = u != v
        pairs = torch.stack([torch.minimum(u, v)[mask], torch.maximum(u, v)[mask]])
        labels = data.labels.cpu().numpy()
        row.update(nodes=int(data.features.shape[0]), edge_index_entries=int(ei.shape[1]), undirected_edges=int(torch.unique(pairs, dim=1).shape[1]),
                   feature_dim=int(data.features.shape[1]), classes=int(len(np.unique(labels))),
                   classes_train=len(data.class_list_train), classes_valid=len(data.class_list_valid), classes_test=len(data.class_list_test))
        caches = {}
        for k in (5, 1):
            p = Path(f"save/downstream_tasks/{ds}/5way_{k}shot.pkl")
            if p.exists():
                valid, test = pickle.load(open(p, "rb"))
                caches[f"5way_{k}shot"] = {"md5": md5(p), "valid_tasks": len(valid), "test_tasks": len(test),
                                           "k_shot": int(np.asarray(test[0][0]).size // 5), "q_query": int(np.asarray(test[0][2]).size // 5),
                                           "test_classes": pool_classes(test, labels), "valid_classes": pool_classes(valid, labels)}
            else:
                caches[f"5way_{k}shot"] = None
        p = Path(f"save/valid500/{ds}.pkl")
        caches["valid500"] = {"md5": md5(p), "tasks": len(pickle.load(open(p, "rb")))} if p.exists() else None
        if caches["5way_5shot"] and caches["5way_1shot"]:
            caches["test_classes_1shot_eq_5shot"] = caches["5way_1shot"]["test_classes"] == caches["5way_5shot"]["test_classes"]
            caches["n_test_classes_used"] = len(caches["5way_5shot"]["test_classes"])
        row["caches"] = caches
        if cli.build_tables:
            row["kmeans_k100_cluster_count"] = int(kmeans_labels(ds, data.features, data.adj).unique().numel())
            row["rawcos_top500_shape"] = list(raw_cos_table(ds, data.features)["ids"].shape)
        row["tables"] = {str(p): md5(p) for p in (Path(f"cache/kmeans/{ds}_k100.pt"), Path(f"cache/rawcos/{ds}_top500.pt")) if p.exists()}
        out[ds] = row
        print(ds, {k: v for k, v in row.items() if k not in ("caches", "files")}, flush=True)
    for ds in DATASETS:  # compact cache print (no class lists)
        print(ds, {k: ({kk: vv for kk, vv in v.items() if not kk.endswith("classes")} if isinstance(v, dict) else v) for k, v in out[ds]["caches"].items()})
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
