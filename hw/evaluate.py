"""5-way 5-shot mean-prototype (squared Euclidean) evaluation.

Transcribed from choi_exps/zip_20261107/exp015.py `evaluate`.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(embeddings, task_pool, n_way, device):
    accuracies = []
    margins = []
    for id_spt, y_spt, id_query, y_query in task_pool:
        supports = embeddings[id_spt]
        queries = embeddings[id_query]
        labels_s = y_spt.to(device)
        labels_q = y_query.to(device)
        prototypes = torch.stack([supports[labels_s == cls].mean(dim=0) for cls in range(n_way)])
        dists = torch.cdist(queries, prototypes).square()
        prediction = dists.argmin(dim=1)
        accuracies.append(float((prediction == labels_q).float().mean().item()))
        correct = dists.gather(1, labels_q[:, None]).squeeze(1)
        wrong = dists.clone()
        wrong.scatter_(1, labels_q[:, None], float("inf"))
        margins.append(float((wrong.min(dim=1).values - correct).mean().item()))
    x = np.asarray(accuracies, dtype=np.float64)
    return {
        "accuracy_mean": float(x.mean()),
        "accuracy_std": float(x.std(ddof=0)),
        "accuracy_sem": float(x.std(ddof=0) / math.sqrt(len(x))),
        "accuracy_ci95_half_width": float(1.96 * x.std(ddof=0) / math.sqrt(len(x))),
        "mean_true_class_distance_margin": float(np.mean(margins)),
        "num_tasks": len(x),
        "n_way": n_way,
        "k_shot": int(task_pool[0][0].size // n_way),
    }


@torch.no_grad()
def evaluate_l2(embeddings, task_pool, n_way, device):
    """HW08 l2 evaluation: L2 row normalization (same expression as hw/trainfree.py
    `F.normalize(feat, p=2, dim=1)`, eps 1e-12), then the unchanged `evaluate`."""
    return evaluate(F.normalize(embeddings, p=2, dim=1, eps=1e-12), task_pool, n_way, device)
