"""hl016_norm_diag: HL13 §3 norm diagnostics (no training). encoder_embeddings/powers imported from hl/hl012_feat_eval.py. Labels: fixed test episode y_support/y_query for scoring and GT classes for the between/within norm variance only."""
import csv
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hl_common as H  # noqa: E402
import hl012_feat_eval as E  # noqa: E402
from exp027 import episode_accuracy  # noqa: E402
from exp049 import load_splits  # noqa: E402
from utils import seed_everything  # noqa: E402

CKPTS = {
    "encb": "results/hl/p0/Amazon_clothing/ours/tau_0.3/seed_1212/best_1shot.pt",
    "encn": "results/hl/p0/Amazon_clothing/naq/baseline/seed_1212/best_1shot.pt",
    "gcn_cos_naq": "results/hl/hl12/curve/gcn_cos_naq/seed_1212/best_l2_1.pt",
    "gcn_cos_ours": "results/hl/hl12/curve/gcn_cos_ours/seed_1212/best_l2_1.pt",
    "gcn_raw_sup": "results/hl/hl12/curve/gcn_raw_sup/seed_1212/best_l2_1.pt",
}
OUT = Path("reports/HL13_tables/T30_norm_diag.csv")


def predictions(z, task):
    """Formula of exp027.episode_accuracy for 1-shot: (correct per query [40])."""
    supports, y_support, queries, y_query = (torch.as_tensor(np.asarray(x), dtype=torch.long, device=z.device) for x in task)
    distances = (z[queries, None, :] - z[supports][None, :, :]).square().sum(-1)
    targets = (y_query[:, None] == y_support[None, :]).long().argmax(dim=1)
    return distances.argmin(dim=-1) == targets, targets


def variance_ratio(norm, labels, classes):
    means, within = [], []
    for c in classes:
        values = norm[labels == c]
        means.append(values.mean())
        within.append(values.var())
    return float(np.var(means) / np.mean(within))


def main():
    torch.set_num_threads(4)
    seed_everything(1212)
    args = H.base_args()
    splits = load_splits(args)
    data, device = E.load_nodiff(args)
    labels = H.labels_np(data)
    edges = data.edge_index.detach().cpu().numpy()
    n = data.features.size(0)
    adjacency = sp.csr_matrix((np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])), shape=(n, n))
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.float32).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    _, _, h2, _ = E.powers(data, device)
    h2_norm = h2.norm(dim=1).cpu().numpy()
    del h2
    tasks = splits[1][1]
    test_classes = sorted(set(labels[np.concatenate([t[0] for t in tasks])].tolist()))
    rows = []
    for name, path in CKPTS.items():
        z = E.encoder_embeddings(data, path, args.latent, device, False)
        l2 = F.normalize(z, dim=1)
        norm = z.norm(dim=1).cpu().numpy()
        gaps, raw_wrong, rescued, raw_ok, l2_ok, total = [], 0, 0, 0, 0, 0
        with torch.no_grad():
            for task in tasks:
                c_raw, targets = predictions(z, task)
                c_l2, _ = predictions(l2, task)
                raw_wrong += int((~c_raw).sum())
                rescued += int(((~c_raw) & c_l2).sum())
                raw_ok += int(c_raw.sum())
                l2_ok += int(c_l2.sum())
                total += len(c_raw)
                supports, queries = np.asarray(task[0]), np.asarray(task[2])
                y_support, y_query = np.asarray(task[1]), np.asarray(task[3])
                for way in range(5):
                    gaps.append(norm[supports[way]] - norm[queries[y_query == y_support[way]]].mean())
            check_raw = float(np.mean([episode_accuracy(z, t, 5, 1) for t in tasks]) * 100)
        assert abs(check_raw - raw_ok / total * 100) < 1e-3
        q10, q50, q90 = np.quantile(norm, [0.1, 0.5, 0.9])
        gaps = np.asarray(gaps)
        rows.append({
            "checkpoint": name, "norm_q10": f"{q10:.4f}", "norm_q50": f"{q50:.4f}", "norm_q90": f"{q90:.4f}", "norm_std_over_mean": f"{norm.std() / norm.mean():.4f}",
            "spearman_norm_degree": f"{spearmanr(norm, degree)[0]:.4f}", "spearman_norm_a2x_rownorm": f"{spearmanr(norm, h2_norm)[0]:.4f}",
            "support_minus_query_norm_mean": f"{gaps.mean():.4f}", "support_minus_query_norm_std": f"{gaps.std():.4f}",
            "test_1shot_raw": f"{raw_ok / total * 100:.4f}", "test_1shot_l2": f"{l2_ok / total * 100:.4f}",
            "raw_wrong_queries": raw_wrong, "norm_induced_error": f"{rescued / raw_wrong:.4f}",
            "between_over_within_all77": f"{variance_ratio(norm, labels, sorted(set(labels.tolist()))):.4f}",
            "between_over_within_test20": f"{variance_ratio(norm, labels, test_classes):.4f}",
        })
        print(rows[-1], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
