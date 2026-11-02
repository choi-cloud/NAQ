"""hl012_feat_eval: HL11 §1 no-training feature arms; load_nodiff copied from hl/hl_common.py: load_everything, encoder_embeddings copied from hl/hl005_diag_distance_purity.py, h3 extended from hl/hl_common.py: structural_scores.

Labels: only the y_support / y_query of the fixed valid/test episodes (exp027.episode_accuracy). PCA, projection and diffusion never read labels.
Writes results/hl/hl11/feat/<arm>_<shot>shot_<pool>.npy (per-episode accuracy), selfcheck.json, selection.json.
"""
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hl_common as H  # noqa: E402
from data import load_data  # noqa: E402
from exp027 import episode_accuracy  # noqa: E402
from exp049 import load_splits  # noqa: E402
from model import GNNEncoder  # noqa: E402
from utils import seed_everything  # noqa: E402

OUT = Path("results/hl/hl11/feat")
SHOTS = (1, 5)
POOLS = ("valid", "test")
PKL = "save/downstream_tasks/Amazon_clothing/5way_{}shot.pkl"
MD5 = {1: "5f901727f81903cc12159d280be6915d", 5: "ca3edaca02d541cbc0ee15a2122ef24a"}
CKPTS = {
    "encb": "results/hl/p0/Amazon_clothing/ours/tau_0.3/seed_1212/best_1shot.pt",
    "encn": "results/hl/p0/Amazon_clothing/naq/baseline/seed_1212/best_1shot.pt",
}
# Selection candidates in table order (§1-2 then §1-3); fixed before any evaluation. Valid ties go to the earliest arm.
ARMS_FEATURE = ["x_l2", "x_raw", "ax_l2", "ax_raw", "a2x_l2", "a2x_raw", "a3x_l2", "a3x_raw", "cat_ax_a2x_l2", "sgc_mean_l2"]
ARMS_DIM = ["a2x_pca256_l2", "a2x_pca256_raw", "a2x_rp256_l2_s0", "a2x_rp256_l2_s1", "a2x_rp256_l2_s2", "a2x_pca64_l2", "a2x_pca1024_l2"]
ARMS_ENC = ["encb_raw", "encb_l2", "encn_raw", "encn_l2"]
RP_DIM = 256


def gpu_free_mib():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits", "-i", "0"], text=True)
    total, used = (int(x) for x in out.strip().split(","))
    return total - used


def load_nodiff(args):
    """hl_common.load_everything without add_diffusion (no dense X cosine matrix)."""
    args.setting, args.type, args.query_generation = "unsup", "feature", "NaQ"
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    return data, device


def powers(data, device):
    """X, ÂX, Â²X, Â³X, unnormalised, with the Â of hl_common.structural_scores."""
    edges = data.edge_index.detach().cpu().numpy()
    x = data.features.detach().float().to(device)
    n = x.size(0)
    adjacency = sp.csr_matrix((np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])), shape=(n, n))
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.float32).tocsr()
    adjacency.setdiag(1)
    adjacency.eliminate_zeros()
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    scale = np.maximum(degrees, 1) ** -0.5
    adjacency = (sp.diags(scale) @ adjacency @ sp.diags(scale)).tocoo()
    indices = torch.as_tensor(np.stack((adjacency.row, adjacency.col)), dtype=torch.long, device=device)
    values = torch.as_tensor(adjacency.data, dtype=torch.float32, device=device)
    a = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    h1 = torch.sparse.mm(a, x)
    h2 = torch.sparse.mm(a, h1)
    h3 = torch.sparse.mm(a, h2)
    return x, h1, h2, h3


def encoder_embeddings(data, path, latent, device, normalize):
    """hl005.encoder_embeddings with F.normalize optional."""
    config = [("GCN", (data.features.shape[1], 2 * latent), True), ("GCN", (2 * latent, latent), True), ("linear", (latent, 5), True)]
    net = GNNEncoder(config).to(device)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
    net.eval()
    with torch.no_grad():
        out = net(data.features, adj=data.adj)
    return F.normalize(out, dim=1) if normalize else out


def check_pkl_json(splits):
    """Exact equality (order included) of the pkl pools and the JSON episodes; stops on any mismatch."""
    result = {}
    for shot in SHOTS:
        with open(PKL.format(shot), "rb") as fh:
            valid_pool, test_pool = pickle.load(fh)
        for name, pool, episodes in (("valid", valid_pool, splits[shot][0]), ("test", test_pool, splits[shot][1])):
            assert len(pool) == len(episodes), f"{shot}-shot {name}: {len(pool)} != {len(episodes)}"
            for i, (p, e) in enumerate(zip(pool, episodes)):
                for j in range(4):
                    a = torch.as_tensor(p[j]).cpu().numpy().reshape(-1)
                    b = np.asarray(e[j]).reshape(-1)
                    assert np.array_equal(a, b), f"{shot}-shot {name} episode {i} field {j} differs"
            result[f"{shot}shot_{name}"] = {"episodes": len(pool), "equal_with_order": True}
    return result


class Evaluator:
    def __init__(self, splits):
        self.splits = splits
        self.summary = {}

    def __call__(self, arm, z):
        """Saves per-episode accuracy for every shot/pool. Returns {key: mean %}."""
        t = time.time()
        z = z.detach()
        row = {"dim": int(z.size(1))}
        with torch.no_grad():
            for shot in SHOTS:
                for i, pool in enumerate(POOLS):
                    scores = np.asarray([episode_accuracy(z, task, 5, shot) for task in self.splits[shot][i]], dtype=np.float64)
                    np.save(OUT / f"{arm}_{shot}shot_{pool}.npy", scores)
                    row[f"{pool}_{shot}shot"] = float(scores.mean() * 100)
        self.summary[arm] = row
        print(f"[hl012] {arm} dim={row['dim']} valid1={row['valid_1shot']:.2f} test1={row['test_1shot']:.2f} test5={row['test_5shot']:.2f} ({time.time()-t:.1f}s)", flush=True)
        return row


def first_best(summary, arms):
    values = {a: round(summary[a]["valid_1shot"], 6) for a in arms if a in summary}
    top = max(values.values())
    tied = [a for a, v in values.items() if v == top]
    return tied[0], tied, values


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    free = gpu_free_mib()
    print(f"[hl012] GPU free {free} MiB", flush=True)
    if free < 10240:
        raise SystemExit(f"STOP: GPU free {free} MiB < 10GB")
    torch.set_num_threads(4)
    seed_everything(1212)
    args = H.base_args()
    splits = load_splits(args)
    check = {"pkl_json": check_pkl_json(splits)}
    print(f"[hl012] pkl-JSON check {check['pkl_json']}", flush=True)
    data, device = load_nodiff(args)

    x, h1, h2, h3 = powers(data, device)
    z0, z1, z2, z3 = (F.normalize(t, dim=1) for t in (x, h1, h2, h3))
    r0, r1, r2 = H.structural_scores(data, device)
    diffs = [float((a - b).abs().max()) for a, b in ((z0, r0), (z1, r1), (z2, r2))]
    check["z_vs_structural_scores_max_abs"] = diffs
    assert max(diffs) < 1e-4, diffs  # CLAUDE.md §9: z1/z2 may differ from the 5th decimal between computations
    evaluate = Evaluator(splits)

    evaluate("x_l2", z0)
    evaluate("x_raw", x)
    evaluate("ax_l2", z1)
    evaluate("ax_raw", h1)
    a2x = evaluate("a2x_l2", z2)
    check["a2x_l2_with_structural_scores_z2"] = {
        f"test_{s}shot": float(np.mean([episode_accuracy(r2, t, 5, s) for t in splits[s][1]]) * 100) for s in SHOTS
    }
    check["a2x_l2_nd01"] = {"test_1shot": 85.0950, "test_5shot": 92.7150}
    check["a2x_l2_this"] = {k: a2x[k] for k in ("test_1shot", "test_5shot")}
    del r0, r1, r2
    evaluate("a2x_raw", h2)
    evaluate("a3x_l2", z3)
    evaluate("a3x_raw", h3)
    evaluate("cat_ax_a2x_l2", torch.cat((z1, z2), dim=1))
    evaluate("sgc_mean_l2", F.normalize((x + h1 + h2) / 3, dim=1))
    del h1, h2, h3, z3

    source = z2.cpu().numpy().astype(np.float32)
    for d in (256, 64, 1024):
        t = time.time()
        try:
            pca = PCA(n_components=d, svd_solver="randomized", random_state=1212)
            low = torch.as_tensor(pca.fit_transform(source), dtype=torch.float32, device=device)
        except MemoryError as exc:
            check[f"pca{d}_error"] = repr(exc)
            print(f"[hl012] PCA {d} failed: {exc!r}", flush=True)
            continue
        check[f"pca{d}"] = {"seconds": round(time.time() - t, 1), "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()), "dtype": str(low.dtype)}
        print(f"[hl012] PCA {d}: {check[f'pca{d}']}", flush=True)
        evaluate(f"a2x_pca{d}_l2", F.normalize(low, dim=1))
        if d == 256:
            evaluate("a2x_pca256_raw", low)
        del low
    del source
    for s in range(3):
        rng = np.random.default_rng(1212 + s)
        r = torch.as_tensor(rng.normal(0.0, np.sqrt(1.0 / RP_DIM), size=(z2.size(1), RP_DIM)), dtype=torch.float32, device=device)
        evaluate(f"a2x_rp256_l2_s{s}", F.normalize(z2 @ r, dim=1))
        del r

    for name, path in CKPTS.items():
        evaluate(f"{name}_raw", encoder_embeddings(data, path, args.latent, device, False))
        evaluate(f"{name}_l2", encoder_embeddings(data, path, args.latent, device, True))

    best, tied, values = first_best(evaluate.summary, ARMS_FEATURE + ARMS_DIM)
    best12, tied12, _ = first_best(evaluate.summary, ARMS_FEATURE)
    selection = {
        "rule": "highest valid 50 1-shot among ARMS_FEATURE + ARMS_DIM; ties go to the earliest arm in table order",
        "feat_best": best, "tied": tied, "feat_best_section1_2_only": best12, "tied_section1_2_only": tied12, "valid_1shot": values,
    }
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(evaluate.summary, indent=2), encoding="utf-8")
    check["seconds"] = round(time.time() - t0, 1)
    check["gpu_peak_allocated_mib"] = int(torch.cuda.max_memory_allocated() // 2**20) if device.type == "cuda" else None
    (OUT / "selfcheck.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    print(f"[hl012] feat_best={best} tied={tied}; section 1-2 only {best12}", flush=True)
    print(f"[hl012] DONE {check['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
