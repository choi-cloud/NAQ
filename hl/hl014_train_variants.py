"""hl014_train_variants: copied from hl/hl013_curve.py (itself from choi_exps/exp049.py: main); ProtoNetV copied from meta.py: ProtoNet with encoder and loss branches; adds episode sources sup/sup_strat and selection rules R/L.

Labels: base-class labels create sup/sup_strat episodes. Base = (train.mat classes) − (classes of the fixed valid pkl) = 40 classes, the same for every seed.
data.class_list_train is NOT used: load_data re-splits train.mat by seed, so for seeds 1213-1216 it holds 10-12 classes of the fixed valid episodes (CLAUDE.md §4).
Fixed valid/test episode labels are read for the meta-test evaluation and, class ids of the valid pool only, to exclude those classes. query_diagnostics reads labels for diagnostics. PCA reads no labels.
Epoch 0 is recorded in history but is not a checkpoint candidate (as hl013).
"""
import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import optim

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hl_common as H  # noqa: E402
from data import load_data  # noqa: E402
from exp027 import episode_accuracy  # noqa: E402
from exp049 import build_structural_queries, load_splits, query_diagnostics  # noqa: E402
from model import GNNEncoder  # noqa: E402
from task_generator import proto_task_generator, supervised_task_generator  # noqa: E402
from utils import seed_everything  # noqa: E402

FEAT = Path("results/hl/hl12/feat")
HL11_T18 = Path("reports/HL11_tables/T18_feat.csv")
PKL = "save/downstream_tasks/Amazon_clothing/5way_{}shot.pkl"
DIMS = {"pca64_lin": 64, "pca256_lin": 256, "pca256_mlp": 256}
SHOTS = (1, 5)
RULES = {"R": "", "L": "_l2"}
STRATA = 5
PER_STRATUM = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prepare_feat", action="store_true")
    p.add_argument("--dataset", default="Amazon_clothing")
    p.add_argument("--episodes", choices=["naq", "ours", "sup", "sup_strat"])
    p.add_argument("--encoder", choices=["gcn", "pca64_lin", "pca256_lin", "pca256_mlp"], default="gcn")
    p.add_argument("--loss", choices=["raw", "cos"], default="raw")
    p.add_argument("--scale", type=float, default=10.0)
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
    p.add_argument("--output_root", default="results/hl/hl12/curve")
    return p.parse_args()


def arm_name(args):
    name = f"{args.encoder}_{args.loss}_{args.episodes}"
    return name if args.scale == 10.0 else f"{name}_s{args.scale:g}"


def hl11_value(arm, column="test_1shot"):
    with HL11_T18.open(encoding="utf-8") as fh:
        return float(next(r for r in csv.DictReader(fh) if r["arm"] == arm)[column])


def load_nodiff(args, device):
    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    return data


def prepare_feat(args):
    """PCA(d) of the row-normalised Â²X (no labels) -> row l2, saved once. Checked against the HL11 per-episode accuracies."""
    FEAT.mkdir(parents=True, exist_ok=True)
    args.setting, args.type, args.query_generation = "unsup", "feature", "NaQ"
    torch.set_num_threads(4)
    seed_everything(1212)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    data = load_nodiff(args, device)
    _, _, z2 = H.structural_scores(data, device)
    source = z2.cpu().numpy().astype(np.float32)
    del z2
    args.seed = 1212
    splits = load_splits(args)
    meta = {}
    for d in (64, 256):
        path = FEAT / f"pca{d}.npy"
        if path.exists():
            raise FileExistsError(path)
        pca = PCA(n_components=d, svd_solver="randomized", random_state=1212)
        low = F.normalize(torch.as_tensor(pca.fit_transform(source), dtype=torch.float32, device=device), dim=1)
        np.save(path, low.cpu().numpy())
        row = {"md5": hashlib.md5(path.read_bytes()).hexdigest(), "shape": list(low.shape), "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum())}
        with torch.no_grad():
            for shot in SHOTS:
                scores = np.asarray([episode_accuracy(low, t, 5, shot) for t in splits[shot][1]])
                old = np.load(f"results/hl/hl11/feat/a2x_pca{d}_l2_{shot}shot_test.npy")
                row[f"test_{shot}shot"] = float(scores.mean() * 100)
                row[f"hl11_test_{shot}shot"] = float(old.mean() * 100)
                row[f"episodes_differing_from_hl11_{shot}shot"] = int((scores != old).sum())
        meta[f"pca{d}"] = row
        print(f"[hl014] pca{d} {row}", flush=True)
        assert abs(row["test_1shot"] - hl11_value(f"a2x_pca{d}_l2")) <= 0.05, row
    (FEAT / "pca_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


class PcaEncoder(nn.Module):
    """Input = saved PCA matrix. `lin`: Linear(d, d) with identity weight and zero bias. `mlp`: Linear -> ReLU -> Linear, default init."""

    def __init__(self, kind, dim):
        super().__init__()
        if kind == "lin":
            self.net = nn.Linear(dim, dim, bias=True)
            with torch.no_grad():
                self.net.weight.copy_(torch.eye(dim))
                self.net.bias.zero_()
        else:
            self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x):
        return self.net(x)


class ProtoNetV(nn.Module):
    """meta.ProtoNet with an encoder branch and a loss branch (`raw` = meta.ProtoNet.forward; `cos` = -s * squared distance of normalised vectors)."""

    def __init__(self, args, config, data, device, features=None):
        super().__init__()
        self.Data = data
        self.encoder = args.encoder
        self.loss_kind = args.loss
        self.scale = args.scale
        self.n, self.k = args.n_way, args.k_shot
        self.device_name = device
        if args.encoder == "gcn":
            self.network = GNNEncoder(config)
        else:
            self.network = PcaEncoder("lin" if args.encoder.endswith("lin") else "mlp", DIMS[args.encoder])
        self.pca_features = features
        self.meta_optimizer = optim.Adam(self.network.parameters(), lr=args.lr)

    def encode(self):
        if self.encoder == "gcn":
            return self.network(self.Data.features, adj=self.Data.adj)
        return self.network(self.pca_features)

    def forward(self, id_spt, y_spt, id_query, y_query):
        encodings = self.encode()
        if self.loss_kind == "cos":
            encodings = F.normalize(encodings, dim=1)
        x_spt, x_query = encodings[id_spt], encodings[id_query]
        prototypes = x_spt.view(self.n, self.k, x_spt.size(1)).mean(dim=1)
        if self.loss_kind == "cos":
            prototypes = F.normalize(prototypes, dim=1)
        dists = self.euclidean_dist(x_query, prototypes)
        output = F.log_softmax(-dists if self.loss_kind == "raw" else -self.scale * dists, dim=1)
        label_new = torch.LongTensor([y_spt.tolist().index(i) for i in y_query.tolist()]).to(self.device_name)
        loss = F.nll_loss(output, label_new)
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()
        preds = output.cpu().detach().max(1)[1].type_as(label_new.cpu())
        return loss, preds.eq(label_new.cpu()).double().sum() / len(label_new)

    def euclidean_dist(self, x, y):
        n, m, d = x.size(0), y.size(0), x.size(1)
        return torch.pow(x.unsqueeze(1).expand(n, m, d) - y.unsqueeze(0).expand(n, m, d), 2).sum(dim=2)


def base_classes(data, labels):
    """(train.mat classes = class_list_train ∪ class_list_valid) − classes of the fixed valid pkl; disjoint from the fixed test classes."""
    train_mat = {float(c) for c in data.class_list_train} | {float(c) for c in data.class_list_valid}
    classes = {}
    for shot in SHOTS:
        with open(PKL.format(shot), "rb") as fh:
            valid_pool, test_pool = pickle.load(fh)
        for name, pool in (("valid", valid_pool), ("test", test_pool)):
            found = set(labels[np.concatenate([np.asarray(e[0]).reshape(-1) for e in pool])].astype(float).tolist())
            classes.setdefault(name, set()).update(found)
    base = sorted(train_mat - classes["valid"] - classes["test"])
    assert len(train_mat) == 57 and len(classes["valid"]) == 17 and len(classes["test"]) == 20 and len(base) == 40, (len(train_mat), len(base))
    assert not set(base) & classes["test"] and not set(base) & classes["valid"]
    return base


def make_sup_episodes(args, data, labels, device):
    """sup and sup_strat share classes and supports for one seed: both consume `random` only through supervised_task_generator."""
    base = base_classes(data, labels)
    id_by_class = data.id_by_class
    _, _, z2 = H.structural_scores(data, device)
    rng = np.random.default_rng(args.seed)
    tasks = []
    q = args.q_query
    query_cos, strata_cos, drawn_with_replacement = [], [[] for _ in range(STRATA)], 0
    classes_used = {}
    with torch.no_grad():
        for _ in range(args.epoch):
            id_spt, y_spt, id_query, y_query = supervised_task_generator(id_by_class, base, labels, args.n_way, args.k_shot, q)
            classes = labels[id_spt].tolist()
            queries = []
            for way, (s, cla) in enumerate(zip(id_spt.tolist(), classes)):
                classes_used[cla] = classes_used.get(cla, 0) + 1
                nodes = np.asarray([v for v in id_by_class[float(cla)] if v != s], dtype=np.int64)
                ids = torch.as_tensor(nodes, device=device)
                cos = (z2[ids] @ z2[s]).cpu().numpy()
                if args.episodes == "sup":
                    picked = id_query[way * q:(way + 1) * q]
                    query_cos.append(float((z2[torch.as_tensor(picked, device=device)] @ z2[s]).mean()))
                    continue
                order = nodes[np.argsort(-cos, kind="stable")]
                sorted_cos = cos[np.argsort(-cos, kind="stable")]
                if len(order) <= q:
                    chosen = order.tolist()
                    extra = q - len(chosen)
                    if extra:
                        chosen += rng.choice(order, extra, replace=True).tolist()
                        drawn_with_replacement += extra
                    queries.extend(chosen)
                    query_cos.append(float((z2[torch.as_tensor(chosen, device=device)] @ z2[s]).mean()))
                    continue
                layers = np.array_split(np.arange(len(order)), STRATA)
                chosen = []
                for layer_index, layer in enumerate(layers):
                    picks = rng.choice(layer, PER_STRATUM, replace=False)
                    chosen.extend(order[picks].tolist())
                    strata_cos[layer_index].append(float(sorted_cos[picks].mean()))
                queries.extend(chosen)
                query_cos.append(float((z2[torch.as_tensor(chosen, device=device)] @ z2[s]).mean()))
            if args.episodes == "sup_strat":
                id_query = np.asarray(queries, dtype=np.int64)
            tasks.append((id_spt, y_spt, id_query, y_query))
    sizes = sorted(len(id_by_class[c]) for c in base)
    stats = {
        "episodes": args.episodes, "seed": args.seed, "n_episodes": len(tasks), "n_base_classes": len(base), "base_classes": base,
        "class_size_min": sizes[0], "class_size_median": float(np.median(sizes)), "class_size_max": sizes[-1], "class_sizes": sizes,
        "classes_sampled_min": min(classes_used.values()), "classes_sampled_max": max(classes_used.values()), "n_classes_sampled": len(classes_used),
        "query_a2x_cos_mean": float(np.mean(query_cos)),
        "strata_a2x_cos_mean": [float(np.mean(x)) if x else None for x in strata_cos],
        "drawn_with_replacement": drawn_with_replacement,
    }
    return tasks, stats


def episode_sha(tasks):
    h = hashlib.sha256()
    for t in tasks:
        h.update(np.asarray(t[0], dtype=np.int64).tobytes())
        h.update(np.asarray(t[2], dtype=np.int64).tobytes())
    return h.hexdigest()


def pool_scores(z, episodes, args, shot):
    return np.asarray([episode_accuracy(z, task, args.n_way, shot) for task in episodes], dtype=np.float64)


def evaluate_row(meta, splits, args):
    meta.network.eval()
    row = {}
    with torch.no_grad():
        raw = meta.encode()
        l2 = F.normalize(raw, dim=1)
        for shot in SHOTS:
            valid, test = splits[shot][0], splits[shot][1]
            row[f"valid_{shot}shot"] = float(pool_scores(raw, valid, args, shot).mean())
            row[f"test_{shot}shot"] = float(pool_scores(raw, test, args, shot).mean())
            row[f"valid_{shot}shot_l2"] = float(pool_scores(l2, valid, args, shot).mean())
            row[f"test_{shot}shot_l2"] = float(pool_scores(l2, test, args, shot).mean())
    return row


def main():
    args = parse_args()
    args.setting, args.type, args.query_generation = "unsup", "feature", "NaQ"
    if args.prepare_feat:
        prepare_feat(args)
        return
    if args.episodes is None:
        raise ValueError("--episodes is required")
    if args.k_shot != 1 or args.k_shot_test != 1 or args.n_way != 5:
        raise ValueError("Training remains 5-way 1-shot")
    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    splits = load_splits(args)
    arm = arm_name(args)
    out = Path(args.output_root) / arm / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    if report_path.exists():
        raise FileExistsError(f"Existing result cannot be overwritten: {report_path}")

    start = time.perf_counter()
    print(f"[hl014] arm={arm} seed={args.seed} loading", flush=True)
    data = load_nodiff(args, device)
    if args.episodes in ("naq", "ours"):
        data.add_diffusion(args)
    config = [
        ("GCN", (data.features.shape[1], 2 * args.latent), True),
        ("GCN", (2 * args.latent, args.latent), True),
        ("linear", (args.latent, args.n_way), True),
    ]
    features = None
    if args.encoder != "gcn":
        features = torch.as_tensor(np.load(FEAT / f"pca{DIMS[args.encoder]}.npy"), dtype=torch.float32, device=device)
    meta = ProtoNetV(args, config, data, device, features).to(device)

    stats = None
    if args.episodes in ("naq", "ours"):
        original = [proto_task_generator(args, data) for _ in range(args.epoch)]
        if args.episodes == "ours":
            train_pool, details = build_structural_queries(data, original, args, device)
        else:
            train_pool, details = original, None
    else:
        train_pool, stats = make_sup_episodes(args, data, H.labels_np(data), device)
        original, details = train_pool, None
    if len(train_pool) != args.epoch:
        raise RuntimeError("Training episode count changed")
    diag = query_diagnostics(original, train_pool, data, details, args.q_query)
    sha = episode_sha(train_pool)
    print(f"[hl014] query diagnostics: {diag} sha={sha[:12]}", flush=True)
    del original
    pool_matches_first = None
    if stats is not None:
        stats["episode_sha256"] = sha
        path = Path(args.output_root) / f"episodes_{args.episodes}_seed{args.seed}.json"
        # sup_strat ranks by Â²X cosine, whose float values differ slightly between processes (CLAUDE.md §9), so its pool may differ between runs: recorded, not fatal.
        if path.exists():
            pool_matches_first = json.loads(path.read_text())["episode_sha256"] == sha
        elif args.epoch == 2000:
            path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    best = {rule: {shot: {"accuracy": -1.0, "epoch": None} for shot in SHOTS} for rule in RULES}
    history = []
    row = {"epoch": 0, "train_loss": None, "train_acc": None, **evaluate_row(meta, splits, args)}
    history.append(row)
    print(f"[hl014] epoch=0 valid1={row['valid_1shot']:.4%} test1={row['test_1shot']:.4%} valid1_l2={row['valid_1shot_l2']:.4%} test1_l2={row['test_1shot_l2']:.4%}", flush=True)
    if args.encoder in ("pca64_lin", "pca256_lin"):
        reference = hl11_value(f"a2x_pca{DIMS[args.encoder]}_l2")
        assert abs(row["test_1shot_l2"] * 100 - reference) <= 0.05, (row["test_1shot_l2"], reference)

    for epoch, (id_spt, y_spt, id_query, y_query) in enumerate(train_pool, 1):
        meta.network.train()
        loss, train_acc = meta(id_spt, y_spt.to(device), id_query, y_query.to(device))
        if epoch % args.eval_interval != 0 and epoch != args.epoch:
            continue
        row = {"epoch": epoch, "train_loss": float(loss.detach()), "train_acc": float(train_acc), **evaluate_row(meta, splits, args)}
        for rule, suffix in RULES.items():
            for shot in SHOTS:
                value = row[f"valid_{shot}shot{suffix}"]
                if value > best[rule][shot]["accuracy"]:
                    best[rule][shot] = {"accuracy": value, "epoch": epoch}
                    name = "best_raw" if rule == "R" else "best_l2"
                    torch.save({"epoch": epoch, "model": meta.network.state_dict(), "config": vars(args).copy()}, out / f"{name}_{shot}.pt")
        history.append(row)
        print(f"[hl014] epoch={epoch} valid1={row['valid_1shot']:.4%} test1={row['test_1shot']:.4%} valid1_l2={row['valid_1shot_l2']:.4%} test1_l2={row['test_1shot_l2']:.4%}", flush=True)

    evaluations = {}
    for rule in RULES:
        evaluations[rule] = {}
        name = "best_raw" if rule == "R" else "best_l2"
        for shot in SHOTS:
            state = torch.load(out / f"{name}_{shot}.pt", map_location="cpu", weights_only=True)
            meta.network.load_state_dict(state["model"])
            meta.network.eval()
            with torch.no_grad():
                raw = meta.encode()
                test_raw = pool_scores(raw, splits[shot][1], args, shot)
                test_l2 = pool_scores(F.normalize(raw, dim=1), splits[shot][1], args, shot)
            evaluations[rule][str(shot)] = {
                "best_epoch": best[rule][shot]["epoch"],
                "valid_accuracy": best[rule][shot]["accuracy"],
                "test_raw": float(test_raw.mean()),
                "test_l2": float(test_l2.mean()),
                "test_raw_ci95": float(1.96 * test_raw.std(ddof=0) / np.sqrt(len(test_raw))),
                "test_l2_ci95": float(1.96 * test_l2.std(ddof=0) / np.sqrt(len(test_l2))),
                "test_raw_episodes": test_raw.tolist(),
                "test_l2_episodes": test_l2.tolist(),
                "checkpoint": str(out / f"{name}_{shot}.pt"),
            }
            print(f"[hl014] rule {rule} {shot}-shot epoch={best[rule][shot]['epoch']} test_raw={test_raw.mean():.4%} test_l2={test_l2.mean():.4%}", flush=True)
    report = {
        "experiment": "hl014", "arm": arm, "args": vars(args).copy(), "dataset": args.dataset, "seed": args.seed,
        "diagnostics": diag, "episode_sha256": sha, "episode_pool_matches_first": pool_matches_first, "episode_stats": stats, "evaluation": evaluations, "history": history,
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "gpu_peak_allocated_mib": int(torch.cuda.max_memory_allocated() // 2**20) if device.type == "cuda" else None,
    }
    tmp = out / "report.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(report_path)
    print(f"[hl014] DONE {report_path}", flush=True)


if __name__ == "__main__":
    main()
