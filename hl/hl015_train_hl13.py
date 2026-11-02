"""hl015_train_hl13: copied from hl/hl014_train_variants.py (main, ProtoNetV); helpers imported from it. Adds sgc*/pca1024_lin64 encoders, --lr for linear encoders, raw_soft/cos_soft losses, mix_r/oracle_strat5 episodes, --pca_fit base.

Labels: base 40 classes (hl014.base_classes) for sup/mix episodes; all classes for oracle_strat5 only (via hl011_pdd_train.build_pdd_queries, oracle); fixed valid/test episode labels for meta-test evaluation; query_diagnostics for diagnostics.
PCA reads no labels; --pca_fit base uses only the node set of the base classes.
Epoch 0 is recorded in history but is not a checkpoint candidate (as hl013/hl014). Episode pool hashes are recorded, never asserted.
"""
import argparse
import hashlib
import json
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
import hl014_train_variants as V  # noqa: E402
from exp027 import episode_accuracy  # noqa: E402
from exp049 import build_structural_queries, load_splits, query_diagnostics  # noqa: E402
from model import GNNEncoder  # noqa: E402
from task_generator import proto_task_generator, supervised_task_generator  # noqa: E402
from utils import seed_everything  # noqa: E402

FEAT12 = Path("results/hl/hl12/feat")
FEAT = Path("results/hl/hl13/feat")
SHOTS = (1, 5)
RULES = {"R": "", "L": "_l2"}
LR_NAME = {1e-4: "1e-4", 1e-3: "1e-3", 1e-2: "1e-2"}
T_STRUCT_TEMPERATURE = 0.1
LINEAR = ("pca64_lin", "pca256_lin", "pca256_mlp", "sgc64_rand", "sgc256_rand", "sgc64_pcainit", "pca1024_lin64")
ORACLE_ARM = "o_grid_strat5_p10"
TEACHER_B = "results/hl/p0/Amazon_clothing/ours/tau_0.3/seed_1212/best_1shot.pt"
TEACHER_N = "results/hl/p0/Amazon_clothing/naq/baseline/seed_1212/best_1shot.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prepare_feat", action="store_true")
    p.add_argument("--dataset", default="Amazon_clothing")
    p.add_argument("--episodes")
    p.add_argument("--encoder", choices=("gcn",) + LINEAR, default="gcn")
    p.add_argument("--loss", default="raw")
    p.add_argument("--scale", type=float, default=10.0)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--pca_fit", choices=["all", "base"], default="all")
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
    p.add_argument("--output_root", default="results/hl/hl13/curve")
    return p.parse_args()


def parse_loss(text):
    """(`raw`|`cos`, A or None)."""
    if text in ("raw", "cos"):
        return text, None
    kind, _, value = text.partition("_soft")
    if kind not in ("raw", "cos") or not value:
        raise ValueError(f"Unknown loss: {text}")
    return kind, float(value)


def arm_name(args):
    name = f"{args.encoder}_{args.loss}_{args.episodes}"
    if args.pca_fit == "base":
        name += "_pcabase"
    if args.lr != 1e-4 or args.encoder.startswith("sgc"):
        name += f"_lr{LR_NAME[args.lr]}"
    return name


def eval_pool(z, splits, shot, which):
    return np.asarray([episode_accuracy(z, t, 5, shot) for t in splits[shot][which]], dtype=np.float64)


def prepare_feat(args):
    """Saved once: PCA-64 components and mean (for sgc64_pcainit), PCA-1024 l2, PCA-64 fit on the base-class node set only l2. Labels: base class ids select nodes; values are not used."""
    FEAT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    seed_everything(1212)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    data = V.load_nodiff(args, device)
    _, _, z2 = H.structural_scores(data, device)
    source = z2.cpu().numpy().astype(np.float32)
    args.seed = 1212
    splits = load_splits(args)
    meta = {}

    pca = PCA(n_components=64, svd_solver="randomized", random_state=1212)
    pca.fit(source)
    np.save(FEAT / "pca64_components.npy", pca.components_.astype(np.float32))
    np.save(FEAT / "pca64_mean.npy", pca.mean_.astype(np.float32))
    w = torch.as_tensor(pca.components_, dtype=torch.float32, device=device)
    mu = torch.as_tensor(pca.mean_, dtype=torch.float32, device=device)
    low = F.normalize((z2 - mu) @ w.T, dim=1)
    ref = torch.as_tensor(np.load(FEAT12 / "pca64.npy"), device=device)
    meta["pcainit"] = {
        "max_abs_diff_vs_hl12_pca64": float((low - ref).abs().max()),
        "test_1shot": float(eval_pool(low, splits, 1, 1).mean() * 100),
        "test_5shot": float(eval_pool(low, splits, 5, 1).mean() * 100),
        "hl12_pca64_test_1shot": float(eval_pool(ref, splits, 1, 1).mean() * 100),
    }
    print(f"[hl015] pcainit {meta['pcainit']}", flush=True)

    pca = PCA(n_components=1024, svd_solver="randomized", random_state=1212)
    low = F.normalize(torch.as_tensor(pca.fit_transform(source), dtype=torch.float32, device=device), dim=1)
    np.save(FEAT / "pca1024.npy", low.cpu().numpy())
    meta["pca1024"] = {"md5": hashlib.md5((FEAT / "pca1024.npy").read_bytes()).hexdigest(), "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
                       "test_1shot": float(eval_pool(low, splits, 1, 1).mean() * 100), "test_5shot": float(eval_pool(low, splits, 5, 1).mean() * 100)}
    print(f"[hl015] pca1024 {meta['pca1024']}", flush=True)

    labels = H.labels_np(data)
    base = V.base_classes(data, labels)
    nodes = np.sort(np.concatenate([np.asarray(data.id_by_class[c], dtype=np.int64) for c in base]))
    pca = PCA(n_components=64, svd_solver="randomized", random_state=1212)
    pca.fit(source[nodes])
    low = F.normalize(torch.as_tensor(pca.transform(source), dtype=torch.float32, device=device), dim=1)
    np.save(FEAT / "pca64_base.npy", low.cpu().numpy())
    row = {"md5": hashlib.md5((FEAT / "pca64_base.npy").read_bytes()).hexdigest(), "n_fit_nodes": int(len(nodes)), "n_base_classes": len(base),
           "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum())}
    for shot in SHOTS:
        for i, pool in enumerate(("valid", "test")):
            scores = eval_pool(low, splits, shot, i)
            row[f"{pool}_{shot}shot"] = float(scores.mean() * 100)
            row[f"{pool}_{shot}shot_ci95"] = float(1.96 * scores.std(ddof=0) / np.sqrt(len(scores)) * 100)
            if pool == "test":
                np.save(FEAT / f"pca64_base_{shot}shot_test.npy", scores)
    (FEAT / "pca_base_eval.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    meta["pca64_base"] = row
    print(f"[hl015] pca64_base {row}", flush=True)
    (FEAT / "pca_meta_hl13.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


class NormLinear(nn.Module):
    """Linear -> row l2 (sgc*, pca1024_lin64): raw and l2 evaluation coincide."""

    def __init__(self, dim_in, dim_out, bias):
        super().__init__()
        self.net = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class ProtoNetV(nn.Module):
    """hl014.ProtoNetV (meta.ProtoNet copy) + soft targets: t = (1-A) one-hot + A t_struct, loss = -sum t log_softmax(logits)."""

    def __init__(self, args, config, data, device, features):
        super().__init__()
        self.Data = data
        self.encoder = args.encoder
        self.loss_kind, self.soft_a = parse_loss(args.loss)
        self.scale = args.scale
        self.n, self.k = args.n_way, args.k_shot
        self.device_name = device
        self.features = features
        if args.encoder == "gcn":
            self.network = GNNEncoder(config)
        elif args.encoder == "pca64_lin":
            self.network = V.PcaEncoder("lin", 64)
        elif args.encoder == "pca256_lin":
            self.network = V.PcaEncoder("lin", 256)
        elif args.encoder == "pca256_mlp":
            self.network = V.PcaEncoder("mlp", 256)
        elif args.encoder in ("sgc64_rand", "sgc64_pcainit"):
            self.network = NormLinear(features.size(1), 64, bias=False)
        elif args.encoder == "sgc256_rand":
            self.network = NormLinear(features.size(1), 256, bias=False)
        else:
            self.network = NormLinear(1024, 64, bias=True)
        if args.encoder == "sgc64_pcainit":
            with torch.no_grad():
                self.network.net.weight.copy_(torch.as_tensor(np.load(FEAT / "pca64_components.npy"), dtype=torch.float32))
        self.meta_optimizer = optim.Adam(self.network.parameters(), lr=args.lr)

    def encode(self):
        if self.encoder == "gcn":
            return self.network(self.Data.features, adj=self.Data.adj)
        return self.network(self.features)

    def forward(self, id_spt, y_spt, id_query, y_query, t_struct=None):
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
        if self.soft_a is None:
            loss = F.nll_loss(output, label_new)
        else:
            target = (1 - self.soft_a) * F.one_hot(label_new, self.n).float() + self.soft_a * t_struct
            loss = -(target * output).sum(dim=1).mean()
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()
        preds = output.cpu().detach().max(1)[1].type_as(label_new.cpu())
        return loss, preds.eq(label_new.cpu()).double().sum() / len(label_new)

    def euclidean_dist(self, x, y):
        n, m, d = x.size(0), y.size(0), x.size(1)
        return torch.pow(x.unsqueeze(1).expand(n, m, d) - y.unsqueeze(0).expand(n, m, d), 2).sum(dim=2)


def clean_nan(value):
    """Diagnostics from hl011 can hold NaN (empty groups); report.json is strict JSON."""
    if isinstance(value, dict):
        return {k: clean_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_nan(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    return value


def t_struct_of(z2, id_spt, id_query):
    return F.softmax(z2[id_query] @ z2[id_spt].T / T_STRUCT_TEMPERATURE, dim=1)


def make_mix(args, data, labels, ratio):
    """naq tasks first (a prefix of the naq pool of this seed), then sup tasks (base 40 classes); order shuffled by default_rng(seed)."""
    n_sup = int(round(args.epoch * ratio))
    tasks = [proto_task_generator(args, data) for _ in range(args.epoch - n_sup)]
    base = V.base_classes(data, labels)
    tasks += [supervised_task_generator(data.id_by_class, base, labels, args.n_way, args.k_shot, args.q_query) for _ in range(n_sup)]
    flags = np.array([0] * (args.epoch - n_sup) + [1] * n_sup, dtype=np.int8)
    order = np.random.default_rng(args.seed).permutation(args.epoch)
    return [tasks[i] for i in order], flags[order]


def main():
    args = parse_args()
    args.setting, args.type, args.query_generation = "unsup", "feature", "NaQ"
    if args.prepare_feat:
        prepare_feat(args)
        return
    if not args.episodes:
        raise ValueError("--episodes is required")
    if args.encoder == "gcn" and args.lr != 1e-4:
        raise ValueError("GCN lr is fixed at 1e-4")
    if args.pca_fit == "base" and args.encoder != "pca64_lin":
        raise ValueError("--pca_fit base is defined for pca64_lin")
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
    print(f"[hl015] arm={arm} seed={args.seed} loading", flush=True)
    data = V.load_nodiff(args, device)
    needs_diffusion = args.episodes in ("naq", "ours", "oracle_strat5") or args.episodes.startswith("mix_r")
    if needs_diffusion:
        data.add_diffusion(args)
    soft_a = parse_loss(args.loss)[1]
    z2 = None
    if args.encoder.startswith("sgc") or soft_a is not None:
        z2 = H.structural_scores(data, device)[2]

    features = None
    if args.encoder == "pca64_lin":
        features = np.load(FEAT / "pca64_base.npy") if args.pca_fit == "base" else np.load(FEAT12 / "pca64.npy")
    elif args.encoder in ("pca256_lin", "pca256_mlp"):
        features = np.load(FEAT12 / "pca256.npy")
    elif args.encoder == "pca1024_lin64":
        features = np.load(FEAT / "pca1024.npy")
    if features is not None:
        features = torch.as_tensor(features, dtype=torch.float32, device=device)
    elif args.encoder == "sgc64_pcainit":
        features = z2 - torch.as_tensor(np.load(FEAT / "pca64_mean.npy"), dtype=torch.float32, device=device)
    elif args.encoder.startswith("sgc"):
        features = z2
    config = [
        ("GCN", (data.features.shape[1], 2 * args.latent), True),
        ("GCN", (2 * args.latent, args.latent), True),
        ("linear", (args.latent, args.n_way), True),
    ]
    meta = ProtoNetV(args, config, data, device, features).to(device)

    extra = {}
    labels = H.labels_np(data)
    if args.episodes in ("naq", "ours"):
        original = [proto_task_generator(args, data) for _ in range(args.epoch)]
        if args.episodes == "ours":
            train_pool, details = build_structural_queries(data, original, args, device)
        else:
            train_pool, details = original, None
    elif args.episodes == "sup":
        sup_args = argparse.Namespace(**vars(args))
        train_pool, extra["episode_stats"] = V.make_sup_episodes(sup_args, data, labels, device)
        original, details = train_pool, None
    elif args.episodes.startswith("mix_r"):
        train_pool, flags = make_mix(args, data, labels, float(args.episodes[len("mix_r"):]))
        original, details = train_pool, None
        extra["mix"] = {"n_sup": int(flags.sum()), "n_naq": int((flags == 0).sum()), "order_sha256": hashlib.sha256(flags.tobytes()).hexdigest(), "first_sup_epochs": (np.flatnonzero(flags)[:10] + 1).tolist()}
    elif args.episodes == "oracle_strat5":
        import hl011_pdd_train as O

        original = [proto_task_generator(args, data) for _ in range(args.epoch)]
        oracle_args = argparse.Namespace(**vars(args))
        oracle_args.arm, oracle_args.teacher, oracle_args.teacher_n, oracle_args.x_band = ORACLE_ARM, TEACHER_B, TEACHER_N, "201,1000"
        oracle_args.ppr_alpha, oracle_args.ppr_iters = 0.15, 30
        t_query = time.perf_counter()
        train_pool, details = O.build_pdd_queries(data, original, oracle_args, device, labels)
        extra["oracle"] = {"query_build_seconds": round(time.perf_counter() - t_query, 2), "pdd": O.pdd_diagnostics(original, details, args.q_query)}
    else:
        raise ValueError(f"Unknown episodes: {args.episodes}")
    if len(train_pool) != args.epoch:
        raise RuntimeError("Training episode count changed")
    diag = query_diagnostics(original, train_pool, data, details, args.q_query)
    sha = V.episode_sha(train_pool)
    print(f"[hl015] query diagnostics: {diag} sha={sha[:12]}", flush=True)
    del original

    agreement = None
    if soft_a is not None:
        values = []
        with torch.no_grad():
            for id_spt, y_spt, id_query, y_query in train_pool:
                t = t_struct_of(z2, id_spt, id_query)
                position = torch.as_tensor([y_spt.tolist().index(i) for i in y_query.tolist()], device=device)
                values.append(float(t[torch.arange(len(position), device=device), position].mean()))
        agreement = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        print(f"[hl015] t_struct agreement {agreement}", flush=True)

    best = {rule: {shot: {"accuracy": -1.0, "epoch": None} for shot in SHOTS} for rule in RULES}
    history = []
    row = {"epoch": 0, "train_loss": None, "train_acc": None, **V.evaluate_row(meta, splits, args)}
    history.append(row)
    print(f"[hl015] epoch=0 valid1={row['valid_1shot']:.4%} test1={row['test_1shot']:.4%} valid1_l2={row['valid_1shot_l2']:.4%} test1_l2={row['test_1shot_l2']:.4%}", flush=True)
    if args.encoder == "sgc64_pcainit":
        assert abs(row["test_1shot_l2"] * 100 - 89.05) <= 0.05, row["test_1shot_l2"]
    elif args.encoder in ("pca64_lin", "pca256_lin") and args.pca_fit == "all":
        reference = V.hl11_value(f"a2x_pca{V.DIMS[args.encoder]}_l2")
        assert abs(row["test_1shot_l2"] * 100 - reference) <= 0.05, (row["test_1shot_l2"], reference)

    for epoch, (id_spt, y_spt, id_query, y_query) in enumerate(train_pool, 1):
        meta.network.train()
        t_struct = t_struct_of(z2, id_spt, id_query).detach() if soft_a is not None else None
        loss, train_acc = meta(id_spt, y_spt.to(device), id_query, y_query.to(device), t_struct)
        if epoch % args.eval_interval != 0 and epoch != args.epoch:
            continue
        row = {"epoch": epoch, "train_loss": float(loss.detach()), "train_acc": float(train_acc), **V.evaluate_row(meta, splits, args)}
        for rule, suffix in RULES.items():
            for shot in SHOTS:
                value = row[f"valid_{shot}shot{suffix}"]
                if value > best[rule][shot]["accuracy"]:
                    best[rule][shot] = {"accuracy": value, "epoch": epoch}
                    name = "best_raw" if rule == "R" else "best_l2"
                    torch.save({"epoch": epoch, "model": meta.network.state_dict(), "config": vars(args).copy()}, out / f"{name}_{shot}.pt")
        history.append(row)
        print(f"[hl015] epoch={epoch} loss={row['train_loss']:.4f} valid1_l2={row['valid_1shot_l2']:.4%} test1={row['test_1shot']:.4%} test1_l2={row['test_1shot_l2']:.4%}", flush=True)

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
                test_raw = V.pool_scores(raw, splits[shot][1], args, shot)
                test_l2 = V.pool_scores(F.normalize(raw, dim=1), splits[shot][1], args, shot)
            evaluations[rule][str(shot)] = {
                "best_epoch": best[rule][shot]["epoch"], "valid_accuracy": best[rule][shot]["accuracy"],
                "test_raw": float(test_raw.mean()), "test_l2": float(test_l2.mean()),
                "test_raw_ci95": float(1.96 * test_raw.std(ddof=0) / np.sqrt(len(test_raw))),
                "test_l2_ci95": float(1.96 * test_l2.std(ddof=0) / np.sqrt(len(test_l2))),
                "test_raw_episodes": test_raw.tolist(), "test_l2_episodes": test_l2.tolist(),
                "checkpoint": str(out / f"{name}_{shot}.pt"),
            }
            print(f"[hl015] rule {rule} {shot}-shot epoch={best[rule][shot]['epoch']} test_raw={test_raw.mean():.4%} test_l2={test_l2.mean():.4%}", flush=True)
    report = {
        "experiment": "hl015", "arm": arm, "args": vars(args).copy(), "dataset": args.dataset, "seed": args.seed,
        "diagnostics": diag, "episode_sha256": sha, "t_struct_agreement": agreement, **extra, "evaluation": evaluations, "history": history,
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "gpu_peak_allocated_mib": int(torch.cuda.max_memory_allocated() // 2**20) if device.type == "cuda" else None,
    }
    tmp = out / "report.json.tmp"
    tmp.write_text(json.dumps(clean_nan(report), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(report_path)
    print(f"[hl015] DONE {report_path}", flush=True)


if __name__ == "__main__":
    main()
