"""HW track runner (single entry point).

    python -m hw.run --arm <arm> --dataset <ds> [--seed S] [--device D]

Training loop transcribed from choi_exps/zip_20261107/exp015.py / exp016.py main().
Outputs $HW_RESULTS_ROOT (default results/hw)/<arm>/<ds>/seed<S>/{metrics.json, train.log}. No checkpoints.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import load_data
from meta import ProtoNet
from utils import seed_everything

from hw.config import DEFAULT_CONFIG, build_args, load_config
from hw.episodes import build_eval_pools, build_train_pool
from hw.evaluate import evaluate
from hw.losses import episode_loss
from hw.negatives import EXTRA_NEG_MODES, NBR_TERMS, ExtraNegatives
from hw.valid500 import NUM_EXISTING, load_valid500
from hw.neighbors import ALL_PICKS, SUP_COS, build_neighbor_lookup, build_sup_cos, load_selection_embeddings, sup_cos_space
from hw.weights import DIAG_KEYS

RESULTS_ROOT = Path(os.environ.get("HW_RESULTS_ROOT", "results/hw"))
FIRST_EPISODES_LOGGED = 5
NBR_DIAG_KEYS = (
    "nbr_eq_support", "nbr_eq_query", "query_eq_support", "pos_nbr_cos_raw", "fallback_frac", "multi_nbr_frac",
    "nbr_in_own_queries", "nbr_in_any_queries", "nbr_is_query", "pos_nbr_cos_a2x",  # HW04a
    "xneg_eq_query", "xneg_eq_support", "xneg_cos_raw", "xneg_fill_frac",  # HW05
)
GEOM_SAMPLE = 4096  # HW05 geom_diag: fixed node sample (dedicated CPU generator, seed GEOM_SEED)
GEOM_SEED = 2026
RANDOM_PICK_SEED_OFFSET = 7919
GLOBAL_RANDOM_PICK_SEED_OFFSET = 104729  # HW03b
VALID_TASKS = (50, 500)  # HW03b; 500 = save/valid500/<ds>.pkl, first 50 = the existing validation tasks


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _mean_or_none(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


@torch.no_grad()
def neighbor_diagnostics(labels, raw_normalized, id_spt, y_spt, id_query, y_query, selection_out, a2x_normalized=None):
    """HW03a purity / hardness of the positive (own pseudo-class) selections of one episode.

    measurement-only label use: GT labels are read here only; the values go to metrics.json
    (epochs[].nbr_diag) and never into the loss, the neighbor selection or checkpoint selection.
    selection_out None (loss "original"): only query_eq_support.
    """
    device = labels.device
    support_ids = torch.as_tensor(id_spt, dtype=torch.long, device=device).flatten()
    query_ids = torch.as_tensor(id_query, dtype=torch.long, device=device).flatten()
    support_pseudo = torch.as_tensor(y_spt, dtype=torch.long, device=device).flatten()
    query_pseudo = torch.as_tensor(y_query, dtype=torch.long, device=device).flatten()
    qi, si = (query_pseudo[:, None] == support_pseudo[None, :]).nonzero(as_tuple=True)
    gt_query = labels[query_ids[qi]]  # measurement-only label use
    gt_support = labels[support_ids[si]]  # measurement-only label use
    out = {"query_eq_support": float((gt_query == gt_support).float().mean().item())}
    if selection_out:
        chosen = selection_out["selected"][qi, si]
        gt_chosen = labels[chosen]  # measurement-only label use
        out["nbr_eq_support"] = float((gt_chosen == gt_support).float().mean().item())
        out["nbr_eq_query"] = float((gt_chosen == gt_query).float().mean().item())
        out["pos_nbr_cos_raw"] = float((raw_normalized[query_ids[qi]] * raw_normalized[chosen]).sum(-1).mean().item())
        out["fallback_frac"] = float((~selection_out["has_neighbor"][qi, si]).float().mean().item())
        out["multi_nbr_frac"] = float((selection_out["valid_count"][qi, si] >= 2).float().mean().item())
        # HW04a (no labels): overlap of the chosen positive with this episode's pseudo-queries
        own = (chosen[:, None] == query_ids[None, :]) & (query_pseudo[None, :] == support_pseudo[si][:, None])
        out["nbr_in_own_queries"] = float(own.any(dim=1).float().mean().item())
        out["nbr_in_any_queries"] = float((chosen[:, None] == query_ids[None, :]).any(dim=1).float().mean().item())
        out["nbr_is_query"] = float((chosen == query_ids[qi]).float().mean().item())
        if a2x_normalized is not None:
            out["pos_nbr_cos_a2x"] = float((a2x_normalized[query_ids[qi]] * a2x_normalized[chosen]).sum(-1).mean().item())
    return out


@torch.no_grad()
def xneg_diagnostics(labels, raw_normalized, id_spt, y_spt, id_query, y_query, xneg_out):
    """HW05 extra negatives of one episode. measurement-only label use: GT labels feed only metrics.json.
    xneg_eq_query = share of (query, extra negative) pairs with the same GT class (false-negative rate);
    xneg_eq_support = same, against the query's own (pseudo-class) support."""
    device = labels.device
    support_ids = torch.as_tensor(id_spt, dtype=torch.long, device=device).flatten()
    query_ids = torch.as_tensor(id_query, dtype=torch.long, device=device).flatten()
    support_pseudo = torch.as_tensor(y_spt, dtype=torch.long, device=device).flatten()
    query_pseudo = torch.as_tensor(y_query, dtype=torch.long, device=device).flatten()
    extra = xneg_out["extra"]  # [Q, m]
    out = {"xneg_fill_frac": float(xneg_out["fill"])}
    if extra.numel() == 0:
        return out
    own_support = support_ids[(query_pseudo[:, None] == support_pseudo[None, :]).float().argmax(dim=1)]  # [Q]
    gt_extra = labels[extra]  # measurement-only label use
    out["xneg_eq_query"] = float((gt_extra == labels[query_ids][:, None]).float().mean().item())  # measurement-only label use
    out["xneg_eq_support"] = float((gt_extra == labels[own_support][:, None]).float().mean().item())  # measurement-only label use
    out["xneg_cos_raw"] = float((raw_normalized[query_ids][:, None, :] * raw_normalized[extra]).sum(-1).mean().item())
    return out


@torch.no_grad()
def geometry(embeddings):
    """HW05 geom_diag on a fixed sample of GEOM_SAMPLE nodes (CPU generator seeded GEOM_SEED; same rule
    for every dataset): norm mean/std, mean pairwise Euclidean distance (u != v), uniformity
    log mean_{u != v} exp(-2 ||û - v̂||²) (Wang & Isola 2020), effective rank exp(H(σ/Σσ)) of the centered
    sample, and the norm of the sample mean."""
    n = embeddings.shape[0]
    generator = torch.Generator()
    generator.manual_seed(GEOM_SEED)
    index = torch.randperm(n, generator=generator)[: min(GEOM_SAMPLE, n)].to(embeddings.device)
    z = embeddings[index].double()
    k = z.shape[0]
    off = ~torch.eye(k, dtype=torch.bool, device=z.device)
    norms = z.norm(dim=1)
    dist = torch.cdist(z, z)
    zn = z / norms.clamp_min(1e-12)[:, None]
    sq = torch.cdist(zn, zn).square()
    uniformity = torch.logsumexp(-2.0 * sq[off], dim=0) - math.log(off.sum().item())
    sigma = torch.linalg.svdvals(z - z.mean(dim=0, keepdim=True))
    prob = sigma / sigma.sum()
    prob = prob[prob > 0]
    return {
        "sample": int(k),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std(unbiased=False)),
        "pair_dist_mean": float(dist[off].mean()),
        "uniformity": float(uniformity),
        "eff_rank": float(torch.exp(-(prob * prob.log()).sum())),
        "center_norm": float(z.mean(dim=0).norm()),
    }


def parse_cli():
    parser = argparse.ArgumentParser(description="HW track runner")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--dataset", required=True, choices=["Cora_full", "Amazon_clothing"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=int, default=None)
    return parser.parse_args()


def run(arm, dataset, seed=None, device_index=None, config_path=DEFAULT_CONFIG):
    cfg = load_config(arm, config_path)
    if seed is not None:
        cfg["seed"] = seed
    if device_index is not None:
        cfg["device"] = device_index
    seed = int(cfg["seed"])
    if "episodes" in cfg:  # HW05: per-arm number of training episodes (eval_every unchanged)
        cfg["epochs"] = int(cfg["episodes"])
    args = build_args(cfg, dataset)
    if (args.n_way, args.k_shot, args.k_shot_test) != (5, 1, 5):
        raise ValueError("Only 5-way/1-shot train, 5-way/5-shot test supported")
    if args.query_generation != "NaQ" or args.type != "feature":
        raise ValueError("Requires original NaQ-Feat pseudo-queries")
    loss_kind = cfg["loss"]
    eval_every = int(cfg["eval_every"])
    temperature = float(cfg["temperature"])
    lambda_neighbor = float(cfg.get("lambda_neighbor", 0.0))
    max_neighbors = int(cfg.get("max_neighbors", 0))
    selector = cfg.get("selector")
    hsan = cfg.get("hsan")
    pick = cfg.get("pick", "max")
    if pick not in ALL_PICKS:
        raise ValueError(f"pick must be one of {ALL_PICKS}")
    pick_hop = cfg.get("pick_hop") if pick == SUP_COS else None
    pick_space = cfg.get("pick_space") if pick == SUP_COS else None
    if pick == SUP_COS and max_neighbors != 0:
        raise ValueError("pick sup_cos requires max_neighbors == 0")
    valid_tasks = int(cfg.get("valid_tasks", 50))
    if valid_tasks not in VALID_TASKS:
        raise ValueError(f"valid_tasks must be one of {VALID_TASKS}")
    v500 = valid_tasks == 500
    nbr_term = cfg.get("nbr_term", "infonce")  # HW05
    if nbr_term not in NBR_TERMS:
        raise ValueError(f"nbr_term must be one of {NBR_TERMS}")
    if nbr_term != "infonce" and loss_kind != "original+neighbor":
        raise ValueError("nbr_term pull/push needs loss 'original+neighbor'")
    extra_neg_cfg = cfg.get("extra_neg") or {"mode": "none", "m": 0}
    if extra_neg_cfg.get("mode", "none") not in EXTRA_NEG_MODES:
        raise ValueError(f"extra_neg.mode must be one of {EXTRA_NEG_MODES}")
    geom_diag = bool(cfg.get("geom_diag", False))
    hsan_diag_keys = [f"{term}_{key}" for term in ("orig", "nbr") for key in DIAG_KEYS]

    out_dir = RESULTS_ROOT / arm / dataset / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "metrics.json"
    log_file = open(out_dir / "train.log", "w", encoding="utf-8")
    stdout = sys.stdout
    sys.stdout = _Tee(stdout, log_file)

    torch.set_num_threads(4)
    seed_everything(seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    result = {
        "experiment": f"hw.{arm}",
        "runner": "hw.run",
        "arm": arm,
        "config_path": str(config_path),
        "config": cfg,
        "dataset": dataset,
        "status": "running",
        "seed": seed,
        "device": str(device),
        "settings": vars(args),
        "neighbor_contrastive": {"temperature": temperature, "lambda_neighbor": lambda_neighbor, "max_neighbors": max_neighbors, "selector": selector, "pick": pick, "pick_hop": pick_hop, "pick_space": pick_space},
        "hsan": hsan,
        "protocol": {
            "sampling": "original NaQ-Feat top-10 raw-feature pseudo-queries",
            "augmentation": "none; single original graph",
            "encoder": "one randomly initialized 2-layer GCN",
            "loss": loss_kind,
            "selection_embedding": selector,
            "eval_every_epochs": eval_every,
            "checkpoint_selection": f"best validation of {args.num_valid_tasks} tasks; test once on {args.num_downstream_tasks} tasks",
            "code_changes": "none to original NaQ modules",
        },
        "epochs": [],
        "first_episodes": [],
        "best": None,
    }
    save_json(json_path, result)
    started = time.time()
    try:
        data = load_data(args)
        if device.type == "cuda":
            data.set_device(str(device))
        data.add_diffusion(args)
        features = data.features
        num_nodes, d = features.shape
        selection = None
        if loss_kind != "original":
            selection, selection_path = load_selection_embeddings(selector, features, dataset, num_nodes, device, cfg.get("bgrl_dir"))
            if selection_path is not None:
                result["selection_embedding_path"] = selection_path
        raw_normalized = F.normalize(features.detach(), p=2, dim=1)  # diagnostics only; no RNG
        generator = None
        if pick == "random":  # dedicated generator: the global RNG stream is not consumed
            generator = torch.Generator(device=device)
            generator.manual_seed(seed + RANDOM_PICK_SEED_OFFSET)
        if pick == "global_random":  # HW03b: dedicated generator, the global RNG stream is not consumed
            generator = torch.Generator(device=device)
            generator.manual_seed(seed + GLOBAL_RANDOM_PICK_SEED_OFFSET)
        sup_cos = None
        if pick == SUP_COS:  # selection table (HW04a): deterministic, no RNG
            table_started = time.time()
            sup_cos = build_sup_cos(pick_hop, pick_space, features, data.adj, data.edge_index, num_nodes, neighbor_lookup=None)
            result["sup_cos_table"] = {"hop": int(pick_hop), "space": pick_space, "seconds": round(time.time() - table_started, 2)}
            if sup_cos.top2 is not None:
                result["sup_cos_table"].update(
                    nodes_with_hop2=float((sup_cos.hop2_size > 0).float().mean().item()),
                    nodes_with_two_hop2=float((sup_cos.top2[:, 1] >= 0).float().mean().item()),
                    hop2_size_median=float(sup_cos.hop2_size.float().median().item()),
                )
        extra_neg = None
        if extra_neg_cfg.get("mode", "none") != "none":  # HW05 (selection-table slot): own generator / deterministic table, no global RNG
            table_started = time.time()
            extra_neg = ExtraNegatives(extra_neg_cfg["mode"], extra_neg_cfg["m"], num_nodes, device, seed, features=features, adj=data.adj)
            result["extra_neg"] = {"mode": extra_neg_cfg["mode"], "m": int(extra_neg_cfg["m"]), "setup_seconds": round(time.time() - table_started, 2)}
        a2x_normalized = None
        if loss_kind != "original":  # diagnostics only (pos_nbr_cos_a2x); no RNG
            a2x_normalized = sup_cos.embeddings if (sup_cos is not None and pick_space == "a2x") else sup_cos_space("a2x", features, data.adj)
        space_features = None
        if hsan is not None and hsan.get("space") == "raw":
            space_features = F.normalize(features.detach(), p=2, dim=1)  # no RNG
        config = [("GCN", (d, 2 * args.latent), True), ("GCN", (2 * args.latent, args.latent), True), ("linear", (args.latent, args.n_way), True)]
        model = ProtoNet(args, config, data).to(device)
        neighbor_lookup = build_neighbor_lookup(data.edge_index, num_nodes) if loss_kind != "original" else None
        if sup_cos is not None:  # sup_cos candidates travel in the neighbor_lookup slot
            sup_cos.neighbor_lookup = neighbor_lookup
            neighbor_lookup = sup_cos
        train_pool = build_train_pool(args, data)
        valid_pool, test_pool = build_eval_pools(args, data)
        if v500:  # HW03b: cached file only, no RNG; tasks 0..49 = the validation pool above
            valid_pool = load_valid500(dataset)
            best_validation_50, best_state_50 = -float("inf"), None
        print(f"[hw.run:{arm}] {dataset} seed={seed} train={len(train_pool)} valid={len(valid_pool)} test={len(test_pool)}", flush=True)

        best_validation = -float("inf")
        best_state = None
        losses, accuracies, original_losses, neighbor_losses, neighbor_counts, valid_fractions = [], [], [], [], [], []
        hsan_buffers = {key: [] for key in hsan_diag_keys}
        nbr_buffers = {key: [] for key in NBR_DIAG_KEYS}
        recent_diagnostics = deque(maxlen=5)
        for epoch, (id_spt, y_spt, id_query, y_query) in tqdm(enumerate(train_pool, 1), total=len(train_pool), desc=dataset, file=sys.__stderr__):
            model.train()
            embeddings = model.network(features, adj=data.adj)
            selection_out = {} if loss_kind != "original" else None
            xneg_out = {} if extra_neg is not None else None
            loss, accuracy, diagnostics = episode_loss(
                embeddings,
                id_spt,
                y_spt,
                id_query,
                y_query,
                temperature,
                loss_kind,
                selection_embeddings_normalized=selection,
                neighbor_lookup=neighbor_lookup,
                lambda_neighbor=lambda_neighbor,
                max_neighbors=max_neighbors,
                hsan=hsan,
                space_features=space_features,
                pick=pick,
                generator=generator,
                selection_out=selection_out,
                nbr_term=nbr_term,
                extra_neg=extra_neg,
                xneg_out=xneg_out,
            )
            if extra_neg is not None:
                for key, value in xneg_diagnostics(data.labels, raw_normalized, id_spt, y_spt, id_query, y_query, xneg_out).items():
                    nbr_buffers[key].append(value)
            for key, value in neighbor_diagnostics(data.labels, raw_normalized, id_spt, y_spt, id_query, y_query, selection_out, a2x_normalized).items():
                nbr_buffers[key].append(value)
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value):  # G4: stop this run, record status "nan"
                result["status"] = "nan"
                result["nan_episode"] = epoch
                result["nan_episode_diagnostics"] = {k: (v if v is None or math.isfinite(v) else str(v)) for k, v in diagnostics.items()}
                result["nan_previous_diagnostics"] = list(recent_diagnostics)
                result["elapsed_seconds"] = round(time.time() - started, 2)
                save_json(json_path, result)
                print(f"[hw.run:{arm}] {dataset} seed={seed} non-finite loss at episode {epoch}; status=nan", flush=True)
                return result
            recent_diagnostics.append({"epoch": epoch, **diagnostics})
            model.meta_optimizer.zero_grad()
            loss.backward()
            model.meta_optimizer.step()
            losses.append(float(loss.detach().item()))
            accuracies.append(accuracy)
            original_losses.append(diagnostics["original_loss"])
            neighbor_losses.append(diagnostics["neighbor_loss"])
            neighbor_counts.append(diagnostics["mean_support_neighbors_considered"])
            valid_fractions.append(diagnostics["fraction_valid_neighbor_selections"])
            for key in hsan_diag_keys:
                if diagnostics.get(key) is not None:
                    hsan_buffers[key].append(diagnostics[key])
            if epoch <= FIRST_EPISODES_LOGGED:
                result["first_episodes"].append({"epoch": epoch, "loss": losses[-1], **diagnostics})
                if hsan is not None:
                    stats = " ".join(f"{key}={diagnostics[key]:.4f}" for key in hsan_diag_keys if diagnostics.get(key) is not None)
                    print(f"[episode {epoch}] loss={losses[-1]:.4f} original={diagnostics['original_loss']:.4f} neighbor={diagnostics['neighbor_loss']:.4f} {stats}", flush=True)
            if epoch % eval_every != 0 and epoch != args.epoch:
                continue
            model.eval()
            with torch.no_grad():
                clean = model.network(features, adj=data.adj)
                validation = evaluate(clean, valid_pool, args.n_way, device)
                validation_50 = evaluate(clean, valid_pool[:NUM_EXISTING], args.n_way, device) if v500 else None
            record = {
                "epoch": epoch,
                "train_loss_mean": float(np.mean(losses)),
                "train_accuracy_support_euclidean_mean": float(np.mean(accuracies)),
                "train_original_loss_mean": float(np.mean(original_losses)),
                "train_neighbor_loss_mean": _mean_or_none(neighbor_losses),
                "mean_support_neighbors_considered": _mean_or_none(neighbor_counts),
                "fraction_valid_neighbor_selections": _mean_or_none(valid_fractions),
                "validation": validation,
                "elapsed_seconds": round(time.time() - started, 2),
            }
            if hsan is not None:
                record["hsan_diag"] = {key: (float(np.mean(values)) if values else None) for key, values in hsan_buffers.items()}
                for values in hsan_buffers.values():
                    values.clear()
            record["nbr_diag"] = {key: (float(np.mean(values)) if values else None) for key, values in nbr_buffers.items()}
            for values in nbr_buffers.values():
                values.clear()
            if v500:
                record.update(val_acc_500=validation["accuracy_mean"], val_acc_50=validation_50["accuracy_mean"], validation_50=validation_50)
            result["epochs"].append(record)
            print(
                f"[Epoch {epoch:5d}] loss={record['train_loss_mean']:.4f} "
                f"acc={record['train_accuracy_support_euclidean_mean']:.4f} "
                f"val={validation['accuracy_mean']:.4f}",
                flush=True,
            )
            for buffer in (losses, accuracies, neighbor_counts, original_losses, neighbor_losses, valid_fractions):
                buffer.clear()
            if validation["accuracy_mean"] > best_validation:
                best_validation = validation["accuracy_mean"]
                best_state = deepcopy(model.network.state_dict())
                result["best"] = {"epoch": epoch, "validation": validation}
            if v500 and validation_50["accuracy_mean"] > best_validation_50:  # HW03b: second checkpoint (measurement)
                best_validation_50 = validation_50["accuracy_mean"]
                best_state_50 = {key: value.detach().cpu().clone() for key, value in model.network.state_dict().items()}
                result["best_50"] = {"epoch": epoch, "validation": validation_50}
            save_json(json_path, result)

        model.network.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            clean = model.network(features, adj=data.adj)
            testing = evaluate(clean, test_pool, args.n_way, device)
        result["best"]["test"] = testing
        if geom_diag:  # HW05: geometry of the 500-based (main) best checkpoint
            result["geom"] = geometry(clean)
        if v500:  # HW03b: test once per checkpoint; same epoch -> value copied
            same_epoch = result["best_50"]["epoch"] == result["best"]["epoch"]
            if same_epoch:
                testing_50 = testing
            else:
                model.network.load_state_dict(best_state_50)
                model.eval()
                with torch.no_grad():
                    clean = model.network(features, adj=data.adj)
                    testing_50 = evaluate(clean, test_pool, args.n_way, device)
            result["best_50"]["test"] = testing_50
            result.update(
                valid_tasks=valid_tasks,
                test_acc=testing["accuracy_mean"],
                test_acc_at_best50=testing_50["accuracy_mean"],
                best_epoch_500=result["best"]["epoch"],
                best_epoch_50=result["best_50"]["epoch"],
                same_epoch=bool(same_epoch),
            )
        result["status"] = "completed"
        result["elapsed_seconds"] = round(time.time() - started, 2)
        save_json(json_path, result)
        print(
            f"[hw.run:{arm}] {dataset} best={result['best']['epoch']} "
            f"test={testing['accuracy_mean']:.4f} "
            f"±{testing['accuracy_ci95_half_width']:.4f}; {json_path}",
            flush=True,
        )
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        result["elapsed_seconds"] = round(time.time() - started, 2)
        save_json(json_path, result)
        raise
    finally:
        sys.stdout = stdout
        log_file.close()
    return result


def main():
    cli = parse_cli()
    result = run(cli.arm, cli.dataset, cli.seed, cli.device)
    if result["status"] == "nan":
        sys.exit(3)


if __name__ == "__main__":
    main()
