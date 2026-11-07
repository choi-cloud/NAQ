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
from hw.neighbors import build_neighbor_lookup, load_selection_embeddings
from hw.weights import DIAG_KEYS

RESULTS_ROOT = Path(os.environ.get("HW_RESULTS_ROOT", "results/hw"))
FIRST_EPISODES_LOGGED = 5


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
        "neighbor_contrastive": {"temperature": temperature, "lambda_neighbor": lambda_neighbor, "max_neighbors": max_neighbors, "selector": selector},
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
        space_features = None
        if hsan is not None and hsan.get("space") == "raw":
            space_features = F.normalize(features.detach(), p=2, dim=1)  # no RNG
        config = [("GCN", (d, 2 * args.latent), True), ("GCN", (2 * args.latent, args.latent), True), ("linear", (args.latent, args.n_way), True)]
        model = ProtoNet(args, config, data).to(device)
        neighbor_lookup = build_neighbor_lookup(data.edge_index, num_nodes) if loss_kind != "original" else None
        train_pool = build_train_pool(args, data)
        valid_pool, test_pool = build_eval_pools(args, data)
        print(f"[hw.run:{arm}] {dataset} seed={seed} train={len(train_pool)} valid={len(valid_pool)} test={len(test_pool)}", flush=True)

        best_validation = -float("inf")
        best_state = None
        losses, accuracies, original_losses, neighbor_losses, neighbor_counts, valid_fractions = [], [], [], [], [], []
        hsan_buffers = {key: [] for key in hsan_diag_keys}
        recent_diagnostics = deque(maxlen=5)
        for epoch, (id_spt, y_spt, id_query, y_query) in tqdm(enumerate(train_pool, 1), total=len(train_pool), desc=dataset, file=sys.__stderr__):
            model.train()
            embeddings = model.network(features, adj=data.adj)
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
            )
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
            save_json(json_path, result)

        model.network.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            clean = model.network(features, adj=data.adj)
            testing = evaluate(clean, test_pool, args.n_way, device)
        result["best"]["test"] = testing
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
