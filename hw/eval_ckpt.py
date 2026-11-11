"""HW09 §1.3: re-evaluate saved best checkpoints (no training) on 5-shot and 1-shot test tasks.

    PYTHONPATH=. python -m hw.eval_ckpt --arm <arm> --dataset <ds> --seed <S> [--results-root results/hw_iislab4]
    PYTHONPATH=. python -m hw.eval_ckpt --hw08      # the 80 HW08 targets (4 l2_* arms x 2 ds x 10 seeds)

Reads <results root>/<arm>/<ds>/seed<S>/metrics.json (never written), loads the files in its `ckpt` record
(best_l2 null -> the l2 best is best_raw.pt), builds the encoder exactly as hw/run.py does
(seed -> load_data -> set_device -> add_diffusion -> ProtoNet; weights then replaced by the state_dict),
and evaluates with hw.evaluate.evaluate / evaluate_l2 on save/downstream_tasks/<ds>/5way_{5,1}shot.pkl test tasks.
Output: <results root>/ckpt_eval/<arm>/<ds>/seed<S>.json with the four test keys for each shot
(5-shot names as in metrics.json, 1-shot names + "_1shot") and the 5-shot |Δ| against metrics.json.
"""

import argparse
import json
from pathlib import Path

import torch

from data import load_data
from meta import ProtoNet
from utils import seed_everything

from hw.config import build_args, load_config
from hw.evaluate import evaluate, evaluate_l2
from hw.run import load_test_pool

HW08_ARMS = ("l2_naq", "l2_rand_m20", "l2_rand_m50", "l2_clus_m20")
HW08_DATASETS = ("Cora_full", "Amazon_clothing")
SEEDS = tuple(range(1212, 1222))
KEYS = ("test_acc", "test_acc_l2_at_bestraw", "test_acc_l2", "test_acc_raw_at_bestl2")


class Context:
    """Data, encoder and test pools of one (arm, dataset); built once and reused across seeds."""

    def __init__(self, arm, dataset, seed, device):
        cfg = load_config(arm)
        cfg["seed"] = seed
        if "episodes" in cfg:
            cfg["epochs"] = int(cfg["episodes"])
        self.args = build_args(cfg, dataset)
        self.device = device
        seed_everything(seed)
        self.data = load_data(self.args)
        self.data.set_device(str(device))
        self.data.add_diffusion(self.args)
        d = self.data.features.shape[1]
        config = [("GCN", (d, 2 * self.args.latent), True), ("GCN", (2 * self.args.latent, self.args.latent), True), ("linear", (self.args.latent, self.args.n_way), True)]
        self.model = ProtoNet(self.args, config, self.data).to(device)
        self.pools = {k: load_test_pool(dataset, self.args.n_way, k) for k in (5, 1)}

    @torch.no_grad()
    def measure(self, path):
        blob = torch.load(path, map_location=self.device)
        self.model.network.load_state_dict(blob["state_dict"])
        self.model.eval()
        z = self.model.network(self.data.features, adj=self.data.adj)
        out = {"epoch": int(blob["epoch"])}
        for k, pool in self.pools.items():
            out[k] = {"raw": evaluate(z, pool, self.args.n_way, self.device)["accuracy_mean"],
                      "l2": evaluate_l2(z, pool, self.args.n_way, self.device)["accuracy_mean"]}
        return out


def evaluate_run(ctx, arm, dataset, seed, root):
    metrics = json.loads((root / arm / dataset / f"seed{seed}" / "metrics.json").read_text())
    ck = metrics["ckpt"]
    raw = ctx.measure(ck["best_raw"])
    l2 = ctx.measure(ck["best_l2"]) if ck["best_l2"] else raw
    out = {"arm": arm, "dataset": dataset, "seed": seed, "ckpt": ck, "epoch_raw": raw["epoch"], "epoch_l2": l2["epoch"],
           "recorded_best_epoch_500": metrics["best_epoch_500"], "recorded_best_epoch_500_l2": metrics["best_epoch_500_l2"]}
    for shot in (5, 1):
        suffix = "" if shot == 5 else "_1shot"
        out[f"test_acc{suffix}"] = raw[shot]["raw"]
        out[f"test_acc_l2_at_bestraw{suffix}"] = raw[shot]["l2"]
        out[f"test_acc_l2{suffix}"] = l2[shot]["l2"]
        out[f"test_acc_raw_at_bestl2{suffix}"] = l2[shot]["raw"]
    out["recorded_5shot"] = {k: metrics[k] for k in KEYS}
    out["abs_diff_5shot"] = {k: abs(out[k] - metrics[k]) for k in KEYS}
    out["test_tasks"] = {str(k): len(p) for k, p in ctx.pools.items()}
    path = root / "ckpt_eval" / arm / dataset / f"seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm")
    parser.add_argument("--dataset")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--results-root", default="results/hw_iislab4")
    parser.add_argument("--hw08", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    cli = parser.parse_args()
    root = Path(cli.results_root)
    device = torch.device(f"cuda:{cli.device}")
    torch.cuda.set_device(device)
    targets = [(a, d, s) for a in HW08_ARMS for d in HW08_DATASETS for s in SEEDS] if cli.hw08 else [(cli.arm, cli.dataset, cli.seed)]
    ctxs, worst = {}, 0.0
    for arm, ds, seed in targets:
        if (arm, ds) not in ctxs:
            ctxs.clear()
            ctxs[(arm, ds)] = Context(arm, ds, seed, device)
        out = evaluate_run(ctxs[(arm, ds)], arm, ds, seed, root)
        worst = max(worst, max(out["abs_diff_5shot"].values()))
        print(f"{arm} {ds} {seed} l2={100 * out['test_acc_l2']:.2f} l2_1s={100 * out['test_acc_l2_1shot']:.2f} max|d5|={max(out['abs_diff_5shot'].values()):.2e}", flush=True)
    print(f"targets {len(targets)} max abs diff (5-shot) {worst:.3e}")


if __name__ == "__main__":
    main()
