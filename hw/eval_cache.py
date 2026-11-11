"""HW09 §1.1: evaluation-episode caches save/downstream_tasks/<ds>/5way_<k>shot.pkl for missing files only.

    PYTHONPATH=. python -m hw.eval_cache --dataset <ds> --shot <k> [--episodes 2000] [--workdir DIR] [--install]

Code path = the one that wrote the existing caches: the first full training run (seed 1212) reached
task_generator.downstream_task_generator after seed -> load_data -> set_device -> add_diffusion -> train pool
(HW01: Cora_full 5way_5shot.pkl was written by the exp015 seed-1212 run with 2000 episodes; hw/run.py follows
the same RNG order, hw/episodes.py). This script replays that order with `naq_base` settings, only
k_shot_test changed, and calls the original NaQ function unchanged (valid 50 / test 500, q 8).

The file is written under --workdir (default: a scratch dir; ./data is symlinked there) so an existing cache
under save/ is never opened for writing. --install copies the new file into save/downstream_tasks/<ds>/
only when that path does not exist yet.
"""

import argparse
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from data import load_data
from hw.config import build_args, load_config
from hw.episodes import build_train_pool
from task_generator import downstream_task_generator
from utils import seed_everything

REPO = Path(__file__).resolve().parent.parent
CACHE_SEED = 1212


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def build(dataset, shot, episodes, workdir):
    workdir = Path(workdir).resolve()
    (workdir / "data").exists() or (workdir / "data").symlink_to(REPO / "data")
    cfg = load_config("naq_base")
    cfg = {**cfg, "episodes": episodes}
    args = build_args(cfg, dataset)
    args.k_shot_test = shot
    args.epoch = episodes
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        if (workdir / f"save/downstream_tasks/{dataset}/{args.n_way}way_{shot}shot.pkl").exists():
            raise FileExistsError("workdir already holds this cache")
        seed_everything(CACHE_SEED)
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        data = load_data(args)
        data.set_device(str(device))
        data.add_diffusion(args)
        build_train_pool(args, data)  # consumes the RNG exactly as the first training run did
        valid_pool, test_pool = downstream_task_generator(args, data, args.n_way, shot, args.q_query_test)
    finally:
        os.chdir(cwd)
    out = workdir / f"save/downstream_tasks/{dataset}/{args.n_way}way_{shot}shot.pkl"
    labels = data.labels.cpu().numpy()
    # measurement-only label use: class sets of the cached tasks
    classes = lambda pool: sorted({int(c) for t in pool for c in labels[np.concatenate([t[0], t[2]])]})  # noqa: E731
    return out, {"dataset": dataset, "shot": shot, "seed": CACHE_SEED, "episodes_before": episodes, "md5": md5(out),
                 "valid_tasks": len(valid_pool), "test_tasks": len(test_pool),
                 "k_shot": int(valid_pool[0][0].size // args.n_way), "q_query": int(valid_pool[0][2].size // args.n_way),
                 "valid_classes": classes(valid_pool), "test_classes": classes(test_pool),
                 "class_list_valid": sorted(int(c) for c in data.class_list_valid), "class_list_test": sorted(int(c) for c in data.class_list_test)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--install", action="store_true")
    cli = parser.parse_args()
    workdir = cli.workdir or tempfile.mkdtemp(prefix="hw_cache_")
    Path(workdir).mkdir(parents=True, exist_ok=True)
    out, info = build(cli.dataset, cli.shot, cli.episodes, workdir)
    target = REPO / f"save/downstream_tasks/{cli.dataset}/5way_{cli.shot}shot.pkl"
    info["existing_target_md5"] = md5(target) if target.exists() else None
    info["installed"] = False
    if cli.install and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out, target)
        info["installed"] = md5(target) == info["md5"]
    info["workdir_file"] = str(out)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
