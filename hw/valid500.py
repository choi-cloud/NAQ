"""HW03b: validation-500 cache save/valid500/<ds>.pkl (outside git).

    python -m hw.valid500 --dataset <ds>

Layout: list of 500 tasks (id_support np.ndarray, y_spt LongTensor, id_query np.ndarray, y_query LongTensor),
the same tuple format as the validation pool of save/downstream_tasks/<ds>/5way_5shot.pkl.
- tasks 0..49: the existing 50 validation tasks, copied unchanged.
- tasks 50..499: task_generator.supervised_task_generator on the existing validation classes
  = the classes that appear in the existing 50 validation tasks (sorted). Data.class_list_valid is
  not used: data.py draws the class split with the global `random` after the run seed, so it
  depends on the seed and differs from the split that produced 5way_5shot.pkl.
  5-way, k = 5, q = 8 (the existing format), drawn with a dedicated
  random.Random(20261108) swapped into task_generator for the duration of the call
  (supervised_task_generator draws only through `random.sample`); the global RNGs are not consumed.
The existing 5way_5shot.pkl is only read. G-valid checks (a)-(c) are written to
save/valid500/<ds>.json and printed.
"""

import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch

import task_generator
from data import load_data
from hw.config import build_args, load_config

VALID500_SEED = 20261108
VALID500_DIR = Path("save/valid500")
NUM_TASKS = 500
NUM_EXISTING = 50


def existing_path(dataset, n_way=5, k_shot=5):
    return Path(f"save/downstream_tasks/{dataset}/{n_way}way_{k_shot}shot.pkl")


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def load_valid500(dataset):
    with open(VALID500_DIR / f"{dataset}.pkl", "rb") as file:
        pool = pickle.load(file)
    if len(pool) != NUM_TASKS:
        raise ValueError(f"valid500 cache for {dataset} has {len(pool)} tasks")
    return pool


def _task_equal(a, b):
    return all(
        np.array_equal(np.asarray(x), np.asarray(y)) if not torch.is_tensor(x) else (torch.is_tensor(y) and torch.equal(x, y))
        for x, y in zip(a, b)
    ) and len(a) == len(b)


def build(dataset):
    cfg = load_config("naq_base")
    args = build_args(cfg, dataset)
    src = existing_path(dataset, args.n_way, args.k_shot_test)
    md5_before = md5(src)
    with open(src, "rb") as file:
        valid_pool, _ = pickle.load(file)
    if len(valid_pool) != NUM_EXISTING:
        raise ValueError(f"expected {NUM_EXISTING} existing validation tasks, got {len(valid_pool)}")
    data = load_data(args)  # CPU; no device needed for task sampling
    k_shot = int(valid_pool[0][0].size // args.n_way)
    q_query = int(valid_pool[0][2].size // args.n_way)
    labels = np.asarray(data.labels)
    # measurement-only label use: class membership of the cached validation tasks (defines the class set)
    classes_of = lambda task: set(int(c) for c in labels[np.concatenate([task[0], task[2]])])  # noqa: E731
    existing_classes = set().union(*(classes_of(t) for t in valid_pool))
    class_list = sorted(existing_classes)
    rng = random.Random(VALID500_SEED)
    saved = task_generator.random
    try:
        task_generator.random = rng  # dedicated stream; module-level `random` restored below
        new = [
            task_generator.supervised_task_generator(data.id_by_class, class_list, data.labels, args.n_way, k_shot, q_query)
            for _ in range(NUM_TASKS - NUM_EXISTING)
        ]
    finally:
        task_generator.random = saved
    pool = list(valid_pool) + new

    new_classes = set().union(*(classes_of(t) for t in new))
    key = lambda t: (frozenset(classes_of(t)), frozenset(np.concatenate([t[0], t[2]]).tolist()))  # noqa: E731
    existing_keys = {key(t) for t in valid_pool}
    VALID500_DIR.mkdir(parents=True, exist_ok=True)
    out = VALID500_DIR / f"{dataset}.pkl"
    with open(out, "wb") as file:
        pickle.dump(pool, file)
    reloaded = load_valid500(dataset)
    report = {
        "dataset": dataset,
        "seed": VALID500_SEED,
        "format": {"n_way": args.n_way, "k_shot": k_shot, "q_query": q_query, "tasks": len(reloaded)},
        "existing_md5_before": md5_before,
        "existing_md5_after": md5(src),
        "valid500_md5": md5(out),
        "existing_validation_classes": sorted(existing_classes),
        "G_valid": {
            "a_first50_identical": all(_task_equal(reloaded[i], valid_pool[i]) for i in range(NUM_EXISTING)),
            "b_new_classes_in_existing_validation_classes": new_classes <= existing_classes,
            "c_existing_md5_unchanged": md5(src) == md5_before,
        },
        "new_task_classes_used": len(new_classes),
        "new_tasks_identical_to_a_first50_task": int(sum(key(t) in existing_keys for t in new)),
    }
    report["G_valid"]["pass"] = bool(report["G_valid"]["a_first50_identical"] and report["G_valid"]["b_new_classes_in_existing_validation_classes"] and report["G_valid"]["c_existing_md5_unchanged"])
    (VALID500_DIR / f"{dataset}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["Cora_full", "Amazon_clothing"])
    report = build(parser.parse_args().dataset)
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("classes")}, indent=2))


if __name__ == "__main__":
    main()
