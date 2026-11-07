"""exp026: freeze downstream validation/test episodes for future GFSL experiments.

Run from the NAQ repository root. Does not train a model.
Future experiments should import load_fixed_episodes from this module and never
resample downstream tasks during normal training or final evaluation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data import load_data
from task_generator import downstream_task_generator
from utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Persist one canonical GFSL evaluation split")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", default="Amazon_clothing")
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot_test", type=int, default=1)
    parser.add_argument("--q_query_test", type=int, default=8)
    parser.add_argument("--num_valid_tasks", type=int, default=50)
    parser.add_argument("--num_downstream_tasks", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--latent", type=int, default=256)
    parser.add_argument("--query_generation", default="NaQ")
    parser.add_argument("--type", default="feature")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--json_path", default="exp026.json")
    return parser.parse_args()


def _integers(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value).reshape(-1)
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.all(array == np.round(array)):
            raise ValueError("Episode IDs and labels must be integers")
    return [int(v) for v in array.tolist()]


def _serialize_episode(task, n_way, k_shot, q_query):
    if len(task) != 4:
        raise ValueError("Each episode must contain (supports, y_support, queries, y_query)")
    supports, y_support, queries, y_query = [_integers(v) for v in task]
    if len(supports) != n_way * k_shot or len(y_support) != len(supports):
        raise ValueError(f"Invalid support shape: {len(supports)}")
    if len(queries) != n_way * q_query or len(y_query) != len(queries):
        raise ValueError(f"Invalid query shape: {len(queries)}")
    classes = y_support[::k_shot]
    if len(set(classes)) != n_way:
        raise ValueError("Support labels must contain exactly n_way distinct classes")
    for i, label in enumerate(classes):
        if y_support[i * k_shot : (i + 1) * k_shot] != [label] * k_shot:
            raise ValueError("Support nodes must be grouped by class")
    if any(label not in classes for label in y_query):
        raise ValueError("Query label is not among support classes")
    return {"supports": supports, "y_support": y_support, "queries": queries, "y_query": y_query}


def _digest(episodes):
    payload = json.dumps(episodes, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _save_split(path, split, tasks, args):
    episodes = [_serialize_episode(t, args.n_way, args.k_shot_test, args.q_query_test) for t in tasks]
    document = {
        "format": "gfsl_fixed_episodes_v1",
        "split": split,
        "dataset": args.dataset,
        "n_way": args.n_way,
        "k_shot": args.k_shot_test,
        "q_query": args.q_query_test,
        "seed": args.seed,
        "num_episodes": len(episodes),
        "sha256": _digest(episodes),
        "episodes": episodes,
    }
    # Exclusive creation prevents accidental changes to the canonical evaluation episodes.
    with path.open("x", encoding="utf-8") as file:
        json.dump(document, file, ensure_ascii=False, indent=2)
    return {"path": str(path), "num_episodes": len(episodes), "sha256": document["sha256"]}


def load_fixed_episodes(path, dataset=None, n_way=None, k_shot=None, q_query=None):
    """Return NAQ-compatible list of (supports, y_support, queries, y_query) tuples.

    Use this in all subsequent experiments for validation and testing.
    """
    with Path(path).open("r", encoding="utf-8") as file:
        document = json.load(file)
    if document.get("format") != "gfsl_fixed_episodes_v1":
        raise ValueError(f"Unsupported fixed-episode format: {path}")
    for field, expected in [("dataset", dataset), ("n_way", n_way), ("k_shot", k_shot), ("q_query", q_query)]:
        if expected is not None and document[field] != expected:
            raise ValueError(f"{path}: {field} mismatch: {document[field]} != {expected}")
    episodes = document["episodes"]
    if _digest(episodes) != document["sha256"]:
        raise RuntimeError(f"Fixed episode SHA-256 mismatch: {path}")
    if len(episodes) != document["num_episodes"]:
        raise RuntimeError(f"Fixed episode count mismatch: {path}")
    return [
        (
            np.asarray(ep["supports"], dtype=np.int64),
            np.asarray(ep["y_support"], dtype=np.int64),
            np.asarray(ep["queries"], dtype=np.int64),
            np.asarray(ep["y_query"], dtype=np.int64),
        )
        for ep in episodes
    ]


def main():
    args = parse_args()
    args.setting = "unsup"
    seed_everything(args.seed)
    torch.set_num_threads(4)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    output_dir = Path(
        args.output_dir or (f"save/fixed_episodes/{args.dataset}/" f"{args.n_way}way_{args.k_shot_test}shot_{args.q_query_test}query_seed{args.seed}")
    )
    valid_path = output_dir / "valid_episodes.json"
    test_path = output_dir / "test_episodes.json"
    for path in [valid_path, test_path]:
        if path.exists():
            raise FileExistsError(f"Canonical episode file already exists; refusing overwrite: {path}")

    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    data.add_diffusion(args)
    valid_tasks, test_tasks = downstream_task_generator(args, data, args.n_way, args.k_shot_test)
    if len(valid_tasks) != args.num_valid_tasks or len(test_tasks) != args.num_downstream_tasks:
        raise RuntimeError(f"Unexpected episode counts: valid={len(valid_tasks)}, test={len(test_tasks)}")

    # Validate both splits completely before creating either file.
    for split_tasks in [valid_tasks, test_tasks]:
        for task in split_tasks:
            _serialize_episode(task, args.n_way, args.k_shot_test, args.q_query_test)

    output_dir.mkdir(parents=True, exist_ok=True)
    valid = _save_split(valid_path, "valid", valid_tasks, args)
    test = _save_split(test_path, "test", test_tasks, args)
    report = {
        "experiment": "exp026",
        "description": "Canonical frozen evaluation episodes; no GNN training",
        "config": vars(args),
        "valid": valid,
        "test": test,
        "usage": "from exp026 import load_fixed_episodes; valid_tasks = load_fixed_episodes(valid_path, ...); test_tasks = load_fixed_episodes(test_path, ...)",
    }
    with Path(args.json_path).open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    result_dir = Path(f"results/{args.dataset}/exp026")
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "result.txt").open("w", encoding="utf-8") as file:
        file.write(f"VALID {valid['path']} sha256={valid['sha256']}\n")
        file.write(f"TEST {test['path']} sha256={test['sha256']}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
