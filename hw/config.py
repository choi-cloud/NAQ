"""configs/hw.json loader: defaults + arms[arm] -> one dict, and NaQ `args`.

`build_args` goes through arg.parse_args_proto("unsup") (same as Choi's
parse_args, which swaps sys.argv), so vars(args) has the same keys as the
`settings` dict in Choi's result JSON.
"""

import json
import sys
from pathlib import Path

from arg import parse_args_proto

DEFAULT_CONFIG = Path("configs/hw.json")

# config key -> parse_args_proto flag (only keys that exist in NaQ's parser)
_ARG_FLAGS = {
    "device": "--device",
    "n_way": "--n_way",
    "k_shot": "--k_shot",
    "k_shot_test": "--k_shot_test",
    "query_generation": "--query_generation",
    "type": "--type",
    "q_query": "--q_query",
    "q_query_test": "--q_query_test",
    "lr": "--lr",
    "epochs": "--epoch",
    "valid_tasks": "--num_valid_tasks",
    "test_tasks": "--num_downstream_tasks",
}


def load_config(arm, path=DEFAULT_CONFIG):
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    if arm not in raw["arms"]:
        raise KeyError(f"Unknown arm '{arm}'. Available: {sorted(raw['arms'])}")
    cfg = dict(raw["defaults"])
    cfg.update(raw["arms"][arm])
    cfg["arm"] = arm
    return cfg


def build_args(cfg, dataset):
    argv = ["hw.run", "--dataset", dataset]
    for key, flag in _ARG_FLAGS.items():
        argv += [flag, str(cfg[key])]
    original = sys.argv
    try:
        sys.argv = argv
        args = parse_args_proto("unsup")
    finally:
        sys.argv = original
    return args
