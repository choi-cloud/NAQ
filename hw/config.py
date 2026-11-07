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


HSAN_ALLOWED = {
    "mode": {"p", "pn", "none"},
    "apply": {"orig", "both"},
    "beta": {1, 2, 3},
    "space": {"gcn", "raw"},
    "logit": {"euclid_gradscale", "cos_hsan"},
}
HSAN_DEFAULTS = {"space": "gcn", "logit": "euclid_gradscale", "tau_cos": 0.5}


def validate_hsan(hsan, loss_kind):
    """Fill defaults and check the hsan block. mode "none" = no weights (cos-logit control only)."""
    if hsan is None:
        return None
    if loss_kind != "original+neighbor":
        raise ValueError("hsan requires loss 'original+neighbor'")
    hsan = {**HSAN_DEFAULTS, **hsan}
    unknown = set(hsan) - set(HSAN_ALLOWED) - {"tau_cos"}
    if unknown:
        raise ValueError(f"Unknown hsan keys: {sorted(unknown)}")
    if hsan.get("mode") not in HSAN_ALLOWED["mode"]:
        raise ValueError(f"hsan.mode must be one of {sorted(HSAN_ALLOWED['mode'])}")
    if hsan["mode"] == "none":
        if hsan["logit"] != "cos_hsan":
            raise ValueError("hsan.mode 'none' is only valid with logit 'cos_hsan'")
        hsan.pop("apply", None)
        hsan.pop("beta", None)
        hsan.pop("space", None)
    else:
        for key in ("apply", "beta", "space", "logit"):
            if hsan.get(key) not in HSAN_ALLOWED[key]:
                raise ValueError(f"hsan.{key}={hsan.get(key)!r} not in {sorted(HSAN_ALLOWED[key], key=str)}")
    if not float(hsan["tau_cos"]) > 0:
        raise ValueError("hsan.tau_cos must be positive")
    return hsan


def load_config(arm, path=DEFAULT_CONFIG):
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    if arm not in raw["arms"]:
        raise KeyError(f"Unknown arm '{arm}'. Available: {sorted(raw['arms'])}")
    cfg = dict(raw["defaults"])
    spec = raw["arms"][arm]
    if "base" in spec:  # one level only
        base = spec["base"]
        if base not in raw["arms"] or "base" in raw["arms"][base]:
            raise ValueError(f"arm '{arm}': base '{base}' must be an existing arm without its own base")
        cfg.update({k: v for k, v in raw["arms"][base].items() if k != "note"})
    cfg.update({k: v for k, v in spec.items() if k != "base"})
    if "base" in spec:
        cfg["base"] = spec["base"]
    cfg["arm"] = arm
    cfg["hsan"] = validate_hsan(cfg.get("hsan"), cfg.get("loss"))
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
