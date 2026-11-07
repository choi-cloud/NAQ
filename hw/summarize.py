"""$HW_RESULTS_ROOT (default results/hw)/<arm>/<ds>/seed<S>/metrics.json -> results/hw/summary.{csv,md}.

    python -m hw.summarize [--ref <arm>]

--ref <arm> also writes results/hw/paired_<arm>.md: per arm and dataset, the
same-seed paired Δ = arm − ref (mean, std ddof=0, positive seeds k/n).
Accuracies are reported in % (x100). Only status == "completed" runs are aggregated;
other runs are listed in summary.md under "not completed".
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_ROOT = Path(os.environ.get("HW_RESULTS_ROOT", "results/hw"))
FIELDS = [
    "arm", "ds", "seed", "best_epoch", "val", "test", "ci95", "elapsed", "orig_pos_weight_mean", "nbr_pos_weight_mean",
    "collapsed", "nbr_eq_support", "nbr_eq_query", "pos_nbr_cos_raw", "nbr_in_own_queries", "nbr_in_any_queries",
]
COLLAPSE_VAL_MAX = 25.0  # best-epoch validation accuracy (%) <= this -> collapsed=1 (CLAUDE.md §4.3)
SKIP_DIRS = {"repro", "_regress"}


def _final_nbr_diag(result, key):
    """HW03a neighbor diagnostic of the last eval interval (None if not recorded)."""
    epochs = result.get("epochs") or []
    if not epochs or not epochs[-1].get("nbr_diag"):
        return None
    value = epochs[-1]["nbr_diag"].get(key)
    return None if value is None else round(value, 4)


def _final_diag(result, key):
    """HSAN diagnostic of the last eval interval (None for arms without weights)."""
    epochs = result.get("epochs") or []
    if not epochs or "hsan_diag" not in epochs[-1]:
        return None
    value = epochs[-1]["hsan_diag"].get(key)
    return None if value is None else round(value, 4)


def paired(rows, ref):
    """{arm: {ds: (deltas list, seeds)}} for arms other than ref, matched on seed."""
    by_key = {(r["arm"], r["ds"], r["seed"]): r["test"] for r in rows}
    out = defaultdict(dict)
    for arm in sorted({r["arm"] for r in rows} - {ref}):
        for ds in sorted({r["ds"] for r in rows}):
            seeds = sorted(s for (a, d, s) in by_key if a == arm and d == ds and (ref, ds, s) in by_key)
            if seeds:
                out[arm][ds] = ([by_key[(arm, ds, s)] - by_key[(ref, ds, s)] for s in seeds], seeds)
    return out


def write_paired(rows, ref):
    table = paired(rows, ref)
    datasets = sorted({r["ds"] for r in rows})
    entries = []
    for arm, per_ds in table.items():
        means = [float(np.mean(per_ds[ds][0])) for ds in datasets if ds in per_ds]
        positives = sum(int(np.sum(np.array(per_ds[ds][0]) > 0)) for ds in datasets if ds in per_ds)
        avg = float(np.mean(means)) if len(means) == len(datasets) else None
        entries.append((arm, per_ds, avg, positives))
    entries.sort(key=lambda e: (-(e[2] if e[2] is not None else -1e9), -e[3], e[0]))
    lines = [
        f"# Paired Δ vs `{ref}`",
        "",
        "Δ = arm − ref test accuracy (%p), matched on seed. Cell: mean ± std (ddof=0), positive seeds k/n.",
        "Sorted by the mean of the per-dataset Δ means (ties: total positive seeds).",
        "",
        "| arm | " + " | ".join(datasets) + " | mean of ds means | positive total |",
        "|---|" + "---|" * len(datasets) + "---|---|",
    ]
    for arm, per_ds, avg, positives in entries:
        cells = []
        for ds in datasets:
            if ds not in per_ds:
                cells.append("—")
                continue
            d = np.array(per_ds[ds][0])
            cells.append(f"{d.mean():+.2f} ± {d.std(ddof=0):.2f} ({int((d > 0).sum())}/{len(d)})")
        avg_text = f"{avg:+.2f}" if avg is not None else "—"
        lines.append(f"| {arm} | " + " | ".join(cells) + f" | {avg_text} | {positives} |")
    lines += ["", "## per-seed Δ", "", "| arm | ds | " + " | ".join(f"seed {s}" for s in _all_seeds(rows)) + " |", "|---|---|" + "---|" * len(_all_seeds(rows))]
    for arm, per_ds, _, _ in entries:
        for ds in datasets:
            if ds not in per_ds:
                continue
            values = dict(zip(per_ds[ds][1], per_ds[ds][0]))
            lines.append(f"| {arm} | {ds} | " + " | ".join(f"{values[s]:+.2f}" if s in values else "—" for s in _all_seeds(rows)) + " |")
    path = RESULTS_ROOT / f"paired_{ref}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def _all_seeds(rows):
    return sorted({r["seed"] for r in rows})


def collect():
    rows, incomplete = [], []
    for path in sorted(RESULTS_ROOT.glob("*/*/seed*/metrics.json")):
        if path.parts[len(RESULTS_ROOT.parts)] in SKIP_DIRS:
            continue
        with open(path, "r", encoding="utf-8") as file:
            result = json.load(file)
        if result.get("status") != "completed":
            incomplete.append((str(path), result.get("status"), result.get("error")))
            continue
        best = result["best"]
        rows.append(
            {
                "arm": result["arm"],
                "ds": result["dataset"],
                "seed": result["seed"],
                "best_epoch": best["epoch"],
                "val": round(100 * best["validation"]["accuracy_mean"], 4),
                "test": round(100 * best["test"]["accuracy_mean"], 4),
                "ci95": round(100 * best["test"]["accuracy_ci95_half_width"], 4),
                "elapsed": result["elapsed_seconds"],
                "orig_pos_weight_mean": _final_diag(result, "orig_pos_weight_mean"),
                "nbr_pos_weight_mean": _final_diag(result, "nbr_pos_weight_mean"),
                "collapsed": int(100 * best["validation"]["accuracy_mean"] <= COLLAPSE_VAL_MAX),
                "nbr_eq_support": _final_nbr_diag(result, "nbr_eq_support"),
                "nbr_eq_query": _final_nbr_diag(result, "nbr_eq_query"),
                "pos_nbr_cos_raw": _final_nbr_diag(result, "pos_nbr_cos_raw"),
                "nbr_in_own_queries": _final_nbr_diag(result, "nbr_in_own_queries"),
                "nbr_in_any_queries": _final_nbr_diag(result, "nbr_in_any_queries"),
            }
        )
    return rows, incomplete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", action="append", default=[], help="reference arm for paired Δ (repeatable)")
    cli = parser.parse_args()
    rows, incomplete = collect()
    with open(RESULTS_ROOT / "summary.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["arm"], row["ds"])].append(row)
    arms = sorted({row["arm"] for row in rows})
    datasets = sorted({row["ds"] for row in rows})
    lines = [
        "# HW summary",
        "",
        "test accuracy (%), seed mean ± std (ddof=0); n = number of seeds. 5-way 5-shot, 500 test tasks.",
        f"[c=k]: k runs collapsed (best-epoch validation ≤ {COLLAPSE_VAL_MAX:.0f}%), included in the mean.",
        "",
        "| arm | " + " | ".join(datasets) + " |",
        "|---|" + "---|" * len(datasets),
    ]
    for arm in arms:
        cells = []
        for ds in datasets:
            group = groups.get((arm, ds), [])
            if not group:
                cells.append("—")
                continue
            tests = np.array([row["test"] for row in group])
            collapsed = sum(row["collapsed"] for row in group)
            mark = f" [c={collapsed}]" if collapsed else ""
            cells.append(f"{tests.mean():.2f} ± {tests.std(ddof=0):.2f} (n={len(group)}){mark}")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += ["", "## runs", "", "| arm | ds | seed | best_epoch | val | test | ci95 | elapsed (s) | collapsed |", "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['arm']} | {row['ds']} | {row['seed']} | {row['best_epoch']} | {row['val']:.2f} | {row['test']:.2f} | {row['ci95']:.2f} | {row['elapsed']:.0f} | {row['collapsed']} |")
    if incomplete:
        lines += ["", "## not completed", ""]
        lines += [f"- `{path}`: {status} {error or ''}" for path, status, error in incomplete]
    (RESULTS_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    for ref in cli.ref:
        write_paired(rows, ref)


if __name__ == "__main__":
    main()
