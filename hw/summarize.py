"""results/hw/<arm>/<ds>/seed<S>/metrics.json -> results/hw/summary.{csv,md}.

    python -m hw.summarize

Accuracies are reported in % (x100). Only status == "completed" runs are aggregated;
other runs are listed in summary.md under "not completed".
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_ROOT = Path("results/hw")
FIELDS = ["arm", "ds", "seed", "best_epoch", "val", "test", "ci95", "elapsed"]


def collect():
    rows, incomplete = [], []
    for path in sorted(RESULTS_ROOT.glob("*/*/seed*/metrics.json")):
        if path.parts[len(RESULTS_ROOT.parts)] == "repro":
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
            }
        )
    return rows, incomplete


def main():
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
            cells.append(f"{tests.mean():.2f} ± {tests.std(ddof=0):.2f} (n={len(group)})")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += ["", "## runs", "", "| arm | ds | seed | best_epoch | val | test | ci95 | elapsed (s) |", "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['arm']} | {row['ds']} | {row['seed']} | {row['best_epoch']} | {row['val']:.2f} | {row['test']:.2f} | {row['ci95']:.2f} | {row['elapsed']:.0f} |")
    if incomplete:
        lines += ["", "## not completed", ""]
        lines += [f"- `{path}`: {status} {error or ''}" for path, status, error in incomplete]
    (RESULTS_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
