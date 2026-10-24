"""hl001_hl01_tables: builds HL01 tables T1-T3 and pre-registration values from exp049 report.json files."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def load(root, dataset, method, name, seed):
    path = root / dataset / method / name / f"seed_{seed}" / "report.json"
    return json.loads(path.read_text()) if path.exists() else None


def write(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[tables] wrote {path}")


def pct(x):
    return round(100 * x, 2)


def mean_std(values):
    if not values:
        return "", ""
    return round(statistics.mean(values), 2), round(statistics.stdev(values), 2) if len(values) > 1 else ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/hl/p0")
    p.add_argument("--dataset", default="Amazon_clothing")
    p.add_argument("--taus", default="0.2,0.3,0.4")
    p.add_argument("--tau_star", type=float, required=True)
    p.add_argument("--seeds", default="1212,1213,1214")
    p.add_argument("--choi_ref", type=float, default=80.66)
    p.add_argument("--out_dir", default="reports/HL01_tables")
    args = p.parse_args()
    root, out_dir = Path(args.root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",")]
    taus = [float(x) for x in args.taus.split(",")]
    star = f"tau_{args.tau_star:g}"

    # T1: smoke
    doc = load(root, args.dataset, "naq", "baseline", 1212)
    ev1, ev5 = doc["evaluation"]["1"], doc["evaluation"]["5"]
    t1 = [
        dict(
            run="naq/baseline/seed_1212",
            valid_1shot=pct(ev1["valid_accuracy"]),
            test_1shot=pct(ev1["test_accuracy"]),
            valid_5shot=pct(ev5["valid_accuracy"]),
            test_5shot=pct(ev5["test_accuracy"]),
            best_epoch_1shot=ev1["best_epoch"],
            best_epoch_5shot=ev5["best_epoch"],
            elapsed_seconds=doc["elapsed_seconds"],
            choi_ref_1shot_test=args.choi_ref,
            delta_1shot_test=round(pct(ev1["test_accuracy"]) - args.choi_ref, 2),
        )
    ]
    write(out_dir / "T1_smoke.csv", list(t1[0]), t1)

    # tau search
    tau_rows = []
    for tau in taus:
        d = load(root, args.dataset, "ours", f"tau_{tau:g}", 1212)
        tau_rows.append(
            dict(
                tau=tau,
                valid_1shot=pct(d["evaluation"]["1"]["valid_accuracy"]),
                valid_5shot=pct(d["evaluation"]["5"]["valid_accuracy"]),
                best_epoch_1shot=d["evaluation"]["1"]["best_epoch"],
            )
        )
    write(out_dir / "tau_search.csv", list(tau_rows[0]), tau_rows)

    # T2: per-seed + mean/std
    arms = [("naq", "baseline"), ("ours", star)]
    t2, per_arm = [], {}
    for method, name in arms:
        cols = {k: [] for k in ("test_1shot", "test_5shot", "valid_1shot", "best_epoch_1shot", "best_epoch_5shot")}
        for seed in seeds:
            d = load(root, args.dataset, method, name, seed)
            if d is None:
                t2.append(dict(arm=f"{method}/{name}", seed=seed, status="missing"))
                continue
            row = dict(
                arm=f"{method}/{name}",
                seed=seed,
                status="complete",
                test_1shot=pct(d["evaluation"]["1"]["test_accuracy"]),
                test_5shot=pct(d["evaluation"]["5"]["test_accuracy"]),
                valid_1shot=pct(d["evaluation"]["1"]["valid_accuracy"]),
                best_epoch_1shot=d["evaluation"]["1"]["best_epoch"],
                best_epoch_5shot=d["evaluation"]["5"]["best_epoch"],
            )
            for k in cols:
                cols[k].append(row[k])
            t2.append(row)
        per_arm[method] = cols
        mean_row, std_row = dict(arm=f"{method}/{name}", seed="mean"), dict(arm=f"{method}/{name}", seed="std")
        for k, v in cols.items():
            mean_row[k], std_row[k] = mean_std(v)
        t2 += [mean_row, std_row]
    fields = ["arm", "seed", "status", "test_1shot", "test_5shot", "valid_1shot", "best_epoch_1shot", "best_epoch_5shot"]
    write(out_dir / "T2_p0.csv", fields, t2)

    # T3: diagnostics of ours tau*
    keys = ["query_gt_match", "original_gt_match", "original_x_rank", "coverage_without_fallback", "fallback_occurrence_rate"]
    t3 = []
    for seed in seeds:
        d = load(root, args.dataset, "ours", star, seed)
        if d is not None:
            t3.append(dict(seed=seed, **{k: d["diagnostics"][k] for k in keys}))
    write(out_dir / "T3_diagnostics.csv", ["seed"] + keys, t3)

    # Pre-registration computed values
    naq, ours = per_arm["naq"], per_arm["ours"]
    paired = [o - n for o, n in zip(ours["test_1shot"], naq["test_1shot"])]
    paired5 = [o - n for o, n in zip(ours["test_5shot"], naq["test_5shot"])]
    d1 = round(statistics.mean(ours["test_1shot"]) - statistics.mean(naq["test_1shot"]), 2)
    d5 = round(statistics.mean(ours["test_5shot"]) - statistics.mean(naq["test_5shot"]), 2)
    prereg = [
        dict(id="HLP001", value=f"delta={t1[0]['delta_1shot_test']}"),
        dict(id="HLP002", value="; ".join(f"tau{r['tau']:g}={r['valid_1shot']}" for r in tau_rows)),
        dict(
            id="HLP003",
            value=f"delta_mean={d1}; per_seed_sign="
            + ",".join(f"{s}:{'+' if x > 0 else ('-' if x < 0 else '0')}({x:.2f})" for s, x in zip(seeds, paired)),
        ),
        dict(
            id="HLP004",
            value=f"delta_1shot={d1}; delta_5shot={d5}; per_seed_5shot="
            + ",".join(f"{s}:{x:.2f}" for s, x in zip(seeds, paired5)),
        ),
        dict(
            id="HLP005",
            value="; ".join(f"{r['seed']}:query={r['query_gt_match']},original={r['original_gt_match']}" for r in t3)
            + f"; mean query={statistics.mean(r['query_gt_match'] for r in t3):.5f}"
            + f", original={statistics.mean(r['original_gt_match'] for r in t3):.5f}",
        ),
        dict(
            id="HLP006",
            value="; ".join(f"{r['seed']}:{r['coverage_without_fallback']:.4f}" for r in t3)
            + f"; mean={statistics.mean(r['coverage_without_fallback'] for r in t3):.4f}",
        ),
    ]
    write(out_dir / "prereg.csv", ["id", "value"], prereg)


if __name__ == "__main__":
    main()
