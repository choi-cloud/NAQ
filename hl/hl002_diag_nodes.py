"""hl002_diag_nodes: HL02 §2 training-free diagnostics (node table, pair and episode statistics). Reads GT labels (diag only)."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hl_common as H  # noqa: E402
from task_generator import spt_sampling  # noqa: E402
from utils import seed_everything  # noqa: E402

QS = (10, 25, 50, 75, 90)


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[hl002] wrote {path}", flush=True)


def describe(v):
    v = np.asarray(v, dtype=np.float64)
    out = {f"p{q}": float(np.percentile(v, q)) for q in QS}
    out["mean"] = float(v.mean())
    return out


def spearman(a, b):
    return float(stats.spearmanr(a, b).correlation)


def partial_spearman(a, b, c):
    """Spearman partial correlation of a, b given c (Pearson partial correlation on ranks)."""
    rab, rac, rbc = spearman(a, b), spearman(a, c), spearman(b, c)
    return float((rab - rac * rbc) / np.sqrt((1 - rac**2) * (1 - rbc**2)))


def quintiles(values, seed):
    """Equal-size quintiles 1..5 of `values` (1 = lowest); ties broken by a seeded random permutation."""
    n = len(values)
    tiebreak = np.random.default_rng(seed).permutation(n)
    order = np.lexsort((tiebreak, values))
    q = np.empty(n, dtype=np.int64)
    q[order] = np.arange(n) * 5 // n + 1
    return q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--L", type=int, default=50)
    p.add_argument("--n_random_pairs", type=int, default=100000)
    p.add_argument("--out_dir", default="results/hl/hl02/diag")
    p.add_argument("--table_dir", default="reports/HL02_tables")
    a = p.parse_args()
    out_dir, table_dir = Path(a.out_dir), Path(a.table_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    timer = H.Timer()
    timing = {}

    args = H.base_args(seed=a.seed, tau=a.tau)
    seed_everything(args.seed)
    data, device = H.load_everything(args)
    # Uniform anchors drawn right after load_data, as exp049 does (same python RNG state for proto_task_generator).
    episodes_anchors = [np.asarray(spt_sampling(data.id_train, args.n_way, 1)[0], dtype=np.int64) for _ in range(args.epoch)]
    timing["load"] = timer()

    z0, z1, z2 = H.structural_scores(data, device)
    check = H.verify_against_exp049(data, args, device, z=(z0, z1, z2), seed=args.seed)
    print(f"[hl002] exp049 identity check passed: {check}", flush=True)

    labels = H.labels_np(data)
    n = len(labels)
    nodes = np.arange(n)
    t = H.Timer()
    mined = H.mine_queries_for(nodes, data, args, device)
    timing["mine_all_nodes_seconds"] = t()
    assert len(mined) == n

    # degree: neighbours in the symmetrised graph without self-loops (the graph exp049 filters on)
    edges = data.edge_index.detach().cpu().numpy()
    pairs = np.unique(np.concatenate([edges.T, edges[::-1].T]), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    degree = np.bincount(pairs[:, 0], minlength=n)

    t = H.Timer()
    top_ids, top_cos = H.x_topL(data, nodes, a.L)
    top10, _ = H.x_topL(data, nodes, args.q_query)
    src = np.repeat(nodes, a.L)
    acc = H.accepted_mask(z1, z2, src, top_ids.reshape(-1), args.tau).reshape(n, a.L)
    timing["topL_filter_seconds"] = t()
    pass50 = acc.sum(1)
    h50 = a.L - pass50
    selected = np.asarray([mined[i]["selected_ids"] for i in nodes])
    pi_bchoi = (labels[selected] == labels[:, None]).mean(1)
    pi_naq = (labels[top10] == labels[:, None]).mean(1)
    raw = np.asarray([mined[i]["raw_retained_count"] for i in nodes])
    xrank = np.asarray([mined[i]["mean_original_x_rank"] for i in nodes])
    fallback = np.asarray([mined[i]["fallback"] for i in nodes])

    rows = [
        dict(node=int(i), label=int(labels[i]), degree=int(degree[i]), raw_retained=int(raw[i]),
             mean_x_rank=float(xrank[i]), fallback=int(fallback[i]), pass50=int(pass50[i]), H50=int(h50[i]),
             pi_bchoi=float(pi_bchoi[i]), pi_naq=float(pi_naq[i]))
        for i in nodes
    ]
    write_csv(out_dir / f"node_table_tau{args.tau:g}.csv", list(rows[0]), rows)
    np.save(out_dir / "bchoi_selected_ids.npy", selected)

    # D1
    cols = {"raw_retained": raw, "H50": h50, "pass50": pass50, "pi_bchoi": pi_bchoi, "pi_naq": pi_naq, "degree": degree}
    d1 = [dict(stat=k, **describe(v)) for k, v in cols.items()]
    rho = {
        "raw_retained~degree": spearman(raw, degree),
        "raw_retained~pi_bchoi": spearman(raw, pi_bchoi),
        "pass50~pi_bchoi": spearman(pass50, pi_bchoi),
        "degree~pi_bchoi": spearman(degree, pi_bchoi),
        "H50~pi_bchoi": spearman(h50, pi_bchoi),
        "raw_retained~H50": spearman(raw, h50),
        "raw_retained~pi_bchoi|degree": partial_spearman(raw, pi_bchoi, degree),
    }
    d1 += [dict(stat=f"spearman:{k}", mean=v) for k, v in rho.items()]
    d1.append(dict(stat="fallback_rate", mean=float(fallback.mean())))
    write_csv(table_dir / "D1_node_stats.csv", ["stat"] + [f"p{q}" for q in QS] + ["mean"], d1)

    quint = quintiles(pi_bchoi, args.seed)
    write_csv(out_dir / "pi_bchoi_quintiles.csv", ["node", "quintile"],
              [dict(node=int(i), quintile=int(quint[i])) for i in nodes])
    qrows = []
    for q in range(1, 6):
        m = quint == q
        qrows.append(dict(quintile=q, n_nodes=int(m.sum()), pi_bchoi_min=float(pi_bchoi[m].min()),
                          pi_bchoi_max=float(pi_bchoi[m].max()), pi_bchoi_mean=float(pi_bchoi[m].mean()),
                          degree_mean=float(degree[m].mean()), raw_retained_mean=float(raw[m].mean())))
    write_csv(table_dir / "D1b_pi_bchoi_quintiles.csv", list(qrows[0]), qrows)

    # D2
    same = labels[top_ids] == labels[:, None]
    rng = np.random.default_rng(args.seed)
    rs = rng.integers(0, n, a.n_random_pairs)
    rd = rng.integers(0, n - 1, a.n_random_pairs)
    rd = rd + (rd >= rs)  # uniform over dst != src
    d2 = [
        dict(set="topL_accepted", n_pairs=int(acc.sum()), same_gt_rate=float(same[acc].mean())),
        dict(set="topL_rejected", n_pairs=int((~acc).sum()), same_gt_rate=float(same[~acc].mean())),
        dict(set="uniform_random", n_pairs=a.n_random_pairs, same_gt_rate=float((labels[rs] == labels[rd]).mean())),
    ]
    for lo, hi in ((1, 10), (11, 30), (31, 50)):
        band = np.zeros_like(acc)
        band[:, lo - 1:hi] = True
        m = band & ~acc
        d2.append(dict(set=f"topL_rejected_rank{lo}-{hi}", n_pairs=int(m.sum()), same_gt_rate=float(same[m].mean())))
    write_csv(table_dir / "D2_pair_stats.csv", ["set", "n_pairs", "same_gt_rate"], d2)

    # D3: uniform anchors (exp049 RNG order), B_Choi queries from the all-node mining
    all_distinct, one_pair, multi_pair, q_overlap, anchor_in_q, xcos, a2cos = [], [], [], [], [], [], []
    iu = np.triu_indices(args.n_way, 1)
    for anchors in episodes_anchors:
        g = labels[anchors]
        n_same = int((g[iu[0]] == g[iu[1]]).sum())
        all_distinct.append(n_same == 0)
        one_pair.append(n_same == 1)
        multi_pair.append(n_same >= 2)
        qs = [set(mined[int(s)]["selected_ids"]) for s in anchors]
        q_overlap.append(any(qs[i] & qs[j] for i, j in zip(*iu)))
        anchor_in_q.append(any(int(anchors[i]) in qs[j] for i in range(args.n_way) for j in range(args.n_way) if i != j))
        xcos.append(H.pair_cos(z0, anchors[iu[0]], anchors[iu[1]]).mean())
        a2cos.append(H.pair_cos(z2, anchors[iu[0]], anchors[iu[1]]).mean())
    d3 = [
        dict(stat="anchor_gt_all_distinct", value=float(np.mean(all_distinct))),
        dict(stat="anchor_gt_exactly_one_pair_same", value=float(np.mean(one_pair))),
        dict(stat="anchor_gt_two_or_more_pairs_same", value=float(np.mean(multi_pair))),
        dict(stat="query_sets_overlap_between_anchors", value=float(np.mean(q_overlap))),
        dict(stat="anchor_in_other_anchor_queries", value=float(np.mean(anchor_in_q))),
        dict(stat="anchor_pair_mean_x_cosine", value=float(np.mean(xcos))),
        dict(stat="anchor_pair_mean_a2x_cosine", value=float(np.mean(a2cos))),
        dict(stat="n_episodes", value=len(episodes_anchors)),
    ]
    write_csv(table_dir / "D3_episode_stats.csv", ["stat", "value"], d3)

    timing["total_seconds"] = timer()
    summary = dict(seed=args.seed, tau=args.tau, L=a.L, n_nodes=int(n), n_classes=int(labels.max() + 1),
                   identity_check=check, timing=timing, D1=d1, D1b=qrows, D2=d2, D3=d3,
                   degree_definition="neighbours in symmetrised edge_index without self-loops",
                   quintile_definition="equal-size rank quintiles of pi_bchoi, ties broken by default_rng(seed).permutation")
    (out_dir / "diag_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(table_dir / "D0_timing.csv", ["item", "seconds"], [dict(item=k, seconds=v) for k, v in timing.items()])
    print(f"[hl002] DONE {timing}", flush=True)


if __name__ == "__main__":
    main()
