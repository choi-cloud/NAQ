"""hl011_pdd_train: copied from hl/hl008_distant_query_train.py; anchors stay uniform (hl003 code path), only query selection is replaced by --arm. Label-free arms never pass labels to selection; GT labels are read for oracle arms and post-hoc diagnostics only.

Arms (instructions/HL06_purity_distance_split.md §2, extended by HL07 §1-2 and HL10):
  label-free  LABEL_FREE[arm] = (judge, rule): judge/rule functions from hl/hl_pdd.py, teacher = --teacher (enc_b) / --teacher_n (enc_n);
              ARM_BAND overrides the X band, encb_strat5_r2 swaps the teacher, encb_strat5_drop2 drops the 2 highest-loss queries per anchor
              after a warm-up; choi_far / choi_strat re-select inside the exp049 filter; rand10 (hl008 rule)
  oracle      o_topsame, o_combswap, o_far50_p03, o_far50_p05, o_grid_<region>_p<pp>; reference oracles g1_p04, g1_p10, near, far
Queries: 10 per anchor from its X-rank band (--x_band, default 201-1000), chosen once per anchor and cached (as exp049).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hl_common as H  # noqa: E402
import hl_pdd as P  # noqa: E402
from data import load_data  # noqa: E402
from exp027 import evaluate  # noqa: E402
from exp049 import load_splits, query_diagnostics  # noqa: E402
from hl003_oracle_train import anchor_diagnostics, read_column  # noqa: E402
from hl005_diag_distance_purity import encoder_embeddings, sym_adjacency  # noqa: E402
from hl006_purity_oracle_train import centre, purity_oracle_queries  # noqa: E402
from hl008_distant_query_train import b4_scores, label_free_queries  # noqa: E402
from hl009_structural_oracle_train import structural_oracle_queries  # noqa: E402
from hl_pdd_diag import Describer, aggregate  # noqa: E402
from meta import ProtoNet  # noqa: E402
from task_generator import proto_task_generator  # noqa: E402
from utils import seed_everything  # noqa: E402

LABEL_FREE = {
    "encb_top10": ("enc_b", "top10"),
    "encb_strat5": ("enc_b", "strat5"),
    "encb_ex20": ("enc_b", "ex20"),
    "encb_far50": ("enc_b", "far50"),
    "knnppr_strat5": ("knnppr_b", "strat5"),
    "knnppr_ex20": ("knnppr_b", "ex20"),
    # HL07 §2
    "rand_strat5": (None, "rand_strat5"),
    "encb_strat5_wide": ("enc_b", "strat5"),
    "encb_strat5_b45": ("enc_b", "strat5"),
    "encb_strat5_r2": ("enc_b", "strat5"),
    "encb_strat5_drop2": ("enc_b", "strat5"),
    # HL10
    "encb_strat10": ("enc_b", "strat10"),
    "encb_strat3": ("enc_b", "strat3"),
    "encb_ex10": ("enc_b", "ex10"),
    "encb_ex40": ("enc_b", "ex40"),
    "nbr_strat5": ("nbr_b", "strat5"),
    "xidf_strat5": ("xidf", "strat5"),
    "clu_strat5": ("clu_b", "strat5"),
    "knnppr_far50": ("knnppr_b", "far50"),
}
ARM_BAND = {"encb_strat5_wide": "51,1000", "encb_strat5_b45": "201,3000"}
R2_TEACHERS = ["results/hl/hl06/train/Amazon_clothing/encb_strat5/seed_1212/best_1shot.pt",
               "results/hl/hl06/train/Amazon_clothing/encb_ex20/seed_1212/best_1shot.pt"]
CHOI = ["choi_far", "choi_strat"]
GRID = [f"o_grid_{r}_p{p}" for r in ("t1", "strat5", "far50") for p in ("03", "05", "07", "10")]
ORACLE = ["o_topsame", "o_combswap", "o_far50_p03", "o_far50_p05", "g1_p04", "g1_p10", "near", "far"] + GRID
LABEL_FREE_OTHER = ["rand10"] + CHOI
ALL_ARMS = list(LABEL_FREE) + LABEL_FREE_OTHER + ORACLE
DROP_WARMUP, DROP_K = 400, 2
TEACHER_B = "results/hl/p0/Amazon_clothing/ours/tau_0.3/seed_1212/best_1shot.pt"
TEACHER_N = "results/hl/p0/Amazon_clothing/naq/baseline/seed_1212/best_1shot.pt"
B5 = (1001, 3000)
SCORE_BATCH = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1212)
    p.add_argument("--epoch", type=int, default=2000)
    p.add_argument("--eval_interval", type=int, default=160)
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=1)
    p.add_argument("--k_shot_test", type=int, default=1)
    p.add_argument("--q_query", type=int, default=10)
    p.add_argument("--q_query_test", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--latent", type=int, default=256)
    p.add_argument("--l2_penalty", type=float, default=1e-4)
    p.add_argument("--num_valid_tasks", type=int, default=50)
    p.add_argument("--num_downstream_tasks", type=int, default=500)
    p.add_argument("--mine_batch_size", type=int, default=64)
    p.add_argument("--min_candidate_policy", choices=["relax"], default="relax")
    p.add_argument("--output_root", default="results/hl/hl06/train/Amazon_clothing")
    # hl011 additions
    p.add_argument("--arm", required=True, choices=ALL_ARMS)
    p.add_argument("--teacher", default=TEACHER_B)
    p.add_argument("--teacher_n", default=TEACHER_N)
    p.add_argument("--x_band", default="201,1000")
    p.add_argument("--ppr_alpha", type=float, default=0.15)
    p.add_argument("--ppr_iters", type=int, default=30)
    p.add_argument("--node_table", default="results/hl/hl02/diag/node_table_tau0.3.csv")
    return p.parse_args()


def rand_pick(rng, ids, k):
    return rng.choice(ids, k, replace=False) if k > 0 else ids[:0]


def o_topsame(a, band_ids, s2_band, ranked, s2_full, labels, q):
    """B4 GT-same by s2 descending; short -> B5 GT-same by s2 descending; then B4 GT-diff by s2 descending."""
    order = np.argsort(-s2_band, kind="stable")
    same = band_ids[order][labels[band_ids[order]] == labels[a]][:q]
    counts = {"primary": len(same), "b5": 0, "diff_fill": 0}
    chosen = list(same)
    if len(chosen) < q:
        b5 = ranked[B5[0] - 1:B5[1]]
        b5 = b5[labels[b5] == labels[a]]
        take = b5[np.argsort(-s2_full[b5], kind="stable")][:q - len(chosen)]
        chosen += list(take)
        counts["b5"] = len(take)
    if len(chosen) < q:
        diff = band_ids[order][labels[band_ids[order]] != labels[a]][:q - len(chosen)]
        chosen += list(diff)
        counts["diff_fill"] = len(diff)
    return np.asarray(chosen, dtype=np.int64), counts


def o_far50(a, n_same, band_ids, s2rank, labels, rng, q):
    """s2 rank 401-800: n_same GT-same + rest GT-diff at random; same shortfall from s2 rank 161-400 GT-same, then far50 GT-diff."""
    m = len(band_ids)
    far = band_ids[s2rank > m // 2]
    mid = band_ids[(s2rank > m // P.N_STRATA) & (s2rank <= m // 2)]
    is_same = labels[far] == labels[a]
    same = rand_pick(rng, far[is_same], min(n_same, int(is_same.sum())))
    counts = {"primary": len(same), "mid_same": 0, "diff_fill": 0}
    short = n_same - len(same)
    mid_same = mid[labels[mid] == labels[a]]
    take = rand_pick(rng, mid_same, min(short, len(mid_same)))
    counts["mid_same"] = len(take)
    short -= len(take)
    diff = rand_pick(rng, far[~is_same], q - n_same + short)
    counts["primary"] += len(diff) - short
    counts["diff_fill"] = short
    return np.concatenate([same, take, diff]).astype(np.int64), counts


def choi_select(kind, a, s0, s1, s2, tau, q):
    """Label-free. exp049 filter over all nodes (s1, s2 >= tau; self excluded); choi_far: lower half by s2 -> X-rank centre q
    (lower half < q: all passed); choi_strat: passed split into 5 s2 quantiles -> X-rank centre 2 each, short quantiles filled from the
    nearest quantile. Passed < q: exp049 relax (top q by min(s1, s2), X order)."""
    cand = np.delete(np.arange(len(s0)), a)
    c0, c1, c2 = s0[cand], s1[cand], s2[cand]
    xsort = lambda ids: ids[np.argsort(-c0[ids], kind="stable")]  # noqa: E731
    kept = np.flatnonzero((c1 >= tau) & (c2 >= tau))
    info = {"n_passed": int(len(kept)), "relax": 0, "half_fallback": 0}
    if len(kept) < q:
        info["relax"] = 1
        return cand[xsort(np.argsort(-np.minimum(c1, c2), kind="stable")[:q])], info
    by_s2 = xsort(kept)
    by_s2 = by_s2[np.argsort(-c2[by_s2], kind="stable")]  # s2 descending, ties by X rank
    if kind == "choi_far":
        low = by_s2[len(by_s2) - len(by_s2) // 2:]
        if len(low) < q:
            low, info["half_fallback"] = by_s2, 1
        return cand[centre(xsort(low), q)], info
    parts = [xsort(x) for x in np.array_split(by_s2, P.N_STRATA)]
    per = q // P.N_STRATA
    taken = [list(centre(x, min(per, len(x)))) for x in parts]
    for i in range(P.N_STRATA):
        for j in sorted(range(P.N_STRATA), key=lambda j: (abs(j - i), -j)):
            if len(taken[i]) >= per:
                break
            if j == i:
                continue
            free = np.asarray([x for x in parts[j] if x not in set(sum(taken, []))], dtype=np.int64)
            taken[i] += list(centre(free, min(per - len(taken[i]), len(free))))
            info["half_fallback"] = 1
    return cand[np.asarray(sum(taken, []), dtype=np.int64)], info


def o_grid(a, region, n_same, band_ids, s2rank, ranked, labels, rng, q):
    """HL07 §1 oracle grid. Regions: t1 (s2 rank 1-160), strat5 (T1-T5, 2 slots each), far50 (401-800). Same/diff drawn at random;
    shortfalls from the secondary pool (t1: T2, far50: s2 rank 161-400, strat5: nearest strata), then B5 GT-same, then region GT-diff."""
    strata = P.strata_of(s2rank)
    same_of = lambda cols: cols[labels[band_ids[cols]] == labels[a]]  # noqa: E731
    diff_of = lambda cols: cols[labels[band_ids[cols]] != labels[a]]  # noqa: E731
    counts = {"primary": 0, "secondary": 0, "b5": 0, "diff_fill": 0}
    used = set()

    def draw(pool, k):
        pool = np.asarray([c for c in pool if c not in used], dtype=np.int64)
        got = rand_pick(rng, pool, min(k, len(pool)))
        used.update(got.tolist())
        return list(got)

    if region in ("t1", "far50"):
        m = len(band_ids)
        main = strata[0] if region == "t1" else np.arange(m)[s2rank > m // 2]
        second = strata[1] if region == "t1" else np.arange(m)[(s2rank > m // P.N_STRATA) & (s2rank <= m // 2)]
        slots = [("same", main, second)] * n_same + [("diff", main, second)] * (q - n_same)
    else:
        kinds = np.zeros(q, dtype=bool)
        kinds[rng.choice(q, n_same, replace=False)] = True
        per = q // P.N_STRATA
        slots = []
        for i in range(P.N_STRATA):
            near = np.concatenate([strata[j] for j in sorted(range(P.N_STRATA), key=lambda j: (abs(j - i), -j)) if j != i])
            for k in range(per):
                slots.append(("same" if kinds[i * per + k] else "diff", strata[i], near))
    chosen, short_same, short_diff = [], 0, 0
    for kind, main, second in slots:
        f = same_of if kind == "same" else diff_of
        got = draw(f(main), 1)
        if got:
            counts["primary"] += 1
        else:
            got = draw(f(second), 1) if region != "strat5" else []
            if region == "strat5":  # nearest strata in order
                for j in sorted(range(P.N_STRATA), key=lambda j: (abs(j - int(np.flatnonzero([np.array_equal(main, s) for s in strata])[0])), -j)):
                    got = draw(f(strata[j]), 1)
                    if got:
                        break
            if got:
                counts["secondary"] += 1
            elif kind == "same":
                short_same += 1
            else:
                short_diff += 1
        chosen += [int(band_ids[c]) for c in got]
    if short_same:
        b5 = ranked[B5[0] - 1:B5[1]]
        b5 = b5[(labels[b5] == labels[a]) & ~np.isin(b5, chosen)]
        take = rand_pick(rng, b5, min(short_same, len(b5)))
        chosen += take.tolist()
        counts["b5"] = len(take)
        short_diff += short_same - len(take)
    if short_diff:
        region_cols = np.concatenate(strata) if region == "strat5" else (strata[0] if region == "t1" else np.arange(len(band_ids))[s2rank > len(band_ids) // 2])
        got = draw(diff_of(region_cols), short_diff)
        chosen += [int(band_ids[c]) for c in got]
        counts["diff_fill"] = len(got)
        short_diff -= len(got)
    if short_diff:  # not in the instruction: region has too few GT-diff (e.g. t1 of a dense same-class neighbourhood) -> any band GT-diff
        got = draw(diff_of(np.arange(len(band_ids))), short_diff)
        chosen += [int(band_ids[c]) for c in got]
        counts["band_diff_fill"] = len(got)
    return np.asarray(chosen, dtype=np.int64), counts


def build_pdd_queries(data, tasks, args, device, labels):
    """Replaces exp049.build_structural_queries; `labels` used only by oracle arms (and Describer, after selection)."""
    unique = np.unique(np.concatenate([np.asarray(t[0], dtype=np.int64).reshape(-1) for t in tasks]))
    band = tuple(int(x) for x in ARM_BAND.get(args.arm, args.x_band).split(","))
    args.band_used = list(band)
    teacher = args.teacher
    if args.arm == "encb_strat5_r2":
        found = [t for t in R2_TEACHERS if Path(t).exists()]
        if not found:
            raise FileNotFoundError(f"encb_strat5_r2: no teacher among {R2_TEACHERS}")
        teacher = args.teacher_used = found[0]
    judges = P.Judges(data, device, teacher, args.teacher_n, args.latent, args.ppr_alpha, args.ppr_iters)
    z0, z1, z2 = judges.z
    emb_desc = judges.emb_b if teacher == args.teacher else encoder_embeddings(data, args.teacher, args.latent, device)
    desc = Describer(data, z1, z2, labels, args.tau, emb_b=emb_desc)  # descriptors always use the fixed enc_b teacher
    names = [LABEL_FREE[args.arm][0]] if args.arm in LABEL_FREE and LABEL_FREE[args.arm][0] else []
    n = data.features.size(0)
    depth = n - 1 if args.arm in CHOI else B5[1]
    rng = np.random.default_rng(args.seed)
    rng_o = np.random.default_rng(args.seed + 1)
    rng_g = np.random.default_rng(args.seed + 7)
    if args.arm == "o_combswap":
        a_sym = sym_adjacency(data, device)
        emb_n = encoder_embeddings(data, args.teacher_n, args.latent, device)
    cache, details = {}, {}
    for begin in range(0, len(unique), SCORE_BATCH):
        ids = unique[begin:begin + SCORE_BATCH]
        ranked, band_ids, s2b, s2r, scores = judges.band(names, ids, band, depth=depth)
        if args.arm in CHOI:
            with torch.no_grad():
                full = [(z[torch.as_tensor(ids, device=device)] @ z.T).cpu().numpy() for z in (z0, z1, z2)]
        if args.arm in ("o_topsame", "near", "far"):
            with torch.no_grad():
                s2_full = judges.rows("a2x", ids).cpu().numpy()
        if args.arm == "o_combswap":
            _, s8 = b4_scores(data, device, ids, z1, z2, a_sym, emb_n, args)  # HL04 comb_top10 scores (teacher enc_n)
        for j, a in enumerate(ids):
            a = int(a)
            counts = {}
            if args.arm in LABEL_FREE:
                judge, rule = LABEL_FREE[args.arm]
                chosen = band_ids[j][P.select(rule, scores[judge][j] if judge else None, s2r[j], rng, args.q_query)]
            elif args.arm in CHOI:
                chosen, counts = choi_select(args.arm, a, full[0][j], full[1][j], full[2][j], args.tau, args.q_query)
            elif args.arm in GRID:
                _, _, region, pp = args.arm.split("_")
                chosen, counts = o_grid(a, region, int(round(args.q_query * int(pp[1:]) / 10)), band_ids[j], s2r[j], ranked[j], labels, rng_g, args.q_query)
            elif args.arm == "rand10":
                chosen = band_ids[j][label_free_queries("rand10", {"comb": np.zeros(band_ids.shape[1])}, rng, args.q_query)]
            elif args.arm == "o_topsame":
                chosen, counts = o_topsame(a, band_ids[j], s2b[j], ranked[j], s2_full[j], labels, args.q_query)
            elif args.arm == "o_combswap":
                picked = band_ids[j][label_free_queries("comb_top10", {k: v[j] for k, v in s8.items()}, None, args.q_query)]
                is_diff = labels[picked] != labels[a]
                pool = band_ids[j][(labels[band_ids[j]] != labels[a]) & ~np.isin(band_ids[j], picked)]
                chosen = picked.copy()
                chosen[is_diff] = rand_pick(rng_o, pool, int(is_diff.sum()))
                counts = {"swapped": int(is_diff.sum())}
            elif args.arm in ("o_far50_p03", "o_far50_p05"):
                chosen, counts = o_far50(a, 3 if args.arm.endswith("p03") else 5, band_ids[j], s2r[j], labels, rng_o, args.q_query)
            elif args.arm in ("g1_p04", "g1_p10"):
                chosen, st = purity_oracle_queries(a, 0.4 if args.arm == "g1_p04" else 1.0, ranked[j][:B5[1]], labels, args.q_query)
                counts = {"primary": st["n_same_b4"] + (args.q_query - st["n_same_target"]), "b5": st["n_same_b5"], "diff_fill": st["n_diff_fill"]}
            elif args.arm in ("near", "far"):
                chosen, counts = structural_oracle_queries(a, args.arm, ranked[j][:B5[1]], s2_full[j], labels, args.q_query)
            else:
                raise ValueError(args.arm)
            cache[a] = np.asarray(chosen, dtype=np.int64)
            xr = np.full(n, np.nan)
            xr[ranked[j]] = np.arange(1, ranked.shape[1] + 1)
            d = desc.describe(a, cache[a], None, band_ids[j], s2r[j], xrank=xr)  # post-hoc (labels)
            d.update(counts)
            d["mean_original_x_rank"] = d["x_rank"]
            d["fallback"] = bool(counts["relax"]) if "relax" in counts else bool(counts.get("primary", args.q_query) < args.q_query)
            details[a] = d
        if begin == 0 or begin + SCORE_BATCH >= len(unique):
            print(f"[hl011] scored {min(begin + SCORE_BATCH, len(unique))}/{len(unique)} anchors", flush=True)
    del judges, desc
    torch.cuda.empty_cache()
    modified = []
    for supports, ys, queries, yq in tasks:
        supports = np.asarray(supports, dtype=np.int64).reshape(-1)
        selected = np.concatenate([cache[int(x)] for x in supports])
        if len(selected) != len(np.asarray(queries).reshape(-1)):
            raise ValueError("Query count mismatch")
        modified.append((supports, ys, selected, yq))
    return modified, details


def drop_step(meta, id_spt, y_spt, id_query, y_query, q, k):
    """meta.ProtoNet.forward (k_shot 1) with the k highest-loss queries of each anchor's block of q left out of the mean loss.
    Losses for ranking are detached; returns (loss, train_acc, flat indices of dropped queries)."""
    enc = meta.network(meta.Data.features, adj=meta.Data.adj)
    x_spt, x_query = enc[id_spt], enc[id_query]
    prototypes = x_spt.view(meta.n, meta.k, x_spt.size(1)).mean(dim=1)
    output = F.log_softmax(-meta.euclidean_dist(x_query, prototypes), dim=1)
    label_new = torch.LongTensor([y_spt.tolist().index(i) for i in y_query.tolist()]).to(meta.device)
    nll = -output.gather(1, label_new[:, None]).squeeze(1)
    blocks = nll.detach().view(-1, q)
    drop = (blocks.topk(k, dim=1).indices + torch.arange(blocks.size(0), device=blocks.device)[:, None] * q).reshape(-1)
    keep = torch.ones_like(nll, dtype=torch.bool)
    keep[drop] = False
    loss = nll[keep].mean()
    meta.meta_optimizer.zero_grad()
    loss.backward()
    meta.meta_optimizer.step()
    return loss, meta.accuracy(output.cpu().detach(), label_new.cpu().detach()), drop.cpu().numpy()


def pdd_diagnostics(tasks, details, q):
    """Per anchor occurrence (as exp049.query_diagnostics counts them)."""
    rows = [details[int(s)] for t in tasks for s in np.asarray(t[0]).reshape(-1)]
    out = aggregate(rows, q)
    out["mean_purity"] = out["purity_pooled"]
    stage_keys = sorted({k for r in rows for k in r if k in ("primary", "b5", "diff_fill", "mid_same", "swapped", "adjacent", "across", "b4_centre", "secondary", "band_diff_fill")})
    for k in ("n_passed", "relax", "half_fallback"):
        if k in rows[0]:
            out[f"{k}_mean"] = float(np.mean([r[k] for r in rows]))
    for k in stage_keys:
        out[f"{k}_query_frac"] = float(np.sum([r.get(k, 0) for r in rows]) / (q * len(rows)))
    out["supplement_query_frac"] = 1.0 - out["primary_query_frac"] if "primary_query_frac" in out else 0.0
    return out


def main():
    args = parse_args()
    if args.k_shot != 1 or args.k_shot_test != 1 or args.n_way != 5:
        raise ValueError("Training remains 5-way 1-shot; evaluation separately tests 1 and 5 shot")
    if args.epoch <= 0 or args.eval_interval <= 0:
        raise ValueError("Invalid training schedule")
    args.policy = "uniform"
    args.method = "ours"
    args.setting, args.type, args.query_generation = "unsup", "feature", "NaQ"
    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    splits = load_splits(args)
    out = Path(args.output_root) / args.arm / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    if report_path.exists():
        raise FileExistsError(f"Existing result cannot be overwritten: {report_path}")

    start = time.perf_counter()
    print(f"[hl011] dataset={args.dataset} arm={args.arm} loading", flush=True)
    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    data.add_diffusion(args)
    config = [
        ("GCN", (data.features.shape[1], 2 * args.latent), True),
        ("GCN", (2 * args.latent, args.latent), True),
        ("linear", (args.latent, args.n_way), True)
    ]
    meta = ProtoNet(args, config, data).to(device)

    # anchors: hl003 `uniform` path unchanged (same python RNG state)
    labels = H.labels_np(data)
    pi_bchoi = read_column(args.node_table, "pi_bchoi", float) if args.node_table else np.zeros(len(labels))
    z2 = H.structural_scores(data, device)[2]
    original = [proto_task_generator(args, data) for _ in range(args.epoch)]
    stats = [{"k_eff": None, "fallback": False} for _ in original]
    anchor_diag = anchor_diagnostics(original, stats, labels, pi_bchoi, data, z2, args.n_way)
    if not args.node_table:
        anchor_diag["anchor_pi_bchoi_mean"] = None
    print(f"[hl011] anchor diagnostics: {anchor_diag}", flush=True)
    del z2
    torch.cuda.empty_cache()

    # ---- the only change vs hl003 uniform: query selection ----
    t_query = time.perf_counter()
    train_pool, details = build_pdd_queries(data, original, args, device, labels)
    if len(train_pool) != len(original):
        raise RuntimeError("Training episode count changed")
    diag = query_diagnostics(original, train_pool, data, details, args.q_query)
    qdiag = pdd_diagnostics(original, details, args.q_query)
    qdiag["query_build_seconds"] = round(time.perf_counter() - t_query, 2)
    print(f"[hl011] query diagnostics: {diag} {qdiag}", flush=True)
    del original
    # ------------------------------------------------------------

    best = {shot: {"accuracy": -1.0, "epoch": None} for shot in (1, 5)}
    history = []
    drop_n, drop_fp, post_n, post_fp = 0, 0, 0, 0
    for epoch, (id_spt, y_spt, id_query, y_query) in enumerate(train_pool, 1):
        args.k_shot_test = 1
        meta.network.train()
        if args.arm == "encb_strat5_drop2" and epoch > DROP_WARMUP:
            loss, train_acc, dropped = drop_step(meta, id_spt, y_spt.to(device), id_query, y_query.to(device), args.q_query, DROP_K)
            fp = labels[np.asarray(id_query)] != labels[np.repeat(np.asarray(id_spt).reshape(-1), args.q_query)]  # post-hoc (labels)
            drop_n, drop_fp, post_n, post_fp = drop_n + len(dropped), drop_fp + int(fp[dropped].sum()), post_n + len(fp), post_fp + int(fp.sum())
        else:
            loss, train_acc = meta(id_spt, y_spt.to(device), id_query, y_query.to(device))
        if epoch % args.eval_interval != 0 and epoch != args.epoch:
            continue
        row = {"epoch": epoch, "train_loss": float(loss.detach()), "train_acc": float(train_acc)}
        for shot in (1, 5):
            args.k_shot_test = shot
            valid = evaluate(meta, data, splits[shot][0], args)
            row[f"valid_{shot}shot"] = valid["accuracy"]
            if valid["accuracy"] > best[shot]["accuracy"]:
                best[shot] = {"accuracy": valid["accuracy"], "epoch": epoch}
                torch.save({"epoch": epoch, "model": meta.network.state_dict(), "config": vars(args).copy()}, out / f"best_{shot}shot.pt")
        history.append(row)
        print(f"[hl011] epoch={epoch} valid1={row['valid_1shot']:.4%} valid5={row['valid_5shot']:.4%}", flush=True)

    evaluations = {}
    for shot in (1, 5):
        state = torch.load(out / f"best_{shot}shot.pt", map_location="cpu", weights_only=True)
        meta.network.load_state_dict(state["model"])
        args.k_shot_test = shot
        test = evaluate(meta, data, splits[shot][1], args, with_episodes=True)
        evaluations[str(shot)] = {
            "best_epoch": best[shot]["epoch"],
            "valid_accuracy": best[shot]["accuracy"],
            "test_accuracy": test["accuracy"],
            "test_ci95_half_width": test["ci95_half_width"],
            "test_episode_accuracies": test["episode_accuracies"],
            "valid_episode_file": splits[shot][2],
            "test_episode_file": splits[shot][3],
            "checkpoint": str(out / f"best_{shot}shot.pt")
        }
        print(f"[hl011] {shot}-shot test={test['accuracy']:.4%} ± {test['ci95_half_width']:.4%}", flush=True)
    if args.arm == "encb_strat5_drop2":
        qdiag["drop_warmup"], qdiag["drop_k"] = DROP_WARMUP, DROP_K
        qdiag["dropped_queries"] = drop_n
        qdiag["dropped_fp_frac"] = drop_fp / max(drop_n, 1)
        qdiag["post_warmup_query_fp_frac"] = post_fp / max(post_n, 1)
    report = {
        "experiment": "hl011",
        "args": vars(args).copy(),
        "dataset": args.dataset,
        "method": "ours",
        "policy": "uniform",
        "arm": args.arm,
        "label_free": args.arm not in ORACLE,
        "tau": args.tau,
        "seed": args.seed,
        "diagnostics": diag,
        "query_diagnostics": qdiag,
        "anchor_diagnostics": anchor_diag,
        "evaluation": evaluations,
        "history": history,
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "max_cuda_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2) if device.type == "cuda" else None,
        "note": "hl011: hl003 uniform anchors with band queries; label-free arms select without labels, labels used only for oracle arms and post-hoc diagnostics."
    }
    tmp = out / "report.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    tmp.replace(report_path)
    print(f"[hl011] DONE {report_path}", flush=True)


if __name__ == "__main__":
    main()
