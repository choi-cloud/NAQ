"""HW04m: train-free purity of positive candidates (hop / PPR / degree / curvature).

    python -m hw.measure_positive --dataset <ds> [--seed 1212] [--episodes 2000]
    python -m hw.measure_positive --selftest
    python -m hw.measure_positive --energy --dataset <ds>      (HW03b: per-dimension Dirichlet energy)
    python -m hw.measure_positive --energy --selftest          (HW03b: E_raw = f^T L f check)
    python -m hw.measure_positive --overlap --dataset <ds>     (HW05: idea-A neighborhood overlap)
    python -m hw.measure_positive --exclusion --dataset <ds>   (HW07: cluster / far exclusion sets)

No training. Graph / features / labels come from hw.purity.build_pool (the hw/run.py path:
seed -> load_data -> ... -> train_pool), so the episode-conditioned rows use the seed-<S> pool.
measurement-only label use: every GT label read here feeds only these measurements.
Aggregation: macro = per-support-node ratio averaged over nodes whose set is non-empty;
micro = pooled over all (support, candidate) pairs. Supports = all nodes.
GPU (device 0 of the visible set): PPR power iteration (and the episode pool, as in hw.purity).
Outputs: results/measure/HW04m_<ds>.{json,md}, results/measure/HW04m_<ds>_nodes.npz,
cache/ppr/<ds>_a{015,050}_top100.pt, cache/curv/<ds>.pt.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from hw import structure as st

OUT_DIR = Path("results/measure")
PPR_DIR = Path("cache/ppr")
CURV_DIR = Path("cache/curv")
ALPHAS = (0.15, 0.5)
PPR_BINS = ((1, 5), (6, 10), (11, 20), (21, 50), (51, 100))
TOPM = (1, 5)
SPACES = ("raw", "a2x")
NODE_CHUNK = 1024


# ====================================================================== selftest (G-toy)
def _graph(n, edges):
    e = np.array(edges, dtype=np.int64).T
    return st.normalize_graph(e, n)


def selftest():
    checks = []

    def check(name, got, want, tol=1e-9):
        ok = abs(float(got) - float(want)) <= tol
        checks.append({"check": name, "got": float(got), "want": float(want), "ok": bool(ok)})

    def edge_curv(a, u, v):
        c = st.curvature(a)
        i = int(np.nonzero((c["u"] == min(u, v)) & (c["v"] == max(u, v)))[0][0])
        return c["t"][i], c["jl"][i], c["forman"][i], st.degrees(a)[u], st.degrees(a)[v]

    p5 = _graph(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    t, jl, fo, du, dv = edge_curv(p5, 1, 2)
    check("P5 middle edge t", t, 0); check("P5 middle edge d_u", du, 2); check("P5 middle edge d_v", dv, 2)
    check("P5 middle edge jl", jl, 0.0); check("P5 middle edge forman", fo, 0.0)
    # tree: u=0, v=1 each with two more leaves
    tree = _graph(6, [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)])
    t, jl, fo, du, dv = edge_curv(tree, 0, 1)
    check("3-3 tree edge t", t, 0); check("3-3 tree edge d_u", du, 3); check("3-3 tree edge d_v", dv, 3)
    check("3-3 tree edge jl", jl, -2.0 / 3.0); check("3-3 tree edge forman", fo, -2.0)
    k4 = _graph(4, [(i, j) for i in range(4) for j in range(i + 1, 4)])
    t, jl, fo, du, dv = edge_curv(k4, 0, 1)
    check("K4 edge t", t, 2); check("K4 edge d_u", du, 3); check("K4 edge d_v", dv, 3)
    check("K4 edge jl", jl, 2.0 / 3.0); check("K4 edge forman", fo, 4.0)
    # hop sets: P5 middle node (2): H1 = {1,3}, H2 = {0,4}, H3 = {}
    h1, h2, h3 = st.hop_sets_dense(p5, np.array([2]))
    check("P5 node 2 |H1|", h1.sum(), 2); check("P5 node 2 |H2|", h2.sum(), 2); check("P5 node 2 |H3|", h3.sum(), 0)
    check("P5 node 2 H1 == {1,3}", float(set(np.nonzero(h1[0])[0]) == {1, 3}), 1.0)
    check("P5 node 2 H2 == {0,4}", float(set(np.nonzero(h2[0])[0]) == {0, 4}), 1.0)
    # PPR: row sum ~ 1, self score >= alpha
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for gname, g in (("P5", p5), ("K4", k4), ("tree", tree)):
        for alpha in ALPHAS:
            r = st.ppr_topk(g, alpha, k=3, chunk=2, device=device)
            check(f"PPR {gname} a={alpha} max |row_sum - 1|", np.abs(r["row_sum"] - 1).max(), 0.0, tol=1e-5)
            check(f"PPR {gname} a={alpha} min(self - alpha) >= 0", float((r["self_score"] - alpha).min() >= -1e-7), 1.0)
    ok = all(c["ok"] for c in checks)
    return ok, checks


# ====================================================================== helpers
def summarize_row(per_node_same, per_node_size, mask=None):
    """macro / micro purity, coverage, median size (over non-empty nodes) from per-node counts."""
    same = np.asarray(per_node_same, dtype=np.float64)
    size = np.asarray(per_node_size, dtype=np.float64)
    if mask is not None:
        same, size = same[mask], size[mask]
    nonempty = size > 0
    n = size.size
    return {
        "macro": float(np.mean(same[nonempty] / size[nonempty])) if nonempty.any() else None,
        "micro": float(same.sum() / size.sum()) if size.sum() > 0 else None,
        "coverage": float(nonempty.mean()) if n else None,
        "size_median": float(np.median(size[nonempty])) if nonempty.any() else None,
        "nodes": int(n),
        "pairs": int(size.sum()),
    }


def edge_row(same_edge, mask):
    m = np.asarray(mask, dtype=bool)
    return {"edges": int(m.sum()), "purity": float(same_edge[m].mean()) if m.any() else None}


def topm_counts(score, valid, same, m):
    """score/valid/same [B, C]; choose top-m valid by score per row -> (same count, chosen count)."""
    s = np.where(valid, score, -np.inf)
    mm = min(m, s.shape[1])
    order = np.argpartition(-s, kth=mm - 1, axis=1)[:, :mm] if mm < s.shape[1] else np.argsort(-s, axis=1)[:, :mm]
    rows = np.arange(s.shape[0])[:, None]
    chosen_valid = valid[rows, order]
    return (same[rows, order] & chosen_valid).sum(1), chosen_valid.sum(1)


# ====================================================================== main measurement
def measure(dataset, seed, episodes):
    from hw.purity import build_pool  # noqa: import here so --selftest needs no data
    from hw.neighbors import select_neighbors_pick

    t0 = time.time()
    timings = {}
    pool = build_pool(dataset, seed, episodes)
    data, device = pool["data"], pool["device"]
    timings["build_pool"] = round(time.time() - t0, 2)
    n = pool["num_nodes"]
    y = pool["labels"].detach().cpu().numpy().astype(np.int64)  # measurement-only label use
    num_classes = int(y.max()) + 1
    a = st.normalize_graph(data.edge_index.detach().cpu().numpy(), n)
    deg = st.degrees(a)
    eu, ev = st.undirected_edges(a)
    same_edge = y[eu] == y[ev]  # measurement-only label use
    class_frac = np.bincount(y, minlength=num_classes) / n  # measurement-only label use
    chance = float((class_frac ** 2).sum())
    raw_edge = data.edge_index.detach().cpu().numpy()
    graph = {
        "nodes": int(n),
        "undirected_edges": int(eu.size),
        "isolated_nodes": int((deg == 0).sum()),
        "edge_index_columns": int(raw_edge.shape[1]),
        "edge_index_self_loops": int((raw_edge[0] == raw_edge[1]).sum()),
        "edge_homophily": float(same_edge.mean()),
        "chance": chance,
        "num_classes": num_classes,
        "degree_mean": float(deg.mean()),
        "degree_median": float(np.median(deg)),
        "degree_max": int(deg.max()),
    }

    # ---- features (CPU)
    torch.set_num_threads(32)
    x = pool["features"].detach().float().cpu()
    adj_cpu = data.adj.detach().cpu().coalesce()
    a2x = st.a2x_trainfree(adj_cpu, x)
    a2x_inline = torch.sparse.mm(adj_cpu, torch.sparse.mm(adj_cpu, x))  # literal hw/trainfree.py expression
    # independent Â = D^-1/2 (A + I) D^-1/2 from the normalized graph
    ai = (a + sp.eye(n, format="csr")).tocoo()
    dd = np.asarray(ai.sum(1)).ravel()
    dis = 1.0 / np.sqrt(dd)
    ref_adj = torch.sparse_coo_tensor(np.vstack([ai.row, ai.col]), (dis[ai.row] * ai.data * dis[ai.col]).astype(np.float32), (n, n)).coalesce()
    a2x_indep = torch.sparse.mm(ref_adj, torch.sparse.mm(ref_adj, x))
    feature_check = {
        "a2x_source": "hw/trainfree.py has no exposed function; formula moved to hw.structure.a2x_trainfree",
        "max_abs_diff_vs_trainfree_expression": float((a2x - a2x_inline).abs().max().item()),
        "max_abs_diff_vs_independent_adj_build": float((a2x - a2x_indep).abs().max().item()),
        "feature_dim": int(x.shape[1]),
    }
    feats = {"raw": st.l2_rows(x), "a2x": st.l2_rows(a2x)}
    del a2x_inline, a2x_indep
    timings["features"] = round(time.time() - t0, 2)

    # ---- PPR (GPU)
    nonisolated = np.nonzero(deg > 0)[0]
    ppr = {}
    ppr_meta = {}
    PPR_DIR.mkdir(parents=True, exist_ok=True)
    for alpha in ALPHAS:
        r = st.ppr_topk(a, alpha, k=100, chunk=2048, device=str(device), sources=nonisolated)
        full_idx = np.full((n, r["idx"].shape[1]), -1, dtype=np.int64)
        full_score = np.zeros((n, r["idx"].shape[1]), dtype=np.float32)
        full_idx[nonisolated] = r["idx"]
        full_score[nonisolated] = r["score"]
        ppr[alpha] = full_idx
        tag = f"a{int(round(alpha * 100)):03d}"
        torch.save({"dataset": dataset, "alpha": alpha, "idx": torch.from_numpy(full_idx), "score": torch.from_numpy(full_score),
                    "isolated_excluded": torch.from_numpy(np.nonzero(deg == 0)[0]), "note": "self excluded; -1 = none"},
                   PPR_DIR / f"{dataset}_{tag}_top100.pt")
        ppr_meta[tag] = {
            "iters_max": int(max(r["iters"])), "iters_min": int(min(r["iters"])),
            "final_max_delta_max": float(max(r["max_delta"])),
            "row_sum_max_abs_dev": float(np.abs(r["row_sum"] - 1).max()),
            "self_score_min": float(r["self_score"].min()),
            "sources": int(nonisolated.size), "isolated_skipped": int((deg == 0).sum()),
            "path": str(PPR_DIR / f"{dataset}_{tag}_top100.pt"),
        }
    timings["ppr"] = round(time.time() - t0, 2)

    # ---- per-node counts (CPU, node chunks)
    rows_same, rows_size = {}, {}

    def put(key, idx, same_cnt, size_cnt):
        if key not in rows_same:
            rows_same[key] = np.zeros(n, dtype=np.int64)
            rows_size[key] = np.zeros(n, dtype=np.int64)
        rows_same[key][idx] = same_cnt
        rows_size[key][idx] = size_cnt

    hop_comp = {}  # (alpha tag, bin) -> counts of hop1 / hop2 / hop>=3
    feat_np = {s: feats[s].numpy() for s in SPACES}
    for start in range(0, n, NODE_CHUNK):
        idx = np.arange(start, min(n, start + NODE_CHUNK))
        h = st.hop_sets_dense(a, idx)
        same_lab = y[None, :] == y[idx][:, None]  # measurement-only label use
        cos = {s: (feats[s][idx] @ feats[s].T).numpy() for s in SPACES}
        for k, hk in enumerate(h, 1):
            put(f"hop{k}", idx, (hk & same_lab).sum(1), hk.sum(1))
            if k <= 3:
                for s in SPACES:
                    for m in TOPM:
                        sc, sz = topm_counts(cos[s], hk, same_lab, m)
                        put(f"hop{k}_top{m}_{s}", idx, sc, sz)
        rows_b = np.arange(idx.size)[:, None]
        for alpha in ALPHAS:
            tag = f"a{int(round(alpha * 100)):03d}"
            cand = ppr[alpha][idx]  # [B, 100]
            valid = cand >= 0
            cc = np.where(valid, cand, 0)
            same_c = same_lab[rows_b, cc] & valid
            in1 = h[0][rows_b, cc] & valid
            in2 = h[1][rows_b, cc] & valid
            for lo, hi in PPR_BINS:
                sl = slice(lo - 1, hi)
                key = f"ppr{tag[1:]}_{lo}-{hi}"
                put(key, idx, same_c[:, sl].sum(1), valid[:, sl].sum(1))
                c = hop_comp.setdefault((tag, f"{lo}-{hi}"), np.zeros(4, dtype=np.int64))
                c += [in1[:, sl].sum(), in2[:, sl].sum(), (valid[:, sl] & ~in1[:, sl] & ~in2[:, sl]).sum(), valid[:, sl].sum()]
            put(f"ppr{tag[1:]}_1-10", idx, same_c[:, :10].sum(1), valid[:, :10].sum(1))
            if alpha == 0.15:
                c20, v20, s20 = cc[:, :20], valid[:, :20], same_c[:, :20]
                for s in SPACES:
                    sc20 = cos[s][rows_b, c20]
                    for m in TOPM:
                        a_same, a_size = topm_counts(sc20, v20, s20, m)
                        put(f"ppr015_top20_top{m}_{s}", idx, a_same, a_size)
    timings["node_loop"] = round(time.time() - t0, 2)

    # ---- M1 / M2 tables
    m1 = {key: summarize_row(rows_same[key], rows_size[key]) for key in rows_same if key.startswith("hop")}
    m2 = {key: summarize_row(rows_same[key], rows_size[key]) for key in rows_same if key.startswith("ppr")}
    for (tag, b), c in hop_comp.items():
        key = f"ppr{tag[1:]}_{b}"
        m2[key]["hop_composition"] = {"hop1": float(c[0] / c[3]) if c[3] else None, "hop2": float(c[1] / c[3]) if c[3] else None, "hop3plus": float(c[2] / c[3]) if c[3] else None}
    g_consist = {
        "hop1_micro": m1["hop1"]["micro"],
        "edge_homophily_both_directions": float(np.concatenate([same_edge, same_edge]).mean()),
    }
    g_consist["abs_diff"] = abs(g_consist["hop1_micro"] - g_consist["edge_homophily_both_directions"])
    g_consist["pass"] = bool(g_consist["abs_diff"] < 1e-12)

    # ---- M3 degree quintiles
    dbin, dedges = st.quantile_bins(deg, 5)
    m3 = {"edges": [float(e) for e in dedges], "bins": []}
    for b in range(5):
        mask = dbin == b
        row = {"bin": f"Q{b + 1}", "nodes": int(mask.sum()),
               "degree_min": int(deg[mask].min()) if mask.any() else None, "degree_max": int(deg[mask].max()) if mask.any() else None}
        for key in ("hop1", "hop2", "hop2_top1_a2x", "ppr015_1-10"):
            row[key] = summarize_row(rows_same[key], rows_size[key], mask)
        m3["bins"].append(row)
    # episode-conditioned rows (pool of seed, raw selector, pick max)
    ep = {k: [] for k in ("support", "query_eq_support", "nbr_eq_support", "nbr_eq_query", "fallback")}
    labels_t = pool["labels"]
    sel_raw = pool["selection"]["raw"]
    with torch.no_grad():
        for id_spt, y_spt, id_query, y_query in pool["train_pool"]:
            s_ids = torch.as_tensor(id_spt, dtype=torch.long, device=device).flatten()
            q_ids = torch.as_tensor(id_query, dtype=torch.long, device=device).flatten()
            sp_ = torch.as_tensor(y_spt, dtype=torch.long, device=device).flatten()
            qp_ = torch.as_tensor(y_query, dtype=torch.long, device=device).flatten()
            out = {}
            select_neighbors_pick(sel_raw, s_ids, q_ids, pool["neighbor_lookup"], 0, "max", None, out)
            qi, si = (qp_[:, None] == sp_[None, :]).nonzero(as_tuple=True)
            gq, gs = labels_t[q_ids[qi]], labels_t[s_ids[si]]  # measurement-only label use
            gn = labels_t[out["selected"][qi, si]]  # measurement-only label use
            ep["support"].append(s_ids[si].cpu().numpy())
            ep["query_eq_support"].append((gq == gs).cpu().numpy())
            ep["nbr_eq_support"].append((gn == gs).cpu().numpy())
            ep["nbr_eq_query"].append((gn == gq).cpu().numpy())
            ep["fallback"].append((~out["has_neighbor"][qi, si]).cpu().numpy())
    ep = {k: np.concatenate(v) for k, v in ep.items()}
    ep_bin = dbin[ep["support"]]
    m3["episode_rows"] = []
    for b in range(5):
        mask = ep_bin == b
        m3["episode_rows"].append({"bin": f"Q{b + 1}", "pairs": int(mask.sum()),
                                   **{k: (float(ep[k][mask].mean()) if mask.any() else None) for k in ("query_eq_support", "nbr_eq_support", "nbr_eq_query", "fallback")}})
    m3["episode_all"] = {"pairs": int(ep_bin.size), **{k: float(ep[k].mean()) for k in ("query_eq_support", "nbr_eq_support", "nbr_eq_query", "fallback")}}
    timings["degree_episode"] = round(time.time() - t0, 2)

    # ---- M4 curvature
    cv = st.curvature(a)
    assert np.array_equal(cv["u"], eu) and np.array_equal(cv["v"], ev)
    CURV_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"dataset": dataset, "edge_index": torch.from_numpy(np.vstack([eu, ev])), "t": torch.from_numpy(cv["t"]),
                "forman": torch.from_numpy(cv["forman"]), "jl": torch.from_numpy(cv["jl"]), "note": "undirected, u < v"}, CURV_DIR / f"{dataset}.pt")
    jl, fo, t = cv["jl"], cv["forman"], cv["t"]
    jl_sign = np.where(np.abs(jl) < st.ZERO_TOL, 0, np.sign(jl))
    m4 = {"cache": str(CURV_DIR / f"{dataset}.pt"),
          "sign": {"jl<0": edge_row(same_edge, jl_sign < 0), "jl=0": edge_row(same_edge, jl_sign == 0), "jl>0": edge_row(same_edge, jl_sign > 0)},
          "t": {"t=0": edge_row(same_edge, t == 0), "t>=1": edge_row(same_edge, t >= 1)}}
    for name, vals, mask in (("jl_quintile", jl, None), ("forman_quintile", fo, None), ("jl_quintile_t>=1", jl, t >= 1)):
        sel = np.ones(jl.size, dtype=bool) if mask is None else mask
        bins, edges = st.quantile_bins(vals[sel], 5)
        m4[name] = {"edges": [float(e) for e in edges],
                    "bins": [{"bin": f"Q{b + 1}", "min": float(vals[sel][bins == b].min()) if (bins == b).any() else None,
                              "max": float(vals[sel][bins == b].max()) if (bins == b).any() else None,
                              **edge_row(same_edge[sel], bins == b)} for b in range(5)]}
    # node-centric: neighbors via jl > 0 edges vs jl <= 0 edges
    yoh = sp.csr_matrix((np.ones(n), (np.arange(n), y)), shape=(n, num_classes))  # measurement-only label use
    node_rows = {}
    for name, emask in (("nbr_jl>0", jl_sign > 0), ("nbr_jl<=0", jl_sign <= 0)):
        uu, vv = eu[emask], ev[emask]
        sub = sp.coo_matrix((np.ones(2 * uu.size), (np.concatenate([uu, vv]), np.concatenate([vv, uu]))), shape=(n, n)).tocsr()
        cnt = sub @ yoh
        same_cnt = np.asarray(cnt[np.arange(n), y]).ravel()
        size_cnt = np.asarray(sub.sum(1)).ravel()
        node_rows[name] = summarize_row(same_cnt, size_cnt)
        rows_same[name], rows_size[name] = same_cnt.astype(np.int64), size_cnt.astype(np.int64)
    m4["node_centric"] = node_rows
    # per-node curvature summary
    inc_sum = np.bincount(eu, weights=jl, minlength=n) + np.bincount(ev, weights=jl, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        node_jl_mean = np.where(deg > 0, inc_sum / np.maximum(deg, 1), np.nan)
    timings["curvature"] = round(time.time() - t0, 2)

    result = {
        "dataset": dataset, "seed": seed, "episodes": episodes,
        "graph": graph, "feature_check": feature_check, "ppr": ppr_meta,
        "M1": m1, "M2": m2, "M3": m3, "M4": m4, "G_consist": g_consist,
        "timings_cumulative_s": timings,
        "note": "no training; measurement-only label use; macro primary, micro secondary; size_median over non-empty nodes",
    }
    nodes = {"degree": deg, "label": y, "degree_bin": dbin, "node_jl_mean": node_jl_mean.astype(np.float32)}
    for key in ("hop1", "hop2", "hop3", "hop2_top1_a2x", "ppr015_1-10", "nbr_jl>0", "nbr_jl<=0"):
        safe = key.replace("-", "_").replace(">", "gt").replace("<=", "le")
        nodes[f"{safe}_same"] = rows_same[key].astype(np.int32)
        nodes[f"{safe}_size"] = rows_size[key].astype(np.int32)
    return result, nodes


# ====================================================================== markdown
def _f(v, d=4):
    return "—" if v is None else f"{v:.{d}f}"


def _pr(r):
    """purity cell: macro (micro)."""
    return f"{_f(r['macro'])} ({_f(r['micro'])})"


def to_markdown(r):
    g = r["graph"]
    L = [f"# HW04m — {r['dataset']} (train-free, seed {r['seed']}, {r['episodes']} episodes)", "",
         f"chance = Σ p_c² = **{g['chance']:.4f}** · edge homophily = {g['edge_homophily']:.4f} · 순도 셀 = macro (micro)", "",
         "## 그래프", "", "| 노드 | 무향 edge | 고립 노드 | 클래스 | 차수 평균 / 중앙값 / 최대 |", "|---|---|---|---|---|",
         f"| {g['nodes']} | {g['undirected_edges']} | {g['isolated_nodes']} | {g['num_classes']} | {g['degree_mean']:.2f} / {g['degree_median']:.0f} / {g['degree_max']} |", "",
         f"Â²X 확인: trainfree 식 대비 max |Δ| {r['feature_check']['max_abs_diff_vs_trainfree_expression']:.2e}, 독립 Â 구성 대비 {r['feature_check']['max_abs_diff_vs_independent_adj_build']:.2e}", "",
         "## M1 hop", "", f"chance {g['chance']:.4f}", "", "| 행 | 순도 macro (micro) | coverage | 크기 중앙값 |", "|---|---|---|---|"]
    order = [f"hop{k}" for k in (1, 2, 3)] + [f"hop{k}_top{m}_{s}" for k in (1, 2, 3) for s in SPACES for m in TOPM]
    for key in order:
        x = r["M1"][key]
        L.append(f"| {key} | {_pr(x)} | {_f(x['coverage'])} | {_f(x['size_median'], 0)} |")
    gc = r["G_consist"]
    L += ["", f"G-consist: hop1 micro {gc['hop1_micro']:.6f} vs edge homophily {gc['edge_homophily_both_directions']:.6f}, |Δ| {gc['abs_diff']:.1e} → {'통과' if gc['pass'] else '실패'}", "",
          "## M2 PPR", "", f"chance {g['chance']:.4f}", "", "| 행 | 순도 macro (micro) | coverage | hop1 / hop2 / hop≥3 비율 |", "|---|---|---|---|"]
    for alpha in ("015", "050"):
        for lo, hi in PPR_BINS:
            x = r["M2"][f"ppr{alpha}_{lo}-{hi}"]
            hc = x["hop_composition"]
            L.append(f"| ppr{alpha}_{lo}-{hi} | {_pr(x)} | {_f(x['coverage'])} | {_f(hc['hop1'], 3)} / {_f(hc['hop2'], 3)} / {_f(hc['hop3plus'], 3)} |")
        x = r["M2"][f"ppr{alpha}_1-10"]
        L.append(f"| ppr{alpha}_1-10 | {_pr(x)} | {_f(x['coverage'])} | |")
    for s in SPACES:
        for m in TOPM:
            x = r["M2"][f"ppr015_top20_top{m}_{s}"]
            L.append(f"| ppr015_top20_top{m}_{s} | {_pr(x)} | {_f(x['coverage'])} | |")
    m3 = r["M3"]
    L += ["", "## M3 차수 5분위", "", f"chance {g['chance']:.4f} · 분위 경계 {m3['edges']}", "",
          "| bin | 노드 | 차수 범위 | hop1 | hop2 | hop2_top1_a2x | ppr015_1-10 | coverage hop1 / hop2 / ppr |", "|---|---|---|---|---|---|---|---|"]
    for b in m3["bins"]:
        L.append(f"| {b['bin']} | {b['nodes']} | {b['degree_min']}–{b['degree_max']} | {_pr(b['hop1'])} | {_pr(b['hop2'])} | {_pr(b['hop2_top1_a2x'])} | {_pr(b['ppr015_1-10'])} | {_f(b['hop1']['coverage'], 3)} / {_f(b['hop2']['coverage'], 3)} / {_f(b['ppr015_1-10']['coverage'], 3)} |")
    L += ["", "에피소드 조건 행 (seed 풀, raw·max 선택, 자기 클래스 (query, support) 쌍, support 차수 bin)", "",
          "| bin | 쌍 | query_eq_support | nbr_eq_support | nbr_eq_query | fallback_frac |", "|---|---|---|---|---|---|"]
    for b in m3["episode_rows"] + [{"bin": "전체", **m3["episode_all"]}]:
        L.append(f"| {b['bin']} | {b['pairs']} | {_f(b['query_eq_support'])} | {_f(b['nbr_eq_support'])} | {_f(b['nbr_eq_query'])} | {_f(b['fallback'])} |")
    m4 = r["M4"]
    L += ["", "## M4 곡률 (edge 순도 = y_u == y_v)", "", f"chance {g['chance']:.4f}", "", "| bin | edge 수 | 순도 |", "|---|---|---|"]
    for k, v in list(m4["sign"].items()) + list(m4["t"].items()):
        L.append(f"| {k} | {v['edges']} | {_f(v['purity'])} |")
    for name in ("jl_quintile", "forman_quintile", "jl_quintile_t>=1"):
        L += ["", f"{name} (경계 {[round(e, 4) for e in m4[name]['edges']]})", "", "| bin | 범위 | edge 수 | 순도 |", "|---|---|---|---|"]
        for b in m4[name]["bins"]:
            L.append(f"| {b['bin']} | {_f(b['min'])} – {_f(b['max'])} | {b['edges']} | {_f(b['purity'])} |")
    L += ["", "노드 중심 (support의 1-hop을 곡률 부호로 나눔)", "", "| 행 | 순도 macro (micro) | coverage | 크기 중앙값 |", "|---|---|---|---|"]
    for k, v in m4["node_centric"].items():
        L.append(f"| {k} | {_pr(v)} | {_f(v['coverage'])} | {_f(v['size_median'], 0)} |")
    return "\n".join(L) + "\n"


# ====================================================================== HW03b: per-dimension Dirichlet energy
ENERGY_TOPK = (16, 64, 512)


def energy_selftest():
    """E_raw (edge sum via incidence) vs f^T L f on small graphs with random columns (tol 1e-9)."""
    rng = np.random.default_rng(0)
    checks = []
    graphs = {
        "path5": _graph(5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
        "loops_dups": st.normalize_graph(np.array([[0, 1, 1, 2, 2, 3, 3], [1, 0, 2, 2, 3, 0, 0]]), 5),
        "random30": st.normalize_graph(rng.integers(0, 30, size=(2, 80)), 30),
    }
    for name, a in graphs.items():
        x = rng.integers(0, 4, size=(a.shape[0], 6)).astype(np.float64)
        x[:, 0] = 0.0
        got = st.dirichlet_energy_per_dim(a, x)
        for m in range(x.shape[1]):
            want = st.laplacian_quadratic(a, x[:, m])
            checks.append({"check": f"{name} dim{m}", "got": float(got[m]), "want": want, "ok": bool(abs(got[m] - want) <= 1e-9)})
    return all(c["ok"] for c in checks), checks


def _quantiles(v):
    v = np.asarray(v, dtype=np.float64)
    q = np.quantile(v, [0, 0.25, 0.5, 0.75, 1.0])
    return {"min": float(q[0]), "q25": float(q[1]), "median": float(q[2]), "q75": float(q[3]), "max": float(q[4]), "n": int(v.size)}


def _topk_overlap(a, b, k):
    ta = set(np.argsort(-a, kind="stable")[:k].tolist())
    tb = set(np.argsort(-b, kind="stable")[:k].tolist())
    return len(ta & tb) / k


def energy_measure(dataset, seed):
    """X exactly as hw/run.py reads it: load_data -> set_device -> add_diffusion -> data.features (seed 1212)."""
    from scipy.stats import spearmanr

    from data import load_data
    from hw.config import build_args, load_config
    from utils import seed_everything

    args = build_args(load_config("naq_base"), dataset)
    seed_everything(seed)
    data = load_data(args)
    data.add_diffusion(args)
    x = data.features.detach().cpu().numpy().astype(np.float64)
    n, dim = x.shape
    a = st.normalize_graph(data.edge_index.detach().cpu().numpy(), n)
    values = np.unique(x)
    row_l2 = np.sqrt((x ** 2).sum(1))
    feature_info = {
        "shape": [int(n), int(dim)],
        "dtype_in_training": str(data.features.dtype),
        "distinct_values": int(values.size),
        "binary": bool(np.isin(values, [0.0, 1.0]).all()),
        "integer_valued": bool(np.all(values == np.round(values))),
        "min": float(x.min()),
        "max": float(x.max()),
        "row_sum": _quantiles(x.sum(1)),
        "row_l2_norm": _quantiles(row_l2),
        "row_normalized_l1": bool(np.allclose(x.sum(1), 1.0)),
        "row_normalized_l2": bool(np.allclose(row_l2, 1.0)),
        "density": float((x != 0).mean()),
    }
    e_raw = st.dirichlet_energy_per_dim(a, x)
    norm2 = (x ** 2).sum(0)
    nnz = (x != 0).sum(0).astype(np.float64)
    nonzero = norm2 > 0
    e_rq = np.full(dim, np.nan)
    e_rq[nonzero] = e_raw[nonzero] / norm2[nonzero]
    er, rq, nz = e_raw[nonzero], e_rq[nonzero], nnz[nonzero]
    sp_ = lambda p, q: float(spearmanr(p, q).correlation)  # noqa: E731
    result = {
        "dataset": dataset,
        "seed": seed,
        "graph": {"nodes": int(n), "undirected_edges": int(a.nnz // 2), "source": "hw.structure.normalize_graph(data.edge_index)"},
        "features": feature_info,
        "definitions": {
            "E_raw": "1/2 sum_(i,j) A_ij (x_im - x_jm)^2 = sum over undirected edges (SPGCL eq. 13)",
            "E_rq": "E_raw / ||x_:,m||^2 (dims with ||x_:,m|| = 0 excluded)",
            "nnz": "number of nodes with x_im != 0",
            "statistics_over": "dims with ||x_:,m|| > 0",
        },
        "dims_total": int(dim),
        "dims_zero_norm": int((~nonzero).sum()),
        "dims_used": int(nonzero.sum()),
        "dims_E_raw_zero": int((e_raw[nonzero] == 0).sum()),
        "summary": {"E_raw": _quantiles(er), "E_rq": _quantiles(rq), "nnz": _quantiles(nz)},
        "spearman": {"E_raw_vs_nnz": sp_(er, nz), "E_raw_vs_E_rq": sp_(er, rq), "E_rq_vs_nnz": sp_(rq, nz)},
        "topk_overlap": {
            str(k): {"E_raw_vs_nnz": _topk_overlap(er, nz, k), "E_raw_vs_E_rq": _topk_overlap(er, rq, k)} for k in ENERGY_TOPK
        },
        "topk_tie_rule": "descending, stable argsort (ties broken by dimension index)",
    }
    return result


def energy_markdown(r):
    f = r["features"]
    L = [f"# HW03b per-dimension Dirichlet energy — {r['dataset']}", "",
         f"X: {f['shape'][0]} × {f['shape'][1]}, {f['dtype_in_training']}, binary {f['binary']}, integer {f['integer_valued']}, "
         f"range {f['min']:g}–{f['max']:g}, distinct values {f['distinct_values']}, density {f['density']:.5f}, "
         f"row L1-normalized {f['row_normalized_l1']}, row L2-normalized {f['row_normalized_l2']}",
         f"row sum min/median/max {f['row_sum']['min']:g} / {f['row_sum']['median']:g} / {f['row_sum']['max']:g}",
         f"graph: {r['graph']['nodes']} nodes, {r['graph']['undirected_edges']} undirected edges ({r['graph']['source']})",
         f"dims: total {r['dims_total']}, zero norm {r['dims_zero_norm']} (excluded), used {r['dims_used']}, E_raw = 0 among used {r['dims_E_raw_zero']}", "",
         "| 값 | min | q25 | median | q75 | max |", "|---|---|---|---|---|---|"]
    for k, v in r["summary"].items():
        L.append(f"| {k} | {v['min']:.6g} | {v['q25']:.6g} | {v['median']:.6g} | {v['q75']:.6g} | {v['max']:.6g} |")
    L += ["", "| Spearman | ρ |", "|---|---|"] + [f"| {k} | {v:.4f} |" for k, v in r["spearman"].items()]
    L += ["", "| K | top-K overlap E_raw ∩ nnz | top-K overlap E_raw ∩ E_rq |", "|---|---|---|"]
    L += [f"| {k} | {v['E_raw_vs_nnz']:.4f} | {v['E_raw_vs_E_rq']:.4f} |" for k, v in r["topk_overlap"].items()]
    return "\n".join(L) + "\n"


def energy_main(cli):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if cli.selftest:
        ok, checks = energy_selftest()
        (OUT_DIR / "HW03b_energy_selftest.json").write_text(json.dumps({"pass": ok, "checks": checks}, indent=2), encoding="utf-8")
        print(f"energy selftest {'PASS' if ok else 'FAIL'} ({sum(c['ok'] for c in checks)}/{len(checks)}), max |diff| "
              f"{max(abs(c['got'] - c['want']) for c in checks):.3g}")
        return 0 if ok else 1
    if not cli.dataset:
        raise SystemExit("--dataset required")
    started = time.time()
    r = energy_measure(cli.dataset, cli.seed)
    r["elapsed_seconds"] = round(time.time() - started, 2)
    (OUT_DIR / f"HW03b_energy_{cli.dataset}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    (OUT_DIR / f"HW03b_energy_{cli.dataset}.md").write_text(energy_markdown(r), encoding="utf-8")
    print(energy_markdown(r))
    return 0


# ====================================================================== HW05: idea-A neighborhood overlap
OVERLAP_THETAS = (0.1, 0.2, 0.3, 0.5)
OVERLAP_PPR_TOP = 20
OVERLAP_PAIRS = 100_000
OVERLAP_SEED = 20261108


def _set_jaccard_pairs(member):
    """member: binary CSR [N, K-space] (row i = set S_i). Returns (i, j, J) for i < j with |S_i ∩ S_j| > 0."""
    member = member.tocsr().astype(np.float64)
    inter = sp.triu(member @ member.T, k=1).tocoo()
    size = np.asarray(member.sum(axis=1)).ravel()
    c = inter.data
    return inter.row.astype(np.int64), inter.col.astype(np.int64), c / (size[inter.row] + size[inter.col] - c)


def _pair_jaccard(member, i, j, chunk=20000):
    member = member.tocsr().astype(np.float64)
    size = np.asarray(member.sum(axis=1)).ravel()
    out = np.zeros(i.size)
    for s0 in range(0, i.size, chunk):
        e = slice(s0, s0 + chunk)
        c = np.asarray(member[i[e]].multiply(member[j[e]]).sum(axis=1)).ravel()
        den = size[i[e]] + size[j[e]] - c
        out[e] = np.where(den > 0, c / np.where(den > 0, den, 1), 0.0)
    return out


def _theta_rows(n, pi, pj, jac, same):
    rows = {}
    for theta in OVERLAP_THETAS:
        keep = jac >= theta
        partners = np.bincount(pi[keep], minlength=n) + np.bincount(pj[keep], minlength=n)
        covered = partners > 0
        rows[str(theta)] = {
            "pairs": int(keep.sum()),
            "same_class": float(same[keep].mean()) if keep.any() else None,
            "coverage": float(covered.mean()),
            "partners_median_all_nodes": float(np.median(partners)),
            "partners_median_covered_nodes": float(np.median(partners[covered])) if covered.any() else None,
        }
    return rows


def _dist(v):
    v = np.asarray(v, dtype=np.float64)
    return {"zero_frac": float((v == 0).mean()), "median": float(np.median(v)), "q90": float(np.quantile(v, 0.9)), "mean": float(v.mean()), "n": int(v.size)}


def overlap_measure(dataset, seed):
    from data import load_data
    from hw.config import build_args, load_config
    from utils import seed_everything

    args = build_args(load_config("naq_base"), dataset)
    seed_everything(seed)
    data = load_data(args)
    n = data.features.shape[0]
    y = np.asarray(data.labels.detach().cpu().numpy()).astype(np.int64)  # measurement-only label use
    num_classes = int(y.max()) + 1
    chance = float(((np.bincount(y, minlength=num_classes) / n) ** 2).sum())  # measurement-only label use
    a = st.normalize_graph(data.edge_index.detach().cpu().numpy(), n)
    gi, gj, gjac = _set_jaccard_pairs(a)  # N(i) sets: c_ij = (A A)_ij, J = c / (d_i + d_j - c)
    gsame = y[gi] == y[gj]  # measurement-only label use
    ppr = torch.load(PPR_DIR / f"{dataset}_a015_top100.pt", weights_only=False)
    idx = ppr["idx"][:, :OVERLAP_PPR_TOP].numpy()
    rows_, cols_ = np.nonzero(idx >= 0)
    pm = sp.csr_matrix((np.ones(rows_.size), (rows_, idx[rows_, cols_])), shape=(n, n))
    pi, pj, pjac = _set_jaccard_pairs(pm)
    psame = y[pi] == y[pj]  # measurement-only label use

    rng = np.random.default_rng(OVERLAP_SEED)
    by_class = [np.nonzero(y == c)[0] for c in range(num_classes)]  # measurement-only label use
    sizes = np.array([len(b) for b in by_class], dtype=np.float64)
    w = sizes * (sizes - 1)
    cls = rng.choice(num_classes, size=OVERLAP_PAIRS, p=w / w.sum())  # same-class pair uniform over all same-class pairs
    si = np.empty(OVERLAP_PAIRS, dtype=np.int64)
    sj = np.empty(OVERLAP_PAIRS, dtype=np.int64)
    for c in np.unique(cls):
        k = np.nonzero(cls == c)[0]
        members = by_class[c]
        a1 = rng.integers(0, members.size, size=k.size)
        a2 = rng.integers(0, members.size - 1, size=k.size)
        a2 = a2 + (a2 >= a1)
        si[k], sj[k] = members[a1], members[a2]
    di = np.empty(0, dtype=np.int64)
    dj = np.empty(0, dtype=np.int64)
    while di.size < OVERLAP_PAIRS:
        u = rng.integers(0, n, size=OVERLAP_PAIRS)
        v = rng.integers(0, n, size=OVERLAP_PAIRS)
        ok = y[u] != y[v]  # measurement-only label use
        di, dj = np.concatenate([di, u[ok]]), np.concatenate([dj, v[ok]])
    di, dj = di[:OVERLAP_PAIRS], dj[:OVERLAP_PAIRS]
    deg = st.degrees(a)
    return {
        "dataset": dataset,
        "seed": seed,
        "graph": {"nodes": int(n), "undirected_edges": int(a.nnz // 2), "degree_median": float(np.median(deg)), "source": "hw.structure.normalize_graph(data.edge_index)"},
        "chance": chance,
        "definitions": {
            "graph_jaccard": "J_ij = c_ij / (d_i + d_j - c_ij), c_ij = (A A)_ij = |N(i) ∩ N(j)|; pairs i < j with c_ij > 0 (others J = 0)",
            "ppr_jaccard": f"Jaccard of the PPR (alpha 0.15) top-{OVERLAP_PPR_TOP} sets (cache/ppr/<ds>_a015_top100.pt, self excluded); pairs with overlap > 0",
            "same_class": "share of the pairs with J >= theta whose GT classes match (merge precision)",
            "coverage": "share of all nodes with at least one partner at J >= theta",
            "partners_median": "median number of such partners per node (all nodes incl. 0; and covered nodes only)",
            "pair_distributions": f"{OVERLAP_PAIRS} uniform same-class pairs (i != j) and {OVERLAP_PAIRS} uniform different-class pairs, numpy Generator seed {OVERLAP_SEED}",
        },
        "graph_pairs_nonzero": int(gi.size),
        "ppr_pairs_nonzero": int(pi.size),
        "graph_theta": _theta_rows(n, gi, gj, gjac, gsame),
        "ppr_theta": _theta_rows(n, pi, pj, pjac, psame),
        "pair_distributions": {
            "graph": {"same_class": _dist(_pair_jaccard(a, si, sj)), "different_class": _dist(_pair_jaccard(a, di, dj))},
            "ppr": {"same_class": _dist(_pair_jaccard(pm, si, sj)), "different_class": _dist(_pair_jaccard(pm, di, dj))},
        },
    }


def overlap_markdown(r):
    f = lambda v, d=4: "—" if v is None else f"{v:.{d}f}"  # noqa: E731
    L = [f"# HW05 idea-A neighborhood overlap — {r['dataset']}", "",
         f"graph: {r['graph']['nodes']} nodes, {r['graph']['undirected_edges']} undirected edges, degree median {r['graph']['degree_median']:g}; chance {r['chance']:.4f}",
         f"pairs with overlap > 0: graph {r['graph_pairs_nonzero']}, PPR top-{OVERLAP_PPR_TOP} {r['ppr_pairs_nonzero']}", ""]
    for name in ("graph", "ppr"):
        L += [f"## {name} Jaccard", "", "| θ | pairs J ≥ θ | same class (merge precision) | coverage | partners median (all / covered) |", "|---|---|---|---|---|"]
        for t, v in r[f"{name}_theta"].items():
            L.append(f"| {t} | {v['pairs']} | {f(v['same_class'])} | {f(v['coverage'])} | {v['partners_median_all_nodes']:g} / {f(v['partners_median_covered_nodes'], 0)} |")
        L.append("")
    L += ["## J of random pairs", "", "| set | pairs | zero frac | median | q90 | mean |", "|---|---|---|---|---|---|"]
    for name, d in r["pair_distributions"].items():
        for k, v in d.items():
            L.append(f"| {name} | {k} | {v['zero_frac']:.4f} | {v['median']:.4f} | {v['q90']:.4f} | {v['mean']:.4f} |")
    return "\n".join(L) + "\n"


def overlap_main(cli):
    if not cli.dataset:
        raise SystemExit("--dataset required")
    started = time.time()
    r = overlap_measure(cli.dataset, cli.seed)
    r["elapsed_seconds"] = round(time.time() - started, 2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"HW05_overlap_{cli.dataset}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    (OUT_DIR / f"HW05_overlap_{cli.dataset}.md").write_text(overlap_markdown(r), encoding="utf-8")
    print(overlap_markdown(r))
    return 0


# ---------------------------------------------------------------- HW07: exclusion sets of cluster / far
EXCLUSION_MS = (20, 100, 200)
EXCLUSION_EPISODE_EXCLUDED = 56  # 5 supports + 50 pseudo-queries + q (as N − 56 in the instruction)


def _dist3(v):
    v = np.asarray(v, dtype=np.float64)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "q90": float(np.quantile(v, 0.9))}


@torch.no_grad()
def exclusion_measure(dataset, seed=1212, episodes=2000):
    """HW07-c. measurement-only label use: every GT label read here feeds only these measurements.
    C(q) = q's cluster (k100, kmeans seed 20261108) without q; F(q) = shortest-path ≤ 2 ∪ raw-cos top-100,
    F_S(q) = shortest-path ≤ 2 ∪ raw-cos top-1000 (both without q). Node values are node macro (mean over nodes)."""
    from hw.negatives import FAR_STRONG_TABLE_K, FAR_TOP, adjacency_torch, kmeans_labels, raw_cos_table, within_two_hops
    from hw.purity import build_pool

    pool = build_pool(dataset, seed, episodes, arm="naq_base")
    data, device = pool["data"], pool["device"]
    features = data.features.detach().float().cpu()
    adj_cpu = data.adj.detach().cpu()
    labels = torch.as_tensor(np.asarray(pool["labels"].cpu() if torch.is_tensor(pool["labels"]) else pool["labels"])).long().to(device)
    n = features.shape[0]
    clus = kmeans_labels(dataset, features, adj_cpu).to(device)
    t1000 = raw_cos_table(dataset, features, k=FAR_STRONG_TABLE_K)["ids"].to(device)
    a = adjacency_torch(data.edge_index, n, device)
    sizes = torch.bincount(clus)
    acc = {key: torch.zeros(n, dtype=torch.float64, device=device) for key in ("c", "c_same", "f", "f_same", "fs", "fs_same", "fc", "class_other")}
    chunk = 1024
    for start in range(0, n, chunk):
        rows = torch.arange(start, min(n, start + chunk), device=device)
        r = rows.numel()
        selfm = torch.zeros((r, n), dtype=torch.bool, device=device)
        selfm[torch.arange(r, device=device), rows] = True
        hop = within_two_hops(a, rows)
        f = hop.clone()
        f.scatter_(1, t1000[rows, :FAR_TOP], True)
        fs = hop.clone()
        fs.scatter_(1, t1000[rows], True)
        f &= ~selfm
        fs &= ~selfm
        c = (clus[None, :] == clus[rows][:, None]) & ~selfm
        same = (labels[None, :] == labels[rows][:, None]) & ~selfm  # measurement-only label use
        for key, m in (("c", c), ("f", f), ("fs", fs)):
            acc[key][rows] = m.sum(dim=1).double()
            acc[key + "_same"][rows] = (m & same).sum(dim=1).double()
        acc["fc"][rows] = (f & c).sum(dim=1).double()
        acc["class_other"][rows] = same.sum(dim=1).double()
    v = {key: t.cpu().numpy() for key, t in acc.items()}
    ratio = lambda num, den: np.where(den > 0, num / np.maximum(den, 1), np.nan)  # noqa: E731
    denom = n - EXCLUSION_EPISODE_EXCLUDED
    node = {}
    for key, name in (("c", "C(q)"), ("f", "F(q)"), ("fs", "F_S(q)")):
        node[name] = {
            "size": _dist3(v[key]),
            "same_class_ratio_node_macro": float(np.nanmean(ratio(v[key + "_same"], v[key]))),
            "nodes_with_empty_set": int((v[key] == 0).sum()),
            "expected_random_hits": {str(m): float((m * v[key] / denom).mean()) for m in EXCLUSION_MS},
            "expected_random_same_class_hits": {str(m): float((m * v[key + "_same"] / denom).mean()) for m in EXCLUSION_MS},
        }
    node["overlap"] = {
        "F_cap_C_over_C_node_macro": float(np.nanmean(ratio(v["fc"], v["c"]))),
        "F_cap_C_over_F_node_macro": float(np.nanmean(ratio(v["fc"], v["f"]))),
    }
    node["random_expected_same_class_hits_all"] = {str(m): float((m * v["class_other"] / denom).mean()) for m in EXCLUSION_MS}

    # (b) episode level: (q, s⁺) pairs of the seed-<seed> training pool
    same_cluster, diff_size, diff_same, other_support_in_cq = [], [], [], []
    for id_spt, y_spt, id_query, y_query in pool["train_pool"]:
        sid = torch.as_tensor(id_spt, dtype=torch.long, device=device).flatten()
        qid = torch.as_tensor(id_query, dtype=torch.long, device=device).flatten()
        ys = torch.as_tensor(y_spt, dtype=torch.long, device=device).flatten()
        yq = torch.as_tensor(y_query, dtype=torch.long, device=device).flatten()
        match = yq[:, None] == ys[None, :]
        s_pos = sid[match.float().argmax(dim=1)]
        cq, cs = clus[qid], clus[s_pos]
        same_cluster.append((cq == cs).cpu())
        in_cs_not_cq = (clus[None, :] == cs[:, None]) & (clus[None, :] != cq[:, None])  # [Q, N]
        diff_size.append(in_cs_not_cq.sum(dim=1).cpu())
        diff_same.append((in_cs_not_cq & (labels[None, :] == labels[qid][:, None])).sum(dim=1).cpu())  # measurement-only label use
        other = ~match
        other_support_in_cq.append(((clus[sid][None, :] == cq[:, None]) & other).any(dim=1).cpu())
    same_cluster = torch.cat(same_cluster).numpy()
    diff_size = torch.cat(diff_size).double().numpy()
    diff_same = torch.cat(diff_same).double().numpy()
    nonempty = diff_size > 0
    episode = {
        "pairs": int(same_cluster.size),
        "episodes": len(pool["train_pool"]),
        "C_spos_eq_C_q_frac": float(same_cluster.mean()),
        "C_spos_minus_C_q_size_mean_all_pairs": float(diff_size.mean()),
        "C_spos_minus_C_q_size_mean_nonempty": float(diff_size[nonempty].mean()) if nonempty.any() else None,
        "C_spos_minus_C_q_same_class_ratio_nonempty_mean": float((diff_same[nonempty] / diff_size[nonempty]).mean()) if nonempty.any() else None,
        "C_q_contains_other_pseudo_class_support_frac": float(torch.cat(other_support_in_cq).float().mean()),
    }
    return {"dataset": dataset, "num_nodes": n, "seed": seed, "random_denominator": denom, "node": node, "episode": episode}


def exclusion_markdown(r):
    lines = [f"# HW07 exclusion sets — {r['dataset']} (N {r['num_nodes']}, seed {r['seed']} pool)", "",
             "Node values: mean over all nodes (node macro). Sets exclude q. Expected hits = m · |S(q)| / (N − 56).", "",
             "| set | size mean / median / q90 | same-class ratio | m20 hits | m100 hits | m200 hits | m20 same | m100 same | m200 same |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in ("C(q)", "F(q)", "F_S(q)"):
        x = r["node"][name]
        h, s = x["expected_random_hits"], x["expected_random_same_class_hits"]
        lines.append(f"| {name} | {x['size']['mean']:.1f} / {x['size']['median']:.0f} / {x['size']['q90']:.0f} | {x['same_class_ratio_node_macro']:.4f} | "
                     f"{h['20']:.3f} | {h['100']:.3f} | {h['200']:.3f} | {s['20']:.4f} | {s['100']:.4f} | {s['200']:.4f} |")
    o = r["node"]["overlap"]
    a = r["node"]["random_expected_same_class_hits_all"]
    lines += ["", f"- |F ∩ C| / |C| {o['F_cap_C_over_C_node_macro']:.4f}, |F ∩ C| / |F| {o['F_cap_C_over_F_node_macro']:.4f}",
              f"- random: expected same-class hits over all nodes m20 {a['20']:.4f}, m100 {a['100']:.4f}, m200 {a['200']:.4f}", ""]
    e = r["episode"]
    lines += [f"Episode level ({e['episodes']} episodes, {e['pairs']} (q, s⁺) pairs):",
              f"- C(s⁺) = C(q): {e['C_spos_eq_C_q_frac']:.4f}",
              f"- |C(s⁺) \\ C(q)| mean (all pairs) {e['C_spos_minus_C_q_size_mean_all_pairs']:.1f}, (non-empty) {e['C_spos_minus_C_q_size_mean_nonempty']:.1f}; same-class ratio (non-empty) {e['C_spos_minus_C_q_same_class_ratio_nonempty_mean']:.4f}",
              f"- C(q) contains another pseudo-class support of the episode: {e['C_q_contains_other_pseudo_class_support_frac']:.4f}"]
    return "\n".join(lines) + "\n"


def exclusion_main(cli):
    if not cli.dataset:
        raise SystemExit("--dataset required")
    started = time.time()
    r = exclusion_measure(cli.dataset, cli.seed, cli.episodes)
    r["elapsed_seconds"] = round(time.time() - started, 2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"HW07_exclusion_{cli.dataset}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    (OUT_DIR / f"HW07_exclusion_{cli.dataset}.md").write_text(exclusion_markdown(r), encoding="utf-8")
    print(exclusion_markdown(r))
    return 0


def main():
    parser = argparse.ArgumentParser(description="HW04m train-free positive-candidate purity")
    parser.add_argument("--dataset", choices=["Cora_full", "Amazon_clothing"])
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--energy", action="store_true", help="HW03b per-dimension Dirichlet energy")
    parser.add_argument("--overlap", action="store_true", help="HW05 idea-A neighborhood overlap")
    parser.add_argument("--exclusion", action="store_true", help="HW07 exclusion sets of cluster / far")
    cli = parser.parse_args()
    if cli.exclusion:
        sys.exit(exclusion_main(cli))
    if cli.overlap:
        sys.exit(overlap_main(cli))
    if cli.energy:
        sys.exit(energy_main(cli))
    if cli.selftest:
        ok, checks = selftest()
        for c in checks:
            print(f"{'OK ' if c['ok'] else 'FAIL'} {c['check']}: got {c['got']:.12g} want {c['want']:.12g}")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "HW04m_selftest.json").write_text(json.dumps({"pass": ok, "checks": checks}, indent=2), encoding="utf-8")
        print("G-toy", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    if not cli.dataset:
        parser.error("--dataset required")
    started = time.time()
    result, nodes = measure(cli.dataset, cli.seed, cli.episodes)
    result["elapsed_seconds"] = round(time.time() - started, 2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"HW04m_{cli.dataset}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT_DIR / f"HW04m_{cli.dataset}.md").write_text(to_markdown(result), encoding="utf-8")
    np.savez_compressed(OUT_DIR / f"HW04m_{cli.dataset}_nodes.npz", **nodes)
    print(to_markdown(result))
    print(f"elapsed {result['elapsed_seconds']} s; G-consist {'PASS' if result['G_consist']['pass'] else 'FAIL'}")
    if not result["G_consist"]["pass"]:
        sys.exit(4)


if __name__ == "__main__":
    main()
