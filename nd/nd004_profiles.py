"""ND02 §2-§8: assignment, inter-profile, group pattern, Q1/Q2 classification, Q3 pairs, coverage (no training); A from nd/nd_vote._sym_a, Â² weights from nd/nd_vote.build_weights, AUROC from hl/hl005_diag_distance_purity.

Labels are read only in functions named grade_* (hl_common.labels_np). Assignment, profile, pattern and score code never receives labels.
Usage: python nd/nd004_profiles.py --episodes pseudo_k3 --f x --unassign none --nbr 1hop --score cos [--rewire] [--stage all]
       python nd/nd004_profiles.py --stage stats    (episode statistics, labels read)
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import nd_vote as V
from nd_vote import H
from hl005_diag_distance_purity import auroc_pooled, auroc_rows

G = 5
ROOT = Path("results/nd/nd02")
EPS = ROOT / "episodes"
BUCKETS = (("0", 0, 0), ("1-2", 1, 2), ("3-4", 3, 4), (">=5", 5, np.inf))
PAIR_SCORES = ("cos_inter5", "cos_full5", "cos_x", "cos_a2x")
Q2_SCORES = ("s_own", "h", "a2x", "x", "comb")
PATTERN_ORDER = ["assign", "profile", "pattern", "classify", "pairs", "coverage", "all"]


def rewire_edges(data, seed=V.SEED):
    """Degree-preserving double edge swap, 10 x |E| attempts with numpy only; self-loops and duplicate edges are skipped."""
    a = V._sym_a(data).tocoo()
    keep = a.row < a.col
    u, v = a.row[keep].astype(np.int64), a.col[keep].astype(np.int64)
    m = len(u)
    rng = np.random.default_rng(seed)
    attempts = 10 * m
    i1, i2 = rng.integers(0, m, attempts), rng.integers(0, m, attempts)
    flip = rng.integers(0, 2, attempts).astype(bool)
    present = {(int(x), int(y)) for x, y in zip(u, v)}
    accepted = 0
    for t in range(attempts):
        i, j = int(i1[t]), int(i2[t])
        if i == j:
            continue
        p, q = int(u[i]), int(v[i])
        r, s = (int(v[j]), int(u[j])) if flip[t] else (int(u[j]), int(v[j]))
        # (p-q, r-s) -> (p-s, r-q)
        if p == s or r == q:
            continue
        e1, e2 = (min(p, s), max(p, s)), (min(r, q), max(r, q))
        if e1 == e2 or e1 in present or e2 in present:
            continue
        present.discard((min(p, q), max(p, q)))
        present.discard((min(r, s), max(r, s)))
        present.add(e1)
        present.add(e2)
        u[i], v[i] = e1
        u[j], v[j] = e2
        accepted += 1
    n = data.features.size(0)
    b = sp.csr_matrix((np.ones(m, dtype=np.float32), (u, v)), shape=(n, n))
    b = ((b + b.T) > 0).astype(np.float32).tocsr()
    deg_a = np.asarray(V._sym_a(data).sum(1)).ravel()
    deg_b = np.asarray(b.sum(1)).ravel()
    assert np.array_equal(deg_a, deg_b), "degree changed by rewiring"
    assert b.diagonal().sum() == 0 and b.nnz == 2 * m
    still = int(((V._sym_a(data).multiply(b)).nnz) / 2)
    return np.concatenate((np.stack((u, v)), np.stack((v, u))), axis=1), {
        "edges": m, "attempts": attempts, "accepted": accepted, "edges_unchanged": still}


def count_weights(data, device, nbr):
    """W for c_v: binary A (1hop) or the row-normalised Â² weights with the diagonal removed (a2)."""
    if nbr == "1hop":
        a = V._sym_a(data).tocoo()
        idx = torch.as_tensor(np.stack((a.row, a.col)), dtype=torch.long, device=device)
        coo = torch.sparse_coo_tensor(idx, torch.as_tensor(a.data, dtype=torch.float32, device=device), a.shape)
    else:
        w, _ = V.build_weights(data, device)
        c = w["a2"].coo
        keep = c.indices()[0] != c.indices()[1]
        coo = torch.sparse_coo_tensor(c.indices()[:, keep], c.values()[keep], c.shape)
    return V.Weight(coo)


def assign_and_profile(zf, W, support, query, unassign, device):
    """§2-§3. c [E, G*(K+M), 5] (query slots with id -1 stay zero), u [E, G*(K+M)]. No labels."""
    E, _, K = support.shape
    M = query.shape[2]
    n = zf.size(0)
    nodes = np.concatenate((support, query), axis=2)
    c = np.zeros((E, G * (K + M), G), dtype=np.float32)
    u = np.zeros((E, G * (K + M)), dtype=np.float32)
    with torch.no_grad():
        for e in range(E):
            sup = torch.as_tensor(support[e], device=device)
            P = F.normalize(zf[sup].mean(dim=1), dim=1)
            s = zf @ P.T
            grp = s.argmax(dim=1)
            ids = torch.as_tensor(nodes[e].ravel(), device=device)
            valid = ids >= 0
            own = torch.arange(G, device=device).repeat_interleave(K + M)
            if unassign == "q30":
                top = s.topk(2, dim=1).values
                margin = top[:, 0] - top[:, 1]
                outside = torch.ones(n, dtype=torch.bool, device=device)
                outside[ids[valid]] = False
                thr = torch.quantile(margin[outside], 0.30)
                grp = grp.clone()
                grp[outside & (margin < thr)] = -1
            grp[ids[valid]] = own[valid]
            Q = torch.zeros(n, G + 1, device=device)
            Q[torch.arange(n, device=device), torch.where(grp < 0, G, grp)] = 1.0  # last column: unassigned
            D = V.vote_rows(Q, W, ids[valid])
            ce = torch.zeros(len(ids), G + 1, device=device)
            ce[valid] = D
            c[e] = ce[:, :G].cpu().numpy()
            u[e] = ce[:, G].cpu().numpy()
    return nodes.reshape(E, -1), c, u


def derive(c, nodes):
    """h, n_inter, inter5, full5, own for each slot; undefined entries are NaN. No labels."""
    E, S, _ = c.shape
    per = S // G
    own = np.repeat(np.arange(G), per)
    total = c.sum(2)
    own_c = c[:, np.arange(S), own]
    n_inter = total - own_c
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(total > 0, own_c / total, np.nan)
        full5 = np.where(total[..., None] > 0, c / total[..., None], np.nan)
        inter5 = c.copy()
        inter5[:, np.arange(S), own] = 0
        inter5 = np.where(n_inter[..., None] > 0, inter5 / n_inter[..., None], np.nan)
    invalid = nodes < 0
    h[invalid], n_inter[invalid] = np.nan, np.nan
    full5[invalid], inter5[invalid] = np.nan, np.nan
    return {"own": own, "h": h, "n_inter": n_inter, "inter5": inter5, "full5": full5}


def group_patterns(inter5, own, K):
    """§4. pi, sigma [E, G, 4] (slot g removed), has [E, G], n_def [E, G]. Supports are the first K slots of each group. No labels."""
    E = inter5.shape[0]
    per = inter5.shape[1] // G
    pi = np.full((E, G, G - 1), np.nan, dtype=np.float64)
    sig = np.full((E, G, G - 1), np.nan, dtype=np.float64)
    n_def = np.zeros((E, G), dtype=np.int64)
    for g in range(G):
        keep = [j for j in range(G) if j != g]
        sup = inter5[:, g * per:g * per + K][:, :, keep]
        ok = ~np.isnan(sup).any(2)
        n_def[:, g] = ok.sum(1)
        for e in range(E):
            if ok[e].any():
                pi[e, g] = sup[e][ok[e]].mean(0)
                sig[e, g] = sup[e][ok[e]].std(0, ddof=0) if ok[e].sum() > 1 else 1.0
    return pi, sig, n_def > 0, n_def


def pattern_stats(pi, has):
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(pi > 0, pi * np.log(pi), 0.0), axis=2)
    E = pi.shape[0]
    cosines = []
    for e in range(E):
        full = np.zeros((G, G))  # slot-aligned: pi_g with 0 at slot g
        for g in range(G):
            if has[e, g]:
                full[g, [j for j in range(G) if j != g]] = pi[e, g]
        for a in range(G):
            for b in range(a + 1, G):
                if has[e, a] and has[e, b]:
                    cosines.append(full[a] @ full[b] / (np.linalg.norm(full[a]) * np.linalg.norm(full[b])))
    return float(ent[has].mean()), float(np.mean(cosines)) if cosines else None


def hypothesis_scores(cq, pi, sig, has, mode):
    """score(q, g) [E, G, M, 5]. -inf where the 4-vector for hypothesis g has zero mass or group g has no pattern. No labels."""
    out = np.full(cq.shape, -np.inf, dtype=np.float64)
    for g in range(G):
        keep = [j for j in range(G) if j != g]
        cg = cq[..., keep].astype(np.float64)
        tot = cg.sum(-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            p = cg / tot
            pig = pi[:, g][:, None, None, :]
            if mode == "cos":
                sc = (p * pig).sum(-1) / (np.linalg.norm(p, axis=-1) * np.linalg.norm(pig, axis=-1))
            else:
                sc = -((p - pig) ** 2 / (sig[:, g][:, None, None, :] ** 2 + 0.01)).sum(-1)
        ok = (tot[..., 0] > 0) & has[:, g][:, None, None]
        out[..., g] = np.where(ok, sc, -np.inf)
    return out


def safe_pooled(s, y):
    y = y.astype(bool)
    if y.size == 0 or y.all() or not y.any():
        return None
    return auroc_pooled(s, y)


def rows_mean(score, y, mask):
    """Mean per-episode AUROC over [E, ...]-shaped arrays restricted to mask; episodes lacking positives or negatives are skipped."""
    vals = []
    for e in range(score.shape[0]):
        m = mask[e].ravel()
        s, t = score[e].ravel()[m], y[e].ravel()[m]
        if t.size and t.any() and not t.all():
            vals.append(float(auroc_rows(s[None], t[None])[0]))
    return (float(np.mean(vals)) if vals else None), len(vals)


def within_rank_mean(a, b, mask):
    from scipy.stats import rankdata
    out = np.full(a.shape, np.nan)
    for e in range(a.shape[0]):
        m = mask[e]
        if m.any():
            out[e][m] = (rankdata(a[e][m]) + rankdata(b[e][m])) / 2
    return out


def grade_q1_q2(labels, support, query, s, own_q, q_valid, has, a2x, x, h, n_inter, score_mode):
    """Q1, Q2 and §8 query buckets. Labels read here only for same(q)."""
    E, _, M = query.shape
    same = np.zeros((E, G, M), dtype=bool)
    anchors = support[:, :, 0]
    same[q_valid] = labels[query[q_valid]] == labels[np.repeat(anchors[:, :, None], M, 2)[q_valid]]
    finite = np.isfinite(s).any(-1)
    has_own = np.repeat(has[:, :, None], M, 2)
    q1_mask = q_valid & has_own & finite
    pred = s.argmax(-1)
    hit = pred == own_q
    ep_acc = np.array([hit[e][q1_mask[e]].mean() if q1_mask[e].any() else np.nan for e in range(E)])
    ok_ep = ep_acc[~np.isnan(ep_acc)]
    diff = q1_mask & ~hit
    agree = q1_mask & hit
    q1 = {"acc_pseudo": float(ok_ep.mean()), "ci95": float(1.96 * ok_ep.std(ddof=0) / np.sqrt(len(ok_ep))), "n_episodes": int(len(ok_ep)),
          "excluded_query_frac": float(1 - q1_mask.sum() / q_valid.sum()),
          "excl_no_group_pattern": int((q_valid & ~has_own).sum()), "excl_all_inf": int((q_valid & has_own & ~finite).sum()),
          "gt_mismatch_pred_ne_own": float((~same[diff]).mean()) if diff.any() else None,
          "gt_mismatch_pred_eq_own": float((~same[agree]).mean()) if agree.any() else None,
          "n_pred_ne_own": int(diff.sum()), "n_pred_eq_own": int(agree.sum())}
    s_own = np.take_along_axis(s, own_q[..., None], -1)[..., 0]
    q2_mask = q_valid & has_own & (n_inter > 0) & np.isfinite(s_own)
    comb = within_rank_mean(s_own, a2x, q2_mask)
    cols = {"s_own": s_own, "h": h, "a2x": a2x, "x": x, "comb": comb}
    q2 = {}
    for k, v in cols.items():
        mean, n_ep = rows_mean(v, same, q2_mask)
        # comb ranks are per episode by construction; pooled uses them as they are
        q2[k] = {"auroc_rows": mean, "n_episodes": n_ep, "auroc_pooled": safe_pooled(v[q2_mask], same[q2_mask]), "n_queries": int(q2_mask.sum())}
    incl = q_valid & has_own
    mean, n_ep = rows_mean(s_own, same, incl)
    q2["s_own_incl_ninter0"] = {"auroc_rows": mean, "n_episodes": n_ep, "n_queries": int(incl.sum())}
    cover = {}
    for name, lo, hi in BUCKETS:
        b = q_valid & (n_inter >= lo) & (n_inter <= hi)
        bq1 = b & q1_mask
        bq2 = b & has_own & np.isfinite(s_own)
        ms, ns = rows_mean(s_own, same, bq2)
        ma, na = rows_mean(a2x, same, b & has_own)
        cover[name] = {"query_frac": float(b.sum() / q_valid.sum()), "acc_pseudo": float(hit[bq1].mean()) if bq1.any() else None, "n_q1": int(bq1.sum()),
                       "auroc_s_own": None if name == "0" else ms, "n_ep_s_own": 0 if name == "0" else ns, "auroc_a2x": ma, "n_ep_a2x": na}
    return q1, q2, cover


def grade_pairs(labels, nodes, derived, zs, device):
    """Q3 over all S ∪ Q pairs. Labels read for same_gt."""
    E, S = nodes.shape
    iu, ju = np.triu_indices(S, 1)
    P = len(iu)
    own = derived["own"]
    same_group = own[iu] == own[ju]
    y = np.zeros((E, P), dtype=bool)
    valid = np.zeros((E, P), dtype=bool)
    sc = {k: np.full((E, P), np.nan, dtype=np.float32) for k in PAIR_SCORES}
    ni3 = np.zeros((E, P), dtype=bool)

    def cos_np(v):
        with np.errstate(invalid="ignore", divide="ignore"):
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            return (v[iu] * v[ju]).sum(1)

    with torch.no_grad():
        for e in range(E):
            ok = nodes[e] >= 0
            valid[e] = ok[iu] & ok[ju]
            ids = torch.as_tensor(np.where(ok, nodes[e], 0), device=device)
            lab = labels[np.where(ok, nodes[e], 0)]
            y[e] = lab[iu] == lab[ju]
            for name, z in (("cos_x", zs["x"]), ("cos_a2x", zs["a2x"])):
                m = z[ids] @ z[ids].T
                sc[name][e] = m[torch.as_tensor(iu, device=device), torch.as_tensor(ju, device=device)].cpu().numpy()
            sc["cos_inter5"][e] = cos_np(derived["inter5"][e])
            sc["cos_full5"][e] = cos_np(derived["full5"][e])
            n3 = np.nan_to_num(derived["n_inter"][e]) >= 3
            ni3[e] = n3[iu] & n3[ju]
    thr = np.quantile(sc["cos_x"][valid], [0.25, 0.5, 0.75])
    bins = np.searchsorted(thr, sc["cos_x"], side="left")  # q1 = cos_x <= thr[0], so all ties at the lower quartile stay in q1
    sg = np.broadcast_to(same_group, (E, P))
    res = {"quartile_thresholds": thr.tolist(), "same_gt_rate": float(y[valid].mean()),
           "same_gt_rate_same_group": float(y[valid & sg].mean()), "same_gt_rate_diff_group": float(y[valid & ~sg].mean()),
           "quartile_pair_frac": [float((bins[valid] == q).mean()) for q in range(4)]}
    for k, s in sc.items():
        ok = valid & np.isfinite(s)
        r = {"auroc_all": safe_pooled(s[ok], y[ok]), "auroc_episode_mean": rows_mean(s, y, ok)[0], "n_pairs": int(ok.sum())}
        for q in range(4):
            m = ok & (bins == q)
            r[f"auroc_q{q + 1}"] = safe_pooled(s[m], y[m])
        r["auroc_same_group"] = safe_pooled(s[ok & sg], y[ok & sg])
        r["auroc_diff_group"] = safe_pooled(s[ok & ~sg], y[ok & ~sg])
        m = ok & ni3
        r["auroc_ninter_ge3"] = safe_pooled(s[m], y[m])
        r["n_pairs_ninter_ge3"] = int(m.sum())
        res[k] = r
    return res


def grade_episode_stats():
    """§1-2 episode statistics. Labels read."""
    args = H.base_args()
    data, _ = H.load_everything(args)
    labels = H.labels_np(data)
    rows = []
    for name in ("pseudo_k3", "pseudo_k5", "pseudo_choi", "gt_k3"):
        d = np.load(EPS / f"{name}.npz")
        sup, qry = d["support"], d["query"]
        for e in range(sup.shape[0]):
            anc = labels[sup[e, :, 0]]
            spur = (labels[sup[e, :, 1:]] == anc[:, None]).mean()
            ok = qry[e] >= 0
            qlab = labels[np.where(ok, qry[e], 0)]
            qpur = (qlab == anc[:, None])[ok].mean()
            conflicts = sum(anc[a] == anc[b] for a in range(G) for b in range(a + 1, G))
            rows.append({"episodes": name, "episode": e, "support_purity": float(spur), "query_purity": float(qpur),
                         "anchor_gt_conflicts": int(conflicts), "n_query": int(ok.sum())})
    with open(EPS / "stats.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", default="pseudo_k3", choices=["pseudo_k3", "pseudo_k5", "pseudo_choi", "gt_k3"])
    p.add_argument("--f", default="x", choices=["x", "a2x"])
    p.add_argument("--unassign", default="none", choices=["none", "q30"])
    p.add_argument("--nbr", default="1hop", choices=["1hop", "a2"])
    p.add_argument("--score", default="cos", choices=["cos", "var"])
    p.add_argument("--rewire", action="store_true")
    p.add_argument("--stage", default="all", choices=PATTERN_ORDER + ["stats"])
    p.add_argument("--run", required=False)
    a = p.parse_args()
    if a.stage == "stats":
        grade_episode_stats()
        return
    t0 = time.time()
    upto = PATTERN_ORDER.index(a.stage)
    args = H.base_args()
    data, device = H.load_everything(args)
    info = {"args": vars(a)}
    if a.rewire:
        edges, info["rewire"] = rewire_edges(data)
        (ROOT / "rewire").mkdir(parents=True, exist_ok=True)
        np.save(ROOT / "rewire" / "edge_index.npy", edges)
        data.edge_index = torch.as_tensor(edges, dtype=data.edge_index.dtype, device=data.edge_index.device)
    z0, _, z2 = H.structural_scores(data, device)
    zs = {"x": z0, "a2x": z2}
    ep = np.load(EPS / f"{a.episodes}.npz")
    support, query = ep["support"], ep["query"]
    E, _, K = support.shape
    M = query.shape[2]
    W = count_weights(data, device, a.nbr)
    assert int((W.length == 0).sum()) == 0 or a.nbr != "1hop", "isolated node: vote_rows would fall back to Q"
    nodes, c, u = assign_and_profile(zs[a.f], W, support, query, a.unassign, device)
    suffix = "_rewire" if a.rewire else ""
    (ROOT / "profiles").mkdir(parents=True, exist_ok=True)
    d = derive(c, nodes)
    np.savez_compressed(ROOT / "profiles" / f"{a.episodes}_{a.f}_{a.unassign}_{a.nbr}{suffix}.npz", c=c, h=d["h"], n_inter=d["n_inter"], u=u, nodes=nodes)
    if upto <= PATTERN_ORDER.index("profile"):
        return
    per = K + M
    pi, sig, has, n_def = group_patterns(d["inter5"], d["own"], K)
    ent, picos = pattern_stats(pi, has)
    sup_slots = np.tile(np.arange(per) < K, G)
    valid = nodes >= 0
    defined = valid & (np.nan_to_num(d["n_inter"]) > 0)
    info["pattern"] = {
        "defined_support_frac": float(defined[:, sup_slots].sum() / valid[:, sup_slots].sum()),
        "defined_query_frac": float(defined[:, ~sup_slots].sum() / valid[:, ~sup_slots].sum()),
        "group_pattern_frac": float(has.mean()), "pi_entropy_mean": ent, "pi_cos_mean": picos,
        "h_mean": float(np.nanmean(d["h"])), "n_inter_mean": float(np.nanmean(d["n_inter"])), "n_inter_median": float(np.nanmedian(d["n_inter"])),
        "defined_support_count_hist": {str(i): int((n_def == i).sum()) for i in range(K + 1)},
        "unassigned_nbr_mean": float(u[valid].mean()), "c_empty_frac": float((c[valid].sum(1) == 0).mean()),
    }
    if upto <= PATTERN_ORDER.index("pattern"):
        print(json.dumps(info, indent=1))
        return
    cq = c.reshape(E, G, per, G)[:, :, K:]
    q_valid = query >= 0
    s = hypothesis_scores(cq, pi, sig, has, a.score)
    own_q = np.broadcast_to(np.arange(G)[None, :, None], (E, G, M))
    anchors = support[:, :, 0]
    with torch.no_grad():
        qid = torch.as_tensor(np.where(q_valid, query, 0), device=device)
        aid = torch.as_tensor(anchors, device=device)
        a2x = (zs["a2x"][qid] * zs["a2x"][aid][:, :, None, :]).sum(-1).cpu().numpy().astype(np.float64)
        x = (zs["x"][qid] * zs["x"][aid][:, :, None, :]).sum(-1).cpu().numpy().astype(np.float64)
    h_q = d["h"].reshape(E, G, per)[:, :, K:]
    ni_q = np.nan_to_num(d["n_inter"].reshape(E, G, per)[:, :, K:], nan=-1.0)
    labels = H.labels_np(data)
    info["q1"], info["q2"], info["coverage"] = grade_q1_q2(labels, support, query, s, own_q, q_valid, has, a2x, x, h_q, ni_q, a.score)
    info["q3"] = grade_pairs(labels, nodes, d, zs, device)
    info["elapsed_s"] = time.time() - t0
    info["gpu_max_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else None
    name = a.run or f"{a.episodes}_{a.f}_{a.unassign}_{a.nbr}_{a.score}{suffix}"
    (ROOT / "summary").mkdir(parents=True, exist_ok=True)
    (ROOT / "summary" / f"{name}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("pattern", "q1", "elapsed_s", "gpu_max_allocated_gb")}, indent=1))


if __name__ == "__main__":
    main()
