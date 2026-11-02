"""ND01 gate A: meta-test classification with d_v (no training); accuracy formula from choi_exps/exp027.py: episode_accuracy.

Labels: only the y_support / y_query of the fixed valid/test episodes (functions named eval_*). Arm and λ selection use valid 1-shot only.
Writes results/nd/nd01/gateA/<shot>shot/<arm>_<pool>.npy (per-episode accuracy) and gateA/selection.json.
"""
import json
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F

import nd_vote as V
from nd_vote import H
from exp027 import episode_accuracy

LAMBDAS = (0.25, 0.5, 0.75)
SHOTS = (1, 5)
POOLS = ("valid", "test")
N_WAY = 5
OUT = V.OUT / "gateA"


def eval_parse_task(task, shot, device):
    """Same format checks as exp027.episode_accuracy. Returns supports, queries, targets (class position per query)."""
    supports, y_support, queries, y_query = task
    ids_s = torch.as_tensor(supports, dtype=torch.long, device=device)
    ids_q = torch.as_tensor(queries, dtype=torch.long, device=device)
    y_s = torch.as_tensor(y_support, dtype=torch.long, device=device).reshape(-1)
    y_q = torch.as_tensor(y_query, dtype=torch.long, device=device).reshape(-1)
    class_labels = y_s.reshape(N_WAY, shot)[:, 0]
    if torch.unique(class_labels).numel() != N_WAY:
        raise ValueError("Invalid support classes in fixed episode")
    if not bool((y_s.reshape(N_WAY, shot) == class_labels[:, None]).all()):
        raise ValueError("Supports are not grouped by class")
    targets = y_q[:, None] == class_labels[None, :]
    if not bool(targets.any(dim=1).all()):
        raise ValueError("Query labels missing from support classes")
    return ids_s, ids_q, targets.long().argmax(dim=1)


def eval_pool_logits(tasks, shot, zs, weights, device):
    """{arm: [episodes, 40, 5]} of logit_feat / logit_d on CPU, and targets [episodes, 40]. Asserts feat_<f> reproduces exp027."""
    logits = {}
    targets = []
    for task in tasks:
        ids_s, ids_q, tgt = eval_parse_task(task, shot, device)
        assert len(ids_q) == 40, len(ids_q)
        targets.append(tgt.cpu())
        rows = torch.cat((ids_s, ids_q))
        for f, z in zs.items():
            spt = z[ids_s].reshape(N_WAY, shot, -1)
            proto = spt.mean(dim=1)
            lf = -(z[ids_q, None, :] - proto[None, :, :]).square().sum(-1)
            mine = float((lf.argmax(1) == tgt).float().mean())
            assert abs(mine - episode_accuracy(z, task, N_WAY, shot)) < 1e-6
            logits.setdefault(f"feat_{f}", []).append(lf.cpu())
            S = V.similarity(z, F.normalize(proto, dim=1))
            for t in V.TAU_KEYS:
                Q = V.assign_from_s(S, t)
                for nbr in V.NBRS[f]:
                    if nbr not in weights:
                        continue
                    D = V.vote_rows(Q, weights[nbr], rows)
                    pi = D[: N_WAY * shot].reshape(N_WAY, shot, N_WAY).mean(dim=1)
                    ld = -(D[N_WAY * shot:, None, :] - pi[None, :, :]).square().sum(-1)
                    logits.setdefault(V.arm_name(f, t, nbr), []).append(ld.cpu())
    return {k: torch.stack(v) for k, v in logits.items()}, torch.stack(targets)


def eval_accuracy(logit, targets):
    return (logit.argmax(-1) == targets).float().mean(1).numpy().astype(np.float64)


def eval_tie_frac(logit):
    """Fraction of queries whose two highest logits are exactly equal (argmax then takes the lower class position)."""
    top = logit.topk(2, dim=-1).values
    return float((top[..., 0] == top[..., 1]).float().mean())


def combine(lf, ld, lam):
    return lam * F.log_softmax(lf, dim=-1) + (1 - lam) * F.log_softmax(ld, dim=-1)


def first_best(values):
    """(first key with the highest value in dict order, all tied keys)."""
    top = max(values.values())
    tied = [k for k, v in values.items() if v == top]
    return tied[0], tied


def main():
    t0 = time.time()
    args = H.base_args()
    data, device = H.load_everything(args)
    z0, _, z2 = H.structural_scores(data, device)
    zs = {"x": z0, "a2x": z2}
    weights, info = V.build_weights(data, device)
    pools = {}
    for shot in SHOTS:
        with open(f"save/downstream_tasks/Amazon_clothing/5way_{shot}shot.pkl", "rb") as fh:
            valid_pool, test_pool = pickle.load(fh)
        assert len(valid_pool) == 50 and len(test_pool) == 500
        pools[shot] = {"valid": valid_pool, "test": test_pool}
    logits, targets, acc = {}, {}, {}
    with torch.no_grad():
        for shot in SHOTS:
            for pool in POOLS:
                t1 = time.time()
                logits[shot, pool], targets[shot, pool] = eval_pool_logits(pools[shot][pool], shot, zs, weights, device)
                acc[shot, pool] = {k: eval_accuracy(v, targets[shot, pool]) for k, v in logits[shot, pool].items()}
                print(f"{shot}shot {pool}: {len(logits[shot, pool])} base arms, {time.time() - t1:.1f}s", flush=True)

        ties = {f"{shot}shot": {k: eval_tie_frac(v) for k, v in logits[shot, "test"].items()} for shot in SHOTS}

        # selection: valid 1-shot only
        selection = {}
        comb_sources = {}
        for f in zs:
            arms = [V.arm_name(ff, t, nbr) for ff, t, nbr in V.grid() if ff == f and nbr in weights]
            best, tied = first_best({a: float(acc[1, "valid"][a].mean()) for a in arms})
            main_arm = V.arm_name(f, "0.1", "1hop")
            comb_sources[f] = [main_arm] + ([best] if best != main_arm else [])
            selection[f] = {"vote_best": best, "vote_best_valid_1shot": 100 * float(acc[1, "valid"][best].mean()), "vote_best_tied": tied}

        for shot in SHOTS:
            for pool in POOLS:
                for f, sources in comb_sources.items():
                    for src in sources:
                        for lam in LAMBDAS:
                            name = f"comb_{src[len('vote_'):]}_l{lam}"
                            acc[shot, pool][name] = eval_accuracy(
                                combine(logits[shot, pool][f"feat_{f}"], logits[shot, pool][src], lam), targets[shot, pool])

        for f in zs:
            src = selection[f]["vote_best"][len("vote_"):]
            lam, tied = first_best({lam: float(acc[1, "valid"][f"comb_{src}_l{lam}"].mean()) for lam in LAMBDAS})
            sel = f"comb_{src}_l{lam}"
            selection[f].update({"comb_sel_lambda": lam, "comb_sel_arm": sel, "comb_sel_valid_1shot": 100 * float(acc[1, "valid"][sel].mean()),
                                 "comb_sel_lambda_tied": tied, "comb_sources": comb_sources[f]})
            for shot in SHOTS:
                for pool in POOLS:
                    acc[shot, pool][f"comb_{f}_sel"] = acc[shot, pool][sel]

    for (shot, pool), arms in acc.items():
        d = OUT / f"{shot}shot"
        d.mkdir(parents=True, exist_ok=True)
        for arm, a in arms.items():
            np.save(d / f"{arm}_{pool}.npy", a)
    meta = {"selection": selection, "weights": info, "test_tie_frac": ties, "elapsed_s": time.time() - t0,
            "gpu_max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else None}
    (OUT / "selection.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "test_tie_frac"}, indent=1))


if __name__ == "__main__":
    main()
