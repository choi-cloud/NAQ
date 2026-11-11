# Paired Δ vs `h9_rand_m20`

Δ = arm − ref test accuracy (%p), matched on seed. Cell: mean ± std (ddof=0), positive seeds k/n.
Sorted by the mean of the per-dataset Δ means (ties: total positive seeds).

| arm | Amazon_clothing | Amazon_eletronics | Cora_full | dblp | mean of ds means | positive total |
|---|---|---|---|---|---|---|
| h9_rand_m50 | — | +0.27 ± 0.10 (10/10) | — | +0.13 ± 0.20 (7/10) | — | 17 |
| h9_clus_m20 | — | -0.05 ± 0.22 (5/10) | — | +0.13 ± 0.23 (6/10) | — | 11 |
| h9_naq | — | -1.50 ± 0.34 (0/10) | — | +0.01 ± 0.43 (5/10) | — | 5 |

## per-seed Δ

| arm | ds | seed 1212 | seed 1213 | seed 1214 | seed 1215 | seed 1216 | seed 1217 | seed 1218 | seed 1219 | seed 1220 | seed 1221 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| h9_rand_m50 | Amazon_eletronics | +0.15 | +0.25 | +0.40 | +0.39 | +0.38 | +0.12 | +0.35 | +0.20 | +0.26 | +0.24 |
| h9_rand_m50 | dblp | +0.19 | +0.31 | +0.14 | +0.06 | -0.04 | -0.08 | -0.07 | +0.07 | +0.61 | +0.07 |
| h9_clus_m20 | Amazon_eletronics | -0.42 | +0.22 | +0.10 | -0.26 | +0.14 | -0.22 | -0.30 | -0.02 | +0.06 | +0.20 |
| h9_clus_m20 | dblp | +0.17 | -0.00 | +0.16 | +0.47 | +0.28 | -0.20 | -0.18 | +0.16 | +0.52 | -0.05 |
| h9_naq | Amazon_eletronics | -1.67 | -1.19 | -1.59 | -1.48 | -1.42 | -1.27 | -1.42 | -2.42 | -1.31 | -1.19 |
| h9_naq | dblp | +0.39 | -0.22 | -0.07 | +0.73 | +0.01 | -0.57 | -0.40 | -0.40 | +0.65 | +0.02 |
