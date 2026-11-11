# Paired Δ vs `h9_rand_m20`

Δ = arm − ref test accuracy (%p), matched on seed. Cell: mean ± std (ddof=0), positive seeds k/n.
Sorted by the mean of the per-dataset Δ means (ties: total positive seeds).

| arm | Amazon_clothing | Amazon_eletronics | Cora_full | dblp | mean of ds means | positive total |
|---|---|---|---|---|---|---|
| h9_rand_m50 | — | +0.26 ± 0.21 (10/10) | — | +0.05 ± 0.09 (7/10) | — | 17 |
| h9_clus_m20 | — | +0.08 ± 0.14 (7/10) | — | +0.06 ± 0.13 (7/10) | — | 14 |
| h9_naq | — | -1.04 ± 0.27 (0/10) | — | -0.34 ± 0.32 (3/10) | — | 3 |

## per-seed Δ

| arm | ds | seed 1212 | seed 1213 | seed 1214 | seed 1215 | seed 1216 | seed 1217 | seed 1218 | seed 1219 | seed 1220 | seed 1221 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| h9_rand_m50 | Amazon_eletronics | +0.21 | +0.12 | +0.34 | +0.03 | +0.66 | +0.19 | +0.62 | +0.06 | +0.10 | +0.24 |
| h9_rand_m50 | dblp | +0.18 | -0.03 | +0.08 | +0.06 | +0.02 | +0.18 | -0.06 | +0.13 | +0.05 | -0.09 |
| h9_clus_m20 | Amazon_eletronics | +0.13 | -0.03 | +0.06 | -0.22 | +0.12 | -0.05 | +0.28 | +0.27 | +0.14 | +0.07 |
| h9_clus_m20 | dblp | +0.01 | +0.12 | +0.17 | +0.22 | +0.04 | +0.25 | -0.09 | -0.08 | -0.16 | +0.11 |
| h9_naq | Amazon_eletronics | -0.78 | -0.67 | -1.21 | -1.47 | -0.85 | -1.11 | -1.06 | -1.48 | -0.88 | -0.87 |
| h9_naq | dblp | -0.39 | +0.02 | -0.22 | +0.08 | -0.67 | +0.02 | -1.03 | -0.36 | -0.45 | -0.38 |
