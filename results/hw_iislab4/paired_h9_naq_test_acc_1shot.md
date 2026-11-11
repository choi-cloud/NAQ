# Paired Δ vs `h9_naq`

Δ = arm − ref test accuracy (%p), matched on seed. Cell: mean ± std (ddof=0), positive seeds k/n.
Sorted by the mean of the per-dataset Δ means (ties: total positive seeds).

| arm | Amazon_eletronics | dblp | mean of ds means | positive total |
|---|---|---|---|---|
| h9_clus_m20 | +1.39 ± 0.52 (10/10) | -1.18 ± 0.52 (0/10) | +0.11 | 10 |
| h9_rand_m20 | +1.14 ± 0.51 (10/10) | -1.10 ± 0.73 (1/10) | +0.02 | 11 |
| h9_rand_m50 | +1.04 ± 0.53 (10/10) | -1.56 ± 0.65 (1/10) | -0.26 | 11 |

## per-seed Δ

| arm | ds | seed 1212 | seed 1213 | seed 1214 | seed 1215 | seed 1216 | seed 1217 | seed 1218 | seed 1219 | seed 1220 | seed 1221 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| h9_clus_m20 | Amazon_eletronics | +1.17 | +1.78 | +1.32 | +1.28 | +0.88 | +0.71 | +1.09 | +2.68 | +1.59 | +1.40 |
| h9_clus_m20 | dblp | -1.59 | -1.61 | -0.47 | -1.84 | -0.72 | -0.85 | -1.79 | -0.95 | -1.49 | -0.44 |
| h9_rand_m20 | Amazon_eletronics | +1.00 | +1.60 | +1.11 | +1.23 | +0.51 | +0.25 | +0.94 | +2.16 | +1.12 | +1.50 |
| h9_rand_m20 | dblp | -1.29 | -2.00 | +0.52 | -1.33 | -1.47 | -0.80 | -0.53 | -1.36 | -2.07 | -0.64 |
| h9_rand_m50 | Amazon_eletronics | +0.83 | +1.52 | +1.39 | +0.90 | +0.59 | +0.02 | +0.71 | +1.98 | +1.32 | +1.17 |
| h9_rand_m50 | dblp | -2.17 | -1.95 | +0.04 | -1.59 | -2.19 | -1.24 | -1.32 | -1.80 | -2.17 | -1.23 |
