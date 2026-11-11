# Paired Δ vs `h9_rand_m20`

Δ = arm − ref test accuracy (%p), matched on seed. Cell: mean ± std (ddof=0), positive seeds k/n.
Sorted by the mean of the per-dataset Δ means (ties: total positive seeds).

| arm | Amazon_eletronics | dblp | mean of ds means | positive total |
|---|---|---|---|---|
| h9_clus_m20 | +0.25 ± 0.19 (9/10) | -0.08 ± 0.64 (5/10) | +0.08 | 14 |
| h9_naq | -1.14 ± 0.51 (0/10) | +1.10 ± 0.73 (9/10) | -0.02 | 9 |
| h9_rand_m50 | -0.10 ± 0.21 (3/10) | -0.46 ± 0.28 (1/10) | -0.28 | 4 |

## per-seed Δ

| arm | ds | seed 1212 | seed 1213 | seed 1214 | seed 1215 | seed 1216 | seed 1217 | seed 1218 | seed 1219 | seed 1220 | seed 1221 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| h9_clus_m20 | Amazon_eletronics | +0.17 | +0.18 | +0.21 | +0.05 | +0.37 | +0.46 | +0.15 | +0.52 | +0.47 | -0.10 |
| h9_clus_m20 | dblp | -0.30 | +0.39 | -0.99 | -0.52 | +0.75 | -0.05 | -1.25 | +0.41 | +0.58 | +0.20 |
| h9_naq | Amazon_eletronics | -1.00 | -1.60 | -1.11 | -1.23 | -0.51 | -0.25 | -0.94 | -2.16 | -1.12 | -1.50 |
| h9_naq | dblp | +1.29 | +2.00 | -0.52 | +1.33 | +1.47 | +0.80 | +0.53 | +1.36 | +2.07 | +0.64 |
| h9_rand_m50 | Amazon_eletronics | -0.17 | -0.08 | +0.29 | -0.33 | +0.08 | -0.23 | -0.23 | -0.18 | +0.19 | -0.33 |
| h9_rand_m50 | dblp | -0.88 | +0.05 | -0.48 | -0.26 | -0.72 | -0.44 | -0.78 | -0.44 | -0.11 | -0.59 |
