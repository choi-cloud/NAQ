# Paired Δ vs `h9_rand_m20`

Δ = arm − ref test accuracy (%p), matched on seed. Cell: mean ± std (ddof=0), positive seeds k/n.
Sorted by the mean of the per-dataset Δ means (ties: total positive seeds).

| arm | Amazon_eletronics | dblp | mean of ds means | positive total |
|---|---|---|---|---|
| h9_clus_m20 | +0.32 ± 0.25 (9/10) | +0.32 ± 0.54 (7/10) | +0.32 | 16 |
| h9_rand_m50 | +0.03 ± 0.28 (5/10) | +0.31 ± 0.73 (5/10) | +0.17 | 10 |
| h9_naq | -0.70 ± 0.27 (0/10) | -0.70 ± 1.41 (3/10) | -0.70 | 3 |

## per-seed Δ

| arm | ds | seed 1212 | seed 1213 | seed 1214 | seed 1215 | seed 1216 | seed 1217 | seed 1218 | seed 1219 | seed 1220 | seed 1221 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| h9_clus_m20 | Amazon_eletronics | +0.66 | +0.05 | -0.05 | +0.05 | +0.45 | +0.39 | +0.34 | +0.33 | +0.74 | +0.22 |
| h9_clus_m20 | dblp | +0.16 | +0.21 | +0.21 | +0.53 | +0.48 | +1.80 | -0.14 | -0.12 | -0.12 | +0.19 |
| h9_rand_m50 | Amazon_eletronics | +0.43 | +0.03 | -0.05 | -0.41 | +0.55 | -0.14 | +0.22 | -0.08 | +0.02 | -0.25 |
| h9_rand_m50 | dblp | +0.06 | -0.27 | -0.08 | +0.14 | -0.12 | +1.56 | +0.17 | +1.89 | -0.02 | -0.27 |
| h9_naq | Amazon_eletronics | -0.19 | -0.30 | -0.91 | -0.91 | -0.84 | -0.53 | -0.92 | -0.84 | -1.01 | -0.50 |
| h9_naq | dblp | -3.45 | +1.61 | -0.58 | +0.28 | -1.77 | +0.88 | -2.08 | -0.75 | -0.16 | -0.94 |
