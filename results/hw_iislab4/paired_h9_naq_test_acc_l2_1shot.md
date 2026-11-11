# Paired Δ vs `h9_naq`

Δ = arm − ref test accuracy (%p), matched on seed. Cell: mean ± std (ddof=0), positive seeds k/n.
Sorted by the mean of the per-dataset Δ means (ties: total positive seeds).

| arm | Amazon_eletronics | dblp | mean of ds means | positive total |
|---|---|---|---|---|
| h9_clus_m20 | +1.01 ± 0.36 (10/10) | +1.02 ± 1.29 (9/10) | +1.01 | 19 |
| h9_rand_m50 | +0.73 ± 0.36 (10/10) | +1.00 ± 1.47 (8/10) | +0.86 | 18 |
| h9_rand_m20 | +0.70 ± 0.27 (10/10) | +0.70 ± 1.41 (7/10) | +0.70 | 17 |

## per-seed Δ

| arm | ds | seed 1212 | seed 1213 | seed 1214 | seed 1215 | seed 1216 | seed 1217 | seed 1218 | seed 1219 | seed 1220 | seed 1221 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| h9_clus_m20 | Amazon_eletronics | +0.85 | +0.36 | +0.85 | +0.96 | +1.30 | +0.92 | +1.27 | +1.16 | +1.75 | +0.72 |
| h9_clus_m20 | dblp | +3.61 | -1.39 | +0.79 | +0.25 | +2.25 | +0.92 | +1.94 | +0.62 | +0.04 | +1.13 |
| h9_rand_m50 | Amazon_eletronics | +0.62 | +0.33 | +0.86 | +0.50 | +1.39 | +0.38 | +1.14 | +0.75 | +1.03 | +0.25 |
| h9_rand_m50 | dblp | +3.52 | -1.87 | +0.50 | -0.14 | +1.64 | +0.68 | +2.25 | +2.64 | +0.14 | +0.67 |
| h9_rand_m20 | Amazon_eletronics | +0.19 | +0.30 | +0.91 | +0.91 | +0.84 | +0.53 | +0.92 | +0.84 | +1.01 | +0.50 |
| h9_rand_m20 | dblp | +3.45 | -1.61 | +0.58 | -0.28 | +1.77 | -0.88 | +2.08 | +0.75 | +0.16 | +0.94 |
