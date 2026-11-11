# Paired Δ vs `h9_naq`

Δ = arm − ref test accuracy (%p), matched on seed. Cell: mean ± std (ddof=0), positive seeds k/n.
Sorted by the mean of the per-dataset Δ means (ties: total positive seeds).

| arm | Amazon_clothing | Amazon_eletronics | Cora_full | dblp | mean of ds means | positive total |
|---|---|---|---|---|---|---|
| h9_clus_m20 | — | +1.45 ± 0.37 (10/10) | — | +0.12 ± 0.26 (6/10) | — | 16 |
| h9_rand_m50 | — | +1.77 ± 0.35 (10/10) | — | +0.11 ± 0.36 (6/10) | — | 16 |
| h9_rand_m20 | — | +1.50 ± 0.34 (10/10) | — | -0.01 ± 0.43 (5/10) | — | 15 |

## per-seed Δ

| arm | ds | seed 1212 | seed 1213 | seed 1214 | seed 1215 | seed 1216 | seed 1217 | seed 1218 | seed 1219 | seed 1220 | seed 1221 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| h9_clus_m20 | Amazon_eletronics | +1.25 | +1.41 | +1.69 | +1.22 | +1.56 | +1.05 | +1.12 | +2.41 | +1.37 | +1.39 |
| h9_clus_m20 | dblp | -0.22 | +0.22 | +0.23 | -0.27 | +0.27 | +0.37 | +0.22 | +0.56 | -0.13 | -0.06 |
| h9_rand_m50 | Amazon_eletronics | +1.82 | +1.44 | +1.99 | +1.86 | +1.80 | +1.39 | +1.77 | +2.63 | +1.58 | +1.43 |
| h9_rand_m50 | dblp | -0.20 | +0.54 | +0.21 | -0.67 | -0.05 | +0.48 | +0.33 | +0.47 | -0.03 | +0.05 |
| h9_rand_m20 | Amazon_eletronics | +1.67 | +1.19 | +1.59 | +1.48 | +1.42 | +1.27 | +1.42 | +2.42 | +1.31 | +1.19 |
| h9_rand_m20 | dblp | -0.39 | +0.22 | +0.07 | -0.73 | -0.01 | +0.57 | +0.40 | +0.40 | -0.65 | -0.02 |
