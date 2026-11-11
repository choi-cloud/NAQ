# Paired Δ vs `h9_naq`

Δ = arm − ref test accuracy (%p), matched on seed. Cell: mean ± std (ddof=0), positive seeds k/n.
Sorted by the mean of the per-dataset Δ means (ties: total positive seeds).

| arm | Amazon_clothing | Amazon_eletronics | Cora_full | dblp | mean of ds means | positive total |
|---|---|---|---|---|---|---|
| h9_clus_m20 | — | +1.12 ± 0.29 (10/10) | — | +0.40 ± 0.24 (10/10) | — | 20 |
| h9_rand_m50 | — | +1.29 ± 0.29 (10/10) | — | +0.39 ± 0.30 (8/10) | — | 18 |
| h9_rand_m20 | — | +1.04 ± 0.27 (10/10) | — | +0.34 ± 0.32 (7/10) | — | 17 |

## per-seed Δ

| arm | ds | seed 1212 | seed 1213 | seed 1214 | seed 1215 | seed 1216 | seed 1217 | seed 1218 | seed 1219 | seed 1220 | seed 1221 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| h9_clus_m20 | Amazon_eletronics | +0.91 | +0.64 | +1.28 | +1.25 | +0.97 | +1.06 | +1.34 | +1.75 | +1.03 | +0.94 |
| h9_clus_m20 | dblp | +0.41 | +0.11 | +0.39 | +0.14 | +0.71 | +0.23 | +0.94 | +0.28 | +0.30 | +0.50 |
| h9_rand_m50 | Amazon_eletronics | +0.98 | +0.78 | +1.55 | +1.50 | +1.51 | +1.30 | +1.69 | +1.54 | +0.98 | +1.11 |
| h9_rand_m50 | dblp | +0.57 | -0.05 | +0.31 | -0.02 | +0.68 | +0.16 | +0.97 | +0.50 | +0.50 | +0.30 |
| h9_rand_m20 | Amazon_eletronics | +0.78 | +0.67 | +1.21 | +1.47 | +0.85 | +1.11 | +1.06 | +1.48 | +0.88 | +0.87 |
| h9_rand_m20 | dblp | +0.39 | -0.02 | +0.22 | -0.08 | +0.67 | -0.02 | +1.03 | +0.36 | +0.45 | +0.38 |
