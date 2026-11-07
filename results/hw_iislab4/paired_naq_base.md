# Paired Δ vs `naq_base`

Δ = arm − ref test accuracy (%p), matched on seed. Cell: mean ± std (ddof=0), positive seeds k/n.
Sorted by the mean of the per-dataset Δ means (ties: total positive seeds).

| arm | Amazon_clothing | Cora_full | mean of ds means | positive total |
|---|---|---|---|---|
| nbr_raw | +0.21 ± 0.14 (4/5) | +0.40 ± 0.24 (5/5) | +0.31 | 9 |

## per-seed Δ

| arm | ds | seed 1212 | seed 1213 | seed 1214 | seed 1215 | seed 1216 |
|---|---|---|---|---|---|---|
| nbr_raw | Amazon_clothing | +0.29 | +0.35 | +0.16 | +0.28 | -0.03 |
| nbr_raw | Cora_full | +0.19 | +0.57 | +0.33 | +0.77 | +0.15 |
