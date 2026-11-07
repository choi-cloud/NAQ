# Paired Δ vs `naq_base`

Δ = arm − ref test accuracy (%p), matched on seed. Cell: mean ± std (ddof=0), positive seeds k/n.
Sorted by the mean of the per-dataset Δ means (ties: total positive seeds).

| arm | Amazon_clothing | Cora_full | mean of ds means | positive total |
|---|---|---|---|---|
| nbr_raw | +0.20 ± 0.09 (5/5) | +0.35 ± 0.79 (4/5) | +0.28 | 9 |
| hsan_p_orig_b2_raw | -4.61 ± 0.83 (0/5) | -2.15 ± 0.88 (0/5) | -3.38 | 0 |
| hsan_p_orig_b2 | -9.99 ± 1.14 (0/5) | -4.73 ± 1.11 (0/5) | -7.36 | 0 |
| hsan_pn_orig_b2 | -1.95 ± 0.43 (0/5) | -34.85 ± 21.92 (0/5) | -18.40 | 0 |
| hsan_p_both_b2 | -34.61 ± 1.52 (0/5) | -38.94 ± 1.35 (0/5) | -36.77 | 0 |
| hsan_pn_both_b2 | -42.52 ± 11.01 (0/5) | -37.96 ± 16.02 (0/5) | -40.24 | 0 |

## per-seed Δ

| arm | ds | seed 1212 | seed 1213 | seed 1214 | seed 1215 | seed 1216 |
|---|---|---|---|---|---|---|
| nbr_raw | Amazon_clothing | +0.09 | +0.27 | +0.12 | +0.21 | +0.33 |
| nbr_raw | Cora_full | +0.32 | +1.48 | -0.96 | +0.71 | +0.21 |
| hsan_p_orig_b2_raw | Amazon_clothing | -5.73 | -4.81 | -5.15 | -3.39 | -4.00 |
| hsan_p_orig_b2_raw | Cora_full | -3.12 | -0.60 | -1.84 | -2.70 | -2.48 |
| hsan_p_orig_b2 | Amazon_clothing | -12.07 | -10.23 | -9.28 | -9.55 | -8.80 |
| hsan_p_orig_b2 | Cora_full | -5.14 | -3.59 | -6.12 | -3.27 | -5.53 |
| hsan_pn_orig_b2 | Amazon_clothing | -1.47 | -2.02 | -2.73 | -1.87 | -1.66 |
| hsan_pn_orig_b2 | Cora_full | -52.02 | -6.35 | -55.22 | -9.84 | -50.82 |
| hsan_p_both_b2 | Amazon_clothing | -37.09 | -35.58 | -33.95 | -33.45 | -32.97 |
| hsan_p_both_b2 | Cora_full | -37.01 | -39.49 | -40.99 | -39.14 | -38.06 |
| hsan_pn_both_b2 | Amazon_clothing | -51.07 | -22.25 | -39.31 | -50.22 | -49.75 |
| hsan_pn_both_b2 | Cora_full | -31.66 | -25.36 | -57.08 | -56.81 | -18.88 |
