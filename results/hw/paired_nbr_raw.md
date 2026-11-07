# Paired Δ vs `nbr_raw`

Δ = arm − ref test accuracy (%p), matched on seed. Cell: mean ± std (ddof=0), positive seeds k/n.
Sorted by the mean of the per-dataset Δ means (ties: total positive seeds).

| arm | Amazon_clothing | Cora_full | mean of ds means | positive total |
|---|---|---|---|---|
| naq_base | -0.20 ± 0.09 (0/5) | -0.35 ± 0.79 (1/5) | -0.28 | 1 |
| hsan_p_orig_b2_raw | -4.82 ± 0.78 (0/5) | -2.50 ± 0.95 (0/5) | -3.66 | 0 |
| hsan_p_orig_b2 | -10.19 ± 1.09 (0/5) | -5.08 ± 0.60 (0/5) | -7.64 | 0 |
| hsan_pn_orig_b2 | -2.16 ± 0.42 (0/5) | -35.20 ± 21.28 (0/5) | -18.68 | 0 |
| hsan_p_both_b2 | -34.81 ± 1.47 (0/5) | -39.29 ± 1.31 (0/5) | -37.05 | 0 |
| hsan_pn_both_b2 | -42.73 ± 10.99 (0/5) | -38.31 ± 15.67 (0/5) | -40.52 | 0 |

## per-seed Δ

| arm | ds | seed 1212 | seed 1213 | seed 1214 | seed 1215 | seed 1216 |
|---|---|---|---|---|---|---|
| naq_base | Amazon_clothing | -0.09 | -0.27 | -0.12 | -0.21 | -0.33 |
| naq_base | Cora_full | -0.32 | -1.48 | +0.96 | -0.71 | -0.21 |
| hsan_p_orig_b2_raw | Amazon_clothing | -5.82 | -5.08 | -5.28 | -3.60 | -4.33 |
| hsan_p_orig_b2_raw | Cora_full | -3.44 | -2.08 | -0.89 | -3.42 | -2.69 |
| hsan_p_orig_b2 | Amazon_clothing | -12.16 | -10.50 | -9.40 | -9.77 | -9.13 |
| hsan_p_orig_b2 | Cora_full | -5.45 | -5.07 | -5.17 | -3.98 | -5.74 |
| hsan_pn_orig_b2 | Amazon_clothing | -1.56 | -2.29 | -2.86 | -2.08 | -1.99 |
| hsan_pn_orig_b2 | Cora_full | -52.33 | -7.83 | -54.27 | -10.55 | -51.03 |
| hsan_p_both_b2 | Amazon_clothing | -37.18 | -35.85 | -34.08 | -33.66 | -33.30 |
| hsan_p_both_b2 | Cora_full | -37.32 | -40.97 | -40.03 | -39.86 | -38.27 |
| hsan_pn_both_b2 | Amazon_clothing | -51.16 | -22.52 | -39.44 | -50.43 | -50.08 |
| hsan_pn_both_b2 | Cora_full | -31.98 | -26.84 | -56.12 | -57.53 | -19.09 |
