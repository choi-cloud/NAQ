# Paired Δ vs `naq_base`

Δ = arm − ref test accuracy (%p), matched on seed. Cell: mean ± std (ddof=0), positive seeds k/n.
Sorted by the mean of the per-dataset Δ means (ties: total positive seeds).

| arm | Amazon_clothing | Cora_full | mean of ds means | positive total |
|---|---|---|---|---|
| nbr_bgrl | +0.11 ± 0.33 (3/5) | +0.97 ± 0.68 (5/5) | +0.54 | 8 |
| nbr_raw | +0.20 ± 0.09 (5/5) | +0.35 ± 0.79 (4/5) | +0.28 | 9 |
| hsan_p_orig_b1_raw | -1.28 ± 0.14 (0/5) | +0.05 ± 0.22 (3/5) | -0.62 | 3 |
| hsan_p_orig_b1 | -5.14 ± 1.25 (0/5) | -0.61 ± 0.87 (1/5) | -2.87 | 1 |
| hsan_p_orig_b2_raw | -4.61 ± 0.83 (0/5) | -2.15 ± 0.88 (0/5) | -3.38 | 0 |
| hsan_pn_orig_b2_cos | -4.41 ± 0.44 (0/5) | -2.51 ± 0.69 (0/5) | -3.46 | 0 |
| nbr_raw_cos | -5.54 ± 0.97 (0/5) | -3.97 ± 1.22 (0/5) | -4.75 | 0 |
| hsan_p_orig_b3_raw | -7.29 ± 0.83 (0/5) | -3.75 ± 0.90 (0/5) | -5.52 | 0 |
| hsan_p_orig_b2 | -9.99 ± 1.14 (0/5) | -4.73 ± 1.11 (0/5) | -7.36 | 0 |
| hsan_p_orig_b3 | -11.76 ± 0.95 (0/5) | -7.91 ± 1.74 (0/5) | -9.83 | 0 |
| hsan_pn_orig_b2 | -1.95 ± 0.43 (0/5) | -34.85 ± 21.92 (0/5) | -18.40 | 0 |
| hsan_p_both_b2 | -34.61 ± 1.52 (0/5) | -38.94 ± 1.35 (0/5) | -36.77 | 0 |
| hsan_pn_both_b2 | -42.52 ± 11.01 (0/5) | -37.96 ± 16.02 (0/5) | -40.24 | 0 |

## per-seed Δ

| arm | ds | seed 1212 | seed 1213 | seed 1214 | seed 1215 | seed 1216 |
|---|---|---|---|---|---|---|
| nbr_bgrl | Amazon_clothing | +0.42 | +0.32 | -0.38 | +0.39 | -0.19 |
| nbr_bgrl | Cora_full | +0.47 | +2.18 | +0.22 | +1.17 | +0.81 |
| nbr_raw | Amazon_clothing | +0.09 | +0.27 | +0.12 | +0.21 | +0.33 |
| nbr_raw | Cora_full | +0.32 | +1.48 | -0.96 | +0.71 | +0.21 |
| hsan_p_orig_b1_raw | Amazon_clothing | -1.48 | -1.39 | -1.08 | -1.24 | -1.23 |
| hsan_p_orig_b1_raw | Cora_full | +0.33 | +0.23 | -0.03 | -0.31 | +0.05 |
| hsan_p_orig_b1 | Amazon_clothing | -6.65 | -5.69 | -6.05 | -3.79 | -3.53 |
| hsan_p_orig_b1 | Cora_full | -0.15 | +0.57 | -0.36 | -1.97 | -1.12 |
| hsan_p_orig_b2_raw | Amazon_clothing | -5.73 | -4.81 | -5.15 | -3.39 | -4.00 |
| hsan_p_orig_b2_raw | Cora_full | -3.12 | -0.60 | -1.84 | -2.70 | -2.48 |
| hsan_pn_orig_b2_cos | Amazon_clothing | -4.39 | -3.78 | -5.12 | -4.53 | -4.23 |
| hsan_pn_orig_b2_cos | Cora_full | -2.52 | -1.45 | -3.62 | -2.38 | -2.56 |
| nbr_raw_cos | Amazon_clothing | -5.05 | -7.17 | -5.86 | -4.25 | -5.38 |
| nbr_raw_cos | Cora_full | -3.37 | -2.47 | -6.07 | -3.52 | -4.41 |
| hsan_p_orig_b3_raw | Amazon_clothing | -8.23 | -8.00 | -6.91 | -5.91 | -7.41 |
| hsan_p_orig_b3_raw | Cora_full | -4.50 | -2.33 | -4.51 | -3.04 | -4.38 |
| hsan_p_orig_b2 | Amazon_clothing | -12.07 | -10.23 | -9.28 | -9.55 | -8.80 |
| hsan_p_orig_b2 | Cora_full | -5.14 | -3.59 | -6.12 | -3.27 | -5.53 |
| hsan_p_orig_b3 | Amazon_clothing | -12.67 | -11.65 | -11.65 | -10.10 | -12.70 |
| hsan_p_orig_b3 | Cora_full | -7.55 | -6.41 | -9.37 | -5.80 | -10.40 |
| hsan_pn_orig_b2 | Amazon_clothing | -1.47 | -2.02 | -2.73 | -1.87 | -1.66 |
| hsan_pn_orig_b2 | Cora_full | -52.02 | -6.35 | -55.22 | -9.84 | -50.82 |
| hsan_p_both_b2 | Amazon_clothing | -37.09 | -35.58 | -33.95 | -33.45 | -32.97 |
| hsan_p_both_b2 | Cora_full | -37.01 | -39.49 | -40.99 | -39.14 | -38.06 |
| hsan_pn_both_b2 | Amazon_clothing | -51.07 | -22.25 | -39.31 | -50.22 | -49.75 |
| hsan_pn_both_b2 | Cora_full | -31.66 | -25.36 | -57.08 | -56.81 | -18.88 |
