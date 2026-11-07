# Paired Δ vs `nbr_raw_cos`

Δ = arm − ref test accuracy (%p), matched on seed. Cell: mean ± std (ddof=0), positive seeds k/n.
Sorted by the mean of the per-dataset Δ means (ties: total positive seeds).

| arm | Amazon_clothing | Cora_full | mean of ds means | positive total |
|---|---|---|---|---|
| nbr_raw | +5.74 ± 0.99 (5/5) | +4.32 ± 0.50 (5/5) | +5.03 | 10 |
| naq_base | +5.54 ± 0.97 (5/5) | +3.97 ± 1.22 (5/5) | +4.75 | 10 |
| hsan_p_orig_b1_raw | +4.26 ± 0.96 (5/5) | +4.02 ± 1.16 (5/5) | +4.14 | 10 |
| hsan_p_orig_b1 | +0.40 ± 1.23 (3/5) | +3.36 ± 1.34 (5/5) | +1.88 | 8 |
| hsan_p_orig_b2_raw | +0.93 ± 0.99 (4/5) | +1.82 ± 1.36 (5/5) | +1.37 | 9 |
| hsan_pn_orig_b2_cos | +1.13 ± 1.22 (4/5) | +1.46 ± 0.60 (5/5) | +1.30 | 9 |
| hsan_p_orig_b3_raw | -1.75 ± 0.83 (0/5) | +0.22 ± 0.87 (4/5) | -0.77 | 4 |
| hsan_p_orig_b2 | -4.45 ± 1.51 (0/5) | -0.76 ± 0.75 (1/5) | -2.60 | 1 |
| hsan_p_orig_b3 | -6.22 ± 1.14 (0/5) | -3.94 ± 1.22 (0/5) | -5.08 | 0 |
| hsan_pn_orig_b2 | +3.59 ± 0.91 (5/5) | -30.88 ± 21.09 (0/5) | -13.65 | 5 |
| hsan_p_both_b2 | -29.07 ± 1.58 (0/5) | -34.97 ± 1.28 (0/5) | -32.02 | 0 |
| hsan_pn_both_b2 | -36.98 ± 11.91 (0/5) | -33.99 ± 15.48 (0/5) | -35.49 | 0 |

## per-seed Δ

| arm | ds | seed 1212 | seed 1213 | seed 1214 | seed 1215 | seed 1216 |
|---|---|---|---|---|---|---|
| nbr_raw | Amazon_clothing | +5.14 | +7.44 | +5.98 | +4.46 | +5.71 |
| nbr_raw | Cora_full | +3.69 | +3.95 | +5.11 | +4.23 | +4.61 |
| naq_base | Amazon_clothing | +5.05 | +7.17 | +5.86 | +4.25 | +5.38 |
| naq_base | Cora_full | +3.37 | +2.47 | +6.07 | +3.52 | +4.41 |
| hsan_p_orig_b1_raw | Amazon_clothing | +3.56 | +5.78 | +4.78 | +3.00 | +4.15 |
| hsan_p_orig_b1_raw | Cora_full | +3.69 | +2.70 | +6.04 | +3.20 | +4.45 |
| hsan_p_orig_b1 | Amazon_clothing | -1.60 | +1.48 | -0.19 | +0.45 | +1.85 |
| hsan_p_orig_b1 | Cora_full | +3.22 | +3.05 | +5.71 | +1.55 | +3.28 |
| hsan_p_orig_b2_raw | Amazon_clothing | -0.68 | +2.36 | +0.70 | +0.86 | +1.38 |
| hsan_p_orig_b2_raw | Cora_full | +0.25 | +1.88 | +4.23 | +0.81 | +1.92 |
| hsan_pn_orig_b2_cos | Amazon_clothing | +0.66 | +3.39 | +0.73 | -0.28 | +1.15 |
| hsan_pn_orig_b2_cos | Cora_full | +0.85 | +1.02 | +2.45 | +1.14 | +1.84 |
| hsan_p_orig_b3_raw | Amazon_clothing | -3.19 | -0.83 | -1.05 | -1.66 | -2.03 |
| hsan_p_orig_b3_raw | Cora_full | -1.13 | +0.14 | +1.56 | +0.48 | +0.02 |
| hsan_p_orig_b2 | Amazon_clothing | -7.02 | -3.07 | -3.42 | -5.31 | -3.42 |
| hsan_p_orig_b2 | Cora_full | -1.77 | -1.11 | -0.05 | +0.25 | -1.12 |
| hsan_p_orig_b3 | Amazon_clothing | -7.62 | -4.48 | -5.80 | -5.86 | -7.32 |
| hsan_p_orig_b3 | Cora_full | -4.18 | -3.94 | -3.30 | -2.28 | -6.00 |
| hsan_pn_orig_b2 | Amazon_clothing | +3.58 | +5.14 | +3.12 | +2.38 | +3.72 |
| hsan_pn_orig_b2 | Cora_full | -48.65 | -3.88 | -49.15 | -6.32 | -46.42 |
| hsan_p_both_b2 | Amazon_clothing | -32.04 | -28.41 | -28.09 | -29.20 | -27.59 |
| hsan_p_both_b2 | Cora_full | -33.64 | -37.02 | -34.92 | -35.62 | -33.66 |
| hsan_pn_both_b2 | Amazon_clothing | -46.02 | -15.08 | -33.46 | -45.98 | -44.37 |
| hsan_pn_both_b2 | Cora_full | -28.29 | -22.89 | -51.01 | -53.30 | -14.48 |
