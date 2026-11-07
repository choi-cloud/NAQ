# HW summary

test accuracy (%), seed mean ± std (ddof=0); n = number of seeds. 5-way 5-shot, 500 test tasks.

| arm | Amazon_clothing | Cora_full |
|---|---|---|
| hsan_p_both_b2 | 56.19 ± 1.57 (n=5) | 38.65 ± 1.66 (n=5) |
| hsan_p_orig_b2 | 80.81 ± 1.23 (n=5) | 72.86 ± 0.98 (n=5) |
| hsan_p_orig_b2_raw | 86.18 ± 0.80 (n=5) | 75.44 ± 0.43 (n=5) |
| hsan_pn_both_b2 | 48.27 ± 11.14 (n=5) | 39.63 ± 15.81 (n=5) |
| hsan_pn_orig_b2 | 88.84 ± 0.25 (n=5) | 42.74 ± 21.49 (n=5) |
| naq_base | 90.80 ± 0.20 (n=5) | 77.59 ± 0.70 (n=5) |
| nbr_raw | 91.00 ± 0.20 (n=5) | 77.94 ± 0.61 (n=5) |

## runs

| arm | ds | seed | best_epoch | val | test | ci95 | elapsed (s) |
|---|---|---|---|---|---|---|---|
| hsan_p_both_b2 | Amazon_clothing | 1212 | 160 | 51.25 | 53.51 | 0.87 | 106 |
| hsan_p_both_b2 | Amazon_clothing | 1213 | 160 | 52.45 | 55.34 | 0.90 | 108 |
| hsan_p_both_b2 | Amazon_clothing | 1214 | 160 | 55.20 | 57.17 | 0.92 | 109 |
| hsan_p_both_b2 | Amazon_clothing | 1215 | 160 | 53.20 | 57.22 | 0.94 | 110 |
| hsan_p_both_b2 | Amazon_clothing | 1216 | 160 | 54.40 | 57.70 | 0.89 | 133 |
| hsan_p_both_b2 | Cora_full | 1212 | 160 | 43.55 | 40.84 | 0.81 | 105 |
| hsan_p_both_b2 | Cora_full | 1213 | 160 | 39.15 | 36.70 | 0.78 | 104 |
| hsan_p_both_b2 | Cora_full | 1214 | 160 | 39.55 | 36.86 | 0.76 | 102 |
| hsan_p_both_b2 | Cora_full | 1215 | 160 | 42.35 | 38.81 | 0.80 | 105 |
| hsan_p_both_b2 | Cora_full | 1216 | 160 | 42.45 | 40.03 | 0.83 | 126 |
| hsan_p_orig_b2 | Amazon_clothing | 1212 | 160 | 75.80 | 78.53 | 0.84 | 100 |
| hsan_p_orig_b2 | Amazon_clothing | 1213 | 800 | 78.90 | 80.68 | 0.84 | 99 |
| hsan_p_orig_b2 | Amazon_clothing | 1214 | 640 | 78.00 | 81.85 | 0.82 | 92 |
| hsan_p_orig_b2 | Amazon_clothing | 1215 | 800 | 79.20 | 81.11 | 0.83 | 96 |
| hsan_p_orig_b2 | Amazon_clothing | 1216 | 960 | 80.10 | 81.87 | 0.80 | 118 |
| hsan_p_orig_b2 | Cora_full | 1212 | 480 | 81.80 | 72.70 | 0.89 | 99 |
| hsan_p_orig_b2 | Cora_full | 1213 | 480 | 80.75 | 72.61 | 0.89 | 97 |
| hsan_p_orig_b2 | Cora_full | 1214 | 1600 | 78.90 | 71.72 | 0.92 | 95 |
| hsan_p_orig_b2 | Cora_full | 1215 | 480 | 81.55 | 74.69 | 0.87 | 100 |
| hsan_p_orig_b2 | Cora_full | 1216 | 960 | 80.55 | 72.56 | 0.88 | 122 |
| hsan_p_orig_b2_raw | Amazon_clothing | 1212 | 1120 | 83.25 | 84.87 | 0.78 | 101 |
| hsan_p_orig_b2_raw | Amazon_clothing | 1213 | 800 | 83.90 | 86.11 | 0.81 | 104 |
| hsan_p_orig_b2_raw | Amazon_clothing | 1214 | 1760 | 83.95 | 85.97 | 0.76 | 104 |
| hsan_p_orig_b2_raw | Amazon_clothing | 1215 | 800 | 85.45 | 87.28 | 0.76 | 130 |
| hsan_p_orig_b2_raw | Amazon_clothing | 1216 | 960 | 84.20 | 86.67 | 0.74 | 69 |
| hsan_p_orig_b2_raw | Cora_full | 1212 | 480 | 83.35 | 74.72 | 0.87 | 95 |
| hsan_p_orig_b2_raw | Cora_full | 1213 | 960 | 84.20 | 75.59 | 0.86 | 91 |
| hsan_p_orig_b2_raw | Cora_full | 1214 | 1760 | 83.10 | 76.00 | 0.85 | 89 |
| hsan_p_orig_b2_raw | Cora_full | 1215 | 1120 | 83.40 | 75.25 | 0.85 | 94 |
| hsan_p_orig_b2_raw | Cora_full | 1216 | 640 | 83.95 | 75.61 | 0.87 | 104 |
| hsan_pn_both_b2 | Amazon_clothing | 1212 | 160 | 34.25 | 39.53 | 0.92 | 107 |
| hsan_pn_both_b2 | Amazon_clothing | 1213 | 160 | 64.10 | 68.67 | 0.96 | 104 |
| hsan_pn_both_b2 | Amazon_clothing | 1214 | 160 | 40.55 | 51.81 | 1.04 | 109 |
| hsan_pn_both_b2 | Amazon_clothing | 1215 | 160 | 36.95 | 40.45 | 0.93 | 111 |
| hsan_pn_both_b2 | Amazon_clothing | 1216 | 160 | 39.20 | 40.92 | 0.89 | 128 |
| hsan_pn_both_b2 | Cora_full | 1212 | 160 | 47.95 | 46.18 | 0.85 | 102 |
| hsan_pn_both_b2 | Cora_full | 1213 | 160 | 52.70 | 50.83 | 0.91 | 98 |
| hsan_pn_both_b2 | Cora_full | 1214 | 160 | 21.25 | 20.77 | 0.18 | 105 |
| hsan_pn_both_b2 | Cora_full | 1215 | 160 | 20.55 | 21.14 | 0.21 | 103 |
| hsan_pn_both_b2 | Cora_full | 1216 | 160 | 64.20 | 59.21 | 0.89 | 126 |
| hsan_pn_orig_b2 | Amazon_clothing | 1212 | 2000 | 87.05 | 89.12 | 0.72 | 99 |
| hsan_pn_orig_b2 | Amazon_clothing | 1213 | 2000 | 85.70 | 88.89 | 0.76 | 92 |
| hsan_pn_orig_b2 | Amazon_clothing | 1214 | 2000 | 86.50 | 88.39 | 0.74 | 96 |
| hsan_pn_orig_b2 | Amazon_clothing | 1215 | 1760 | 85.95 | 88.80 | 0.71 | 94 |
| hsan_pn_orig_b2 | Amazon_clothing | 1216 | 2000 | 86.15 | 89.01 | 0.72 | 117 |
| hsan_pn_orig_b2 | Cora_full | 1212 | 160 | 22.65 | 25.82 | 0.48 | 97 |
| hsan_pn_orig_b2 | Cora_full | 1213 | 320 | 75.45 | 69.84 | 0.91 | 90 |
| hsan_pn_orig_b2 | Cora_full | 1214 | 160 | 21.90 | 22.62 | 0.34 | 90 |
| hsan_pn_orig_b2 | Cora_full | 1215 | 160 | 75.15 | 68.12 | 0.90 | 89 |
| hsan_pn_orig_b2 | Cora_full | 1216 | 160 | 26.25 | 27.27 | 0.52 | 110 |
| naq_base | Amazon_clothing | 1212 | 1600 | 87.50 | 90.60 | 0.70 | 90 |
| naq_base | Amazon_clothing | 1213 | 1920 | 87.35 | 90.92 | 0.69 | 89 |
| naq_base | Amazon_clothing | 1214 | 2000 | 88.15 | 91.12 | 0.70 | 83 |
| naq_base | Amazon_clothing | 1215 | 2000 | 87.95 | 90.67 | 0.69 | 86 |
| naq_base | Amazon_clothing | 1216 | 1440 | 87.35 | 90.67 | 0.67 | 107 |
| naq_base | Cora_full | 1212 | 2000 | 86.60 | 77.84 | 0.85 | 80 |
| naq_base | Cora_full | 1213 | 1120 | 86.45 | 76.19 | 0.86 | 83 |
| naq_base | Cora_full | 1214 | 1760 | 85.85 | 77.85 | 0.83 | 74 |
| naq_base | Cora_full | 1215 | 2000 | 86.40 | 77.96 | 0.81 | 79 |
| naq_base | Cora_full | 1216 | 2000 | 86.10 | 78.09 | 0.84 | 96 |
| nbr_raw | Amazon_clothing | 1212 | 1440 | 88.60 | 90.69 | 0.68 | 95 |
| nbr_raw | Amazon_clothing | 1213 | 1760 | 87.80 | 91.19 | 0.71 | 93 |
| nbr_raw | Amazon_clothing | 1214 | 2000 | 88.60 | 91.25 | 0.70 | 87 |
| nbr_raw | Amazon_clothing | 1215 | 1920 | 87.80 | 90.88 | 0.70 | 89 |
| nbr_raw | Amazon_clothing | 1216 | 1600 | 87.75 | 91.00 | 0.68 | 111 |
| nbr_raw | Cora_full | 1212 | 2000 | 86.90 | 78.16 | 0.82 | 89 |
| nbr_raw | Cora_full | 1213 | 1920 | 86.70 | 77.67 | 0.84 | 91 |
| nbr_raw | Cora_full | 1214 | 2000 | 85.95 | 76.89 | 0.86 | 90 |
| nbr_raw | Cora_full | 1215 | 1920 | 87.25 | 78.67 | 0.81 | 93 |
| nbr_raw | Cora_full | 1216 | 2000 | 86.35 | 78.30 | 0.81 | 115 |
