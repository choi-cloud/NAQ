# HW summary

test accuracy (%), seed mean ± std (ddof=0); n = number of seeds. 5-way 5-shot, 500 test tasks.

| arm | Amazon_clothing | Cora_full |
|---|---|---|
| naq_base | 90.77 ± 0.18 (n=5) | 77.68 ± 0.43 (n=5) |
| nbr_raw | 90.98 ± 0.22 (n=5) | 78.08 ± 0.41 (n=5) |

## runs

| arm | ds | seed | best_epoch | val | test | ci95 | elapsed (s) |
|---|---|---|---|---|---|---|---|
| naq_base | Amazon_clothing | 1212 | 1600 | 87.75 | 90.60 | 0.70 | 66 |
| naq_base | Amazon_clothing | 1213 | 2000 | 87.55 | 90.80 | 0.68 | 66 |
| naq_base | Amazon_clothing | 1214 | 2000 | 88.10 | 91.11 | 0.70 | 68 |
| naq_base | Amazon_clothing | 1215 | 1920 | 87.90 | 90.66 | 0.69 | 67 |
| naq_base | Amazon_clothing | 1216 | 1600 | 87.45 | 90.67 | 0.70 | 67 |
| naq_base | Cora_full | 1212 | 1760 | 86.60 | 77.95 | 0.82 | 52 |
| naq_base | Cora_full | 1213 | 1920 | 86.30 | 77.11 | 0.85 | 52 |
| naq_base | Cora_full | 1214 | 1600 | 86.15 | 77.23 | 0.83 | 52 |
| naq_base | Cora_full | 1215 | 2000 | 86.25 | 77.90 | 0.82 | 53 |
| naq_base | Cora_full | 1216 | 2000 | 86.15 | 78.21 | 0.83 | 51 |
| nbr_raw | Amazon_clothing | 1212 | 1280 | 88.60 | 90.89 | 0.68 | 72 |
| nbr_raw | Amazon_clothing | 1213 | 1760 | 87.70 | 91.15 | 0.72 | 72 |
| nbr_raw | Amazon_clothing | 1214 | 2000 | 88.45 | 91.26 | 0.70 | 72 |
| nbr_raw | Amazon_clothing | 1215 | 2000 | 88.05 | 90.94 | 0.69 | 72 |
| nbr_raw | Amazon_clothing | 1216 | 960 | 87.75 | 90.64 | 0.71 | 73 |
| nbr_raw | Cora_full | 1212 | 2000 | 86.70 | 78.14 | 0.83 | 59 |
| nbr_raw | Cora_full | 1213 | 1920 | 86.55 | 77.68 | 0.84 | 57 |
| nbr_raw | Cora_full | 1214 | 1440 | 86.00 | 77.56 | 0.83 | 57 |
| nbr_raw | Cora_full | 1215 | 1920 | 87.20 | 78.67 | 0.82 | 58 |
| nbr_raw | Cora_full | 1216 | 2000 | 86.30 | 78.36 | 0.81 | 57 |
