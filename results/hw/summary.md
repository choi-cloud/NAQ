# HW summary

test accuracy (%), seed mean ± std (ddof=0); n = number of seeds. 5-way 5-shot, 500 test tasks.

| arm | Amazon_clothing | Cora_full |
|---|---|---|
| naq_base | 90.60 ± 0.00 (n=1) | 77.84 ± 0.00 (n=1) |
| nbr_raw | 90.69 ± 0.00 (n=1) | 78.16 ± 0.00 (n=1) |

## runs

| arm | ds | seed | best_epoch | val | test | ci95 | elapsed (s) |
|---|---|---|---|---|---|---|---|
| naq_base | Amazon_clothing | 1212 | 1600 | 87.50 | 90.60 | 0.70 | 90 |
| naq_base | Cora_full | 1212 | 2000 | 86.60 | 77.84 | 0.85 | 80 |
| nbr_raw | Amazon_clothing | 1212 | 1440 | 88.60 | 90.69 | 0.68 | 95 |
| nbr_raw | Cora_full | 1212 | 2000 | 86.90 | 78.16 | 0.82 | 89 |
