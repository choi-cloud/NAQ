# HW summary

test accuracy (%), seed mean ± std (ddof=0); n = number of seeds. 5-way 5-shot, 500 test tasks.

| arm | Amazon_clothing | Cora_full |
|---|---|---|
| naq_base | 90.67 ± 0.00 (n=1) | 77.78 ± 0.00 (n=1) |
| nbr_raw | 90.86 ± 0.00 (n=1) | 78.10 ± 0.00 (n=1) |

## runs

| arm | ds | seed | best_epoch | val | test | ci95 | elapsed (s) |
|---|---|---|---|---|---|---|---|
| naq_base | Amazon_clothing | 1212 | 1600 | 87.70 | 90.67 | 0.69 | 60 |
| naq_base | Cora_full | 1212 | 2000 | 86.90 | 77.78 | 0.84 | 48 |
| nbr_raw | Amazon_clothing | 1212 | 1280 | 88.45 | 90.86 | 0.69 | 64 |
| nbr_raw | Cora_full | 1212 | 2000 | 86.65 | 78.10 | 0.83 | 48 |
