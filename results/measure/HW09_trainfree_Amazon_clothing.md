<!-- HW09-b: copy of results/hw_iislab4/trainfree/HW03a_trainfree_Amazon_clothing{,_1shot}.md (hw.trainfree --shot 5 / 1) -->

# HW03a train-free baselines — Amazon_clothing

5-way 5-shot ProtoNet (squared Euclidean), cached episodes: valid 50 / test 500. Accuracy %.
Â = data.adj = D^-1/2 (A+I) D^-1/2 (max |diff| vs independent build: 8.94e-08). No training.

| 특징 | 행 정규화 | val | test | test ci95 |
|---|---|---|---|---|
| X (raw) | 없음 | 66.25 | 66.58 | 1.07 |
| X (raw) | L2 | 84.20 | 87.61 | 0.77 |
| ÂX | 없음 | 70.10 | 73.64 | 1.01 |
| ÂX | L2 | 89.55 | 91.79 | 0.69 |
| Â²X | 없음 | 78.45 | 83.03 | 0.86 |
| Â²X | L2 | 90.30 | 92.72 | 0.65 |
| naq_base (학습, seio 10 seed 평균 ± 표준편차) | — | 87.92 | 90.74 ± 0.21 | — |
| nbr_raw (학습, seio 10 seed 평균 ± 표준편차) | — | 88.30 | 91.07 ± 0.23 | — |

# HW03a train-free baselines — Amazon_clothing

5-way 1-shot ProtoNet (squared Euclidean), cached episodes: valid 50 / test 500. Accuracy %.
Â = data.adj = D^-1/2 (A+I) D^-1/2 (max |diff| vs independent build: 8.94e-08). No training.

| 특징 | 행 정규화 | val | test | test ci95 |
|---|---|---|---|---|
| X (raw) | 없음 | 35.60 | 36.43 | 0.93 |
| X (raw) | L2 | 68.35 | 72.30 | 1.02 |
| ÂX | 없음 | 44.35 | 47.84 | 1.21 |
| ÂX | L2 | 76.20 | 82.47 | 0.99 |
| Â²X | 없음 | 54.75 | 60.55 | 1.26 |
| Â²X | L2 | 76.70 | 85.10 | 0.99 |
| naq_base (학습, seio 10 seed 평균 ± 표준편차) | — | 87.92 | 90.74 ± 0.21 | — |
| nbr_raw (학습, seio 10 seed 평균 ± 표준편차) | — | 88.30 | 91.07 ± 0.23 | — |
