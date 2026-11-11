# HW03a train-free baselines — Cora_full

5-way 5-shot ProtoNet (squared Euclidean), cached episodes: valid 50 / test 500. Accuracy %.
Â = data.adj = D^-1/2 (A+I) D^-1/2 (max |diff| vs independent build: 8.94e-08). No training.

| 특징 | 행 정규화 | val | test | test ci95 |
|---|---|---|---|---|
| X (raw) | 없음 | 49.90 | 48.07 | 0.92 |
| X (raw) | L2 | 67.80 | 60.50 | 0.83 |
| ÂX | 없음 | 72.95 | 64.51 | 1.02 |
| ÂX | L2 | 85.25 | 77.41 | 0.84 |
| Â²X | 없음 | 77.00 | 67.57 | 1.01 |
| Â²X | L2 | 87.70 | 80.12 | 0.81 |
| naq_base (학습, seio 10 seed 평균 ± 표준편차) | — | 86.44 | 77.85 ± 0.47 | — |
| nbr_raw (학습, seio 10 seed 평균 ± 표준편차) | — | 86.44 | 78.12 ± 0.35 | — |
