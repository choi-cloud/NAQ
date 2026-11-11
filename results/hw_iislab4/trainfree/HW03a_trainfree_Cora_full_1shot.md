# HW03a train-free baselines — Cora_full

5-way 1-shot ProtoNet (squared Euclidean), cached episodes: valid 50 / test 500. Accuracy %.
Â = data.adj = D^-1/2 (A+I) D^-1/2 (max |diff| vs independent build: 8.94e-08). No training.

| 특징 | 행 정규화 | val | test | test ci95 |
|---|---|---|---|---|
| X (raw) | 없음 | 22.85 | 23.30 | 0.44 |
| X (raw) | L2 | 47.35 | 41.44 | 0.84 |
| ÂX | 없음 | 34.80 | 32.64 | 0.88 |
| ÂX | L2 | 53.95 | 50.14 | 1.09 |
| Â²X | 없음 | 41.60 | 38.55 | 1.02 |
| Â²X | L2 | 56.75 | 51.76 | 1.20 |
| naq_base (학습, seio 10 seed 평균 ± 표준편차) | — | 86.44 | 77.85 ± 0.47 | — |
| nbr_raw (학습, seio 10 seed 평균 ± 표준편차) | — | 86.44 | 78.12 ± 0.35 | — |
