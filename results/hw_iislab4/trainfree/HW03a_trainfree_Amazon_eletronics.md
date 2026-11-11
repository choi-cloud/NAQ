# HW03a train-free baselines — Amazon_eletronics

5-way 5-shot ProtoNet (squared Euclidean), cached episodes: valid 50 / test 500. Accuracy %.
Â = data.adj = D^-1/2 (A+I) D^-1/2 (max |diff| vs independent build: 8.94e-08). No training.

| 특징 | 행 정규화 | val | test | test ci95 |
|---|---|---|---|---|
| X (raw) | 없음 | 62.80 | 62.22 | 1.07 |
| X (raw) | L2 | 89.20 | 87.99 | 0.74 |
| ÂX | 없음 | 67.95 | 64.66 | 1.07 |
| ÂX | L2 | 89.90 | 87.75 | 0.70 |
| Â²X | 없음 | 74.10 | 69.86 | 1.07 |
| Â²X | L2 | 90.75 | 88.64 | 0.66 |
