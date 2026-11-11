<!-- HW09-b: copy of results/hw_iislab4/trainfree/HW03a_trainfree_dblp{,_1shot}.md (hw.trainfree --shot 5 / 1) -->

# HW03a train-free baselines — dblp

5-way 5-shot ProtoNet (squared Euclidean), cached episodes: valid 50 / test 500. Accuracy %.
Â = data.adj = D^-1/2 (A+I) D^-1/2 (max |diff| vs independent build: 2.24e-08). No training.

| 특징 | 행 정규화 | val | test | test ci95 |
|---|---|---|---|---|
| X (raw) | 없음 | 54.95 | 53.09 | 1.10 |
| X (raw) | L2 | 65.80 | 67.30 | 0.94 |
| ÂX | 없음 | 75.95 | 76.31 | 1.00 |
| ÂX | L2 | 84.55 | 83.45 | 0.88 |
| Â²X | 없음 | 80.25 | 80.92 | 0.93 |
| Â²X | L2 | 85.75 | 85.65 | 0.87 |

# HW03a train-free baselines — dblp

5-way 1-shot ProtoNet (squared Euclidean), cached episodes: valid 50 / test 500. Accuracy %.
Â = data.adj = D^-1/2 (A+I) D^-1/2 (max |diff| vs independent build: 2.24e-08). No training.

| 특징 | 행 정규화 | val | test | test ci95 |
|---|---|---|---|---|
| X (raw) | 없음 | 26.25 | 26.59 | 0.62 |
| X (raw) | L2 | 45.60 | 48.82 | 0.99 |
| ÂX | 없음 | 46.15 | 47.91 | 1.13 |
| ÂX | L2 | 69.60 | 69.17 | 1.19 |
| Â²X | 없음 | 55.70 | 58.42 | 1.22 |
| Â²X | L2 | 72.90 | 72.97 | 1.21 |
