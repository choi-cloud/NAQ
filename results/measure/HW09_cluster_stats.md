# HW09 cluster statistics (before training; measurement-only label use)

k-means: Â²X L2 정규화, k 100, n_init 4, random_state 20261108 (`cache/kmeans/<ds>_k100.pt`). 정의는 `hw/cluster_stats_hw09.py`.

| 항목 | Cora_full | Amazon_clothing | dblp | Amazon_eletronics |
|---|---|---|---|---|
| 노드 수 | 19793 | 24919 | 40672 | 42318 |
| cluster (k 100): 크기 최소 / 중앙값 / 최대 | 4 / 170 / 1083 | 11 / 185 / 1575 | 78 / 368 / 1394 | 58 / 352 / 2347 |
| cluster: 같은 군집 내 같은 클래스 비율 (노드 macro) | 0.3562 | 0.4560 | 0.1841 | 0.2768 |
| cluster: 다른 군집 노드 중 같은 클래스 비율 (노드 macro) | 0.0175 | 0.0143 | 0.0080 | 0.0049 |
| random 기대 FN rate Σ p_c² | 0.0217 | 0.0202 | 0.0100 | 0.0081 |
| NaQ-Feat top-10 순도, 전체 노드 (노드 macro) | 0.3498 | 0.5972 | 0.1572 | 0.6454 |
| NaQ-Feat top-10 순도, seed 1212 학습 pool 4000 episode (`query_eq_support`, 쌍 비율) | 0.3496 | 0.5989 | 0.1569 | 0.6465 |

HW06 G-table (`results/hw_iislab4/HW06_tables.json`) 값:

| 항목 | Cora_full | Amazon_clothing |
|---|---|---|
| cluster (k 100): 크기 최소 / 중앙값 / 최대 | 4 / 170 / 1083 | 11 / 185 / 1575 |
| cluster: 같은 군집 내 같은 클래스 비율 (노드 macro) | 0.3562 | 0.4560 |
| cluster: 다른 군집 노드 중 같은 클래스 비율 (노드 macro) | 0.0175 | 0.0143 |
| random 기대 FN rate Σ p_c² | 0.0217 | 0.0202 |
