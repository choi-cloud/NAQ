# HW03b per-dimension Dirichlet energy — Cora_full

X: 19793 × 8710, torch.float32, binary True, integer True, range 0–1, distinct values 2, density 0.00653, row L1-normalized False, row L2-normalized False
row sum min/median/max 2 / 54 / 293
graph: 19793 nodes, 63421 undirected edges (hw.structure.normalize_graph(data.edge_index))
dims: total 8710, zero norm 0 (excluded), used 8710, E_raw = 0 among used 0

| 값 | min | q25 | median | q75 | max |
|---|---|---|---|---|---|
| E_raw | 7 | 95.25 | 219 | 656 | 28353 |
| E_rq | 0.7 | 4.8125 | 5.875 | 7 | 27.5385 |
| nnz | 10 | 16 | 35 | 105 | 10002 |

| Spearman | ρ |
|---|---|
| E_raw_vs_nnz | 0.9503 |
| E_raw_vs_E_rq | 0.3703 |
| E_rq_vs_nnz | 0.1029 |

| K | top-K overlap E_raw ∩ nnz | top-K overlap E_raw ∩ E_rq |
|---|---|---|
| 16 | 1.0000 | 0.0000 |
| 64 | 0.8750 | 0.0000 |
| 512 | 0.9297 | 0.0000 |
