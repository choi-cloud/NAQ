# HW03b per-dimension Dirichlet energy — Amazon_clothing

X: 24919 × 9034, torch.float32, binary False, integer True, range 0–99, distinct values 32, density 0.00351, row L1-normalized False, row L2-normalized False
row sum min/median/max 2 / 32 / 527
graph: 24919 nodes, 91680 undirected edges (hw.structure.normalize_graph(data.edge_index))
dims: total 9034, zero norm 0 (excluded), used 9034, E_raw = 0 among used 10

| 값 | min | q25 | median | q75 | max |
|---|---|---|---|---|---|
| E_raw | 0 | 58 | 173 | 595 | 302841 |
| E_rq | 0 | 2.86517 | 4.77896 | 7.42044 | 125.393 |
| nnz | 1 | 8 | 18 | 53.75 | 5463 |

| Spearman | ρ |
|---|---|
| E_raw_vs_nnz | 0.8069 |
| E_raw_vs_E_rq | 0.5327 |
| E_rq_vs_nnz | 0.1919 |

| K | top-K overlap E_raw ∩ nnz | top-K overlap E_raw ∩ E_rq |
|---|---|---|
| 16 | 0.3750 | 0.0000 |
| 64 | 0.5312 | 0.0000 |
| 512 | 0.7148 | 0.0645 |
