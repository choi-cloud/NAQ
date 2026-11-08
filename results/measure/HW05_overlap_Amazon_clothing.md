# HW05 idea-A neighborhood overlap — Amazon_clothing

graph: 24919 nodes, 91680 undirected edges, degree median 3; chance 0.0202
pairs with overlap > 0: graph 1049476, PPR top-20 5457216

## graph Jaccard

| θ | pairs J ≥ θ | same class (merge precision) | coverage | partners median (all / covered) |
|---|---|---|---|---|
| 0.1 | 704381 | 0.6040 | 0.9555 | 31 / 34 |
| 0.2 | 472103 | 0.6237 | 0.9360 | 16 / 18 |
| 0.3 | 297941 | 0.6541 | 0.8813 | 8 / 11 |
| 0.5 | 173810 | 0.6756 | 0.7154 | 3 / 7 |

## ppr Jaccard

| θ | pairs J ≥ θ | same class (merge precision) | coverage | partners median (all / covered) |
|---|---|---|---|---|
| 0.1 | 2256429 | 0.4572 | 0.9698 | 120 / 128 |
| 0.2 | 1465848 | 0.5210 | 0.9698 | 68 / 70 |
| 0.3 | 994200 | 0.5847 | 0.9692 | 42 / 44 |
| 0.5 | 552370 | 0.6510 | 0.9385 | 21 / 24 |

## J of random pairs

| set | pairs | zero frac | median | q90 | mean |
|---|---|---|---|---|---|
| graph | same_class | 0.9045 | 0.0000 | 0.0000 | 0.0263 |
| graph | different_class | 0.9988 | 0.0000 | 0.0000 | 0.0003 |
| ppr | same_class | 0.7167 | 0.0000 | 0.2903 | 0.0748 |
| ppr | different_class | 0.9880 | 0.0000 | 0.0000 | 0.0015 |
