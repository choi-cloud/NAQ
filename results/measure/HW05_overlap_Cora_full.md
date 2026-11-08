# HW05 idea-A neighborhood overlap — Cora_full

graph: 19793 nodes, 63421 undirected edges, degree median 4; chance 0.0217
pairs with overlap > 0: graph 785754, PPR top-20 6939461

## graph Jaccard

| θ | pairs J ≥ θ | same class (merge precision) | coverage | partners median (all / covered) |
|---|---|---|---|---|
| 0.1 | 255474 | 0.4719 | 0.9629 | 17 / 17 |
| 0.2 | 81169 | 0.5238 | 0.8923 | 4 / 5 |
| 0.3 | 29395 | 0.5470 | 0.6563 | 1 / 2 |
| 0.5 | 10928 | 0.5454 | 0.3374 | 0 / 1 |

## ppr Jaccard

| θ | pairs J ≥ θ | same class (merge precision) | coverage | partners median (all / covered) |
|---|---|---|---|---|
| 0.1 | 1077781 | 0.4181 | 0.9753 | 74 / 76 |
| 0.2 | 418891 | 0.4981 | 0.9751 | 29 / 30 |
| 0.3 | 186898 | 0.5374 | 0.9637 | 13 / 14 |
| 0.5 | 62734 | 0.5695 | 0.8002 | 4 / 5 |

## J of random pairs

| set | pairs | zero frac | median | q90 | mean |
|---|---|---|---|---|---|
| graph | same_class | 0.9299 | 0.0000 | 0.0000 | 0.0079 |
| graph | different_class | 0.9976 | 0.0000 | 0.0000 | 0.0002 |
| ppr | same_class | 0.6658 | 0.0000 | 0.1111 | 0.0354 |
| ppr | different_class | 0.9710 | 0.0000 | 0.0000 | 0.0016 |
