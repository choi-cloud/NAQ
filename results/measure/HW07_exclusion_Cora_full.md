# HW07 exclusion sets — Cora_full (N 19793, seed 1212 pool)

Node values: mean over all nodes (node macro). Sets exclude q. Expected hits = m · |S(q)| / (N − 56).

| set | size mean / median / q90 | same-class ratio | m20 hits | m100 hits | m200 hits | m20 same | m100 same | m200 same |
|---|---|---|---|---|---|---|---|---|
| C(q) | 307.3 / 255 / 480 | 0.3562 | 0.311 | 1.557 | 3.113 | 0.0887 | 0.4435 | 0.8869 |
| F(q) | 172.3 / 136 / 297 | 0.2408 | 0.175 | 0.873 | 1.746 | 0.0457 | 0.2284 | 0.4568 |
| F_S(q) | 1054.1 / 1025 / 1150 | 0.0979 | 1.068 | 5.341 | 10.682 | 0.1062 | 0.5308 | 1.0615 |

- |F ∩ C| / |C| 0.2128, |F ∩ C| / |F| 0.2545
- random: expected same-class hits over all nodes m20 0.4342, m100 2.1709, m200 4.3419

Episode level (2000 episodes, 100000 (q, s⁺) pairs):
- C(s⁺) = C(q): 0.3581
- |C(s⁺) \ C(q)| mean (all pairs) 209.6, (non-empty) 326.5; same-class ratio (non-empty) 0.1030
- C(q) contains another pseudo-class support of the episode: 0.0542
