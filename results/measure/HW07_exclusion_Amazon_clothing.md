# HW07 exclusion sets — Amazon_clothing (N 24919, seed 1212 pool)

Node values: mean over all nodes (node macro). Sets exclude q. Expected hits = m · |S(q)| / (N − 56).

| set | size mean / median / q90 | same-class ratio | m20 hits | m100 hits | m200 hits | m20 same | m100 same | m200 same |
|---|---|---|---|---|---|---|---|---|
| C(q) | 440.2 / 325 / 923 | 0.4560 | 0.354 | 1.771 | 3.541 | 0.1220 | 0.6100 | 1.2201 |
| F(q) | 162.7 / 134 / 271 | 0.4416 | 0.131 | 0.654 | 1.309 | 0.0604 | 0.3020 | 0.6040 |
| F_S(q) | 1028.9 / 1011 / 1081 | 0.2202 | 0.828 | 4.138 | 8.277 | 0.1824 | 0.9120 | 1.8240 |

- |F ∩ C| / |C| 0.3413, |F ∩ C| / |F| 0.5290
- random: expected same-class hits over all nodes m20 0.4034, m100 2.0172, m200 4.0344

Episode level (2000 episodes, 100000 (q, s⁺) pairs):
- C(s⁺) = C(q): 0.6782
- |C(s⁺) \ C(q)| mean (all pairs) 173.9, (non-empty) 540.5; same-class ratio (non-empty) 0.1870
- C(q) contains another pseudo-class support of the episode: 0.0659
