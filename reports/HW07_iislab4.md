# HW07 [iislab4] — cluster 검증 (무작위 군집 대조 · k-means seed · anchor 군집 · m 100/200) · far 재확인 (m 200, far_strong) · 제외 집합 측정

2026-11-09 · 서버 iislab4 · 브랜치 `hl/hsan-weighting` · 작업 시작 HEAD `17dd625` · 지시서 `instructions/HW07_iislab4.md` · 사전 등록 HWP070–HWP078

모든 학습 arm은 `valid_tasks: 500`, `episodes: 4000`, `geom_diag: true`, base `naq_base`입니다. 주 결과는 `test_acc`(500 기준 best epoch의 test)입니다. 약칭: rand20 = `e4k_rand_m20`, clus20 = `e4k_clus_m20`(둘 다 HW06). 숫자를 두 개 적을 때는 Cora_full / Amazon_clothing 순서입니다.

## 1. 결론

1. **cluster 검증 (m 20, `− rand20`):** q의 군집 제외 +0.39 (10/10) / +0.00, 무관한 군집 제외(`shuf`) +0.01 / +0.01, k-means seed 2 +0.31 (10/10) / −0.04, 양성 support의 군집 제외(`anc`) +0.03 / +0.04입니다. `clus20 − shuf`는 +0.37 (9/10) / −0.01입니다.
2. **큰 m (같은 m의 무작위 대비):** cluster m 100 +0.43 / −0.18, m 200 +0.39 / −0.31, far m 200 +0.09 / −0.31, far_strong m 200 +0.11 / +0.00입니다. Cora_full 최고값은 `clus_m200` 80.58, Amazon_clothing의 HW07 arm 최고값은 `farS_m200` 92.04(`rand_m200`과 같음)입니다.
3. **제외 집합:** 무작위 음성 m 20이 C(q)\{q}에 떨어지는 기대 개수는 query당 0.311 / 0.354이고, 그중 같은 클래스는 0.0887 / 0.1220개입니다. 무작위 m 20의 전체 같은 클래스 기대 개수는 0.4342 / 0.4034개입니다.

## 2. 핵심 표

### 2.1 test 정확도

- 값: test 정확도(%), 5-way 5-shot, test 500 task. 셀은 10 seed의 평균 ± 표준편차(ddof=0)입니다.
- 붕괴: 500 기준 best validation ≤ 25%인 run 수. NaN: status `nan`인 run 수. completed와 붕괴·NaN은 두 데이터셋 합계입니다.
- 출처: `results/hw_iislab4/summary.md`. HW06 행은 같은 결과 루트의 HW06 run입니다.

| arm | 출처 | Cora_full | Amazon_clothing | completed | 붕괴 | NaN |
|---|---|---|---|---|---|---|
| `e4k_naq` | HW06 | 78.69 ± 0.47 | 91.38 ± 0.26 | 20/20 | 0 | 0 |
| `e4k_rand_m20` | HW06 | 79.84 ± 0.37 | 91.99 ± 0.19 | 20/20 | 0 | 0 |
| `e4k_rand_m100` | HW06 | 80.09 ± 0.32 | 92.04 ± 0.22 | 20/20 | 0 | 0 |
| `e4k_rand_m200` | HW06 | 80.20 ± 0.35 | 92.04 ± 0.15 | 20/20 | 0 | 0 |
| `e4k_clus_m20` | HW06 | 80.23 ± 0.31 | 91.99 ± 0.17 | 20/20 | 0 | 0 |
| `e4k_far_m20` | HW06 | 79.91 ± 0.41 | 91.94 ± 0.25 | 20/20 | 0 | 0 |
| `e4k_clus_shuf_m20` | HW07 | 79.85 ± 0.45 | 92.01 ± 0.18 | 20/20 | 0 | 0 |
| `e4k_clus_ks2_m20` | HW07 | 80.15 ± 0.29 | 91.95 ± 0.16 | 20/20 | 0 | 0 |
| `e4k_clus_anc_m20` | HW07 | 79.87 ± 0.42 | 92.03 ± 0.18 | 20/20 | 0 | 0 |
| `e4k_clus_m100` | HW07 | 80.52 ± 0.23 | 91.86 ± 0.21 | 20/20 | 0 | 0 |
| `e4k_clus_m200` | HW07 | 80.58 ± 0.27 | 91.73 ± 0.24 | 20/20 | 0 | 0 |
| `e4k_far_m200` | HW07 | 80.28 ± 0.26 | 91.73 ± 0.13 | 20/20 | 0 | 0 |
| `e4k_farS_m200` | HW07 | 80.31 ± 0.37 | 92.04 ± 0.12 | 20/20 | 0 | 0 |

### 2.2 cluster 검증 표 (m 20)

- Δ: test 정확도의 같은 seed 짝지은 Δ(%p), "평균 ± 표준편차(ddof=0) (양수 k/10)".
- 진단값: 마지막 평가 구간(episode 3841–4000)의 에피소드 평균을 10 seed 평균한 값입니다.
  - `xneg_eq_query`: (query, 추가 음성) 쌍 중 GT 클래스가 같은 비율 = FN rate(측정 전용 레이블).
  - `xneg_in_qcluster`: 추가 음성 중 q의 기준 군집(k100, kmeans seed 20261108)에 속한 비율.
  - `xneg_in_farset`: 추가 음성 중 q의 기준 far 집합(2-hop ∪ raw cos 상위 100)에 속한 비율.
  - `excl_cluster_size`: 제외된 군집 크기의 query 평균.
  - `anc_same_cluster`: C(s⁺) = C(q)인 query 비율.
- `e4k_clus_m20`은 HW06 run이어서 HW07 진단 키가 없습니다("—"). `_regress/HW07/e4k_clus_m20`(Cora_full, seed 1212)의 `xneg_in_qcluster`는 0.0000이고 `xneg_in_farset`은 0.0062입니다.

**Cora_full**

| arm | Δ−rand20 | Δ−clus20 | xneg_eq_query | xneg_in_qcluster | xneg_in_farset | excl_cluster_size | anc_same_cluster |
|---|---|---|---|---|---|---|---|
| `e4k_clus_m20` | +0.39 ± 0.21 (10/10) | — | 0.0173 | — | — | — | — |
| `e4k_clus_shuf_m20` | +0.01 ± 0.15 (5/10) | −0.37 ± 0.22 (1/10) | 0.0218 | 0.0145 | 0.0090 | 306.5 | — |
| `e4k_clus_ks2_m20` | +0.31 ± 0.27 (10/10) | −0.07 ± 0.20 (2/10) | 0.0173 | 0.0060 | 0.0063 | 311.6 | — |
| `e4k_clus_anc_m20` | +0.03 ± 0.15 (6/10) | −0.36 ± 0.20 (0/10) | 0.0193 | 0.0095 | 0.0076 | 310.0 | 0.3557 |

**Amazon_clothing**

| arm | Δ−rand20 | Δ−clus20 | xneg_eq_query | xneg_in_qcluster | xneg_in_farset | excl_cluster_size | anc_same_cluster |
|---|---|---|---|---|---|---|---|
| `e4k_clus_m20` | +0.00 ± 0.16 (5/10) | — | 0.0141 | — | — | — | — |
| `e4k_clus_shuf_m20` | +0.01 ± 0.10 (7/10) | +0.01 ± 0.15 (6/10) | 0.0203 | 0.0166 | 0.0065 | 439.5 | — |
| `e4k_clus_ks2_m20` | −0.04 ± 0.20 (4/10) | −0.05 ± 0.12 (4/10) | 0.0138 | 0.0043 | 0.0028 | 472.2 | — |
| `e4k_clus_anc_m20` | +0.04 ± 0.13 (8/10) | +0.04 ± 0.15 (7/10) | 0.0152 | 0.0059 | 0.0035 | 441.8 | 0.6828 |

무작위 음성의 FN rate(HW06 진단)는 `e4k_rand_m20` 0.0216 / 0.0202입니다.

### 2.3 큰 m 표

- Δ: test 정확도 짝지은 Δ(%p), 형식은 2.2와 같습니다. 진단값은 "비교" 열 왼쪽 arm의 값이고 정의는 2.2와 같습니다.
- HW06 run(`e4k_clus_m20`, `e4k_far_m20`)은 `xneg_in_qcluster`, `xneg_in_farset` 키가 없습니다("—").

**Cora_full**

| 비교 | Δ | xneg_eq_query | xneg_in_qcluster | xneg_in_farset |
|---|---|---|---|---|
| `clus_m20` − `rand_m20` | +0.39 ± 0.21 (10/10) | 0.0173 | — | — |
| `clus_m100` − `rand_m100` | +0.43 ± 0.27 (9/10) | 0.0173 | 0.0000 | 0.0063 |
| `clus_m200` − `rand_m200` | +0.39 ± 0.23 (10/10) | 0.0173 | 0.0000 | 0.0063 |
| `far_m20` − `rand_m20` | +0.07 ± 0.10 (7/10) | 0.0193 | — | — |
| `far_m200` − `rand_m200` | +0.09 ± 0.13 (7/10) | 0.0193 | 0.0116 | 0.0000 |
| `farS_m200` − `rand_m200` | +0.11 ± 0.21 (6/10) | 0.0169 | 0.0097 | 0.0000 |
| `farS_m200` − `far_m200` | +0.03 ± 0.19 (6/10) | 0.0169 | 0.0097 | 0.0000 |

**Amazon_clothing**

| 비교 | Δ | xneg_eq_query | xneg_in_qcluster | xneg_in_farset |
|---|---|---|---|---|
| `clus_m20` − `rand_m20` | +0.00 ± 0.16 (5/10) | 0.0141 | — | — |
| `clus_m100` − `rand_m100` | −0.18 ± 0.20 (2/10) | 0.0142 | 0.0000 | 0.0027 |
| `clus_m200` − `rand_m200` | −0.31 ± 0.24 (0/10) | 0.0142 | 0.0000 | 0.0027 |
| `far_m20` − `rand_m20` | −0.05 ± 0.19 (4/10) | 0.0172 | — | — |
| `far_m200` − `rand_m200` | −0.31 ± 0.15 (0/10) | 0.0173 | 0.0125 | 0.0000 |
| `farS_m200` − `rand_m200` | +0.00 ± 0.14 (5/10) | 0.0112 | 0.0080 | 0.0000 |
| `farS_m200` − `far_m200` | +0.32 ± 0.12 (10/10) | 0.0112 | 0.0080 | 0.0000 |

무작위 음성의 FN rate(HW06 진단): m 20 / 100 / 200 = 0.0216 / 0.0216 / 0.0216 (Cora_full), 0.0202 / 0.0202 / 0.0203 (Amazon_clothing).

### 2.4 제외 집합 표 (§3, 학습 없음)

출처: `results/measure/HW07_exclusion_<ds>.{json,md}`. 레이블은 측정 전용입니다.

**(a) 노드 단위.** 값은 전체 노드 평균(노드 macro)입니다.
- C(q) = q의 기준 군집(k100, seed 20261108), F(q) = 2-hop ∪ raw cos 상위 100, F_S(q) = 2-hop ∪ raw cos 상위 1000. 세 집합 모두 q 자신을 뺀 값입니다.
- 크기는 평균 / 중앙값 / q90(노드 수)입니다.
- 같은 클래스 비율: 집합 안에서 q와 GT 클래스가 같은 노드 비율의 노드 macro입니다.
- 기대 개수 = m · |S(q)| / (N − 56)입니다. N − 56 = 19737 / 24863입니다. "같은 클래스 기대 개수"는 |S(q)|를 S(q) 안의 같은 클래스 노드 수로 바꾼 값입니다.

| ds | 집합 | 크기 | 같은 클래스 비율 | 기대 개수 m 20 / 100 / 200 | 같은 클래스 기대 개수 m 20 / 100 / 200 |
|---|---|---|---|---|---|
| Cora_full | C(q) | 307.3 / 255 / 480 | 0.3562 | 0.311 / 1.557 / 3.113 | 0.0887 / 0.4435 / 0.8869 |
| Cora_full | F(q) | 172.3 / 136 / 297 | 0.2408 | 0.175 / 0.873 / 1.746 | 0.0457 / 0.2284 / 0.4568 |
| Cora_full | F_S(q) | 1054.1 / 1025 / 1150 | 0.0979 | 1.068 / 5.341 / 10.682 | 0.1062 / 0.5308 / 1.0615 |
| Amazon_clothing | C(q) | 440.2 / 325 / 923 | 0.4560 | 0.354 / 1.771 / 3.541 | 0.1220 / 0.6100 / 1.2201 |
| Amazon_clothing | F(q) | 162.7 / 134 / 271 | 0.4416 | 0.131 / 0.654 / 1.309 | 0.0604 / 0.3020 / 0.6040 |
| Amazon_clothing | F_S(q) | 1028.9 / 1011 / 1081 | 0.2202 | 0.828 / 4.138 / 8.277 | 0.1824 / 0.9120 / 1.8240 |

| 항목 (노드 macro) | Cora_full | Amazon_clothing |
|---|---|---|
| \|F(q) ∩ C(q)\| / \|C(q)\| | 0.2128 | 0.3413 |
| \|F(q) ∩ C(q)\| / \|F(q)\| | 0.2545 | 0.5290 |
| 무작위 m 20 / 100 / 200의 전체 같은 클래스 기대 개수 | 0.4342 / 2.1709 / 4.3419 | 0.4034 / 2.0172 / 4.0344 |

**(b) 에피소드 단위.** seed 1212 학습 풀의 2000 에피소드, (q, s⁺) 쌍 100000개입니다. 비율은 쌍 비율입니다.

| 항목 | Cora_full | Amazon_clothing |
|---|---|---|
| C(s⁺) = C(q)인 비율 | 0.3581 | 0.6782 |
| \|C(s⁺) \ C(q)\| 평균, 전체 쌍 | 209.6 | 173.9 |
| \|C(s⁺) \ C(q)\| 평균, 비지 않은 쌍 | 326.5 | 540.5 |
| C(s⁺) \ C(q) 안의 q와 같은 클래스 비율 (비지 않은 쌍 평균) | 0.1030 | 0.1870 |
| C(q) 안에 같은 에피소드의 다른 pseudo-class support가 있는 비율 | 0.0542 | 0.0659 |

### 2.5 기하 표

- 값: 500 기준 best checkpoint의 z(`dim` 256). 표본은 고정 4096 노드이고 셀은 10 seed 평균입니다.
- 정의: `norm_mean` = ‖z‖ 평균. `uniformity` = log mean exp(−2‖û − v̂‖²). `eff_rank` = 중심화 표본 특이값 분포의 엔트로피 지수. `corr_offdiag_mean` = 차원 간 Pearson 상관 행렬의 비대각 |상관| 평균.

| arm | ds | norm_mean | uniformity | eff_rank | corr_offdiag_mean |
|---|---|---|---|---|---|
| `e4k_clus_shuf_m20` | Cora_full | 1.62 | −2.3413 | 168.47 | 0.1474 |
| `e4k_clus_shuf_m20` | Amazon_clothing | 1.84 | −2.6578 | 180.20 | 0.1235 |
| `e4k_clus_ks2_m20` | Cora_full | 1.68 | −2.3583 | 167.44 | 0.1466 |
| `e4k_clus_ks2_m20` | Amazon_clothing | 1.92 | −2.6593 | 175.86 | 0.1248 |
| `e4k_clus_anc_m20` | Cora_full | 1.62 | −2.3472 | 168.04 | 0.1480 |
| `e4k_clus_anc_m20` | Amazon_clothing | 1.84 | −2.6706 | 178.35 | 0.1250 |
| `e4k_clus_m100` | Cora_full | 1.78 | −2.3971 | 171.75 | 0.1357 |
| `e4k_clus_m100` | Amazon_clothing | 2.04 | −2.6827 | 179.62 | 0.1150 |
| `e4k_clus_m200` | Cora_full | 1.84 | −2.3904 | 173.27 | 0.1333 |
| `e4k_clus_m200` | Amazon_clothing | 2.12 | −2.6730 | 180.18 | 0.1124 |
| `e4k_far_m200` | Cora_full | 1.81 | −2.3424 | 175.47 | 0.1330 |
| `e4k_far_m200` | Amazon_clothing | 2.04 | −2.6675 | 187.08 | 0.1084 |
| `e4k_farS_m200` | Cora_full | 1.86 | −2.3508 | 174.04 | 0.1345 |
| `e4k_farS_m200` | Amazon_clothing | 2.13 | −2.6530 | 182.13 | 0.1139 |

### 2.6 best epoch 분포 (500 기준)

- 값: 10 seed의 best epoch 최소 / 중앙값 / 최대(평가 간격 160), 그리고 best epoch = 4000인 seed 수.

| arm | Cora_full | =4000 | Amazon_clothing | =4000 |
|---|---|---|---|---|
| `e4k_clus_shuf_m20` | 2240 / 3360 / 4000 | 2/10 | 2880 / 3680 / 4000 | 3/10 |
| `e4k_clus_ks2_m20` | 2240 / 3440 / 4000 | 2/10 | 3360 / 3760 / 4000 | 4/10 |
| `e4k_clus_anc_m20` | 2240 / 3520 / 4000 | 4/10 | 3360 / 3680 / 4000 | 4/10 |
| `e4k_clus_m100` | 2240 / 3440 / 4000 | 2/10 | 2880 / 3600 / 4000 | 1/10 |
| `e4k_clus_m200` | 2240 / 3600 / 4000 | 3/10 | 3040 / 3520 / 3840 | 0/10 |
| `e4k_far_m200` | 2240 / 3360 / 4000 | 2/10 | 2560 / 3520 / 4000 | 2/10 |
| `e4k_farS_m200` | 2720 / 3440 / 4000 | 3/10 | 2880 / 3680 / 4000 | 1/10 |

### 2.7 G-table (학습 전, `results/hw_iislab4/HW07_tables.json`)

- 같은 클래스 비율은 측정 전용 레이블 값이고 노드 macro입니다.
- far_strong의 후보 수에는 에피소드 제외(최대 55개)가 반영되어 있지 않으며, "− 55" 행이 그 값을 뺀 것입니다.
- shuffled 표본은 q를 전체 노드에서 균등하게 1만 번 뽑고, u를 C(q) 밖에서 균등하게 뽑은 것입니다(전용 Generator seed 20261109).

| 항목 | Cora_full | Amazon_clothing |
|---|---|---|
| 기준 군집 크기 최소 / 중앙값 / 최대 | 4 / 170 / 1083 | 11 / 185 / 1575 |
| 기준 군집 같은 군집 내 같은 클래스 비율 | 0.3562 | 0.4560 |
| seed 2 군집 크기 최소 / 중앙값 / 최대 | 14 / 174 / 1429 | 26 / 178 / 2114 |
| seed 2 군집 같은 군집 내 같은 클래스 비율 | 0.3540 | 0.4498 |
| 기준 군집과의 ARI | 0.5261 | 0.6619 |
| far_strong 남는 후보 수 최소 / 중앙값 | 17871 / 18767 | 23541 / 23907 |
| far_strong 최소 − 55 (기준 ≥ 1000) | 17816 | 23486 |
| `_top1000` 앞 500열 = `_top500`: 순서까지 같은 노드 비율 | 0.0002 | 0.0004 |
| `_top1000` 앞 500열 = `_top500`: 집합이 같은 노드 비율 | 0.3634 | 0.5107 |
| 집합이 다른 노드 수 / 그중 순위 500 경계 동점인 노드 수 | 12600 / 12600 | 12192 / 12192 |
| 두 표 앞 500열 cos 값의 최대 \|차\| | 0.0 | 0.0 |
| 노드 가중 C(q) 크기 평균 / 중앙값 / q90 | 308.25 / 256 / 481 | 441.24 / 326 / 924 |
| shuffled C(u) 크기 평균 / 중앙값 / q90 | 305.23 / 253 / 481 | 431.20 / 326 / 764 |
| 두 평균의 상대 차이 (기준 ≤ 10%) | 0.98% | 2.28% |

## 3. 게이트·사전 등록 대조표

### 3.1 게이트

| 게이트 | 기준 | 값 | 결과 |
|---|---|---|---|
| G-table | far_strong 최소 − 55 ≥ 1000 | 17816 / 23486 | 통과 |
| G-table | shuffled 평균 상대 차이 ≤ 10% | 0.98% / 2.28% | 통과 |
| G-ident | 같은 음성 인덱스, \|Δ손실\| ≤ 1e-6 | 인덱스 전부 같음, \|Δ\| 0.0 | 통과 |
| 회귀 | \|Δ\| ≤ 0.30 | 최대 0.16 | 통과 |
| G6 | completed ≥ 95% | 140/140 | 통과 |
| G7 (기록) | arm별 붕괴·NaN | 모든 arm 0 / 0 | 기록 |

- **G-ident** (`results/hw_iislab4/HW07_gident.json`): seed 1212, Cora_full, 학습 전 모델, train pool 앞 에피소드 3개입니다. 새 키 없는 HW06 `StrategyNegatives`와, 새 키를 기본값으로 명시한 `StrategyNegativesHW07`을 같은 seed로 만들어 비교했습니다.
  - cluster (`cluster_ref query`, `kmeans_seed 20261108`)와 far (`far_top 100`) 모두 [50, 20] 음성 인덱스가 3개 에피소드에서 같고, 손실 차이는 0.0입니다.
- **회귀** (`_regress/HW07/`, seed 1212, test %):

| arm | ds | HW07 | 기존 | Δ |
|---|---|---|---|---|
| `naq_base` | Cora_full | 77.80 | 77.96 | −0.16 |
| `naq_base` | Amazon_clothing | 90.61 | 90.60 | +0.00 |
| `nbr_raw` | Cora_full | 78.09 | 78.15 | −0.06 |
| `nbr_raw` | Amazon_clothing | 90.84 | 90.89 | −0.05 |
| `e4k_clus_m20` (추가 확인) | Cora_full | 80.60 | 80.51 | +0.09 |
| `e4k_far_m20` (추가 확인) | Cora_full | 79.68 | 79.74 | −0.06 |

- **불변** (`git diff 17dd625 --numstat`):
  - 삭제 줄: `hw/losses.py` 0, `hw/negatives.py` 0, `hw/neighbors.py` 0(변경 없음), `hw/run.py` 0, `hw/measure_positive.py` 0. 추가 줄: 2, 119, 0, 15, 138.
  - `configs/hw.json`은 삭제 1줄입니다. 마지막 arm(`e4k_qpos_qneg`) 줄 끝에 `,`가 붙은 것이고 값은 같습니다.
  - `hw/losses.py`의 추가 2줄은 `extra_neg.sample` 호출 앞의 `if getattr(extra_neg, "needs_episode_labels", False):`와 `extra_neg.set_episode(support_labels, query_labels)`입니다. pseudo-class 레이블만 넘깁니다.
  - `hw/run.py`의 새 분기는 `if any(key in extra_neg_cfg for key in HW07_KEYS):`입니다. 이 키가 없는 arm은 HW06 `StrategyNegatives` 또는 `ExtraNegatives` 객체를 그대로 씁니다.
- **캐시 md5** (작업 후): `5way_5shot.pkl` Cora_full `cfef6a69…`, Amazon_clothing `ca3edaca…`; `valid500` Cora_full `d8fdb82a…`, Amazon_clothing `0920f5d7…`로 기록값과 같습니다. `cache/kmeans/<ds>_k100.pt`와 `cache/rawcos/<ds>_top500.pt`의 수정 시각은 HW06 시점(2026-11-08)입니다. 새로 생긴 캐시는 `cache/kmeans/<ds>_k100_s20261109.pt`, `cache/rawcos/<ds>_top1000.pt`입니다.
- **smoke** (episodes 4, `/tmp/hw_smoke`, 커밋 안 함): 지시서의 `e4k_clus_shuf_m20`, `e4k_clus_anc_m20`, `e4k_clus_ks2_m20`, `e4k_farS_m200`에 `e4k_clus_m200`, `e4k_far_m200`을 더해 Cora_full에서 각 1회 돌렸습니다. 모두 completed이고 `xneg_in_qcluster`, `xneg_in_farset`, `excl_cluster_size`(cluster_ref arm), `anc_same_cluster`(anchor)가 기록되었습니다.

### 3.2 사전 등록 (판정은 사람)

Δ = test_acc 짝지은 Δ의 10 seed 평균입니다.

| ID | 기준 | 값 | 기준식 적용 |
|---|---|---|---|
| HWP070 | \|`clus_shuf_m20 − rand20`\| ≤ 0.15, Cora_full | +0.01 | 충족 |
| HWP071 | `clus20 − clus_shuf_m20` > 0, 양수 ≥ 7/10, Cora_full | +0.37 (9/10) | 충족 |
| HWP072 | `clus_ks2_m20 − rand20` ≥ +0.20, Cora_full | +0.31 | 충족 |
| HWP073 | `clus_anc_m20 − rand20` ≥ +0.20, Cora_full | +0.03 | 미충족 |
| HWP074 | `clus_m200 − rand_m200` > 0, Cora_full | +0.39 | 충족 |
| HWP075 | `far_m200 − rand_m200` > 0, 두 ds | +0.09 / −0.31 | 미충족 (Amazon_clothing) |
| HWP076 | `farS_m200 − far_m200` > 0, 두 ds | +0.03 / +0.32 | 충족 |
| HWP077 | 같은 m의 무작위 대비 \|Δ\| ≤ 0.15, cluster 5 arm, Amazon_clothing | shuf +0.01, ks2 −0.04, anc +0.04, m100 −0.18, m200 −0.31 | 미충족 (m100, m200) |
| HWP078 | 무작위 m 20의 C(q) 기대 개수 ≤ 0.5, 두 ds | 0.311 / 0.354 | 충족 |

## 4. 수행 내용

| 커밋 | 내용 |
|---|---|
| `92d31d0` HW07-a | `cluster_ref`·`kmeans_seed`·`far_top`(`StrategyNegativesHW07`), 진단(`ReferenceSets`), arm 7개, G-table(`hw/neg_tables_hw07.py`), G-ident, 회귀 run, `--exclusion` 코드. CLAUDE.md 개정 9와 지시서(사전 등록 HWP070–078) 포함 |
| `f3402a2` HW07-b | 140 run, `summary.{csv,md}`, `paired_{e4k_naq,e4k_rand_m20,e4k_rand_m100,e4k_rand_m200,e4k_clus_m20}.md`, `HW07_runs.txt` |
| `f021541` HW07-c | 제외 집합 측정 결과, `build_pool` 호출 수정 |
| (이 커밋) HW07-d | 보고서 |

§0 기록: HW06의 cluster·far는 `hw/negatives.py`의 `StrategyNegatives.sample`에 있습니다. 군집 제외는 `sample` 안의 `self.cluster[None, :] == self.cluster[query_ids][:, None]`이고, far 제외는 `within_two_hops(self.adj, query_ids)`와 `self.top`(raw cos 상위 100)의 `scatter_`입니다. q의 군집이나 far 제외 집합만 계산하는 별도 함수는 없습니다.

## 5. 판단한 사항

1. **새 키의 구현 위치**: HW06 `StrategyNegatives`를 고치지 않고 하위 클래스 `StrategyNegativesHW07`을 추가했습니다. `run.py`는 `extra_neg` 블록에 `cluster_ref`, `kmeans_seed`, `far_top` 중 하나라도 있을 때만 이 클래스를 씁니다. 그래서 `e4k_clus_m100`, `e4k_clus_m200`, `e4k_far_m200`은 HW06 클래스로 돌았습니다.
2. **`anchor`의 s⁺ 전달**: `sample(support_ids, query_ids)`에 레이블 인자가 없어서, `losses.py`의 호출 앞에 `set_episode(support_labels, query_labels)` 두 줄을 넣었습니다. 넘기는 값은 에피소드의 pseudo-class 레이블이고 GT 레이블이 아닙니다.
3. **`shuffled`의 u**: 지시서대로 C(q) 밖의 전체 노드에서 뽑았습니다. 에피소드 제외 노드를 u 후보에서 빼지 않았습니다.
4. **`excl_cluster_size`의 크기**: 각 arm이 실제로 쓰는 군집 분할 기준입니다(`ks2`는 seed 2 분할).
5. **`far_top` 1000의 표**: `raw_cos_table(k=1000)`으로 계산해 `cache/rawcos/<ds>_top1000.pt`에 저장했습니다. `farS`의 제외 집합은 이 표의 앞 1000열입니다. 기준 far 집합과 `far_m200`은 기존 `_top500` 표의 앞 100열입니다.
6. **제외 집합 측정의 정의**
   - F(q), F_S(q), C(q) 모두 q 자신을 뺐습니다. F(q)의 raw cos 상위 100은 `_top1000` 표의 앞 100열입니다.
   - `build_pool`은 기본 arm(`nbr_bgrl`)으로 호출했습니다. `naq_base`에는 `bgrl_dir`가 없어 `build_pool`이 오류를 냈습니다. HW04m·HW05와 같은 호출입니다.
   - "C(s⁺) \ C(q) 안의 같은 클래스 비율"은 비지 않은 쌍만의 평균입니다.
7. **HWP077의 "cluster 5 arm"**: HW07에서 추가한 cluster arm 5개(`shuf`, `ks2`, `anc`, `m100`, `m200`)로 읽었습니다. HW06 `clus20`의 값은 +0.00입니다.
8. **HW06 run의 진단 키**: `e4k_clus_m20`, `e4k_far_m20`, 무작위 arm은 HW06 코드로 돌았으므로 `xneg_in_qcluster`, `xneg_in_farset`이 없습니다. 재실행하지 않고 "—"로 두었습니다.
9. **smoke와 회귀에 추가한 항목**: 지시서의 smoke 4개 arm에 `e4k_clus_m200`, `e4k_far_m200`을 더했습니다. 회귀에는 `e4k_clus_m20`, `e4k_far_m20`(Cora_full, seed 1212)을 더해 HW06 값과 비교했습니다.

## 6. 제외·실패·붕괴 run

- 제외 run 없음. 140/140 completed, NaN 0, 붕괴 0입니다. 실패로 인한 재시도는 0입니다.
- **중단과 재실행**: 큐 실행 중 ssh 연결이 끊겨 큐가 종료되었습니다. 큐 로그의 마지막 기록은 06:01:20 UTC이고, 그때 135/140이 completed였습니다.
  - 미완료 5개 run은 모두 seed 1221입니다: `e4k_clus_m200` Amazon_clothing, `e4k_far_m200` 두 ds, `e4k_farS_m200` 두 ds. `metrics.json`이 `status: running`으로 남아 있었습니다.
  - 같은 run 목록으로 큐를 다시 실행했습니다(06:41:06 → 06:44:15 UTC). completed 135개는 건너뛰었고 5개는 처음부터 다시 돌았습니다. 설정, seed, 코드는 그대로입니다.
- `_regress/HW07/`는 집계에서 뺐습니다.

## 7. 소요 시간, 환경, HEAD

- **시간**:
  - G-table: 11분 46초 (05:05:45 → 05:17:31 UTC). seed 2 k-means와 상위 1000 표를 포함합니다.
  - 140 run: 05:18:47 → 06:01:20 UTC (42분 33초, 135 run 완료 시점까지)에 재실행 3분 9초를 더해 45분 42초입니다.
  - run당 시간(arm별 최소 / 중앙값 / 최대): `clus_shuf` 124 / 141 / 156 s, `clus_ks2` 123 / 140 / 158 s, `clus_anc` 125 / 141 / 157 s, `clus_m100` 128 / 145 / 162 s, `clus_m200` 129 / 148 / 165 s, `far_m200` 133 / 148 / 164 s, `farS_m200` 130 / 149 / 168 s.
  - 제외 집합 측정: 11.63 s / 15.57 s.
- **환경**: iislab4, RTX 3090 × 8 (`GPUS=0,1,2,3,4,5,6,7`), conda `taskns`. 시작 시 GPU 메모리 사용 0 MiB × 8.
- **HEAD**: 작업 시작 `17dd625`. HW07-a `92d31d0`, HW07-b `f3402a2`, HW07-c `f021541`.
