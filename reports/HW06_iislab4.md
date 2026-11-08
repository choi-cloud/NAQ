# HW06 [iislab4] — 음성 샘플링 전략 비교: 무작위 m 곡선 · far / band / cluster · 에피소드 안 query 음성·양성 · 차원 상관

2026-11-08 · 서버 iislab4 · 브랜치 `hl/hsan-weighting` · 작업 시작 HEAD `3b62b03` · 지시서 `instructions/HW06_iislab4.md` · 사전 등록 HWP059–HWP069

모든 학습 arm은 `valid_tasks: 500`, `episodes: 4000`, `geom_diag: true`, base `naq_base`입니다. 주 결과는 `test_acc`(500 기준 best epoch의 test)입니다. 약칭: naq = `e4k_naq`, rand20 = `e4k_rand_m20`, qneg = `e4k_qneg`.

## 1. 결론

1. **m 곡선 (`− naq`, Cora_full / Amazon_clothing):** 무작위 음성 m 20 / 50 / 100 / 200은 +1.15 / +0.61, +1.31 / +0.70, +1.39 / +0.66, +1.50 / +0.66 (모두 10/10)입니다. 인접 m 간 Δ는 m 50 이후 Cora +0.09 ~ +0.16, Amazon −0.04 ~ +0.09입니다.
2. **전략 (m 20, `− rand20`):** cluster +0.39 (10/10) / +0.00 (5/10), far +0.07 / −0.05, band −1.04 / −8.44 (0/10, 0/10)입니다. 추가 음성이 query와 같은 GT 클래스인 비율(FN rate)은 rand 0.0216 / 0.0202, cluster 0.0173 / 0.0141, band 0.0987 / 0.2624입니다.
3. **에피소드 안 query:** qneg `− naq` +0.33 / +0.19, qneg + 무작위 20 `− rand20` −0.19 / −0.14, query 양성 추가(`qpos_qneg − qneg`) −2.94 / −2.77입니다. 차원 간 |상관| 평균은 naq 0.1658 / 0.1483, rand20 0.1478 / 0.1231입니다.

## 2. 핵심 표

### 2.1 test 정확도 (11 arm)

- 값: test 정확도(%), 5-way 5-shot, test 500 task. 셀은 10 seed의 평균 ± 표준편차(ddof=0)입니다.
- 붕괴: 500 기준 best validation ≤ 25%인 run 수. NaN: status `nan`인 run 수.
- 출처: `results/hw_iislab4/summary.md`.

| arm | Cora_full | Amazon_clothing | completed | 붕괴 | NaN |
|---|---|---|---|---|---|
| `e4k_naq` | 78.69 ± 0.47 | 91.38 ± 0.26 | 20/20 | 0 | 0 |
| `e4k_rand_m20` | 79.84 ± 0.37 | 91.99 ± 0.19 | 20/20 | 0 | 0 |
| `e4k_rand_m50` | 80.00 ± 0.35 | 92.08 ± 0.24 | 20/20 | 0 | 0 |
| `e4k_rand_m100` | 80.09 ± 0.32 | 92.04 ± 0.22 | 20/20 | 0 | 0 |
| `e4k_rand_m200` | 80.20 ± 0.35 | 92.04 ± 0.15 | 20/20 | 0 | 0 |
| `e4k_far_m20` | 79.91 ± 0.41 | 91.94 ± 0.25 | 20/20 | 0 | 0 |
| `e4k_band_m20` | 78.80 ± 0.40 | 83.56 ± 0.28 | 20/20 | 0 | 0 |
| `e4k_clus_m20` | 80.23 ± 0.31 | 91.99 ± 0.17 | 20/20 | 0 | 0 |
| `e4k_qneg` | 79.02 ± 0.38 | 91.57 ± 0.17 | 20/20 | 0 | 0 |
| `e4k_qneg_rand_m20` | 79.65 ± 0.37 | 91.85 ± 0.19 | 20/20 | 0 | 0 |
| `e4k_qpos_qneg` | 76.08 ± 0.41 | 88.80 ± 0.40 | 20/20 | 0 | 0 |

### 2.2 m 곡선 (무작위 음성)

- 값: test 정확도(%)의 10 seed 평균, 그리고 같은 seed 짝지은 Δ(%p) "평균 ± 표준편차(ddof=0) (양수 k/10)".
- "Δ−이전 m" = 이 행 arm − 바로 윗 행 arm.

| m | arm | Cora acc | Cora Δ−naq | Cora Δ−이전 m | Amazon acc | Amazon Δ−naq | Amazon Δ−이전 m |
|---|---|---|---|---|---|---|---|
| 0 | `e4k_naq` | 78.69 | — | — | 91.38 | — | — |
| 20 | `e4k_rand_m20` | 79.84 | +1.15 ± 0.28 (10/10) | +1.15 ± 0.28 (10/10) | 91.99 | +0.61 ± 0.20 (10/10) | +0.61 ± 0.20 (10/10) |
| 50 | `e4k_rand_m50` | 80.00 | +1.31 ± 0.27 (10/10) | +0.16 ± 0.11 (10/10) | 92.08 | +0.70 ± 0.23 (10/10) | +0.09 ± 0.14 (7/10) |
| 100 | `e4k_rand_m100` | 80.09 | +1.39 ± 0.32 (10/10) | +0.09 ± 0.14 (8/10) | 92.04 | +0.66 ± 0.28 (10/10) | −0.04 ± 0.11 (4/10) |
| 200 | `e4k_rand_m200` | 80.20 | +1.50 ± 0.34 (10/10) | +0.11 ± 0.19 (7/10) | 92.04 | +0.66 ± 0.25 (10/10) | −0.00 ± 0.16 (5/10) |

### 2.3 전략 표 (m 20)

- Δ: test 정확도 짝지은 Δ(%p), 형식은 2.2와 같습니다.
- 진단값(측정 전용 레이블): 마지막 평가 구간(episode 3841–4000)의 에피소드 평균을 10 seed 평균한 값입니다.
  - `xneg_eq_query`: (query, 추가 음성) 쌍 중 GT 클래스가 같은 비율 = FN rate.
  - `xneg_eq_support`: 추가 음성과 그 query의 양성 support의 GT 클래스가 같은 비율.
  - `xneg_cos_raw`: query와 추가 음성의 raw cos(L2 정규화 X) 평균.
  - `band_cos_lo` / `band_cos_hi`: query마다 고른 20개의 raw cos 최솟값 / 최댓값을 query 평균한 값.

**Cora_full**

| arm | Δ−naq | Δ−rand20 | xneg_eq_query | xneg_eq_support | xneg_cos_raw | band_cos_lo | band_cos_hi |
|---|---|---|---|---|---|---|---|
| `e4k_rand_m20` | +1.15 ± 0.28 (10/10) | — | 0.0216 | 0.0212 | 0.0520 | — | — |
| `e4k_far_m20` | +1.22 ± 0.27 (10/10) | +0.07 ± 0.10 (7/10) | 0.0193 | 0.0201 | 0.0512 | — | — |
| `e4k_band_m20` | +0.11 ± 0.66 (6/10) | −1.04 ± 0.54 (0/10) | 0.0987 | 0.0752 | 0.1544 | 0.1386 | 0.1781 |
| `e4k_clus_m20` | +1.53 ± 0.38 (10/10) | +0.39 ± 0.21 (10/10) | 0.0173 | 0.0188 | 0.0516 | — | — |

**Amazon_clothing**

| arm | Δ−naq | Δ−rand20 | xneg_eq_query | xneg_eq_support | xneg_cos_raw | band_cos_lo | band_cos_hi |
|---|---|---|---|---|---|---|---|
| `e4k_rand_m20` | +0.61 ± 0.20 (10/10) | — | 0.0202 | 0.0199 | 0.0319 | — | — |
| `e4k_far_m20` | +0.56 ± 0.20 (10/10) | −0.05 ± 0.19 (4/10) | 0.0172 | 0.0174 | 0.0301 | — | — |
| `e4k_band_m20` | −7.83 ± 0.40 (0/10) | −8.44 ± 0.37 (0/10) | 0.2624 | 0.2367 | 0.2445 | 0.2035 | 0.3091 |
| `e4k_clus_m20` | +0.61 ± 0.25 (10/10) | +0.00 ± 0.16 (5/10) | 0.0141 | 0.0150 | 0.0293 | — | — |

### 2.4 에피소드 안 query 표

- Δ: test 정확도 짝지은 Δ(%p).
- 진단값(측정 전용 레이블, 마지막 평가 구간 10 seed 평균):
  - `qneg_eq_query`: (q, 다른 pseudo-class pseudo-query 40개) 쌍 중 GT 클래스가 같은 비율 = query 음성의 FN rate.
  - `qneg_cos_raw`: 그 쌍들의 raw cos 평균.
  - `qpos_eq_query`: (q, 같은 pseudo-class 다른 pseudo-query 9개) 쌍 중 GT 클래스가 같은 비율 = query 양성 순도(쌍 비율).
- pseudo-query 집합은 학습 손실과 무관하게 seed별 train pool로 정해집니다. 그래서 세 arm의 `qneg_*` 값이 같습니다.

**Cora_full**

| arm | Δ−naq | Δ−rand20 | Δ−qneg | qneg_eq_query | qneg_cos_raw | qpos_eq_query | xneg_eq_query |
|---|---|---|---|---|---|---|---|
| `e4k_qneg` | +0.33 ± 0.31 (9/10) | −0.82 ± 0.11 (0/10) | — | 0.0220 | 0.0580 | — | — |
| `e4k_qneg_rand_m20` | +0.96 ± 0.22 (10/10) | −0.19 ± 0.12 (1/10) | +0.63 ± 0.15 (10/10) | 0.0220 | 0.0580 | — | 0.0216 |
| `e4k_qpos_qneg` | −2.61 ± 0.53 (0/10) | −3.75 ± 0.54 (0/10) | −2.94 ± 0.51 (0/10) | 0.0220 | 0.0580 | 0.2454 | — |

**Amazon_clothing**

| arm | Δ−naq | Δ−rand20 | Δ−qneg | qneg_eq_query | qneg_cos_raw | qpos_eq_query | xneg_eq_query |
|---|---|---|---|---|---|---|---|
| `e4k_qneg` | +0.19 ± 0.23 (8/10) | −0.42 ± 0.15 (0/10) | — | 0.0200 | 0.0342 | — | — |
| `e4k_qneg_rand_m20` | +0.47 ± 0.23 (10/10) | −0.14 ± 0.14 (2/10) | +0.28 ± 0.11 (10/10) | 0.0200 | 0.0342 | — | 0.0202 |
| `e4k_qpos_qneg` | −2.58 ± 0.44 (0/10) | −3.19 ± 0.35 (0/10) | −2.77 ± 0.34 (0/10) | 0.0200 | 0.0342 | 0.5838 | — |

### 2.5 기하 표

- 값: 500 기준 best checkpoint의 z(GCN 출력, `dim` 256). 표본은 고정 4096 노드(CPU Generator seed 2026)이고, 셀은 10 seed 평균입니다.
- 정의:
  - `norm_mean`: ‖z‖ 평균.
  - `uniformity`: log mean exp(−2‖û − v̂‖²). 더 음수일수록 L2 정규화 표본이 더 퍼져 있습니다.
  - `eff_rank`: 중심화 표본 특이값 분포의 엔트로피 지수.
  - `corr_offdiag_mean`: 차원 간 Pearson 상관 행렬 비대각 |상관|의 평균(± seed 간 표준편차, ddof=1). 분산 0인 차원은 제외했습니다.
  - `dead_dims`: 표본에서 분산 0(≤ 1e-12)인 차원 수.

| arm | ds | norm_mean | uniformity | eff_rank | corr_offdiag_mean | dead_dims | dim |
|---|---|---|---|---|---|---|---|
| `e4k_naq` | Cora_full | 1.53 | −2.2338 | 155.02 | 0.1658 ± 0.0056 | 7.0 | 256 |
| `e4k_naq` | Amazon_clothing | 1.72 | −2.5131 | 166.76 | 0.1483 ± 0.0041 | 0.5 | 256 |
| `e4k_rand_m20` | Cora_full | 1.63 | −2.3406 | 168.90 | 0.1478 ± 0.0044 | 1.3 | 256 |
| `e4k_rand_m20` | Amazon_clothing | 1.83 | −2.6665 | 180.37 | 0.1231 ± 0.0019 | 0.0 | 256 |
| `e4k_rand_m50` | Cora_full | 1.67 | −2.3467 | 170.65 | 0.1436 ± 0.0031 | 1.3 | 256 |
| `e4k_rand_m50` | Amazon_clothing | 1.85 | −2.6806 | 183.89 | 0.1180 ± 0.0016 | 0.0 | 256 |
| `e4k_rand_m100` | Cora_full | 1.69 | −2.3636 | 173.11 | 0.1402 ± 0.0040 | 1.5 | 256 |
| `e4k_rand_m100` | Amazon_clothing | 1.88 | −2.6804 | 185.97 | 0.1156 ± 0.0024 | 0.0 | 256 |
| `e4k_rand_m200` | Cora_full | 1.72 | −2.3564 | 174.13 | 0.1382 ± 0.0037 | 1.3 | 256 |
| `e4k_rand_m200` | Amazon_clothing | 1.89 | −2.6741 | 187.56 | 0.1131 ± 0.0020 | 0.0 | 256 |
| `e4k_far_m20` | Cora_full | 1.66 | −2.3452 | 169.21 | 0.1467 ± 0.0046 | 1.3 | 256 |
| `e4k_far_m20` | Amazon_clothing | 1.88 | −2.6556 | 179.50 | 0.1225 ± 0.0026 | 0.0 | 256 |
| `e4k_band_m20` | Cora_full | 1.32 | −2.3509 | 181.63 | 0.1137 ± 0.0039 | 2.4 | 256 |
| `e4k_band_m20` | Amazon_clothing | 1.55 | −2.6015 | 200.62 | 0.0829 ± 0.0014 | 0.1 | 256 |
| `e4k_clus_m20` | Cora_full | 1.67 | −2.3622 | 167.98 | 0.1457 ± 0.0051 | 1.2 | 256 |
| `e4k_clus_m20` | Amazon_clothing | 1.91 | −2.6666 | 176.35 | 0.1242 ± 0.0020 | 0.0 | 256 |
| `e4k_qneg` | Cora_full | 1.68 | −2.3600 | 160.46 | 0.1526 ± 0.0034 | 4.6 | 256 |
| `e4k_qneg` | Amazon_clothing | 2.00 | −2.5802 | 167.10 | 0.1387 ± 0.0028 | 0.5 | 256 |
| `e4k_qneg_rand_m20` | Cora_full | 1.68 | −2.3598 | 167.27 | 0.1467 ± 0.0037 | 1.4 | 256 |
| `e4k_qneg_rand_m20` | Amazon_clothing | 1.93 | −2.6427 | 177.78 | 0.1239 ± 0.0018 | 0.0 | 256 |
| `e4k_qpos_qneg` | Cora_full | 2.18 | −1.8796 | 151.78 | 0.1616 ± 0.0094 | 7.8 | 256 |
| `e4k_qpos_qneg` | Amazon_clothing | 2.58 | −2.1451 | 161.96 | 0.1375 ± 0.0031 | 0.6 | 256 |

### 2.6 복제: `e4k_naq − v500_naq_base_e4000` (HW05)

- 값: test 정확도 seed별 Δ(%p), seed 1212 → 1221 순서입니다.
- 두 arm의 학습 설정은 같고, `e4k_naq`에만 `geom_diag`가 켜져 있습니다(학습 후 측정만 추가).

| ds | seed별 Δ | 평균 Δ | 평균 \|Δ\| | 최대 \|Δ\| |
|---|---|---|---|---|
| Cora_full | −0.04, +0.03, +0.04, −0.06, +0.76, −0.07, −0.02, +0.06, +0.02, −0.12 | +0.06 | 0.12 | 0.76 |
| Amazon_clothing | −0.09, −0.04, +0.05, −0.02, +0.02, +0.04, −0.05, +0.51, −0.02, −0.04 | +0.04 | 0.09 | 0.51 |

### 2.7 best epoch 분포 (500 기준)

- 값: 10 seed의 best epoch 최소 / 중앙값 / 최대(평가 간격 160), 그리고 best epoch = 4000인 seed 수.

| arm | Cora_full | =4000 | Amazon_clothing | =4000 |
|---|---|---|---|---|
| `e4k_naq` | 2240 / 3360 / 4000 | 2/10 | 2560 / 3680 / 4000 | 1/10 |
| `e4k_rand_m20` | 2240 / 3360 / 4000 | 2/10 | 2880 / 3600 / 4000 | 3/10 |
| `e4k_rand_m50` | 2240 / 3280 / 4000 | 3/10 | 2720 / 3520 / 4000 | 1/10 |
| `e4k_rand_m100` | 2240 / 3360 / 4000 | 3/10 | 2880 / 3600 / 4000 | 1/10 |
| `e4k_rand_m200` | 2240 / 3440 / 4000 | 2/10 | 2560 / 3600 / 4000 | 1/10 |
| `e4k_far_m20` | 2240 / 3360 / 4000 | 3/10 | 2560 / 3680 / 4000 | 2/10 |
| `e4k_band_m20` | 1600 / 2480 / 4000 | 1/10 | 2560 / 3680 / 4000 | 2/10 |
| `e4k_clus_m20` | 2240 / 3600 / 4000 | 4/10 | 2880 / 3760 / 4000 | 4/10 |
| `e4k_qneg` | 2240 / 3280 / 4000 | 3/10 | 2880 / 3680 / 4000 | 2/10 |
| `e4k_qneg_rand_m20` | 2240 / 3280 / 4000 | 3/10 | 2560 / 3680 / 4000 | 2/10 |
| `e4k_qpos_qneg` | 1120 / 3360 / 4000 | 2/10 | 2720 / 3600 / 4000 | 2/10 |

### 2.8 G-table (학습 전, `results/hw_iislab4/HW06_tables.json`)

- raw cos = L2 정규화 X의 cos. 순위는 노드마다 자기 자신을 뺀 1-based 순위입니다.
- 에피소드 제외(support 5 + pseudo-query 50)는 이 표에 반영하지 않았습니다. query마다 최대 55개가 더 빠집니다.
- 같은 클래스 비율은 측정 전용 레이블 값이고, 노드별 비율의 노드 평균(노드 macro)입니다.

| 항목 | Cora_full | Amazon_clothing |
|---|---|---|
| 노드 수 | 19793 | 24919 |
| band: 순위 101–500 중 cos > 0인 노드 수 최소 / 중앙값 | 8 / 400 | 0 / 400 |
| band: cos(순위 101) 최소 / Q1 / 중앙값 / Q3 / 최대 | 0.0595 / 0.1374 / 0.1519 / 0.1671 / 0.5774 | 0.0000 / 0.2114 / 0.2681 / 0.3341 / 0.9694 |
| band: cos(순위 500) 최소 / Q1 / 중앙값 / Q3 / 최대 | 0.0000 / 0.1088 / 0.1216 / 0.1339 / 0.3536 | 0.0000 / 0.1333 / 0.1704 / 0.2191 / 0.4328 |
| band: 순위 500 cos가 0인 노드 비율 (멈춤 기준 > 0.20) | 0.0001 | 0.0018 |
| band: 구간 같은 클래스 비율 (학습 전 기대 FN rate) | 0.0935 | 0.2450 |
| far: 제외 후 남는 후보 수 최소 / 중앙값 (기준 ≥ 1000) | 18638 / 19656 | 24250 / 24784 |
| far: 위 최소 − 55 | 18583 | 24195 |
| cluster (k 100): 크기 최소 / 중앙값 / 최대 | 4 / 170 / 1083 | 11 / 185 / 1575 |
| cluster: 같은 군집 내 같은 클래스 비율 | 0.3562 | 0.4560 |
| cluster: 다른 군집 노드 중 같은 클래스 비율 (학습 전 기대 FN rate) | 0.0175 | 0.0143 |
| random 기대 FN rate Σ p_c² | 0.0217 | 0.0202 |

- band의 구간 후보 수는 400개이고, 최대 제외 55개를 빼도 345개 ≥ m 20입니다.
- far의 2-hop 집합은 노드 200개에서 `hw.structure.hop_sets_dense`(H1 ∪ H2 ∪ 자기)와 일치합니다(`hop_check` true).

## 3. 게이트·사전 등록 대조표

### 3.1 게이트

| 게이트 | 기준 | 값 | 결과 |
|---|---|---|---|
| G-table | band 순위 500 cos 0 비율 ≤ 0.20 | 0.0001 / 0.0018 | 통과 |
| G-table | far 제외 후 최소 후보 ≥ 1000 | 18638 / 24250 (−55: 18583 / 24195) | 통과 |
| G-ident | flags 끔 손실 = L_original, \|Δ\| ≤ 1e-6 | 0.0 (에피소드 3개) | 통과 |
| 회귀 | \|Δ\| ≤ 0.30 | 최대 0.18 | 통과 |
| G6 | completed ≥ 95% | 220/220, 재시도 0 | 통과 |
| G7 (기록) | arm별 붕괴·NaN | 모든 arm 0 / 0 | 기록 |

- **G-ident** (`results/hw_iislab4/HW06_gident.json`): seed 1212, Cora_full, 학습 전 모델, train pool 앞 에피소드 3개.
  - 기본 호출, flags를 명시적으로 끈 호출(`extra_neg None, ep_query_neg False, ep_query_pos False`), 직접 계산한 L_original이 1.5539947748 / 1.5262658596 / 1.5647628307로 같습니다(|Δ| 0.0).
  - `ep_query_neg`와 `ep_query_neg + ep_query_pos` 손실은 query별 반복문 참조 계산과 |Δ| 0.0입니다.
- **회귀** (`_regress/HW06/`, seed 1212, test %):

| arm | ds | HW06 | 기존 | Δ |
|---|---|---|---|---|
| `naq_base` | Cora_full | 77.78 | 77.96 | −0.18 |
| `naq_base` | Amazon_clothing | 90.61 | 90.60 | +0.01 |
| `nbr_raw` | Cora_full | 78.07 | 78.15 | −0.08 |
| `nbr_raw` | Amazon_clothing | 90.83 | 90.89 | −0.06 |

- **불변** (`git diff 3b62b03 --numstat`):
  - 삭제 줄: `hw/losses.py` 0, `hw/negatives.py` 0, `hw/neighbors.py` 0(변경 없음), `hw/run.py` 0. 추가 줄: 25, 147, 0, 63.
  - 기존 `ExtraNegatives`(random / hard_raw / hard_a2x), 기존 `extra_neg` 블록, `_info_nce`, L_neighbor 경로는 그대로입니다.
    - 새 코드는 `if ep_query_neg:` 블록(L_original 재계산)과 run.py의 `if extra_neg_cfg["mode"] in STRATEGY_MODES:` 분기입니다. 이 분기는 기존 `ExtraNegatives(...)` 생성 줄 뒤에서 객체를 `StrategyNegatives`로 바꿉니다.
  - `configs/hw.json`은 삭제 1줄입니다. 마지막 arm 줄 끝에 `,`가 붙은 것이고 값은 같습니다.
- **캐시 md5** (작업 후):
  - `5way_5shot.pkl`: Cora_full `cfef6a69…`, Amazon_clothing `ca3edaca…`.
  - `valid500`: Cora_full `d8fdb82a…`, Amazon_clothing `0920f5d7…`.
  - 모두 기록값과 같습니다.
- **smoke** (episodes 4, `/tmp/hw_smoke`, 커밋 안 함): `e4k_band_m20`, `e4k_clus_m20`, `e4k_qpos_qneg`, 추가로 `e4k_far_m20`, `e4k_qneg_rand_m20`을 Cora_full에서 각 1회 돌렸습니다. 모두 completed이고 `band_cos_lo/hi`, `qneg_eq_query`, `qneg_cos_raw`, `qpos_eq_query`, `geom.corr_offdiag_mean`, `geom.dead_dims`, `geom.dim`이 기록되었습니다.

### 3.2 사전 등록 (판정은 사람)

Δ = test_acc 짝지은 Δ의 10 seed 평균(Cora_full / Amazon_clothing). 진단값은 마지막 평가 구간 10 seed 평균입니다.

| ID | 기준 | 값 | 기준식 적용 |
|---|---|---|---|
| HWP059 | `rand_m50 − rand_m20` > 0, 두 ds | +0.16 / +0.09 | 충족 |
| HWP060 | \|`rand_m200 − rand_m100`\| ≤ 0.15, 두 ds | +0.11 / −0.00 | 충족 |
| HWP061 | \|`far_m20 − rand_m20`\| ≤ 0.20, 두 ds | +0.07 / −0.05 | 충족 |
| HWP062 | `band_m20` `xneg_eq_query` ≥ 0.10, 두 ds | 0.0987 / 0.2624 | 미충족 (Cora_full) |
| HWP063 | `band_m20 − rand_m20` < 0, 두 ds | −1.04 / −8.44 | 충족 |
| HWP064 | `clus_m20` `xneg_eq_query` < `rand_m20` 값, 두 ds | 0.0173 < 0.0216 / 0.0141 < 0.0202 | 충족 |
| HWP065 | `qneg_eq_query` ≥ 0.04, 두 ds | 0.0220 / 0.0200 | 미충족 |
| HWP066 | `qneg − naq` > 0, 양수 ≥ 7/10, 두 ds | +0.33 (9/10) / +0.19 (8/10) | 충족 |
| HWP067 | `qneg_rand_m20 − rand_m20` > 0, 두 ds | −0.19 / −0.14 | 미충족 |
| HWP068 | `qpos_qneg − qneg` < 0, 두 ds | −2.94 / −2.77 | 충족 |
| HWP069 | `corr_offdiag_mean`: `rand_m20` < `naq`, 두 ds | 0.1478 < 0.1658 / 0.1231 < 0.1483 | 충족 |

## 4. 수행 내용

| 커밋 | 내용 |
|---|---|
| `5bd6ce5` HW06-a | 구현과 arm 11개, G-table·G-ident, 회귀 run. CLAUDE.md 개정 8과 지시서(사전 등록 HWP059–069)를 함께 넣었습니다. 세부는 아래와 같습니다. |
| `d99f115` HW06-b | 220 run, `summary.{csv,md}`, `paired_{e4k_naq,e4k_rand_m20,e4k_qneg}.md`, `HW06_runs.txt` |
| (이 커밋) HW06-c | 보고서 |

HW06-a의 구현 내용:
- **`hw/negatives.py`**: `StrategyNegatives`(far / band / cluster)와 `raw_cos_table`, `kmeans_labels`, `adjacency_torch`, `within_two_hops`를 추가했습니다.
- **`hw/losses.py`**: `ep_query_neg` / `ep_query_pos`를 추가했습니다.
- **`hw/run.py`**: 다음을 추가했습니다.
  - 새 진단 키, `geometry_corr`, `band_diagnostics`, `qep_diagnostics`.
  - `ep_query_*` 설정 파싱.
- **`hw/neg_tables.py`**: G-table 계산 스크립트입니다.

새 캐시(git 밖):
- `cache/rawcos/<ds>_top500.pt`: raw cos 상위 500의 ids와 값.
- `cache/kmeans/<ds>_k100.pt`: k-means 군집 레이블.
- 둘 다 `python -m hw.neg_tables`가 처음 만들었고, 학습 run은 이 파일을 읽기만 했습니다.

## 5. 판단한 사항

1. **기존 코드를 고치지 않고 새 모드를 붙인 방식**
   - `EXTRA_NEG_MODES`는 파일 끝에서 `EXTRA_NEG_MODES + ("far", "band", "cluster")`로 다시 정의했습니다. 그래서 `run.py`의 기존 검사 줄을 바꾸지 않았습니다.
   - `run.py`의 기존 `ExtraNegatives(...)` 생성 줄은 새 모드에서도 실행됩니다. 이때는 표 없이 Generator 하나만 만듭니다. 그 바로 뒤에 넣은 줄이 객체를 `StrategyNegatives`로 바꿉니다.
2. **far의 "raw cos 상위 100"**: `hard_raw` 표를 다시 계산하지 않고 raw cos 상위 500 표(CPU 계산, 캐시)의 앞 100열을 썼습니다.
   - GPU `hard_table` top-100 집합과 같은 노드 비율은 0.6955 / 0.6849입니다.
   - 다른 노드(6026 / 7851개)는 모두 집합 차이의 원소들이 순위 100의 cos 값과 같은 동점(|차| ≤ 1e-6)입니다.
3. **band 구간**: 순위 101–500 = 상위 500 표의 열 100–499(0-based)입니다.
   - 순위는 자기 자신을 제외한 torch.topk 정렬 순서이고, 동점의 순서는 topk 구현을 따릅니다.
   - band 모드는 band 구간에서 cos > 0 조건을 따로 걸지 않았습니다. G-table에서 순위 500 cos 0 비율이 0.0001 / 0.0018이었기 때문입니다.
4. **cluster**:
   - 군집 계산은 CPU에서 `features.cpu()`, `adj.cpu()`로 Â²X를 만들고 L2 정규화했습니다.
   - 군집 레이블은 seed와 무관하게 데이터셋당 한 번 계산합니다.
5. **같은 query의 음성은 중복 없이 뽑습니다.** 전용 Generator의 균등 점수를 쓰고, 금지 노드를 −1로 가린 뒤 top-m을 고릅니다. 허용 후보가 m보다 적으면 오류를 냅니다(실제로 발생 0).
6. **`ep_query_neg` / `ep_query_pos`의 대상 집합**
   - 음성은 같은 에피소드에서 q와 pseudo-class가 다른 pseudo-query입니다(40개, q_query 10 × 4).
   - 양성은 같은 pseudo-class의 다른 pseudo-query입니다(9개).
   - pseudo-class 크기가 다르면 오류를 냅니다(실제로 발생 0).
   - 두 키는 hsan과 함께 쓰면 오류를 냅니다.
7. **`corr_offdiag_mean`의 표본**: `geometry`와 같은 4096 노드 표본입니다. 분산은 모분산(ddof 0)이고, 분산 ≤ 1e-12인 차원은 `dead_dims`로 셌습니다.
8. **smoke 결과 경로**: 지시서가 정한 `/tmp/hw_smoke`를 썼습니다. 지시서 3개 arm에 far와 qneg_rand도 함께 확인했습니다.
9. **복제 비교 출처**: HW05의 `v500_naq_base_e4000` 결과(같은 결과 루트)와 seed별로 비교했습니다.
10. **G-table 실행 중 회귀 run**: CPU 계산인 G-table과 GPU 4장에서 동시에 돌렸습니다.

## 6. 제외·실패·붕괴 run

없습니다. 220/220 completed, 재시도 0, NaN 0, 붕괴 0입니다. `_regress/HW06/`는 집계에서 뺐습니다.

## 7. 소요 시간, 환경, HEAD

- **시간**:
  - G-table: 11분 48초. 두 데이터셋의 상위 500 표와 k-means(n_init 4)를 포함합니다.
  - 220 run: 12:16:27 → 13:23:12 UTC (66분 45초).
  - run당 시간(arm별 최소 / 중앙값 / 최대): `e4k_naq` 111 / 127 / 142 s, `e4k_rand_m200` 130 / 146 / 160 s, far·band·clus·qpos 120–155 s.
  - `StrategyNegatives` 설정 시간은 캐시를 읽어 0.0–0.1 s입니다.
- **환경**: iislab4, RTX 3090 × 8 (`GPUS=0,1,2,3,4,5,6,7`), conda `taskns`. 시작 시 GPU 메모리 사용 0 MiB × 8.
- **HEAD**: 작업 시작 `3b62b03`. HW06-a `5bd6ce5`, HW06-b `d99f115`.
- **§0 기록**:
  - GCN 출력 차원 z = 256 (`args.latent` 256, `parse_args_proto` 기본값).
  - HW05 `v500_naq_base_e4000` 설정 키(병합 후, defaults 제외): `loss: original`, `valid_tasks: 500`, `episodes: 4000`, `note`.
  - HW05 `v500_xneg_rand_m20` 설정 키: `loss: original`, `valid_tasks: 500`, `extra_neg: {mode: random, m: 20}`, `geom_diag: true`, `note`. episodes는 기본 2000입니다.
