# HW03b [iislab4] — validation 500 + 대조군(`self`, `global_random`) + λ sweep + 차원별 Dirichlet 에너지

2026-11-08 · 서버 iislab4 · 브랜치 `hl/hsan-weighting` · 작업 시작 HEAD `fddf4b0` · 지시서 `instructions/HW03b_iislab4.md` · 사전 등록 HWP041–HWP048

주 결과는 `test_acc`(validation 500 기준 best epoch의 test)입니다. `test_acc_at_best50`(앞 50개 기준 checkpoint의 test)은 측정용이며 따로 적었습니다. 기존 validation 50 run과는 짝짓지 않았습니다.

## 1. 결론

1. 짝지은 `test_acc` Δ, `− v500_naq_base` (Cora_full / Amazon_clothing):

   | arm | λ 0.1 | λ 1.0 | 비고 |
   |---|---|---|---|
   | `raw` | +0.50 / +0.30 | +1.46 / +0.55 | λ에 따라 단조 증가 |
   | `grand` | +0.97 / +0.79 | +1.86 / +0.64 | — |
   | `self` | −0.01 / −0.04 | −0.04 / −0.01 | — |

2. `raw − grand`는 모든 λ·두 ds에서 음수입니다(−0.94 ~ −0.09, 양수 seed 0–4/10).
   - `grand` 양성의 노드 클래스 일치 비율(`nbr_eq_support`)은 0.0225 / 0.0203입니다.
   - `raw` 양성은 0.6257 / 0.6629입니다.
3. 차원별 Dirichlet 에너지 E_raw와 nnz의 Spearman은 0.9503 / 0.8069입니다.
   - E_raw 상위 K와 E_rq 상위 K의 겹침은 K = 16, 64에서 0입니다.

## 2. 핵심 표

### 2.1 test 정확도 (11 arm)

- 값: test 정확도(%), 5-way 5-shot, test 500 task.
- 셀: 10 seed(1212–1221)의 평균 ± 표준편차(ddof=0).
- 붕괴: 500 기준 best validation ≤ 25%인 run 수.
- 출처: `results/hw_iislab4/summary.md`, `summary_test_acc_at_best50.md`.

| arm | Cora_full `test_acc` | 붕괴 | Cora_full `test_acc_at_best50` (측정용) | Amazon_clothing `test_acc` | 붕괴 | Amazon_clothing `test_acc_at_best50` (측정용) |
|---|---|---|---|---|---|---|
| `v500_naq_base` | 77.71 ± 0.52 | 0 | 77.90 ± 0.37 | 90.80 ± 0.21 | 0 | 90.75 ± 0.21 |
| `v500_raw_l01` | 78.21 ± 0.47 | 0 | 78.08 ± 0.39 | 91.10 ± 0.15 | 0 | 91.05 ± 0.25 |
| `v500_self_l01` | 77.70 ± 0.53 | 0 | 77.72 ± 0.64 | 90.76 ± 0.30 | 0 | 90.77 ± 0.21 |
| `v500_grand_l01` | 78.68 ± 0.47 | 0 | 78.66 ± 0.47 | 91.59 ± 0.36 | 0 | 91.65 ± 0.22 |
| `v500_raw_l03` | 78.57 ± 0.58 | 0 | 78.50 ± 0.61 | 91.26 ± 0.21 | 0 | 91.10 ± 0.31 |
| `v500_raw_l07` | 78.93 ± 0.55 | 0 | 78.95 ± 0.49 | 91.28 ± 0.23 | 0 | 91.15 ± 0.30 |
| `v500_raw_l10` | 79.17 ± 0.58 | 0 | 79.09 ± 0.55 | 91.35 ± 0.21 | 0 | 91.18 ± 0.26 |
| `v500_grand_l03` | 79.51 ± 0.38 | 0 | 79.37 ± 0.41 | 91.74 ± 0.33 | 0 | 91.72 ± 0.46 |
| `v500_grand_l07` | 79.56 ± 0.40 | 0 | 79.54 ± 0.53 | 91.61 ± 0.30 | 0 | 91.64 ± 0.20 |
| `v500_grand_l10` | 79.57 ± 0.43 | 0 | 79.55 ± 0.42 | 91.44 ± 0.26 | 0 | 91.27 ± 0.36 |
| `v500_self_l10` | 77.67 ± 0.59 | 0 | 77.78 ± 0.64 | 90.79 ± 0.22 | 0 | 90.81 ± 0.22 |

### 2.2 λ 표

- 값: 같은 seed끼리 짝지은 `test_acc` 차이(%p).
- 셀: 평균 ± 표준편차(ddof=0) (양수 seed k/10).
- 약칭: `raw` = `v500_raw_l*`, `grand` = `v500_grand_l*`, `self` = `v500_self_l*`, `naq` = `v500_naq_base`.

**Cora_full**

| λ | `raw − naq` | `grand − naq` | `raw − grand` | `self − naq` |
|---|---|---|---|---|
| 0.1 | +0.50 ± 0.39 (9/10) | +0.97 ± 0.48 (10/10) | −0.47 ± 0.22 (0/10) | −0.01 ± 0.04 (2/10) |
| 0.3 | +0.86 ± 0.54 (10/10) | +1.80 ± 0.49 (10/10) | −0.94 ± 0.33 (0/10) | — |
| 0.7 | +1.21 ± 0.45 (10/10) | +1.85 ± 0.73 (10/10) | −0.63 ± 0.60 (1/10) | — |
| 1.0 | +1.46 ± 0.49 (10/10) | +1.86 ± 0.63 (10/10) | −0.40 ± 0.47 (2/10) | −0.04 ± 0.18 (6/10) |

**Amazon_clothing**

| λ | `raw − naq` | `grand − naq` | `raw − grand` | `self − naq` |
|---|---|---|---|---|
| 0.1 | +0.30 ± 0.20 (9/10) | +0.79 ± 0.28 (10/10) | −0.49 ± 0.38 (1/10) | −0.04 ± 0.14 (6/10) |
| 0.3 | +0.46 ± 0.23 (9/10) | +0.95 ± 0.30 (10/10) | −0.48 ± 0.37 (1/10) | — |
| 0.7 | +0.49 ± 0.19 (10/10) | +0.81 ± 0.29 (10/10) | −0.33 ± 0.39 (3/10) | — |
| 1.0 | +0.55 ± 0.19 (10/10) | +0.64 ± 0.26 (10/10) | −0.09 ± 0.31 (4/10) | −0.01 ± 0.02 (4/10) |

**그 밖의 짝지은 Δ** (Cora_full / Amazon_clothing)
- `v500_raw_l10 − v500_raw_l01`: +0.96 ± 0.28 (10/10) / +0.25 ± 0.14 (9/10)
- `v500_grand_l01 − v500_raw_l01`: +0.47 ± 0.22 (10/10) / +0.49 ± 0.38 (9/10)

### 2.3 선택 잡음

- `same_epoch`: 500 기준과 50 기준 best epoch이 같은 seed의 비율입니다.
- |두 기준 차|: seed별 |`test_acc` − `test_acc_at_best50`|의 평균(%p)입니다.
- sd 두 열: `paired_v500_naq_base`의 짝지은 Δ 표준편차입니다. 하나는 `test_acc`로, 하나는 `test_acc_at_best50`으로 계산했습니다.

**Cora_full**

| arm | `same_epoch` | \|두 기준 차\| | sd (`test_acc`) | sd (`at_best50`) |
|---|---|---|---|---|
| `v500_naq_base` | 0.4 | 0.28 | — | — |
| `v500_raw_l01` | 0.6 | 0.14 | 0.39 | 0.43 |
| `v500_self_l01` | 0.4 | 0.32 | 0.04 | 0.35 |
| `v500_grand_l01` | 0.8 | 0.03 | 0.48 | 0.28 |
| `v500_raw_l03` | 0.6 | 0.07 | 0.54 | 0.47 |
| `v500_raw_l07` | 0.5 | 0.15 | 0.45 | 0.33 |
| `v500_raw_l10` | 0.4 | 0.20 | 0.49 | 0.37 |
| `v500_grand_l03` | 0.6 | 0.16 | 0.49 | 0.39 |
| `v500_grand_l07` | 0.5 | 0.23 | 0.73 | 0.40 |
| `v500_grand_l10` | 0.7 | 0.08 | 0.63 | 0.30 |
| `v500_self_l10` | 0.4 | 0.36 | 0.18 | 0.35 |
| sd 평균 (10 arm) | | | 0.442 | 0.368 |

**Amazon_clothing**

| arm | `same_epoch` | \|두 기준 차\| | sd (`test_acc`) | sd (`at_best50`) |
|---|---|---|---|---|
| `v500_naq_base` | 0.4 | 0.10 | — | — |
| `v500_raw_l01` | 0.3 | 0.11 | 0.20 | 0.19 |
| `v500_self_l01` | 0.4 | 0.12 | 0.14 | 0.06 |
| `v500_grand_l01` | 0.6 | 0.17 | 0.28 | 0.19 |
| `v500_raw_l03` | 0.6 | 0.16 | 0.23 | 0.19 |
| `v500_raw_l07` | 0.5 | 0.17 | 0.19 | 0.24 |
| `v500_raw_l10` | 0.5 | 0.20 | 0.19 | 0.39 |
| `v500_grand_l03` | 0.5 | 0.12 | 0.30 | 0.42 |
| `v500_grand_l07` | 0.4 | 0.14 | 0.29 | 0.26 |
| `v500_grand_l10` | 0.4 | 0.22 | 0.26 | 0.44 |
| `v500_self_l10` | 0.4 | 0.10 | 0.02 | 0.07 |
| sd 평균 (10 arm) | | | 0.210 | 0.247 |

### 2.4 50 기준 값과 기존 run 비교 (기록만)

- 값: 같은 seed의 |`test_acc_at_best50` − 기존 run test|(%p)입니다.
- 기존 run은 validation 50으로 학습한 결과이고 `results/hw_iislab4/{naq_base,nbr_raw}`에 있습니다.

| 새 arm vs 기존 | ds | 평균 \|Δ\| | 최대 \|Δ\| | best epoch(50 기준)이 기존과 같은 seed |
|---|---|---|---|---|
| `v500_naq_base` vs `naq_base` | Cora_full | 0.20 | 0.57 | 5/10 |
| `v500_naq_base` vs `naq_base` | Amazon_clothing | 0.02 | 0.07 | 7/10 |
| `v500_raw_l01` vs `nbr_raw` | Cora_full | 0.11 | 0.54 | 7/10 |
| `v500_raw_l01` vs `nbr_raw` | Amazon_clothing | 0.07 | 0.35 | 8/10 |

### 2.5 진단

- 값: 마지막 평가 구간(episode 1921–2000)의 episode 평균을 10 seed에 걸쳐 평균한 값입니다.
- 쌍: (pseudo-query, 같은 pseudo-class support) 쌍입니다.
- 쌍 비율 열: `nbr_eq_support`, `nbr_eq_query`, `query_eq_support`(측정 전용 레이블), `nbr_in_own_queries`.
- 쌍 평균 열: `pos_nbr_cos_raw`, `pos_nbr_cos_a2x`(query와 고른 양성 사이 cos, L2(X)·L2(Â²X)).
- 같은 pick 안에서 λ가 다른 arm은 모든 열 값이 같았습니다. 그래서 pick별로 한 행만 적습니다.
- `v500_naq_base`의 `query_eq_support`는 0.3520 / 0.6022입니다.

| pick (arm) | ds | `nbr_eq_support` | `nbr_eq_query` | `query_eq_support` | `nbr_in_own_queries` | `pos_nbr_cos_raw` | `pos_nbr_cos_a2x` |
|---|---|---|---|---|---|---|---|
| max (`v500_raw_l*`) | Cora_full | 0.6257 | 0.3400 | 0.3520 | 0.3393 | 0.1732 | 0.5380 |
| max (`v500_raw_l*`) | Amazon_clothing | 0.6629 | 0.5577 | 0.6022 | 0.4145 | 0.4455 | 0.7055 |
| self (`v500_self_l*`) | Cora_full | 1.0000 | 0.3520 | 0.3520 | 0.0000 | 0.2464 | 0.5188 |
| self (`v500_self_l*`) | Amazon_clothing | 1.0000 | 0.6022 | 0.6022 | 0.0000 | 0.5806 | 0.7111 |
| global_random (`v500_grand_l*`) | Cora_full | 0.0225 | 0.0218 | 0.3520 | 0.0004 | 0.0522 | 0.2852 |
| global_random (`v500_grand_l*`) | Amazon_clothing | 0.0203 | 0.0203 | 0.6022 | 0.0003 | 0.0317 | 0.0971 |

### 2.6 best epoch 분포

- 값: best validation epoch입니다. 셀은 seed 1212 → 1221 순서입니다.

**Cora_full**

| arm | 500 기준 | 50 기준 |
|---|---|---|
| `v500_naq_base` | 1760,1920,1600,1120,1920,2000,2000,2000,1920,2000 | 2000,1920,1760,2000,2000,2000,2000,1760,2000,2000 |
| `v500_raw_l01` | 1760,1920,1600,1760,960,2000,2000,2000,1920,1920 | 2000,1920,1440,1920,960,2000,1600,2000,1920,1920 |
| `v500_self_l01` | 1760,1920,1600,1120,1920,2000,2000,2000,1920,2000 | 2000,1120,1600,2000,2000,2000,2000,2000,2000,1920 |
| `v500_grand_l01` | 1760,1920,1600,2000,1600,2000,2000,2000,1920,1920 | 2000,1920,1600,2000,2000,2000,2000,2000,1920,1920 |
| `v500_raw_l03` | 1760,1920,1600,2000,960,1920,2000,2000,1920,1920 | 2000,1600,2000,1920,960,1920,2000,2000,1920,1920 |
| `v500_raw_l07` | 1760,1920,1600,1920,960,1920,2000,2000,1920,1920 | 2000,1600,1600,1920,1920,2000,2000,2000,1920,2000 |
| `v500_raw_l10` | 1760,1920,2000,2000,1920,1920,2000,2000,1920,2000 | 2000,1600,1440,1920,2000,1440,2000,2000,1920,2000 |
| `v500_grand_l03` | 1760,1920,1760,2000,1280,1920,2000,2000,1920,1920 | 1760,1920,2000,1920,2000,1920,2000,1920,1920,1920 |
| `v500_grand_l07` | 1760,1920,2000,2000,1280,1920,1600,2000,1920,1920 | 1920,1600,2000,2000,1280,1920,2000,2000,1760,1600 |
| `v500_grand_l10` | 1760,1920,2000,2000,1280,1920,2000,2000,1920,1920 | 1760,1600,2000,2000,2000,1920,2000,2000,1600,1920 |
| `v500_self_l10` | 1760,1920,2000,1120,1920,2000,2000,2000,1920,2000 | 2000,1120,1600,2000,2000,2000,2000,2000,1760,2000 |

**Amazon_clothing**

| arm | 500 기준 | 50 기준 |
|---|---|---|
| `v500_naq_base` | 2000,2000,2000,2000,2000,2000,1920,1760,1760,2000 | 1600,1920,2000,2000,1440,1440,1920,1760,2000,1920 |
| `v500_raw_l01` | 2000,1920,1920,2000,1920,2000,1920,1760,2000,1920 | 1280,1760,2000,2000,960,1440,1920,2000,1920,1920 |
| `v500_self_l01` | 2000,2000,2000,2000,2000,1120,1920,1760,1760,2000 | 1600,2000,2000,2000,1600,1440,1920,2000,2000,1920 |
| `v500_grand_l01` | 1920,1760,2000,1920,1760,1120,1920,1920,2000,1920 | 1920,1760,2000,1760,1760,1920,1920,1760,2000,1760 |
| `v500_raw_l03` | 2000,1920,2000,2000,1920,2000,1760,2000,2000,1920 | 1760,1920,2000,2000,960,1280,1920,2000,2000,1920 |
| `v500_raw_l07` | 1760,1760,2000,2000,1920,1760,2000,2000,2000,2000 | 1760,1760,2000,1600,1760,1120,1280,2000,2000,1920 |
| `v500_raw_l10` | 1920,1920,2000,2000,1920,1760,2000,2000,2000,1920 | 1920,1760,1120,1600,1760,1760,1280,2000,2000,1920 |
| `v500_grand_l03` | 1920,1760,2000,1920,1600,1120,1600,1920,2000,2000 | 1920,1760,2000,1440,1760,960,1440,1280,2000,2000 |
| `v500_grand_l07` | 1920,1920,1600,2000,1760,1120,1760,1920,2000,2000 | 1920,1760,1920,2000,1600,2000,1760,2000,2000,1920 |
| `v500_grand_l10` | 1920,1920,1600,2000,1440,1120,1760,2000,2000,2000 | 1280,1760,1760,2000,1440,1280,1120,1280,2000,2000 |
| `v500_self_l10` | 2000,2000,2000,2000,2000,1600,1920,1760,1760,2000 | 1600,1920,2000,2000,1600,1440,1920,1600,2000,2000 |

### 2.7 차원별 Dirichlet 에너지 (학습 없음)

- 출처: `results/measure/HW03b_energy_<ds>.{json,md}`.
- 입력 X: 학습 경로의 `data.features`(load_data → add_diffusion; add_diffusion 전후 동일)입니다.
- 그래프: `normalize_graph(data.edge_index)`입니다.

**정의**
- E_raw = ½ Σ_(i,j) A_ij (x_im − x_jm)² (무향 edge 1회씩)
- E_rq = E_raw / ‖x_:,m‖²
- nnz = x_:,m ≠ 0인 노드 수

**특징 행렬**

| 항목 | Cora_full | Amazon_clothing |
|---|---|---|
| X | 19793 × 8710, float32, 이진(0/1) | 24919 × 9034, float32, 정수 0–99(서로 다른 값 32개), 이진 아님 |
| 행 정규화 | 없음 | 없음 |
| 행 합 최소 / 중앙 / 최대 | 2 / 54 / 293 | 2 / 32 / 527 |
| ‖x_:,m‖ = 0인 차원 | 0 | 0 |
| E_raw = 0인 차원 | 0 | 10 |

**분포** (최소 / q25 / 중앙 / q75 / 최대)

| 값 | Cora_full | Amazon_clothing |
|---|---|---|
| E_raw | 7 / 95.25 / 219 / 656 / 28353 | 0 / 58 / 173 / 595 / 302841 |
| E_rq | 0.70 / 4.81 / 5.88 / 7.00 / 27.54 | 0 / 2.87 / 4.78 / 7.42 / 125.39 |
| nnz | 10 / 16 / 35 / 105 / 10002 | 1 / 8 / 18 / 53.75 / 5463 |

**Spearman ρ**

| 쌍 | Cora_full | Amazon_clothing |
|---|---|---|
| (E_raw, nnz) | 0.9503 | 0.8069 |
| (E_raw, E_rq) | 0.3703 | 0.5327 |
| (E_rq, nnz) | 0.1029 | 0.1919 |

**상위 K 겹침 비율**

| K | E_raw ∩ nnz (Cora_full / Amazon_clothing) | E_raw ∩ E_rq (Cora_full / Amazon_clothing) |
|---|---|---|
| 16 | 1.0000 / 0.3750 | 0.0000 / 0.0000 |
| 64 | 0.8750 / 0.5312 | 0.0000 / 0.0000 |
| 512 | 0.9297 / 0.7148 | 0.0000 / 0.0645 |

- selftest: 작은 그래프 3개 × 6열에서 E_raw와 fᵀLf의 차이가 18/18 모두 0입니다(허용 1e-9).

### 2.8 §0 확인, G-valid, 회귀, 불변, md5

**손실 형태** (`hw/losses.py` `episode_loss`, hsan 없음)
- 기호:
  - z = GCN 임베딩
  - S = support 5개(pseudo-class 0–4), Q = pseudo-query 50개
  - P(q) = {s : y_s = y_q} (한 개)
  - T = 0.5
- L_original:
  - 로짓 ℓ_qs = −‖z_q − z_s‖² / T
  - L_original = mean_q [ logsumexp_s ℓ_qs − logsumexp_{s∈P(q)} ℓ_qs + log |P(q)| ]
- L_neighbor:
  - 같은 InfoNCE에서 support 임베딩 z_s를 (q, s)마다 고른 노드 n(q, s)의 임베딩 z_{n(q,s)}로 바꿉니다.
  - 로짓 ℓ'_qs = −‖z_q − z_{n(q,s)}‖² / T이고, 양성 마스크는 같습니다.
  - 이웃 임베딩은 prototype을 대체하는 별도 항입니다. 음성은 다른 4개 support 각각에 대해 고른 노드(n(q, s′))입니다.
- 총손실: L = L_original + λ · L_neighbor.
- optimizer: `torch.optim.Adam`(lr 1e-4, betas (0.9, 0.999), eps 1e-8, weight_decay 0)입니다.
  - 그 밖의 정규화 항은 없습니다.
  - `args.l2_penalty`는 `meta.py`의 LogisticRegression 미세조정에만 쓰이고, `hw/run.py` 경로에서는 쓰이지 않습니다.
- 동일성 수치: seed 1212, Cora_full, 학습 전 모델, 에피소드 1–3에서 n = s(`pick: self`)일 때 |L_neighbor − L_original| = 0.0, 0.0, 0.0입니다.
  - L_original 값은 1.5540, 1.5263, 1.5648입니다.

**validation 캐시**
- 기존 `save/downstream_tasks/<ds>/5way_5shot.pkl`:
  - 내용은 튜플 (valid_pool, test_pool)이고 키는 없습니다.
  - valid 50, test 500 task입니다.
  - task는 (id_support ndarray[25], y_spt LongTensor[25], id_query ndarray[40], y_query LongTensor[40]) 형식이고 5-way, k 5, q 8입니다.
  - 생성 코드는 `task_generator.downstream_task_generator`(→ `supervised_task_generator`, 전역 `random.sample`)입니다.
- 기존 validation 클래스 수는 Cora_full 20, Amazon_clothing 17입니다.

**G-valid** (`results/hw_iislab4/HW03b_valid500/<ds>.json`)
- (a) 앞 50개가 기존 task와 원소 단위로 같습니다. 두 ds 모두 true입니다.
- (b) 뒤 450개의 클래스가 기존 validation 클래스 안에 있습니다. 두 ds 모두 true이며, 사용한 클래스는 20 / 17입니다.
- (c) 기존 md5가 변하지 않았습니다. 두 ds 모두 true입니다.
- 판정: 통과입니다.
- 새 파일 md5: Cora_full `d8fdb82affdaec55ee7b4f26cb78a63c`, Amazon_clothing `0920f5d7800670c4597697cb0e80feb2`.
- 뒤 450개 중 앞 50개와 클래스·노드 집합이 완전히 같은 task는 0 / 0개입니다.

**회귀 확인** (`_regress/HW03b/`, seed 1212, test 정확도 %)

| arm | ds | 기존 | HW03b | Δ |
|---|---|---|---|---|
| `naq_base` | Cora_full | 77.96 | 77.72 | −0.24 |
| `naq_base` | Amazon_clothing | 90.60 | 90.60 | −0.00 |
| `nbr_raw` | Cora_full | 78.15 | 78.09 | −0.05 |
| `nbr_raw` | Amazon_clothing | 90.89 | 90.87 | −0.02 |

- 판정: 최대 |Δ| 0.24 ≤ 0.30, 통과입니다.

**불변 확인** (`git diff fddf4b0`)

| 파일 | 삭제 줄 | 추가 줄 |
|---|---|---|
| `hw/neighbors.py` | 0 | 40 |
| `hw/evaluate.py` | 0 | 0 |
| `hw/run.py` | 0 | 39 |

- `select_neighbors_pick` 안에 추가한 줄은 아래 2줄입니다. 나머지는 파일 끝의 새 함수입니다.

```diff
     if pick == SUP_COS:  # HW04a: separate path; the max/contrast/random lines below are unchanged
         return select_neighbors_sup_cos(neighbor_lookup, support_ids, query_ids, selection_out)
+    if pick in CONTROL_PICKS:  # HW03b: separate path; the lines below are unchanged
+        return select_neighbors_control(pick, selection_embeddings_normalized.shape[0], support_ids, query_ids, generator, selection_out)
```

- `hw/run.py`의 추가분은 모두 `if v500:` 또는 `if pick == "global_random":` 아래에 있습니다. `valid_tasks: 50`이면 실행되지 않습니다.
- `hw/evaluate.py`는 바꾸지 않았습니다. 앞 50개 정확도는 같은 함수를 `valid_pool[:50]`에 한 번 더 호출해 얻습니다.

**기존 캐시 md5** (작업 전 = 작업 후)
- Cora_full `cfef6a69020acde861b554cd2e8ac9c7`
- Amazon_clothing `ca3edaca02d541cbc0ee15a2122ef24a`

## 3. 게이트·사전 등록 대조

| 게이트 | 값 | 판정 |
|---|---|---|
| G-valid | (a)(b)(c) 두 ds 모두 true | 통과 |
| 회귀 확인 | 최대 \|Δ\| 0.24 | 통과 |
| G6 | 220/220 completed, 재시도 0 | 통과 |
| G7 (기록) | 11 arm 모두 붕괴 0 | — |

태그 `hw03b`를 붙였습니다.

| ID | 값의 종류 | 기준 | 값 (Cora_full / Amazon_clothing) | 기준 충족 |
|---|---|---|---|---|
| HWP041 | 짝지은 Δ | `self_l01`, `self_l10` − naq 모두 \|Δ\| ≤ 0.15 | l01 −0.01 / −0.04, l10 −0.04 / −0.01 | 충족 |
| HWP042 | 짝지은 Δ | \|`grand_l01 − raw_l01`\| ≤ 0.15 | +0.47 / +0.49 | 미충족 |
| HWP043 | 짝지은 Δ | `grand_l01 − naq` > 0, 양수 ≥ 7/10 | +0.97 (10/10) / +0.79 (10/10) | 충족 |
| HWP044 | 짝지은 Δ의 차 | (raw − grand)@1.0 − (raw − grand)@0.1 ≥ +0.3 | +0.07 / +0.40 | 미충족 (Cora_full) |
| HWP045 | 짝지은 Δ | `raw_l10 − raw_l01` ≥ +0.4, 양수 ≥ 7/10, Cora_full | +0.96 (10/10) | 충족 |
| HWP046 | 짝지은 Δ | `grand_l10 − naq` < 0 | +1.86 / +0.64 | 미충족 |
| HWP047 | 짝지은 Δ의 sd 평균 | `test_acc` < `at_best50` | 0.442 vs 0.368 / 0.210 vs 0.247 | 미충족 (Cora_full) |
| HWP048 | Spearman | (E_raw, nnz) ≥ 0.8 | 0.9503 / 0.8069 | 충족 |

판정은 사람이 합니다. "기준 충족" 열은 기준식을 적용한 결과입니다.

## 4. 수행 내용

**`ceae6c4` HW03b-a**
- `hw/valid500.py`(새 파일): validation 500 캐시와 G-valid를 만듭니다.
- `hw/neighbors.py`: `pick` `self`, `global_random`(`select_neighbors_control`)을 추가했습니다.
- `hw/run.py`: `valid_tasks` 500 경로, 두 checkpoint test, `global_random`용 Generator(seed + 104729)를 추가했습니다.
- `hw/summarize.py`: `--metric`을 추가했습니다.
- `configs/hw.json`: 11 arm을 추가했습니다.
- 함께 커밋: G-valid 보고(`HW03b_valid500/`), 회귀 run(`_regress/HW03b/`), CLAUDE.md 개정 6, 지시서.
- smoke(epochs 4, `/tmp/hw_smoke`, 커밋 안 함): `v500_self_l01`, `v500_grand_l10` Cora_full 각 1회 completed.
  - `test_acc`, `test_acc_at_best50`, `same_epoch` 키가 기록되었습니다.
  - `same_epoch`는 `v500_self_l01`에서 true, `v500_grand_l10`에서 false였습니다. false인 경우 두 test 값이 서로 다르게 측정되었습니다.

**`fc0666f` HW03b-b**
- 220 run과 summarize 결과(`test_acc`, `test_acc_at_best50` 두 벌)입니다.

**`c3cbf03` HW03b-c**
- `hw/structure.py`에 `incidence`, `dirichlet_energy_per_dim`, `laplacian_quadratic`을 추가했습니다.
- `hw/measure_positive.py`에 `--energy` 옵션을 추가했습니다.
- `results/measure/HW03b_energy_*`를 커밋했습니다.

**이 커밋 (HW03b-d)**
- 이 보고서입니다.

## 5. 판단한 사항

1. **새 450개를 뽑은 클래스 집합**
   - "기존 validation 클래스 집합"은 기존 50개 task에 나오는 클래스(Cora_full 20, Amazon_clothing 17)로 정했습니다.
   - `data.py`는 run seed 뒤의 전역 `random`으로 클래스를 나눕니다(`load_data`의 `random.sample`/`random.shuffle`). 그래서 `Data.class_list_valid`는 seed마다 다르고, 캐시를 만든 분할과도 다릅니다. seed 없이 불러온 `class_list_valid`를 쓴 첫 시도에서는 그 클래스 목록과 기존 50개 task의 클래스가 달랐습니다(기존 클래스 중 Cora_full 13개, Amazon_clothing 11개가 목록에 없음). 그래서 클래스 집합을 바꿔 캐시를 다시 만들었고, 첫 시도의 파일은 덮어썼습니다.
2. **전용 RNG를 쓴 방법**
   - `supervised_task_generator`는 그대로 재사용했습니다.
   - 대신 호출 동안만 `task_generator.random`을 `random.Random(20261108)` 인스턴스로 바꿨습니다. 이 함수는 `random.sample`로만 뽑습니다.
   - 전역 RNG는 소비하지 않았습니다. `numpy`/`torch` Generator는 이 함수가 쓰지 않아 만들지 않았습니다.
3. **checkpoint 보관 위치**
   - 500 기준 checkpoint는 기존 줄(`deepcopy(state_dict)`, GPU)을 그대로 씁니다.
   - 50 기준 checkpoint만 CPU에 복사합니다. 지시서의 "두 checkpoint를 CPU에" 중 앞쪽은 기존 줄을 바꾸지 않기 위해 GPU 그대로 두었습니다.
4. **`global_random` 추출 방식**
   - 제외할 id(support, query, 둘이 같으면 하나)의 수 m에 대해 U{0..N−1−m}에서 뽑습니다.
   - 그다음 제외 id를 오름차순으로 건너뛰도록 값을 옮깁니다.
   - 7노드 예제 20000회에서 허용 노드 빈도가 1/(N−m) ± 0.01 안에 있었고, 제외 노드 빈도는 0이었습니다.
5. **회귀 결과 위치**
   - 회귀 run은 HW03a·HW04a 기록을 덮지 않도록 `_regress/HW03b/`에 썼습니다.
6. **진단 표 행 정리**
   - §2.5에서 λ만 다른 arm은 진단값이 같아 pick별 한 행으로 줄였습니다.
7. **에너지 계산 방법**
   - 에너지는 부호 있는 incidence B로 ‖B x_:,m‖²를 float64로 계산했습니다.
   - 상위 K 동률은 차원 번호 순으로 정했습니다.

## 6. 제외·실패·붕괴 run

- 없습니다. 220/220 completed, 재시도 0, 붕괴 0입니다.

## 7. 소요 시간, 환경, HEAD

- 실행 큐: 220 run, 8 worker(GPUS=0–7), 07:49:25–08:24:55(35.5분).
  - run당 58–83 s
- 에너지 측정은 CPU에서 22.45 s(Cora_full), 31.65 s(Amazon_clothing)였습니다.
- 환경: iislab4, RTX 3090 × 8, conda `taskns`(python 3.9.25, torch 2.4.1+cu121, scipy 1.13.1).
- 작업 시작 HEAD: `fddf4b0`.
- 커밋: `ceae6c4`(a), `fc0666f`(b), `c3cbf03`(c), 보고서(d).
