# HW08 [iislab4] — raw·l2 평가 병기 (`eval_l2`) + best checkpoint 저장 (`save_best`) + 핵심 5 arm 재실행 + 학습 없는 기준선 재측정

2026-11-11 · 서버 iislab4 · 브랜치 `hl/hsan-weighting` · 작업 시작 HEAD `67c9746` · 지시서 `instructions/HW08_iislab4.md` · 사전 등록 HWP079–HWP087

모든 학습 arm은 base `naq_base`, `valid_tasks: 500`, `episodes: 4000`, `geom_diag: true`, `eval_l2: true`, `save_best: true`입니다. 숫자 두 개는 Cora_full / Amazon_clothing 순서입니다.

지표 네 개의 정의 (CLAUDE.md §4.2, §5):

| 키 | checkpoint 선택 기준 | test 평가 |
|---|---|---|
| `test_acc` | raw validation 500 | raw |
| `test_acc_l2_at_bestraw` | raw validation 500 | l2 |
| `test_acc_l2` | l2 validation 500 | l2 |
| `test_acc_raw_at_bestl2` | l2 validation 500 | raw |

raw = `hw.evaluate.evaluate`(인코더 출력 그대로, 제곱 유클리드), l2 = `hw.evaluate.evaluate_l2`(행 L2 정규화 뒤 같은 `evaluate`)입니다.

## 1. 결론

1. 같은 checkpoint(raw 선택)에서 l2 평가 − raw 평가는 다섯 arm 모두 Cora_full +0.51 ~ +0.65, Amazon_clothing +0.69 ~ +1.16입니다(전부 10/10).
2. `rand_m20 − naq`는 l2(`test_acc_l2`) +1.15 (9/10) / +0.22 (8/10), raw +1.17 (10/10) / +0.62 (10/10)입니다. `clus_m20 − rand_m20`은 l2 +0.46 (10/10) / +0.12 (7/10), raw +0.35 (8/10) / −0.04 (6/10)입니다.
3. l2 평균 최고값은 `l2_clus_m20` 80.74 / 92.94이고, 학습 없는 Â²X(L2)는 80.12 / 92.72입니다. raw `test_acc`의 `l2_X − e4k_X` seed별 |Δ| 평균은 0.05–0.18입니다.

## 2. 핵심 표

### 2.1 정확도 표

- 값: test 정확도(%), 5-way 5-shot, test 500 task. 셀은 10 seed의 평균 ± 표준편차(ddof=0)입니다.
- 붕괴: 선택 기준 validation(500) ≤ 25%인 run 수(raw 기준 / l2 기준). NaN: status `nan`인 run 수.
- 출처: `results/hw_iislab4/HW08_tables.json` (`accuracy`), `summary{,_test_acc_l2,_test_acc_l2_at_bestraw,_test_acc_raw_at_bestl2}.md`.

| arm | ds | raw 선택·raw 평가 (`test_acc`) | raw 선택·l2 평가 (`test_acc_l2_at_bestraw`) | l2 선택·l2 평가 (`test_acc_l2`) | l2 선택·raw 평가 (`test_acc_raw_at_bestl2`) | completed | 붕괴 (raw/l2) | NaN |
|---|---|---|---|---|---|---|---|---|
| `l2_naq` | Cora_full | 78.66 ± 0.52 | 79.17 ± 0.51 | 79.14 ± 0.35 | 78.50 ± 0.38 | 10/10 | 0/0 | 0 |
| `l2_rand_m20` | Cora_full | 79.83 ± 0.45 | 80.37 ± 0.35 | 80.29 ± 0.31 | 79.40 ± 0.73 | 10/10 | 0/0 | 0 |
| `l2_rand_m50` | Cora_full | 79.94 ± 0.39 | 80.59 ± 0.29 | 80.48 ± 0.24 | 79.42 ± 0.67 | 10/10 | 0/0 | 0 |
| `l2_rand_m200` | Cora_full | 80.22 ± 0.32 | 80.74 ± 0.31 | 80.64 ± 0.36 | 79.32 ± 0.88 | 10/10 | 0/0 | 0 |
| `l2_clus_m20` | Cora_full | 80.19 ± 0.42 | 80.74 ± 0.29 | 80.74 ± 0.31 | 80.04 ± 0.53 | 10/10 | 0/0 | 0 |
| `l2_naq` | Amazon_clothing | 91.37 ± 0.28 | 92.54 ± 0.19 | 92.59 ± 0.18 | 91.17 ± 0.66 | 10/10 | 0/0 | 0 |
| `l2_rand_m20` | Amazon_clothing | 91.99 ± 0.18 | 92.91 ± 0.10 | 92.82 ± 0.15 | 91.71 ± 0.37 | 10/10 | 0/0 | 0 |
| `l2_rand_m50` | Amazon_clothing | 92.03 ± 0.21 | 92.85 ± 0.16 | 92.90 ± 0.12 | 91.93 ± 0.39 | 10/10 | 0/0 | 0 |
| `l2_rand_m200` | Amazon_clothing | 92.08 ± 0.18 | 92.77 ± 0.13 | 92.73 ± 0.17 | 91.74 ± 0.62 | 10/10 | 0/0 | 0 |
| `l2_clus_m20` | Amazon_clothing | 91.95 ± 0.20 | 92.97 ± 0.19 | 92.94 ± 0.18 | 91.66 ± 0.52 | 10/10 | 0/0 | 0 |

학습 없는 기준선 (iislab4, §2 HW08-b, `results/measure/HW08_trainfree_<ds>.md`). 값: test 정확도(%), 같은 test 500 task, 학습 없음, 결정적. raw = 정규화 없음, l2 = 행 L2 정규화.

| 특징 | Cora_full raw | Cora_full l2 | Amazon_clothing raw | Amazon_clothing l2 |
|---|---|---|---|---|
| X | 48.07 | 60.50 | 66.58 | 87.61 |
| ÂX | 64.51 | 77.41 | 73.64 | 91.79 |
| Â²X | 67.57 | 80.12 | 83.03 | 92.72 |

- 같은 평가 방식끼리: raw 열의 학습 arm 최고 80.22 / 92.08 vs Â²X raw 67.57 / 83.03, l2 열(`test_acc_l2`)의 학습 arm 최고 80.74 / 92.94 vs Â²X l2 80.12 / 92.72입니다.

### 2.2 짝지은 Δ 표

- 값: test 정확도의 같은 seed 짝지은 Δ(%p), "평균 ± 표준편차(ddof=0) (양수 k/10)". 같은 지표끼리만 뺐습니다.
- 출처: `HW08_tables.json` `paired`, `paired_l2_naq*.md`, `paired_l2_rand_m20*.md`.

Cora_full

| Δ | raw 선택·raw 평가 | raw 선택·l2 평가 | l2 선택·l2 평가 | l2 선택·raw 평가 |
|---|---|---|---|---|
| `rand_m20 − naq` | +1.17 ± 0.41 (10/10) | +1.19 ± 0.44 (10/10) | +1.15 ± 0.50 (9/10) | +0.90 ± 0.88 (9/10) |
| `rand_m50 − naq` | +1.28 ± 0.42 (10/10) | +1.41 ± 0.49 (10/10) | +1.34 ± 0.44 (10/10) | +0.92 ± 0.83 (9/10) |
| `rand_m200 − naq` | +1.55 ± 0.38 (10/10) | +1.57 ± 0.47 (10/10) | +1.50 ± 0.50 (10/10) | +0.82 ± 0.94 (8/10) |
| `clus_m20 − naq` | +1.52 ± 0.40 (10/10) | +1.56 ± 0.35 (10/10) | +1.61 ± 0.23 (10/10) | +1.54 ± 0.45 (10/10) |
| `clus_m20 − rand_m20` | +0.35 ± 0.31 (8/10) | +0.37 ± 0.27 (10/10) | +0.46 ± 0.41 (10/10) | +0.64 ± 0.88 (9/10) |
| `rand_m200 − rand_m20` | +0.38 ± 0.26 (10/10) | +0.37 ± 0.27 (9/10) | +0.35 ± 0.26 (9/10) | −0.08 ± 0.79 (6/10) |

Amazon_clothing

| Δ | raw 선택·raw 평가 | raw 선택·l2 평가 | l2 선택·l2 평가 | l2 선택·raw 평가 |
|---|---|---|---|---|
| `rand_m20 − naq` | +0.62 ± 0.23 (10/10) | +0.37 ± 0.21 (9/10) | +0.22 ± 0.17 (8/10) | +0.53 ± 0.50 (9/10) |
| `rand_m50 − naq` | +0.66 ± 0.26 (10/10) | +0.31 ± 0.23 (9/10) | +0.31 ± 0.17 (9/10) | +0.75 ± 0.39 (10/10) |
| `rand_m200 − naq` | +0.70 ± 0.30 (10/10) | +0.23 ± 0.20 (9/10) | +0.14 ± 0.25 (8/10) | +0.56 ± 0.71 (9/10) |
| `clus_m20 − naq` | +0.57 ± 0.30 (10/10) | +0.44 ± 0.23 (9/10) | +0.34 ± 0.18 (10/10) | +0.49 ± 0.49 (9/10) |
| `clus_m20 − rand_m20` | −0.04 ± 0.17 (6/10) | +0.06 ± 0.17 (6/10) | +0.12 ± 0.21 (7/10) | −0.04 ± 0.47 (6/10) |
| `rand_m200 − rand_m20` | +0.09 ± 0.16 (7/10) | −0.14 ± 0.13 (2/10) | −0.09 ± 0.21 (2/10) | +0.03 ± 0.39 (6/10) |

### 2.3 평가 방식 차이 표 (같은 run 안, CLAUDE.md §4.3 예외)

- 값: 같은 run 안의 차이(%p), "평균 ± 표준편차(ddof=0) (양수 k/10)", 괄호 뒤는 seed별 |Δ| 평균.
- A = `test_acc_l2_at_bestraw − test_acc` (같은 checkpoint, 평가 방식 차이). B = `test_acc_l2 − test_acc_l2_at_bestraw` (같은 l2 평가, 선택 기준 차이).

| arm | ds | A | B | B의 seed별 \|Δ\| 평균 |
|---|---|---|---|---|
| `l2_naq` | Cora_full | +0.51 ± 0.17 (10/10) | −0.04 ± 0.29 (1/10) | 0.17 |
| `l2_rand_m20` | Cora_full | +0.53 ± 0.27 (10/10) | −0.08 ± 0.46 (4/10) | 0.28 |
| `l2_rand_m50` | Cora_full | +0.65 ± 0.25 (10/10) | −0.11 ± 0.32 (4/10) | 0.25 |
| `l2_rand_m200` | Cora_full | +0.52 ± 0.19 (10/10) | −0.10 ± 0.38 (3/10) | 0.32 |
| `l2_clus_m20` | Cora_full | +0.55 ± 0.21 (10/10) | +0.01 ± 0.12 (1/10) | 0.05 |
| `l2_naq` | Amazon_clothing | +1.16 ± 0.18 (10/10) | +0.06 ± 0.16 (3/10) | 0.12 |
| `l2_rand_m20` | Amazon_clothing | +0.92 ± 0.14 (10/10) | −0.09 ± 0.14 (1/10) | 0.13 |
| `l2_rand_m50` | Amazon_clothing | +0.82 ± 0.13 (10/10) | +0.05 ± 0.11 (3/10) | 0.07 |
| `l2_rand_m200` | Amazon_clothing | +0.69 ± 0.16 (10/10) | −0.04 ± 0.14 (3/10) | 0.11 |
| `l2_clus_m20` | Amazon_clothing | +1.02 ± 0.12 (10/10) | −0.04 ± 0.17 (5/10) | 0.11 |

- B의 양수 개수에는 Δ = 0(같은 epoch, 값 복사)인 seed가 들어가지 않습니다. 같은 epoch인 seed 수는 §2.5의 `same_epoch_l2`입니다.

### 2.4 복제 표 (raw `test_acc`, `l2_X − e4k_X`)

- 값: 같은 seed의 raw `test_acc` 차이(%p). `l2_X`와 `e4k_X`는 학습 설정이 같습니다(G-cfg). 같은 서버, 같은 결과 루트입니다.

| 쌍 | ds | 평균 ± sd (양수 k/10) | 평균 \|Δ\| | 최대 \|Δ\| | seed 1212–1221 Δ |
|---|---|---|---|---|---|
| `l2_naq − e4k_naq` | Cora_full | −0.03 ± 0.22 (5/10) | 0.13 | 0.62 | +0.09, −0.08, +0.10, +0.00, −0.62, +0.05, −0.07, −0.01, −0.04, +0.27 |
| `l2_rand_m20 − e4k_rand_m20` | Cora_full | −0.01 ± 0.11 (3/10) | 0.09 | 0.23 | −0.01, −0.13, +0.11, +0.23, −0.01, −0.09, −0.05, −0.05, +0.07, −0.12 |
| `l2_rand_m50 − e4k_rand_m50` | Cora_full | −0.06 ± 0.21 (5/10) | 0.17 | 0.50 | +0.26, −0.13, +0.11, +0.02, +0.03, +0.10, −0.23, −0.50, −0.05, −0.23 |
| `l2_rand_m200 − e4k_rand_m200` | Cora_full | +0.02 ± 0.10 (7/10) | 0.08 | 0.20 | −0.01, +0.01, +0.07, +0.13, −0.09, +0.08, −0.20, +0.12, +0.01, +0.09 |
| `l2_clus_m20 − e4k_clus_m20` | Cora_full | −0.04 ± 0.26 (4/10) | 0.18 | 0.54 | +0.09, +0.01, +0.07, −0.10, −0.04, −0.05, +0.54, −0.05, −0.54, −0.32 |
| `l2_naq − e4k_naq` | Amazon_clothing | −0.01 ± 0.07 (4/10) | 0.05 | 0.17 | −0.17, +0.06, −0.00, −0.00, +0.05, +0.02, −0.01, −0.06, −0.03, +0.06 |
| `l2_rand_m20 − e4k_rand_m20` | Amazon_clothing | −0.00 ± 0.15 (4/10) | 0.10 | 0.37 | +0.02, −0.02, −0.09, +0.05, −0.01, −0.08, +0.37, −0.28, +0.00, +0.02 |
| `l2_rand_m50 − e4k_rand_m50` | Amazon_clothing | −0.05 ± 0.09 (3/10) | 0.10 | 0.16 | −0.10, +0.13, −0.14, −0.08, −0.07, +0.05, −0.07, −0.16, −0.14, +0.02 |
| `l2_rand_m200 − e4k_rand_m200` | Amazon_clothing | +0.04 ± 0.12 (8/10) | 0.10 | 0.23 | +0.05, +0.08, +0.08, +0.17, +0.18, +0.00, −0.08, +0.09, +0.06, −0.23 |
| `l2_clus_m20 − e4k_clus_m20` | Amazon_clothing | −0.04 ± 0.06 (2/10) | 0.06 | 0.14 | −0.04, −0.07, −0.11, −0.03, +0.06, −0.02, −0.12, +0.02, −0.00, −0.14 |

### 2.5 best epoch 분포

- 값: 500 기준 best epoch(평가 간격 160). raw 기준 = raw validation 500, l2 기준 = l2 validation 500. `same_epoch_l2` = 두 기준의 best epoch이 같은 seed 수.

| arm | ds | raw 기준 평균 (4000인 seed) | l2 기준 평균 (4000인 seed) | l2 기준 범위 | `same_epoch_l2` |
|---|---|---|---|---|---|
| `l2_naq` | Cora_full | 3408 (3) | 2960 (0) | 2240–3680 | 6/10 |
| `l2_rand_m20` | Cora_full | 3312 (2) | 2384 (1) | 640–4000 | 3/10 |
| `l2_rand_m50` | Cora_full | 3232 (3) | 1952 (1) | 640–4000 | 2/10 |
| `l2_rand_m200` | Cora_full | 3424 (3) | 1760 (0) | 480–3360 | 1/10 |
| `l2_clus_m20` | Cora_full | 3440 (3) | 3024 (2) | 1440–4000 | 8/10 |
| `l2_naq` | Amazon_clothing | 3680 (2) | 3120 (1) | 800–4000 | 4/10 |
| `l2_rand_m20` | Amazon_clothing | 3664 (3) | 2912 (0) | 1120–3840 | 3/10 |
| `l2_rand_m50` | Amazon_clothing | 3600 (0) | 3152 (1) | 960–4000 | 6/10 |
| `l2_rand_m200` | Amazon_clothing | 3408 (2) | 2752 (1) | 640–4000 | 2/10 |
| `l2_clus_m20` | Amazon_clothing | 3664 (4) | 2656 (0) | 800–3840 | 2/10 |

- seed별 epoch 목록은 `HW08_tables.json` `best_epoch`에 있습니다.

### 2.6 기하·노름 표

- 값: 10 seed 평균. `geom`·`norm_diag` = raw 기준 best checkpoint, `geom_l2`·`norm_diag_l2` = l2 기준 best checkpoint(같은 epoch이면 같은 값).
- `geom` 항목의 정의는 CLAUDE.md §4.2(고정 표본 4096 노드, 인코더 출력 z). `test_norm_*` = test 500 task에 등장하는 노드(6194 / 8299개)의 ‖z‖ 평균·표준편차(ddof=0). `proto_norm_cv` = test task마다 5개 프로토타입(support 평균, 정규화 전) 노름의 표준편차(ddof=0) / 평균, 500 task 평균.

`geom` / `geom_l2`

| arm | ds | norm_mean | norm_std | uniformity | eff_rank | corr_offdiag_mean |
|---|---|---|---|---|---|---|
| `l2_naq` | Cora_full | 1.5222 / 1.4964 | 0.2521 / 0.2642 | −2.2366 / −2.2407 | 155.48 / 154.18 | 0.1648 / 0.1663 |
| `l2_rand_m20` | Cora_full | 1.6238 / 1.5993 | 0.2542 / 0.2914 | −2.3380 / −2.2912 | 168.97 / 165.99 | 0.1480 / 0.1559 |
| `l2_rand_m50` | Cora_full | 1.6563 / 1.5987 | 0.2571 / 0.3147 | −2.3632 / −2.3039 | 170.93 / 167.13 | 0.1437 / 0.1521 |
| `l2_rand_m200` | Cora_full | 1.7190 / 1.6400 | 0.2557 / 0.3329 | −2.3635 / −2.2825 | 174.50 / 168.64 | 0.1372 / 0.1495 |
| `l2_clus_m20` | Cora_full | 1.6678 / 1.6389 | 0.2602 / 0.2726 | −2.3641 / −2.3618 | 168.17 / 167.09 | 0.1465 / 0.1485 |
| `l2_naq` | Amazon_clothing | 1.7264 / 1.6632 | 0.4042 / 0.4162 | −2.5095 / −2.5135 | 167.01 / 166.24 | 0.1472 / 0.1484 |
| `l2_rand_m20` | Amazon_clothing | 1.8471 / 1.7835 | 0.3748 / 0.4003 | −2.6570 / −2.6476 | 180.26 / 178.54 | 0.1233 / 0.1252 |
| `l2_rand_m50` | Amazon_clothing | 1.8628 / 1.8479 | 0.3643 / 0.3845 | −2.6776 / −2.6621 | 183.96 / 182.58 | 0.1181 / 0.1194 |
| `l2_rand_m200` | Amazon_clothing | 1.8807 / 1.8619 | 0.3567 / 0.3990 | −2.6830 / −2.6323 | 187.53 / 185.25 | 0.1138 / 0.1162 |
| `l2_clus_m20` | Amazon_clothing | 1.9265 / 1.8517 | 0.4014 / 0.4447 | −2.6552 / −2.6289 | 175.97 / 173.63 | 0.1245 / 0.1274 |

`norm_diag` / `norm_diag_l2`

| arm | ds | zero_rows_all | zero_rows_test | test_norm_mean | test_norm_std | proto_norm_cv |
|---|---|---|---|---|---|---|
| `l2_naq` | Cora_full | 0 / 0 | 0 / 0 | 1.5392 / 1.5132 | 0.2576 / 0.2689 | 0.1050 / 0.1066 |
| `l2_rand_m20` | Cora_full | 0 / 0 | 0 / 0 | 1.6426 / 1.6164 | 0.2616 / 0.2983 | 0.0995 / 0.1032 |
| `l2_rand_m50` | Cora_full | 0 / 0 | 0 / 0 | 1.6720 / 1.6098 | 0.2638 / 0.3199 | 0.0952 / 0.1098 |
| `l2_rand_m200` | Cora_full | 0 / 0 | 0 / 0 | 1.7322 / 1.6452 | 0.2605 / 0.3373 | 0.0911 / 0.1081 |
| `l2_clus_m20` | Cora_full | 0 / 0 | 0 / 0 | 1.6912 / 1.6603 | 0.2693 / 0.2813 | 0.0988 / 0.1015 |
| `l2_naq` | Amazon_clothing | 0 / 0.1 | 0 / 0 | 1.7431 / 1.6775 | 0.4016 / 0.4128 | 0.1370 / 0.1429 |
| `l2_rand_m20` | Amazon_clothing | 0 / 0 | 0 / 0 | 1.8665 / 1.7902 | 0.3660 / 0.3919 | 0.1226 / 0.1272 |
| `l2_rand_m50` | Amazon_clothing | 0 / 0 | 0 / 0 | 1.8676 / 1.8585 | 0.3555 / 0.3777 | 0.1180 / 0.1207 |
| `l2_rand_m200` | Amazon_clothing | 0 / 0 | 0 / 0 | 1.8797 / 1.8584 | 0.3470 / 0.3849 | 0.1086 / 0.1125 |
| `l2_clus_m20` | Amazon_clothing | 0 / 0 | 0 / 0 | 1.9449 / 1.8639 | 0.3967 / 0.4385 | 0.1284 / 0.1378 |

- `l2_naq` Amazon_clothing의 `zero_rows_all` 0.1은 seed 1215의 l2 기준 checkpoint(전체 노드 중 1행)입니다. 그 행은 test task 노드가 아닙니다(`zero_rows_test` 0).

## 3. 게이트·사전 등록 대조표

### 3.1 게이트

| 게이트 | 기준 | 값 | 결과 |
|---|---|---|---|
| G-cfg | `l2_*` 병합 설정 − {`eval_l2`, `save_best`} = `e4k_*` 병합 설정 | 5/5 같음, 다른 키 없음 | 통과 |
| G-l2 | `evaluate_l2`(미정규화 Â²X) test vs trainfree Â²X L2, \|Δ\| ≤ 1e-4 | \|Δ\| 0.0 / 0.0 | 통과 |
| G-tf | Â²X L2 80.12 / 92.72, 정규화 없음 67.57 / 83.03, \|Δ\| ≤ 0.01 | 80.12 / 92.715, 67.57 / 83.03 (최대 \|Δ\| 0.005) | 통과 |
| 회귀 | \|Δ\| ≤ 0.30 | 최대 0.105 | 통과 |
| 불변 | 학습 경로 4개 파일 diff 없음 | 비어 있음 | 통과 |
| smoke | 새 키·`ckpt` 기록·파일 | 3 + 1 run 모두 기록·파일 확인 | 통과 |
| G6 | completed ≥ 95% (1회 재시도 후) | 100/100, 재시도 0 | 통과 |
| G7 (기록) | arm별 붕괴·NaN, `zero_rows_test` > 0 run 수 | 모든 arm 0 / 0, 0 run | 기록 |

- 전체 기록: `results/hw_iislab4/HW08_checks.json` (`hw/check_hw08.py --start 67c9746…`).
- **G-cfg**: 비교에서 `arm`과 `note` 키를 뺐습니다(§5 판단 1). `l2_*` 다섯 arm의 `eval_l2`, `save_best`는 모두 true입니다.
- **G-l2**: Â²X = `torch.sparse.mm(data.adj, torch.sparse.mm(data.adj, X))`(`hw/trainfree.py:92–93`과 같은 식), test 500 task. 80.12000134 / 92.71500096으로 trainfree JSON과 같습니다.
- **G-tf**: `hw/trainfree.py`는 수정 없이 실행했습니다. L2 행 정규화는 `hw/trainfree.py:109` `emb = F.normalize(feat, p=2, dim=1) if norm == "l2" else feat`(eps 기본값 1e-12)이고, `evaluate_l2`는 `F.normalize(embeddings, p=2, dim=1, eps=1e-12)`입니다.
- **회귀** (`_regress/HW08/`, seed 1212, raw test %):

| arm | ds | HW08 | 기존 | Δ | best epoch (HW08 / 기존) |
|---|---|---|---|---|---|
| `naq_base` | Cora_full | 77.86 | 77.96 | −0.10 | 2000 / 1760 |
| `naq_base` | Amazon_clothing | 90.62 | 90.60 | +0.01 | 1600 / 1600 |
| `nbr_raw` | Cora_full | 78.04 | 78.15 | −0.11 | 2000 / 2000 |
| `nbr_raw` | Amazon_clothing | 90.91 | 90.89 | +0.02 | 1280 / 1280 |

  - 기본 설정(`eval_l2`, `save_best` false) run의 `metrics.json` 키 집합(최상위, `epochs[i]`)은 `_regress/HW07`의 같은 run과 같습니다.
- **불변** (`git diff 67c9746`):
  - `hw/losses.py`, `hw/negatives.py`, `hw/neighbors.py`, `hw/episodes.py`: diff 없음.
  - `hw/evaluate.py`: 삭제 0줄, 추가 8줄(`import torch.nn.functional as F`, `evaluate_l2`). `evaluate` 본문(`hw/evaluate.py:13–39`)은 그대로입니다.
  - `hw/run.py`: 삭제 1줄(모듈 docstring의 `... {metrics.json, train.log}. No checkpoints.`), 추가 90줄. 기존 평가 경로 `clean = model.network(features, adj=data.adj)` → `evaluate(clean, valid_pool, ...)`, `evaluate(clean, valid_pool[:NUM_EXISTING], ...)`, best 갱신 `if validation["accuracy_mean"] > best_validation:`는 그대로이고, 새 코드는 모두 `if eval_l2:` / `if save_best:` 아래에 있습니다.
- **smoke** (episodes 4, `eval_every 1`, `/tmp/hw_smoke`, 커밋 안 함): `l2_naq`, `l2_rand_m20`, `l2_clus_m20` Cora_full 각 1회. 모두 completed, §1.2의 새 키 전부(`epochs[i].val_acc_500_l2`, `val_acc_50_l2`, `best_l2`, `best_epoch_500_l2`, `same_epoch_l2`, 지표 3개, `geom_l2`, `norm_diag`, `norm_diag_l2`)와 `ckpt`가 기록되었습니다. 세 run 모두 `same_epoch_l2` true여서 `best_raw.pt`(18,367,448 B)만 저장되었습니다.
  - 추가 smoke 1회(`/tmp/hw_smoke/forced`): `l2_naq` Cora_full에서 프로세스 안에서만 validation용 `evaluate_l2` 반환값을 바꿔 l2 기준 best를 epoch 1로 만들었습니다. `same_epoch_l2` false, `best_l2.pt` 저장(합계 36,734,696 B = `ckpt.bytes`), `geom_l2` 기록. 저장한 두 checkpoint를 다시 읽어 잰 test가 기록값과 |Δ| 0.0이고, raw `test_acc`는 바꾸지 않은 smoke run과 같습니다.

### 3.2 사전 등록 (판정은 사람)

Δ = 같은 seed 짝지은 차이의 10 seed 평균. "raw" = `test_acc`, "l2" = `test_acc_l2`.

| ID | 기준 | 값 | 기준식 적용 |
|---|---|---|---|
| HWP079 | 다섯 arm × 두 ds 모두 seed별 \|`l2_X − e4k_X`\|(raw) 평균 ≤ 0.20 | 0.05–0.18 (최대 `clus_m20` Cora_full 0.18) | 충족 |
| HWP080 | `l2_naq`: `test_acc_l2_at_bestraw − test_acc` ≥ +0.50, 두 ds | +0.51 / +1.16 | 충족 |
| HWP081 | l2: `rand_m20 − naq` > 0, 양수 ≥ 8/10, 두 ds | +1.15 (9/10) / +0.22 (8/10) | 충족 |
| HWP082 | l2 Δ(`rand_m20 − naq`) ≥ 0.5 × raw Δ, 두 ds | Cora_full 1.15 vs 0.5 × 1.17 = 0.59; Amazon_clothing 0.22 vs 0.5 × 0.62 = 0.31 | 미충족 (Amazon_clothing) |
| HWP083 | l2: `rand_m200 − rand_m20` ≥ 0, Cora_full | +0.35 (9/10) | 충족 |
| HWP084 | l2: `clus_m20 − rand_m20` ≥ +0.20, Cora_full | +0.46 (10/10) | 충족 |
| HWP085 | 다섯 arm 중 l2 평균 최고 ≥ 80.12 / ≥ 92.72 | `l2_clus_m20` 80.74 / 92.94 | 충족 |
| HWP086 | 다섯 arm × 두 ds 모두 \|`test_acc_l2 − test_acc_l2_at_bestraw`\| 평균 ≤ 0.20 | 평균의 절댓값: 0.01–0.11; seed별 절댓값의 평균: 0.05–0.32 (> 0.20: `rand_m20`, `rand_m50`, `rand_m200` Cora_full) | 평균의 절댓값으로 읽으면 충족, seed별 절댓값의 평균으로 읽으면 미충족 |
| HWP087 | `proto_norm_cv`(raw 기준 best): `l2_rand_m20` < `l2_naq`, 두 ds | 0.0995 < 0.1050 / 0.1226 < 0.1370 | 충족 |

## 4. 수행 내용

| 커밋 | 내용 |
|---|---|
| `84b42c0` HW08-a | CLAUDE.md 개정 10, 지시서(사전 등록 HWP079–087), `hw/evaluate.py` `evaluate_l2`, `hw/run.py` `eval_l2`·`save_best`·`norm_diag`, `configs/hw.json` `l2_*` 5 arm, `.gitignore` `ckpt/`, `hw/check_hw08.py`, `HW08_checks.json`, `_regress/HW08/` |
| `fe3eefd` HW08-b | `hw.trainfree` 두 ds → `results/hw_iislab4/trainfree/HW03a_trainfree_<ds>.{json,md}`, `results/measure/HW08_trainfree_<ds>.md` |
| HW08-c (이 커밋) | 100 run(`HW08_runs.txt`), `summary*`, `paired_l2_naq*`, `paired_l2_rand_m20*`, `hw/tables_hw08.py`, `HW08_tables.json`, 이 보고서 |

- 집계 명령: `python -m hw.summarize --ref l2_naq --ref l2_rand_m20 [--metric test_acc_l2 | test_acc_l2_at_bestraw | test_acc_raw_at_bestl2]`. `summarize.py`는 최상위 키를 이미 일반적으로 읽고 키가 없는 run을 빼므로 바꾸지 않았습니다. `summary.csv`/`summary.md`는 HW08 행 추가만 있습니다(삭제 0줄).
- §0 기록:
  - 평가 임베딩 생성 `hw/run.py:461, 506, 520`(작업 시작 HEAD 기준), `evaluate` 호출 `:462–463, 507, 521`, best 갱신 `:493–500` — `>` 비교라 동률이면 앞 epoch이 유지됩니다.
  - `hw/trainfree.py:109` L2 행 정규화 `F.normalize(feat, p=2, dim=1)`.
  - 인코더 state_dict: Cora_full 파라미터 4,591,360개(텐서 4개, 특징 8710, latent 256), 저장 18,367,248 B; Amazon_clothing 4,757,248개(특징 9034), 저장 19,030,800 B.
  - 디스크 여유 38 GB(≥ 20 GB) → `save_best` 유지.
  - git: `hl/hsan-weighting...origin/hl/hsan-weighting` 동기, 태그 hw01–hw07, GPU 8장 메모리 0 MiB, 허용 외 미커밋 변경 없음.

## 5. 판단한 사항

1. G-cfg 비교에서 `note` 키를 `arm`과 함께 뺐습니다. 설명 문자열이고 `hw/run.py`가 읽지 않는 키입니다.
2. `eval_l2: true`는 `valid_tasks: 500`일 때만 허용했습니다(아니면 `ValueError`). 지시서의 `val_acc_500_l2`·`val_acc_50_l2`가 500 캐시를 전제로 합니다.
3. `norm_diag`에 지시서의 다섯 항목 외에 `test_nodes`(test task 노드 수)와 `proto_norm_cv_undefined_tasks`(프로토타입 노름 평균이 0인 task 수, 이번 run 모두 0)를 더했습니다. `proto_norm_cv`의 프로토타입 묶음에는 `evaluate`와 같은 방식으로 test support 레이블을 썼습니다(`# measurement-only label use`).
4. checkpoint 위치는 `HW_CKPT_ROOT`(기본 `ckpt/<결과 루트 이름>`)로 정했습니다. 본실행은 `ckpt/hw_iislab4/`, smoke는 `/tmp/hw_smoke/ckpt`입니다. `.pt` 내용은 `{"state_dict" (CPU), "epoch"}`입니다.
5. l2 기준 best가 raw 기준 best와 같은 epoch이면 l2 test·`geom_l2`·`norm_diag_l2`를 raw 쪽 값에서 복사했습니다(HW03b `best_50` 규칙과 같음).
6. smoke는 `eval_every 1`로 돌렸습니다(지시서는 episodes 4만 지정). 다른 epoch 분기가 자연 발생하지 않아 §3.1의 강제 smoke 1회를 더했습니다. 이 run은 `/tmp`에만 있고 커밋하지 않았습니다.
7. G-tf와 G-l2에 필요한 trainfree 실행은 HW08-a 단계에서 했고, 그 출력은 HW08-b 커밋에 넣었습니다.
8. `hw/trainfree.py`는 수정하지 않았기 때문에, 출력 표의 참고 행 이름이 "seio 10 seed"로 찍혀 있습니다. 실제 값은 `results/hw_iislab4/summary.csv`(iislab4)의 `naq_base`·`nbr_raw` 10 seed입니다.
9. HWP086의 "\|…\| 평균"은 두 가지로 읽을 수 있어 둘 다 적었습니다(§3.2).

## 6. 제외·실패·붕괴 run

- 실패·NaN·붕괴 run 없음(100/100 completed, 재시도 0).
- `l2_rand_m200` Amazon_clothing seed 1217: 04:54:50에 `metrics.json`(status `completed`)과 `best_raw.pt`를 다 쓴 뒤 python 프로세스가 종료되지 않았습니다(메인 스레드 `futex_wait`, GPU 사용률 0%). 05:14에 SIGTERM이 듣지 않아 SIGKILL로 종료했고, 큐 로그에는 `rc=143 status=completed`로 기록되었습니다. 결과 파일은 종료 전에 완성된 것을 그대로 썼고 다시 돌리지 않았습니다. 이 run은 `same_epoch_l2` true(best 4000)라 `best_l2.pt`가 없는 것이 규칙과 맞습니다.

## 7. 소요 시간, 환경, HEAD

- 작업 시작 HEAD `67c974695399431ca581337f61b067256e12cb5a` (HW07-d). HW08-a `84b42c0`, HW08-b `fe3eefd`.
- 회귀·smoke·trainfree 실행 04:28–04:31 UTC, 본실행 큐 04:34:20–05:09:55(마지막 정상 종료), 큐 종료 05:15:01(위 프로세스 종료 대기 포함). run당 126–175 s, 합계 15,132 s.
- `ckpt/hw_iislab4/` 163 파일(`best_raw.pt` 100, `best_l2.pt` 63), 3,048,956,240 B(2.9 GB). 작업 후 디스크 여유 35 GB.
- iislab4, RTX 3090 × 8, conda `taskns`(python 3.9.25, torch 2.4.1+cu121).
