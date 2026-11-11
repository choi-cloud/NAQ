# HW09 [iislab4] — dblp · Amazon_eletronics 확장 + 1-shot test 병기 + HW08 checkpoint 1-shot 재평가 + 학습 없는 기준선

2026-11-11 · 서버 iislab4 · 브랜치 `hl/hsan-weighting` · 지시서 `instructions/HW09_iislab4.md` · CLAUDE.md 개정 11 · 사전 등록 HWP088–095 · 작업 시작 HEAD `50fda1c`

- 주 지표: `test_acc_l2`(l2 validation 500으로 고른 checkpoint, l2 평가, 5-shot), 그다음 `test_acc_l2_1shot`(같은 checkpoint, 1-shot test 500 task).
- 정확도는 %, 셀은 seed 평균 ± 표준편차(ddof=0, n=10). 짝지은 Δ는 같은 seed·같은 지표의 차이, "평균 ± 표준편차 (양수 k/n)".
- dblp·Amazon_eletronics = HW09 학습 run `h9_*`. Cora_full·Amazon_clothing = HW08 `l2_*` run(5-shot 값은 HW08 `metrics.json`, 1-shot 값은 `ckpt_eval/` 재평가). 리포의 데이터셋 이름은 `Amazon_eletronics`(NaQ 원본 표기)입니다.

## 1. 결론

- l2 5-shot `clus_m20 − naq`: Cora_full +1.61 / Amazon_clothing +0.34 / dblp +0.40 / Amazon_eletronics +1.11 (네 ds 모두 10/10). `rand_m20 − naq`: +1.15 / +0.22 / +0.34 (7/10) / +1.04.
- l2 5-shot 최고 arm 대 학습 없는 Â²X L2: dblp 84.71(`clus_m20`) vs 85.65, Amazon_eletronics 89.68(`rand_m50`) vs 88.64. `naq`는 84.31 / 88.38.
- l2 1-shot `clus_m20 − naq`: +1.53 / +0.25 (6/10) / +1.02 (9/10) / +1.01. 1-shot Δ가 5-shot Δ보다 큰 ds는 dblp 하나입니다.

## 2. 핵심 표

### 2.1 데이터 표

값의 종류: 그래프 크기와 클래스 수(`hw/data_hw09.py`, `load_data`, seed 1212). 엣지 = `edge_index`의 u<v 고유 쌍(자기 루프 제외). train / valid / test 클래스 수는 `data.py`의 분할(valid는 seed 1212의 `random.sample`, test는 파일 고정; Cora_full은 셔플).

| 항목 | Cora_full | Amazon_clothing | dblp | Amazon_eletronics |
|---|---|---|---|---|
| 로더 문자열 | `Cora_full` | `Amazon_clothing` | `dblp` | `Amazon_eletronics` |
| 노드 | 19793 | 24919 | 40672 | 42318 |
| 엣지 (무방향) | 63421 | 91680 | 288270 | 43556 |
| 특징 차원 | 8710 | 9034 | 7202 | 8669 |
| 클래스 (전체 / train / valid / test) | 70 / 25 / 20 / 25 | 77 / 40 / 17 / 20 | 137 / 80 / 27 / 30 | 167 / 90 / 37 / 40 |

평가 캐시 md5 (`save/`, git 밖; **새** = HW09에서 만든 파일):

| 파일 | Cora_full | Amazon_clothing | dblp | Amazon_eletronics |
|---|---|---|---|---|
| `5way_5shot.pkl` | `cfef6a69…` | `ca3edaca…` | **새** `aa9fd2e7…` | **새** `91bb9111…` |
| `5way_1shot.pkl` | **새** `d51c0676…` | `5f901727…` | **새** `d00b16ee…` | **새** `e4651dc3…` |
| `valid500/<ds>.pkl` | `d8fdb82a…` | `0920f5d7…` | **새** `ae4c5509…` | **새** `2eb2f743…` |
| test task 수 (5-shot / 1-shot) | 500 / 500 | 500 / 500 | 500 / 500 | 500 / 500 |
| 1-shot test 클래스 집합 = 5-shot test 클래스 집합 | 예 (25) | 예 (20) | 예 (30) | 예 (40) |

전체 md5와 클래스 목록은 `results/hw_iislab4/HW09_data.json`에 있습니다. k-means·raw cos 표: `cache/kmeans/{dblp,Amazon_eletronics}_k100.pt`(md5 `747e599c…`, `2f43da8b…`), `cache/rawcos/<ds>_top500.pt`.

군집 통계 (학습 전, 측정 전용 레이블; `results/measure/HW09_cluster_stats.md`). Cora_full·Amazon_clothing의 군집 네 행은 HW06 G-table 값과 같습니다.

| 항목 (값의 종류) | Cora_full | Amazon_clothing | dblp | Amazon_eletronics |
|---|---|---|---|---|
| 군집 크기 최소 / 중앙값 / 최대 (노드 수, k 100) | 4 / 170 / 1083 | 11 / 185 / 1575 | 78 / 368 / 1394 | 58 / 352 / 2347 |
| 같은 군집 내 같은 클래스 비율 (노드 macro) | 0.3562 | 0.4560 | 0.1841 | 0.2768 |
| 다른 군집 노드 중 같은 클래스 비율 (노드 macro) | 0.0175 | 0.0143 | 0.0080 | 0.0049 |
| 무작위 기대 FN rate Σ p_c² | 0.0217 | 0.0202 | 0.0100 | 0.0081 |
| NaQ-Feat top-10 순도, 전체 노드 (노드 macro) | 0.3498 | 0.5972 | 0.1572 | 0.6454 |
| NaQ-Feat top-10 순도, seed 1212 학습 pool 4000 episode (쌍 비율) | 0.3496 | 0.5989 | 0.1569 | 0.6465 |

### 2.2 주 결과 표 — l2, 5-shot (`test_acc_l2`; l2 validation 500 선택, l2 평가)

| 행 | Cora_full (HW08) | Amazon_clothing (HW08) | dblp | Amazon_eletronics |
|---|---|---|---|---|
| Â²X L2 (학습 없음) | 80.12 | 92.72 | 85.65 | 88.64 |
| naq | 79.14 ± 0.35 | 92.59 ± 0.18 | 84.31 ± 0.23 | 88.38 ± 0.27 |
| rand_m20 | 80.29 ± 0.31 | 92.82 ± 0.15 | 84.65 ± 0.26 | 89.42 ± 0.28 |
| rand_m50 | 80.48 ± 0.24 | 92.90 ± 0.12 | 84.70 ± 0.27 | 89.68 ± 0.28 |
| clus_m20 | 80.74 ± 0.31 | 92.94 ± 0.18 | 84.71 ± 0.24 | 89.50 ± 0.27 |

### 2.3 주 결과 표 — l2, 1-shot (`test_acc_l2_1shot`; 같은 l2 checkpoint, 1-shot test 500 task)

| 행 | Cora_full (재평가) | Amazon_clothing (재평가) | dblp | Amazon_eletronics |
|---|---|---|---|---|
| Â²X L2 (학습 없음) | 51.76 | 85.10 | 72.97 | 76.14 |
| naq | 63.98 ± 0.28 | 86.73 ± 0.48 | 73.29 ± 1.10 | 76.96 ± 0.35 |
| rand_m20 | 65.08 ± 0.33 | 86.31 ± 0.67 | 73.99 ± 1.11 | 77.66 ± 0.38 |
| rand_m50 | 65.19 ± 0.50 | 86.22 ± 0.54 | 74.29 ± 1.07 | 77.69 ± 0.39 |
| clus_m20 | 65.50 ± 0.44 | 86.97 ± 0.56 | 74.31 ± 0.96 | 77.97 ± 0.45 |

### 2.4 짝지은 Δ 표 (%p)

l2 5-shot (`test_acc_l2`):

| Δ | Cora_full | Amazon_clothing | dblp | Amazon_eletronics |
|---|---|---|---|---|
| rand_m20 − naq | +1.15 ± 0.50 (9/10) | +0.22 ± 0.17 (8/10) | +0.34 ± 0.32 (7/10) | +1.04 ± 0.27 (10/10) |
| rand_m50 − naq | +1.34 ± 0.44 (10/10) | +0.31 ± 0.17 (9/10) | +0.39 ± 0.30 (8/10) | +1.29 ± 0.29 (10/10) |
| clus_m20 − naq | +1.61 ± 0.23 (10/10) | +0.34 ± 0.18 (10/10) | +0.40 ± 0.24 (10/10) | +1.11 ± 0.29 (10/10) |
| clus_m20 − rand_m20 | +0.46 ± 0.41 (10/10) | +0.12 ± 0.21 (7/10) | +0.06 ± 0.13 (7/10) | +0.08 ± 0.14 (7/10) |
| rand_m50 − rand_m20 | +0.19 ± 0.16 (9/10) | +0.08 ± 0.10 (8/10) | +0.05 ± 0.09 (7/10) | +0.26 ± 0.21 (10/10) |

l2 1-shot (`test_acc_l2_1shot`):

| Δ | Cora_full | Amazon_clothing | dblp | Amazon_eletronics |
|---|---|---|---|---|
| rand_m20 − naq | +1.11 ± 0.31 (10/10) | −0.41 ± 0.49 (3/10) | +0.70 ± 1.41 (7/10) | +0.70 ± 0.27 (10/10) |
| rand_m50 − naq | +1.22 ± 0.38 (10/10) | −0.51 ± 0.53 (2/10) | +1.00 ± 1.47 (8/10) | +0.73 ± 0.36 (10/10) |
| clus_m20 − naq | +1.53 ± 0.51 (10/10) | +0.25 ± 0.64 (6/10) | +1.02 ± 1.29 (9/10) | +1.01 ± 0.36 (10/10) |
| clus_m20 − rand_m20 | +0.42 ± 0.54 (9/10) | +0.66 ± 0.73 (9/10) | +0.32 ± 0.54 (7/10) | +0.32 ± 0.25 (9/10) |
| rand_m50 − rand_m20 | +0.11 ± 0.43 (7/10) | −0.09 ± 0.45 (3/10) | +0.31 ± 0.73 (5/10) | +0.03 ± 0.28 (5/10) |

### 2.5 학습 없는 기준선 표 (test 정확도 %, 5-shot / 1-shot; `hw.trainfree --shot {5,1}`)

| 특징 · 행 정규화 | Cora_full | Amazon_clothing | dblp | Amazon_eletronics |
|---|---|---|---|---|
| X · 없음 (raw) | 48.07 / 23.30 | 66.58 / 36.43 | 53.09 / 26.59 | 62.22 / 36.77 |
| X · L2 | 60.50 / 41.44 | 87.61 / 72.30 | 67.30 / 48.82 | 87.99 / 73.56 |
| ÂX · 없음 (raw) | 64.51 / 32.64 | 73.64 / 47.84 | 76.31 / 47.91 | 64.66 / 39.91 |
| ÂX · L2 | 77.41 / 50.14 | 91.79 / 82.47 | 83.45 / 69.17 | 87.75 / 74.09 |
| Â²X · 없음 (raw) | 67.57 / 38.55 | 83.03 / 60.55 | 80.92 / 58.42 | 69.86 / 45.41 |
| Â²X · L2 | 80.12 / 51.76 | 92.72 / 85.10 | 85.65 / 72.97 | 88.64 / 76.14 |

표 원본: `results/measure/HW09_trainfree_<ds>.md`(5-shot·1-shot 두 표).

### 2.6 raw 지표 표 (측정용; raw validation 500 선택, raw 평가)

`test_acc`(5-shot) / `test_acc_1shot`(1-shot). Cora_full·Amazon_clothing의 1-shot은 재평가 값.

| 행 | Cora_full | Amazon_clothing | dblp | Amazon_eletronics |
|---|---|---|---|---|
| Â²X raw (학습 없음) | 67.57 / 38.55 | 83.03 / 60.55 | 80.92 / 58.42 | 69.86 / 45.41 |
| naq | 78.66 / 60.83 | 91.37 / 80.93 | 83.27 ± 0.30 / 67.83 ± 0.52 | 86.71 ± 0.42 / 71.82 ± 0.58 |
| rand_m20 | 79.83 / 61.41 | 91.99 / 80.40 | 83.25 ± 0.35 / 66.73 ± 0.75 | 88.21 ± 0.26 / 72.96 ± 0.31 |
| rand_m50 | 79.94 / 61.21 | 92.03 / 79.86 | 83.38 ± 0.21 / 66.27 ± 0.79 | 88.48 ± 0.27 / 72.86 ± 0.40 |
| clus_m20 | 80.19 / 62.06 | 91.95 / 80.71 | 83.38 ± 0.30 / 66.66 ± 0.61 | 88.16 ± 0.31 / 73.21 ± 0.33 |

새 두 ds의 짝지은 Δ (raw, 같은 지표끼리; 5-shot `test_acc` | 1-shot `test_acc_1shot`):

| Δ | dblp 5-shot | dblp 1-shot | Amazon_eletronics 5-shot | Amazon_eletronics 1-shot |
|---|---|---|---|---|
| rand_m20 − naq | −0.01 ± 0.43 (5/10) | −1.10 ± 0.73 (1/10) | +1.50 ± 0.34 (10/10) | +1.14 ± 0.51 (10/10) |
| rand_m50 − naq | +0.11 ± 0.36 (6/10) | −1.56 ± 0.65 (1/10) | +1.77 ± 0.35 (10/10) | +1.04 ± 0.53 (10/10) |
| clus_m20 − naq | +0.12 ± 0.26 (6/10) | −1.18 ± 0.52 (0/10) | +1.45 ± 0.37 (10/10) | +1.39 ± 0.52 (10/10) |
| clus_m20 − rand_m20 | +0.13 ± 0.23 (6/10) | −0.08 ± 0.64 (5/10) | −0.05 ± 0.22 (5/10) | +0.25 ± 0.19 (9/10) |
| rand_m50 − rand_m20 | +0.13 ± 0.20 (7/10) | −0.46 ± 0.28 (1/10) | +0.27 ± 0.10 (10/10) | −0.10 ± 0.21 (3/10) |

Cora_full·Amazon_clothing의 raw 1-shot Δ(`clus_m20 − naq`): +1.22 ± 0.58 (10/10) / −0.22 ± 0.52 (3/10).

### 2.7 best epoch 분포 (새 두 ds, 4000 episode, 평가 간격 160)

| arm / ds | raw 기준 평균 | l2 기준 평균 | `same_epoch_l2` | raw 기준 = 4000 | l2 기준 = 4000 |
|---|---|---|---|---|---|
| naq / dblp | 3040 | 2192 | 6/10 | 0 | 0 |
| rand_m20 / dblp | 2272 | 1152 | 3/10 | 0 | 0 |
| rand_m50 / dblp | 2272 | 816 | 1/10 | 0 | 0 |
| clus_m20 / dblp | 3024 | 992 | 1/10 | 1 | 0 |
| naq / Amazon_eletronics | 3776 | 3568 | 4/10 | 5 | 2 |
| rand_m20 / Amazon_eletronics | 3936 | 3440 | 2/10 | 6 | 3 |
| rand_m50 / Amazon_eletronics | 3824 | 3552 | 3/10 | 4 | 2 |
| clus_m20 / Amazon_eletronics | 3856 | 3680 | 6/10 | 6 | 5 |

seed별 값은 `results/hw_iislab4/HW09_tables.json`의 `best_epoch`.

### 2.8 기하·노름 표 (새 두 ds, seed 평균)

`geom_l2` = l2 기준 best checkpoint의 고정 표본 4096 노드(z, 정규화 전). `norm_diag` = raw 기준 best checkpoint, test 노드.

| arm / ds | 평균 노름 | uniformity | effective rank | 차원 간 |상관| 평균 | test 노름 평균 (raw best) | `proto_norm_cv` (raw best) |
|---|---|---|---|---|---|---|
| naq / dblp | 1.5406 | −2.0840 | 157.09 | 0.1581 | 1.5992 | 0.0970 |
| rand_m20 / dblp | 1.6282 | −2.0711 | 164.41 | 0.1509 | 1.6925 | 0.1016 |
| rand_m50 / dblp | 1.6341 | −2.0623 | 165.12 | 0.1500 | 1.7309 | 0.1001 |
| clus_m20 / dblp | 1.6738 | −2.0705 | 161.44 | 0.1514 | 1.7757 | 0.0934 |
| naq / Amazon_eletronics | 1.6678 | −2.5418 | 175.10 | 0.1173 | 1.6579 | 0.1493 |
| rand_m20 / Amazon_eletronics | 1.8556 | −2.6106 | 188.38 | 0.1089 | 1.8404 | 0.1261 |
| rand_m50 / Amazon_eletronics | 1.9235 | −2.6102 | 191.98 | 0.1052 | 1.8924 | 0.1197 |
| clus_m20 / Amazon_eletronics | 1.9162 | −2.6214 | 185.32 | 0.1097 | 1.8848 | 0.1288 |

## 3. 게이트·사전 등록 대조표

### 3.1 게이트 (`results/hw_iislab4/HW09_checks.json`, `all_pass` true)

| 게이트 | 기준 | 값 | 판정 |
|---|---|---|---|
| G-data | 두 새 ds의 파일·로더·캐시 | `data/{dblp,Amazon_eletronics}_{network,train.mat,test.mat}` 있음, 로더 읽음, 캐시 3종 있음, 1-shot test 클래스 = 5-shot | 통과 |
| G-cfg | `h9_*` − `test_shots` = `l2_*` (`arm`, `note` 제외) | 4 arm 모두 같음, `test_shots` [5, 1] | 통과 |
| G-ckpt | 재평가 5-shot 4 키 × 80 run, |Δ| ≤ 1e-4 | 최대 |Δ| 0.0 | 통과 |
| G-tf | Â²X L2 80.12 / 92.715, raw 67.57 / 83.025, |Δ| ≤ 0.01 | 80.12 / 92.715, 67.57 / 83.025 (|Δ| ≤ 1.4e-6) | 통과 |
| 회귀 | `naq_base`, `nbr_raw` seed 1212, |Δ| ≤ 0.30 | naq_base −0.17 / −0.01, nbr_raw −0.04 / −0.01 (Cora_full / Amazon_clothing); 기록 키 집합 = `_regress/HW08` | 통과 |
| 불변 | losses·negatives·neighbors·episodes·evaluate diff 없음 | 비어 있음. 삭제 줄: `hw/run.py` 1(CLI `choices`), `hw/trainfree.py` 5, `hw/valid500.py` 1 | 통과 |
| smoke | 1-shot 키, checkpoint, 시간·메모리 | 4 run completed, 강제 smoke 1 run 재평가 |Δ| 0 | 통과 |
| 시간 규칙 | 추정 > 150분이면 seed 5개 | 추정 41.2분 → seed 10개 유지 | 적용 안 함 |
| G6 | completed ≥ 95% | 80/80, 재시도 0 | 통과 |
| G7 (기록) | 붕괴·NaN·`zero_rows_test` | 붕괴 0, NaN 0, `zero_rows_test` > 0 run 0 (`zero_rows_all` > 0: 1 run, §6) | 기록 |

### 3.2 사전 등록 (판정은 사람; 아래 "기준 식"은 지시서 기준을 값에 그대로 적용한 결과)

| ID | 기준 | 값 | 기준 식 |
|---|---|---|---|
| HWP088 | l2 `rand_m20 − naq` > 0, 양수 ≥ 7/10, 새 두 ds | dblp +0.34 (7/10), Amazon_eletronics +1.04 (10/10) | 충족 |
| HWP089 | l2 `clus_m20 − naq` > 0, 양수 ≥ 8/10, 새 두 ds | +0.40 (10/10), +1.11 (10/10) | 충족 |
| HWP090 | l2 `clus_m20 − rand_m20` ≥ 0, 새 두 ds | +0.06, +0.08 | 충족 |
| HWP091 | l2 `naq` 평균 < Â²X L2 (5-shot), 새 두 ds | 84.31 < 85.65, 88.38 < 88.64 | 충족 |
| HWP092 | 네 arm 중 l2 최고 ≥ Â²X L2, 새 두 ds 중 하나 이상 | dblp 84.71 (`clus_m20`) < 85.65; Amazon_eletronics 89.68 (`rand_m50`) ≥ 88.64 | 충족 (Amazon_eletronics) |
| HWP093 | l2-1s `clus_m20 − naq` > 0, 네 ds | +1.53, +0.25, +1.02, +1.01 | 충족 |
| HWP094 | `clus_m20 − naq`의 l2-1s Δ > l2 Δ, 네 ds 중 3개 이상 | Cora 1.53 vs 1.61, Amazon_clothing 0.25 vs 0.34, dblp 1.02 vs 0.40, Amazon_eletronics 1.01 vs 1.11 → 1개 | 미충족 |
| HWP095 | l2 `rand_m50 − rand_m20` ≥ 0, 새 두 ds | +0.05, +0.26 | 충족 |

## 4. 수행 내용

| 커밋 | 내용 |
|---|---|
| `24a55b4` HW09-a | CLAUDE.md 개정 11, 지시서(HWP088–095). `hw/eval_cache.py`(없는 평가 캐시 생성), `hw/run.py` `test_shots`, `hw/eval_ckpt.py`, `hw/trainfree.py --shot`, `hw/valid500.py` 데이터셋 선택지, `hw/data_hw09.py`(G-data, md5 목록), `hw/check_hw09.py`, `h9_*` 4 arm, `HW09_data.json`, `HW09_checks.json`, `_regress/HW09/` |
| `19a9869` HW09-b | `ckpt_eval/` 80개, 학습 없는 기준선 4 ds × 2 shot(`trainfree/`, `results/measure/HW09_trainfree_<ds>.md`), `hw/cluster_stats_hw09.py`와 `results/measure/HW09_cluster_stats.{json,md}`, `hw/run.py` CLI 데이터셋 선택지 |
| HW09-c (이 커밋) | 80 run(`HW09_runs.txt`, `.gitignore` 허용 규칙 추가), 요약(`summary_*`, `paired_h9_naq*`, `paired_h9_rand_m20*`), `hw/tables_hw09.py`, `HW09_tables.json`, `HW09_checks.json` 재생성, 이 보고서 |

- `test_shots [5, 1]`: 학습 종료 뒤 raw 기준 best와 l2 기준 best checkpoint를 `save/downstream_tasks/<ds>/5way_1shot.pkl`의 test 500 task로 `evaluate` / `evaluate_l2` 평가. 파일만 읽고 RNG를 쓰지 않습니다. 두 best가 같은 epoch이면 값을 복사합니다. 기본값 `[5]`이면 새 키가 없습니다(회귀 run의 키 집합 = `_regress/HW08`).
- `hw/eval_ckpt.py`: `metrics.json`의 `ckpt` 경로를 읽어 `hw/run.py`와 같은 순서(seed → load_data → set_device → add_diffusion → ProtoNet)로 인코더를 만들고 state_dict를 올립니다. 출력은 `ckpt_eval/<arm>/<ds>/seed<S>.json`이고 `metrics.json`은 열지 않고 읽기만 합니다.

## 5. 판단한 사항

1. **새 평가 캐시의 생성 경로**: 기존 Cora_full `5way_5shot.pkl`은 HW01에서 exp015 seed 1212 run(2000 episode)이 만든 파일입니다. `hw/eval_cache.py`는 이 순서(seed 1212 → load_data → set_device → add_diffusion → train pool 2000 → `task_generator.downstream_task_generator`, valid 50 / test 500 / q 8)를 그대로 재생합니다. 이 경로로 Cora_full 5-shot을 임시 디렉터리에 다시 만들면 550 task 내용이 기존 파일과 모두 같았습니다. md5는 `c5f907fe…`로 달랐는데, pickle 직렬화 바이트 차이입니다. 같은 경로로 k_shot만 바꿔 Cora_full 1-shot, dblp·Amazon_eletronics 5/1-shot을 만들었습니다. 생성은 임시 디렉터리에서 하고, `save/`에 그 파일이 없을 때만 복사했습니다.
2. Amazon_clothing의 기존 캐시(2026-10-24, NaQ 원본 쪽 생성)는 같은 경로의 episode 수 1000 / 2000 / 16000으로 재현되지 않았습니다. 이미 있는 파일이라 그대로 썼습니다.
3. valid500(dblp, Amazon_eletronics)은 HW03b의 `hw/valid500.py`를 그대로 썼습니다(seed 20261108). 선택지에 두 데이터셋을 추가한 한 줄만 바꿨습니다. G-valid (a)–(c)는 모두 통과했습니다.
4. 첫 큐 실행(07:13)에서 `hw/run.py` CLI의 `--dataset` 선택지가 Cora_full·Amazon_clothing뿐이어서, 80 run 모두 argparse 오류(rc 2)로 학습 전에 끝났습니다. smoke는 `hw.run.run()`을 직접 불러 CLI를 거치지 않았습니다. 큐를 멈추고 선택지 한 줄을 고쳐 07:15에 다시 실행했습니다. 첫 큐 로그는 `out/hw_queue/HW09_runs.attempt0_cli_choices.log`로 남겼습니다. 이 수정은 HW09-a push 뒤라 HW09-b 커밋에 들어갔습니다.
5. smoke는 HW08처럼 `eval_every 1`로 돌렸습니다(episodes 8). 네 run 모두 `same_epoch_l2` true여서 강제 smoke를 1회 더했습니다(`/tmp/hw_smoke/hw09_forced`, `h9_naq` dblp). 프로세스 안에서 첫 평가 시점의 l2 validation만 1.0으로 바꿔 l2 best를 epoch 1로 만들었습니다. 저장된 두 checkpoint를 `eval_ckpt`로 다시 잰 8개 키가 기록과 |Δ| 0이었고, raw 값은 바꾸지 않은 smoke와 같았습니다.
6. 시간 추정: t = 고정 + 평가 횟수 × e + episode × p 모형입니다. smoke(8 episode, 평가 8회)와 timing run 두 개(320 episode, 평가 2회 / 10회)로 풀었습니다. 4000 episode 추정은 run당 227–277 s, 합계 41.2분입니다. 첫 smoke 3개는 k-means(CPU)와 동시에 돌아 시간이 늘어서, k-means가 끝난 뒤 다시 돌린 값을 썼습니다. 실제 run 시간은 209–249 s, 큐 전체는 40분 38초였습니다.
7. 1-shot 캐시를 만들 때 seed 규칙은 5-shot과 같습니다. 그래서 1-shot과 5-shot의 test 클래스 집합이 같고(네 ds 모두 확인), task 내용은 다릅니다.
8. NaQ-Feat top-10 순도는 두 가지로 적었습니다. 전체 노드에 `query_generation_NaQ`를 적용한 노드 macro 값과, seed 1212 학습 pool(4000 episode)의 `query_eq_support` 쌍 비율입니다. Cora_full·Amazon_clothing의 쌍 비율 0.3496 / 0.5989는 HW03a·HW03b 값 0.35 / 0.60과 같은 정의입니다.
9. G-tf 재실행으로 `trainfree/HW03a_trainfree_{Cora_full,Amazon_clothing}.json`이 다시 쓰였습니다. 정확도는 같고, `mean_true_class_distance_margin`이 1e-8 수준, `elapsed_seconds`가 다릅니다.
10. `HW09_runs.txt`는 `.gitignore`의 전역 `*`에 걸려서, `git add -f` 대신 허용 규칙 `!results/hw_iislab4/HW09_runs.txt`를 추가했습니다. 이전 HW의 run 목록은 그대로 둡니다.
11. `hw.summarize`의 `test_acc` 집계는 `summary.{csv,md}`를 다시 쓰고, 이번에 `h9_*` 행이 추가되었습니다. 기존 arm 값은 바뀌지 않았습니다.

## 6. 제외·실패·붕괴 run

- 제외·실패·붕괴·NaN 없음. 80/80 completed, 재시도 0.
- `h9_clus_m20` Amazon_eletronics seed 1219: `zero_rows_all` 3(raw 기준 best와 l2 기준 best가 같은 checkpoint). test 노드 8881개 중 0개입니다.
- 첫 큐 실행의 argparse 실패 80건(§5-4)은 학습 전에 끝나 결과 파일이 없습니다.

## 7. 소요 시간, 환경, HEAD

- 작업 시작 HEAD `50fda1c`(HW08-c, 태그 `hw08`). 작업 시각 약 06:30–08:10 UTC.
  - 캐시·k-means: dblp 약 9분, Amazon_eletronics 약 12분(k-means CPU).
  - HW08 checkpoint 재평가 80개: 약 4분.
  - 본실행 07:15:01–07:55:39. run 시간 합 18454 s, run당 209–249 s.
- iislab4 RTX 3090 × 8. run당 GPU 메모리 최대 할당은 dblp 9693 MiB, Amazon_eletronics 11044 MiB(smoke 측정). conda `taskns`이고 패키지 변경은 없습니다.
- `ckpt/hw_iislab4/`: HW09 `h9_*`는 134 파일 2,242,670,816 B이고, HW08과 합친 전체는 5.0 GB입니다. 디스크 여유는 22 GB입니다.
