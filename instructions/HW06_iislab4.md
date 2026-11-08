# HW06 [iislab4] — 음성 샘플링 전략 비교: 무작위 m 곡선 · far / band / cluster · 에피소드 안 query 음성·양성 · 차원 상관 (auto, 4000 episode, 220 run)

2026-11-08 · 작성: 이희태 (Claude 정리) · 서버 **iislab4** · 브랜치 `hl/hsan-weighting` · 선행 `3b62b03`(HW05-d) 이후 · 태그 `hw06` · 사전 등록 HWP059–HWP069 · 선행 문서 `CLAUDE.md` 개정 8, `reports/HW05_iislab4.md`

- auto 모드. 멈추는 조건: §0 허용 외 미커밋 변경, §1.6 G-table 실패, §1.7 G-ident·회귀 확인 실패, G6, 해결 못 한 오류.
- 결과 루트 **`HW_RESULTS_ROOT=results/hw_iislab4`**, `GPUS=0,1,2,3,4,5,6,7`, seed 1212–1221. 모든 학습 arm은 **`valid_tasks: 500`, `episodes: 4000`, `geom_diag: true`**.
- 질문: L_original 분모에 넣는 음성을 **무엇으로, 몇 개** 쓸 때 가장 좋은가. 새 전략은 같은 m(20)에서 무작위와 비교한다. 이긴 전략의 최적 m 확인은 다음 지시서에서 한다.
- 시간 예산 3시간 (구현 60 / 확인 20 / 실행 ~60 / 집계·보고 30).

## 0. 현황 (수정 없음, 보고서에 적고 계속)

`git fetch && git status -sb | head -1`, `git log --oneline -3`, `nvidia-smi --query-gpu=index,memory.used --format=csv`. **작업 시작 HEAD 기록.** 허용 미커밋 변경은 미추적 `instructions/HW06_iislab4.md`와 수정된 `CLAUDE.md`(개정 8)뿐. 그 밖에 있으면 **멈추고 보고**. `git pull --rebase`. CLAUDE.md 개정 8을 다시 읽는다.

기록: GCN 출력 차원(z의 차원), HW05 `v500_naq_base_e4000`·`v500_xneg_rand_m20`의 설정 키 목록.

## 1. HW06-a: 구현

### 1.1 `extra_neg` 새 모드 (`hw/negatives.py`)
기존 `none`/`random`/`hard_raw`/`hard_a2x` 경로는 한 줄도 바꾸지 않는다. 후보 제외(그 에피소드의 support 5개, pseudo-query 50개, q 자신)는 모든 모드에 공통이다. 한 query 안에서는 중복 없이 뽑는다.

- **`far`**: q의 최단거리 ≤ 2 노드(`hw.structure`의 hop 집합, 그래프 `normalize_graph`)와 q의 raw cos 상위 100(`hard_raw` 표 재사용)을 추가로 제외한 뒤 균등. 전용 Generator `seed + 31337`.
- **`band`**: 학습 전에 노드마다 raw cos(L2 정규화 X) 상위 `band_hi`개 표를 만든다(자기 제외, 기본 500). 선택 시 순위 `band_lo`–`band_hi`(기본 101–500) 구간에서 제외 대상을 걸러 균등. 전용 Generator `seed + 40503`. 표 계산은 CPU 청크(N × N 밀집 금지).
- **`cluster`**: 학습 전에 Â²X(L2 정규화, `hw.structure.a2x_trainfree`)로 `sklearn.cluster.KMeans(n_clusters=k, n_init=4, random_state=20261108)`, 기본 k 100. 선택 시 q와 다른 군집 노드에서 균등. 전용 Generator `seed + 52361`. 군집 결과는 `cache/kmeans/<ds>_k100.pt`에 저장하고 재사용한다(데이터셋당 한 번, seed와 무관).
- 표·군집 계산은 CLAUDE.md §4.4의 "(선택 임베딩·선택 표)" 자리. 전역 RNG 소비 없음.

### 1.2 에피소드 안 query 음성·양성 (`hw/losses.py`)
- `ep_query_neg: true`: query q의 로짓 행에 같은 에피소드의 **다른 pseudo-class pseudo-query 40개**를 음성으로 추가(−‖z_q − z_{q'}‖²/T). `extra_neg`와 함께 켜면 둘 다 분모에 들어간다.
- `ep_query_pos: true`: 같은 pseudo-class의 다른 pseudo-query 9개를 양성 집합 P(q)에 추가. 기존 다중 양성 식(logsumexp_all − logsumexp_P + log|P|)을 그대로 쓴다. `ep_query_pos`는 `ep_query_neg`와 함께만 쓴다(단독 사용 시 오류).
- 둘 다 기본 false이면 기존 경로와 같다.

### 1.3 진단 (`nbr_diag`, 측정 전용 레이블)
- 모든 `extra_neg` 모드: 기존 `xneg_eq_query`(FN rate), `xneg_eq_support`, `xneg_cos_raw`, `xneg_fill_frac`.
- `band`: `band_cos_lo`, `band_cos_hi`(선택된 음성의 raw cos 최소·최대 평균).
- `ep_query_neg`: `qneg_eq_query`(다른 pseudo-class query 중 q와 같은 GT 비율 = query 음성의 FN rate), `qneg_cos_raw`.
- `ep_query_pos`: `qpos_eq_query`(같은 pseudo-class query 중 q와 같은 GT 비율 = query 양성의 순도).

### 1.4 `geom_diag` 확장 (`hw/run.py`)
기존 다섯 값에 `corr_offdiag_mean`(4096 표본 z의 차원 간 Pearson 상관 행렬, 비대각 |상관| 평균; 분산 0인 차원은 제외하고 수를 `dead_dims`로 기록)과 `dim`을 추가한다.

### 1.5 arm 추가 (`configs/hw.json`; 모두 `valid_tasks 500`, `episodes 4000`, `geom_diag true`, base `naq_base`)

| 묶음 | arm | 내용 |
|---|---|---|
| 기준 | `e4k_naq` | L_original만 (HW05 `v500_naq_base_e4000`의 복제 + 기하 진단) |
| A | `e4k_rand_m20` / `_m50` / `_m100` / `_m200` | `extra_neg {random, m}` |
| B | `e4k_far_m20` | `extra_neg {far, 20}` |
| B | `e4k_band_m20` | `extra_neg {band, 20, band_lo 101, band_hi 500}` |
| B | `e4k_clus_m20` | `extra_neg {cluster, 20, k 100}` |
| C | `e4k_qneg` | `ep_query_neg true` |
| C | `e4k_qneg_rand_m20` | `ep_query_neg true` + `extra_neg {random, 20}` |
| C | `e4k_qpos_qneg` | `ep_query_neg true`, `ep_query_pos true` |

11 arm × 2 ds × 10 seed = **220 run**.

### 1.6 G-table (학습 전, 실패 시 **멈추고 보고**)
두 데이터셋에서 다음을 계산해 `results/hw_iislab4/HW06_tables.json`에 쓴다.
- `band`: 노드마다 순위 101–500 구간에서 "q 자신을 뺀 cos > 0인 노드 수"의 최소·중앙값, cos(순위 101)·cos(순위 500)의 사분위, 순위 500의 cos가 0인 노드 비율. 구간 후보 수(400)에서 최대 제외 수(55)를 뺀 345 ≥ 20이므로 개수는 구조상 충분하다. **순위 500의 cos가 0인 노드 비율이 0.20을 넘으면 멈추고 보고**(그 경우 band가 무작위에 가까워진다).
- `band` 구간의 같은 클래스 비율(측정 전용, 노드 macro) — 학습 전 기대 FN rate.
- `far`: 노드마다 제외 후 남는 후보 수의 최소·중앙값(≥ 1000이어야 함; 아니면 멈추고 보고).
- `cluster`: 군집 크기 최소·중앙값·최대, 노드 macro "같은 군집 내 같은 클래스 비율", q와 다른 군집 노드 중 같은 클래스 비율(학습 전 기대 FN rate).
- `random`의 기대 FN rate = Σ p_c² (비교용).

### 1.7 G-ident, 회귀, 불변, smoke
- G-ident: seed 1212, Cora_full, 학습 전 모델, 에피소드 3개에서 `ep_query_neg false, ep_query_pos false, extra_neg none`의 손실 = 기존 L_original (|Δ| ≤ 1e-6). 실패 시 멈춤.
- 회귀: `naq_base`, `nbr_raw`(기본 설정) seed 1212 두 ds → `_regress/HW06/`, 기존 값과 |Δ| ≤ 0.30. 실패 시 멈춤.
- 불변: `git diff <작업 시작 HEAD> -- hw/losses.py hw/negatives.py hw/neighbors.py hw/run.py`의 삭제 줄 수와 기존 경로 발췌.
- smoke(episodes 4, `/tmp/hw_smoke`, 커밋 안 함): `e4k_band_m20`, `e4k_clus_m20`, `e4k_qpos_qneg` Cora_full 각 1회. 새 진단 키와 `geom.corr_offdiag_mean` 기록 확인.

커밋 `HW06-a [iislab4]: extra_neg far/band/cluster, episode query neg/pos, geom corr, arms, G-table` (CLAUDE.md 개정 8, 지시서, 사전 등록, G-table·G-ident 결과 포함) → pull --rebase → push.

## 2. HW06-b: 실행

- `results/hw_iislab4/HW06_runs.txt`: 220 run, 순서 seed 바깥·arm 안쪽.
- 집계: `python -m hw.summarize --ref e4k_naq --ref e4k_rand_m20 --ref e4k_qneg` (+ HW05 `v500_naq_base_e4000`과의 복제 비교는 보고서에서 직접 계산).
- 커밋 `HW06-b [iislab4]: 220 runs (4000 ep)` → pull --rebase → push.

## 3. 게이트

- **G-table**(§1.6), **G-ident**·**회귀**(§1.7): 실패 시 멈춤.
- **G6**: run의 95% 이상 `completed`(1회 재시도 후). 못 미치면 멈추고 보고.
- **G7 (기록)**: arm별 붕괴·NaN 수.
- **태그**: 위 게이트 통과 → `git tag hw06 && git push origin hw06`.

## 4. 사전 등록 (HW06-a 커밋에 포함, 판정은 사람)

Δ = 같은 seed 짝지은 `test_acc` 차이의 10 seed 평균. "두 ds" = 두 데이터셋 모두. 진단값은 마지막 평가 구간 10 seed 평균.

| ID | 예측 | 기준 | 근거 | 확신도 |
|---|---|---|---|---|
| HWP059 | 4000 episode에서도 m을 늘리면 오른다 | `rand_m50 − rand_m20` > 0, 두 ds | HW05 m5→m20 +0.46 / +0.26 | 0.55 |
| HWP060 | m 100–200 사이에서 포화한다 | \|`rand_m200 − rand_m100`\| ≤ 0.15, 두 ds | — | 0.50 |
| HWP061 | far는 무작위와 같다 | \|`far_m20 − rand_m20`\| ≤ 0.20, 두 ds | 무작위 FN이 이미 0.02 | 0.55 |
| HWP062 | band의 FN rate는 무작위의 5배 이상 | `band_m20`의 `xneg_eq_query` ≥ 0.10, 두 ds | 상위 20 FN 0.30 / 0.54(HW05) | 0.60 |
| HWP063 | band는 무작위보다 낮다 | `band_m20 − rand_m20` < 0, 두 ds | FN 증가 | 0.60 |
| HWP064 | cluster는 FN rate를 무작위보다 낮춘다 | `clus_m20`의 `xneg_eq_query` < `rand_m20`의 값, 두 ds | 같은 군집 제외 | 0.60 |
| HWP065 | query 음성의 FN rate는 무작위보다 높다 | `qneg_eq_query` ≥ 0.04, 두 ds | 서로 다른 anchor가 같은 GT일 수 있음 | 0.50 |
| HWP066 | query 음성만으로도 NaQ보다 높다 | `qneg − naq` > 0, 양수 ≥ 7/10, 두 ds | 음성 4 → 44 | 0.60 |
| HWP067 | query 음성과 무작위 음성은 더해진다 | `qneg_rand_m20 − rand_m20` > 0, 두 ds | 다른 출처의 음성 | 0.45 |
| HWP068 | query 양성 추가는 해롭다 | `qpos_qneg − qneg` < 0, 두 ds | query 순도 0.35 / 0.60 | 0.60 |
| HWP069 | 음성 보강은 차원 상관을 낮춘다 | `corr_offdiag_mean`: `rand_m20` < `naq`, 두 ds | HW05 effective rank 증가 | 0.60 |

## 5. 범위 밖

- m ≠ 20인 far·band·cluster, band 구간·k 변경, HSAN식 가중, 새 전역 손실, debiased 보정, 커리큘럼, 양성 선택 arm, 2000 episode arm, 기존 캐시 수정, seed 추가, 다른 데이터셋, `results/hw/` 수정.

## 6. 보고 (`reports/HW06_iislab4.md`, CLAUDE.md §7)

값의 종류를 표마다 명시. 주 결과 = `test_acc`(500 기준).
1. 11 arm × ds 평균 ± sd, 붕괴·NaN 수
2. **m 곡선 표**: m ∈ {0(`naq`), 20, 50, 100, 200} × ds의 정확도와 `− naq` Δ, 인접 m 간 Δ
3. **전략 표 (m 20)**: {rand, far, band, clus} − naq, − rand_m20 (Δ ± sd, 양수 k/10) + 각 `xneg_eq_query`(FN rate), `xneg_cos_raw`, band는 `band_cos_lo/hi`
4. **에피소드 안 query 표**: {qneg, qneg_rand_m20, qpos_qneg} − naq, − rand_m20, − qneg + `qneg_eq_query`, `qpos_eq_query`
5. **기하 표**: 11 arm × ds의 `norm_mean`, `uniformity`, `eff_rank`, `corr_offdiag_mean`, `dead_dims`, `dim`
6. G-table 값 (§1.6 전부), G-ident, 회귀, 불변 발췌
7. 복제: `e4k_naq − v500_naq_base_e4000`(HW05) seed별 Δ, 평균 |Δ|
8. best epoch 분포 (arm × ds)
9. HWP059–069 대조
10. 소요 시간, HEAD(작업 시작 HEAD 포함)
