# HW05 [iislab4] — 무작위 항의 기제 확정: pull/push 분해 · 추가 음성(random·hard) · 학습 길이 · 기하 진단 · FN rate · 아이디어 A 겹침 (auto, 220 run)

2026-11-08 · 작성: 이희태 (Claude 정리) · 서버 **iislab4** · 브랜치 `hl/hsan-weighting` · 선행 `e015bc4`(HW03b-d) 이후 · 태그 `hw05` · 사전 등록 HWP049–HWP058 · 선행 문서 `CLAUDE.md` 개정 7, `reports/HW03b_iislab4.md`

- auto 모드. 멈추는 조건: §0 허용 외 미커밋 변경, §1.5 G-ident 실패, §1.6 회귀 확인 실패, G6, 해결 못 한 오류.
- 결과 루트 **`HW_RESULTS_ROOT=results/hw_iislab4`**, `GPUS=0,1,2,3,4,5,6,7`, seed 1212–1221. 모든 학습 arm은 `valid_tasks: 500`.
- 질문은 하나다: **HW03b의 무작위 양성 이득(λ 0.3: Cora +1.80 / Amazon +0.95)은 어디서 오는가.** 후보는 (i) 추가 음성, (ii) 밀어냄 자체(균일성), (iii) 당김. 이 셋을 가르고, 추가 음성이라면 hard negative로 바꿨을 때 무엇이 달라지는지 본다.
- 시간 예산 2시간 30분 (구현 45 / 확인 15 / 실행 ~55 / 측정 10 / 집계·보고 25).

## 0. 현황 (수정 없음, 보고서에 적고 계속)

`git fetch && git status -sb | head -1`, `git log --oneline -3`, `nvidia-smi --query-gpu=index,memory.used --format=csv`. **작업 시작 HEAD 기록.** 허용 미커밋 변경은 미추적 `instructions/HW05_iislab4.md`와 수정된 `CLAUDE.md`(개정 7)뿐. 그 밖에 있으면 **멈추고 보고**. `git pull --rebase`. CLAUDE.md 개정 7을 다시 읽는다.

기록: `hw/losses.py`의 `episode_loss`에서 L_original의 분모가 어떤 텐서로 만들어지는지(로짓 행렬 모양, support 5개 고정인지), `eval_every`, `episodes` 기본값, best checkpoint 보관 방식(HW03b에서 500 기준은 GPU `deepcopy`).

## 1. HW05-a: 구현

### 1.1 `nbr_term ∈ {infonce, pull, push}` (`hw/losses.py`)
- 기본 `infonce` = 기존 L_neighbor 경로 그대로(한 줄도 바꾸지 않는다).
- `pull` = mean_q ‖z_q − z_{n(q,s⁺)}‖² / T (s⁺ = q의 양성 support에 대해 고른 노드).
- `push` = mean_q logsumexp_{s∈5}(−‖z_q − z_{n(q,s)}‖² / T).
- 둘 다 λ(`lambda_neighbor`)로 L_original에 더한다. `pick: global_random`과 함께 쓴다.

### 1.2 `extra_neg` 블록 (`hw/losses.py`, `hw/neighbors.py` 또는 새 `hw/negatives.py`)
- `mode ∈ {none, random, hard_raw, hard_a2x}`, `m`. 기본 `none`이면 기존 경로 불변.
- L_original의 로짓 행렬을 query마다 [support 5개 ‖ 추가 노드 m개]로 확장한다. 양성 마스크는 support 쪽 그대로, 추가 노드는 전부 음성. **λ 없음, 별도 항 없음.** L_neighbor가 켜져 있으면 그 항은 바꾸지 않는다.
- 후보 제외: 그 에피소드의 support 5개, pseudo-query 50개, query 자신.
- `random`: 전체 노드에서 제외 후 균등 m개. 전용 `torch.Generator`(device 맞춤, `manual_seed(seed + 23757)`), 전역 RNG 소비 없음.
- `hard_raw` / `hard_a2x`: 학습 전에 노드마다 cos(raw 또는 Â²X, L2 정규화; `hw.structure.a2x_trainfree`) 상위 100개 표를 만든다(자기 제외). 선택 시 표에서 제외 대상을 걸러 앞 m개. 부족하면 `random`으로 채우고 채운 비율을 기록한다.
- 진단 (`nbr_diag`에 추가, 측정 전용 레이블): `xneg_eq_query`(추가 음성의 GT == query GT 비율 = **false negative rate**), `xneg_eq_support`, `xneg_cos_raw`(query와 추가 음성의 raw cos 평균), `xneg_fill_frac`.

### 1.3 `geom_diag` (`hw/run.py`)
- `geom_diag: true`면 500 기준 best checkpoint의 모델로 고정 표본 4096 노드(전용 Generator seed 2026, 두 ds 공통 규칙)의 z(ProtoNet이 쓰는 임베딩)를 `torch.no_grad()`로 구해 다음을 `metrics.json["geom"]`에 기록한다.
  - `norm_mean`, `norm_std`
  - `pair_dist_mean`: 표본 내 쌍 유클리드 거리 평균
  - `uniformity`: log mean_{u≠v} exp(−2‖û − v̂‖²) (û = z/‖z‖; Wang & Isola 2020 정의)
  - `eff_rank`: 중심화 표본의 특이값 σ_i, p_i = σ_i/Σσ, exp(−Σ p_i log p_i)
  - `center_norm`: 표본 평균 벡터의 노름
- 기존 arm에는 영향 없음(기본 false).

### 1.4 arm 추가 (`configs/hw.json`; 모두 `valid_tasks: 500`, 표기 없는 키는 `v500_grand_l03` / `v500_naq_base`와 같음)

| 단계 | arm | 내용 | run |
|---|---|---|---|
| 복제·기하 | `v500_naq_base_r2` | `v500_naq_base` + `geom_diag` | 20 |
| 복제·기하 | `v500_grand_l03_r2` | `v500_grand_l03` + `geom_diag` | 20 |
| A 분해 | `v500_grand_pull_l03` | `pick global_random`, λ 0.3, `nbr_term: pull`, `geom_diag` | 20 |
| A 분해 | `v500_grand_push_l03` | 같고 `nbr_term: push` | 20 |
| A 음성 | `v500_xneg_rand_m5` | `naq_base` + `extra_neg {random, 5}`, `geom_diag` | 20 |
| A 음성 | `v500_xneg_rand_m20` | 같고 m 20 | 20 |
| B hard | `v500_xneg_hraw_m20` | `naq_base` + `extra_neg {hard_raw, 20}`, `geom_diag` | 20 |
| B hard | `v500_xneg_ha2x_m20` | 같고 `hard_a2x` | 20 |
| C 길이 | `v500_naq_base_e4000` | `naq_base`, `episodes: 4000` | 20 |
| C 길이 | `v500_grand_l03_e4000` | `v500_grand_l03`, `episodes: 4000` | 20 |
| D 결합 | `v500_raw_l10_xrand_m20` | `v500_raw_l10` + `extra_neg {random, 20}`, `geom_diag` | 20 |
| 합계 | | 11 arm | **220** |

### 1.5 G-ident (실패 시 **멈추고 보고**)
seed 1212, Cora_full, 학습 전 모델, 에피소드 3개에서 `pick: global_random`일 때 `|pull + push − infonce|` ≤ 1e-5 (log|P| = 0이므로 상수 없음). 또 `extra_neg {random, 0}`(m = 0)의 L_original이 기존 L_original과 같은지(≤ 1e-6). 스크래치 스크립트, 커밋 안 함.

### 1.6 회귀 확인, 불변, smoke
- 회귀: `naq_base`, `nbr_raw`(`valid_tasks` 기본 50) seed 1212 두 ds → `_regress/HW05/`, 기존 값과 |Δ| ≤ 0.30. 어긋나면 **멈추고 보고**.
- 불변: `git diff <작업 시작 HEAD> -- hw/losses.py hw/neighbors.py hw/run.py`의 삭제 줄 수와 기존 경로 발췌.
- smoke(epochs 4, `/tmp/hw_smoke`, 커밋 안 함): `v500_grand_push_l03`, `v500_xneg_hraw_m20`, `v500_naq_base_e4000`(epochs는 smoke 값으로) Cora_full 각 1회. `geom`·`xneg_*` 키 기록 확인.

커밋 `HW05-a [iislab4]: nbr_term pull/push, extra_neg (random/hard), geom_diag, episodes override, arms` (CLAUDE.md 개정 7, 지시서, 사전 등록, G-ident 결과 포함) → pull --rebase → push.

## 2. HW05-b: 실행

- `results/hw_iislab4/HW05_runs.txt`: 11 arm × 2 ds × 10 seed = 220. 순서: seed 바깥, arm 안쪽. 4000 episode arm은 run당 약 2배 시간이다.
- 집계: `python -m hw.summarize --ref v500_naq_base --ref v500_grand_l03 --ref v500_xneg_rand_m20 --ref v500_raw_l10`.
- 커밋 `HW05-b [iislab4]: 220 runs` → pull --rebase → push.

## 3. HW05-c: 학습 없는 측정 — 아이디어 A 겹침 (`hw/measure_positive.py --overlap`)

- 목적: "이웃이 겹치는 anchor끼리 묶어 K > 1 pseudo-class를 만든다"(아이디어 A)의 전제 측정. 레이블은 측정 전용.
- 그래프 A = `normalize_graph`. 노드 쌍 (i, j), i < j, 공통 이웃 c_ij = (A·A)_ij, Jaccard J = c / (d_i + d_j − c). 희소 행렬로 계산하고 c = 0인 쌍은 제외한다(J = 0).
- **Jaccard 표**: θ ∈ {0.1, 0.2, 0.3, 0.5}마다 (a) J ≥ θ인 쌍 수, (b) 그중 같은 클래스 비율(= 병합 정밀도), (c) J ≥ θ인 상대가 1개 이상 있는 노드 비율(= 커버리지), (d) 노드당 그런 상대 수의 중앙값. chance = Σ p_c².
- **PPR 표**: `cache/ppr/<ds>_a015_top100.pt`의 상위 20 집합으로 같은 네 값(θ는 두 집합의 Jaccard).
- **같은 클래스 쌍의 분포**: 무작위 같은 클래스 쌍 100,000개(전용 Generator)와 다른 클래스 쌍 100,000개의 J 분포(0인 비율, 중앙값, q90).
- 출력 `results/measure/HW05_overlap_<ds>.{json,md}`. 커밋 `HW05-c [iislab4]: idea-A neighborhood overlap measurement` → pull --rebase → push.

## 4. 게이트

- **G-ident**(§1.5), **회귀 확인**(§1.6): 실패 시 멈춤.
- **G6**: run의 95% 이상 `completed`(1회 재시도 후). 못 미치면 멈추고 보고. `pull`·`push` arm의 붕괴·발산(loss NaN/Inf)은 **실패가 아니라 결과**다: 붕괴 표시하고 계속한다. NaN이면 그 run을 `completed(nan)`으로 기록하고 집계에서 test를 NaN으로 둔다.
- **G7 (기록)**: arm별 붕괴·NaN 수.
- **태그**: G-ident + 회귀 + G6 통과 → `git tag hw05 && git push origin hw05`.

## 5. 사전 등록 (HW05-a 커밋에 포함, 판정은 사람)

Δ = 같은 seed 짝지은 `test_acc`(500 기준) 차이의 10 seed 평균. "두 ds" = 두 데이터셋 모두. naq = `v500_naq_base`(HW03b 원본), grand = `v500_grand_l03`(원본).

| ID | 예측 | 기준 | 근거 | 확신도 |
|---|---|---|---|---|
| HWP049 | 복제 잡음은 작다 | `_r2 − 원본` 평균 \|Δ\| ≤ 0.30, 두 arm·두 ds | 같은 서버 | 0.70 |
| HWP050 | 추가 음성만으로 무작위 양성의 이득이 재현된다 | `xneg_rand_m20 − naq` ≥ +1.0 (Cora) / ≥ +0.5 (Amazon), 양수 ≥ 8/10 | H-neg | 0.50 |
| HWP051 | 음성 수에 단조 | `xneg_rand_m20 − xneg_rand_m5` > 0, 두 ds | — | 0.55 |
| HWP052 | 이득은 밀어냄 쪽에 있다 | `grand_push_l03 − naq` > 0 이고 `grand_pull_l03 − naq` ≤ 0, 두 ds (붕괴·NaN이면 pull은 "≤ 0"으로 판정) | 양성 내용이 무관했음 | 0.60 |
| HWP053 | hard negative의 false negative rate는 무작위의 10배 이상 | `xneg_hraw_m20`의 `xneg_eq_query` ≥ 0.20, 두 ds (무작위 ≈ 0.02) | NaQ query 순도 0.35/0.60 = 특징 최근접의 동질도 | 0.75 |
| HWP054 | hard negative는 무작위 음성보다 낮다 | `xneg_hraw_m20 − xneg_rand_m20` < 0, 두 ds | FN이 같은 클래스를 밀어냄 | 0.60 |
| HWP055 | 2000 episode는 짧다 | `naq_base_e4000 − naq` ≥ +0.3, 양수 ≥ 7/10, Cora_full | best epoch 다수 2000 | 0.50 |
| HWP056 | 이웃 항과 추가 음성은 더해진다 | `raw_l10_xrand_m20 − max(raw_l10, xneg_rand_m20)` ≥ +0.2, 두 ds | 다른 기제라면 가산 | 0.35 |
| HWP057 | 이득이 있는 arm은 더 균일하다 | `uniformity`: grand_r2, xneg_rand_m20 < naq_r2 (더 음수), 두 ds | H-uni | 0.55 |
| HWP058 | 아이디어 A의 겹침은 정밀하지만 드물다 | Jaccard θ 0.3: 병합 정밀도 ≥ 0.70, 커버리지 ≤ 0.35, 두 ds | 1-hop 순도 0.59/0.62, 차수 중앙값 3–4 | 0.50 |

## 6. 범위 밖

- 새 양성 선택 arm, λ sweep 추가, hard negative의 다른 정의(semi-hard, 거리 구간), M > 20, 인코더 변경, validation 50 arm(회귀 제외), 기존 캐시 수정, 다른 데이터셋, seed 추가, `results/hw/` 수정.

## 7. 보고 (`reports/HW05_iislab4.md`, CLAUDE.md §7)

값의 종류를 표마다 명시. 주 결과 = `test_acc`(500 기준).
1. 11 arm × ds 평균 ± sd, 붕괴·NaN 수; 원본 `v500_naq_base`, `v500_grand_l03`, `v500_raw_l10`, `v500_xneg_*` 비교 행 포함
2. **분해 표**: {grand(원본), grand_r2, pull, push, xneg_rand_m5, xneg_rand_m20} − naq (Δ ± sd, 양수 k/10) × 2 ds
3. **hard negative 표**: {xneg_rand_m20, xneg_hraw_m20, xneg_ha2x_m20} − naq, 그리고 각 arm의 `xneg_eq_query`(FN rate), `xneg_eq_support`, `xneg_cos_raw`, `xneg_fill_frac`
4. **길이 표**: e4000 − e2000(같은 설정), best epoch 분포(500 기준)
5. **결합 표**: raw_l10_xrand_m20 − {naq, raw_l10, xneg_rand_m20}
6. **기하 표**: `geom_diag` arm × ds의 다섯 값(10 seed 평균 ± sd)
7. **복제 표**: `_r2 − 원본` seed별 Δ, 평균 |Δ|, 최대 |Δ|
8. 겹침 측정 표(§3), G-ident 값, 회귀, 불변 발췌
9. HWP049–058 대조
10. 소요 시간, HEAD(작업 시작 HEAD 포함)
