# HW03a [iislab4] — cos HSAN τ 격자 + hard positive 선택 + 자동 2단계 + 조건부 BGRL (auto, 10 seed)

2026-11-07 · 작성: 이희태 (Claude 정리) · 서버 **iislab4** (RTX 3090 × 8) · 브랜치 `hl/hsan-weighting` · 선행 `d277ea2` 이후 · 태그 `hw03a` · 사전 등록 HWP014–HWP023 · 선행 문서 `CLAUDE.md` 개정 4, `reports/HW02_hsan_weighting.md`, `reports/HW03_prep_iislab4.md`

- auto 모드. 멈추는 조건: §1.6 회귀 확인 실패, G6, 구현 중 해결 못 한 오류. 그 밖에는 규칙대로 판단하고 계속한다.
- 결과 루트: **모든 run은 `HW_RESULTS_ROOT=results/hw_iislab4`**. `results/hw/`(seio)에는 쓰지 않는다.
- 병렬: `GPUS=0,1,2,3,4,5,6,7` (작업자 8개, GPU당 1 run).
- seed: 1212–1221 (10개).
- 시간 예산 3시간 (구현 30 / 회귀 5 / 1단계 ~35 / 2단계 ~8 / BGRL 대기 ≤60 + 실행 ~15 / 집계·보고 20).
- 같은 시간에 seio가 HW03a_seio를 돌린다. seio는 `configs/hw.json`과 기존 `hw/*.py`를 고치지 않고, 새 파일 `hw/purity.py`만 추가한다. 이 서버의 HW03a-a 커밋(이웃 선택기)을 seio가 pull해서 쓰므로, **HW03a-a는 가장 먼저 끝내고 push한다.**

## 0. 현황 (수정 없음, 보고서에 적고 계속)

`git fetch && git status -sb | head -1`(upstream이 `origin/hl/hsan-weighting`이어야 한다), `git log --oneline -3`, `nvidia-smi --query-gpu=index,memory.used --format=csv`, `ls cache/bgrl/ 2>/dev/null`. HEAD가 `d277ea2` 이후이고 작업 트리가 깨끗한지. `hw/`에 미커밋 변경이 있으면 **멈추고 보고**.

## 1. HW03a-a: 구현 (먼저 끝내고 push)

### 1.1 드롭 파일
`~/wkspace/HW_drop/CLAUDE.md`(개정 4) → 루트, 이 지시서 → `instructions/HW03a_iislab4.md`. CLAUDE.md를 다시 읽는다.

### 1.2 이웃 선택 `pick` (`hw/neighbors.py`)
- 새 설정 키 `pick ∈ {max, contrast, random}`, 기본 `max`. `max`는 기존 코드 경로를 그대로 쓴다(한 줄도 바꾸지 않는다).
- 유효 이웃: 기존과 같다(support의 1-hop 이웃, query 자신 제외, `max_neighbors` 적용). 유효 이웃이 없으면 support 자신(기존 fallback).
- `contrast`: 선택 공간(`selector`)의 정규화 임베딩 e로 `score(n) = cos(e_n, e_support) − cos(e_n, e_query)`, argmax. 
- `random`: 유효 이웃 중 균등 무작위. **전용 `torch.Generator`**(device 맞춤, `manual_seed(seed + 7919)`)를 쓰고 전역 RNG를 소비하지 않는다.
- 모든 선택은 `torch.no_grad()`.

### 1.3 순도·난이도 진단 (`hw/run.py`, 측정 전용, CLAUDE.md §5)
- 에피소드마다 query × 자기 클래스(양성) 선택에 대해, `eval_every` 구간 평균을 `epochs[].nbr_diag`에 기록한다.
  - `nbr_eq_support`: 고른 이웃의 GT == 자기 support의 GT 비율
  - `nbr_eq_query`: 고른 이웃의 GT == query의 GT 비율
  - `query_eq_support`: query의 GT == 자기 support의 GT 비율 (NaQ query 순도, arm과 무관한 참고값)
  - `pos_nbr_cos_raw`: query와 고른 양성 이웃의 raw feature cos 평균 (selector와 무관하게 raw로 잰다, 난이도 비교용)
  - `fallback_frac`: 유효 이웃이 없어 support 자신을 쓴 비율
  - `multi_nbr_frac`: 유효 이웃이 2개 이상인 support 비율 (`pick`이 실제로 작동하는 비율)
- GT 레이블은 `data`의 레이블 텐서에서 읽는다. 진단은 손실·선택·체크포인트 선택에 들어가지 않는다. `loss: original`(이웃 항 없음) arm에서는 `query_eq_support`만 기록한다.
- `hw/summarize.py`: `summary.csv`에 마지막 구간 `nbr_eq_support`, `nbr_eq_query`, `pos_nbr_cos_raw` 열 추가. 붕괴 표시(CLAUDE.md §4.3: best-epoch val ≤ 25% → `collapsed=1`, `summary.md` 셀에 `[c=k]`).

### 1.4 cos τ
- `hsan.tau_cos`가 설정에서 그대로 쓰이는지 확인한다(HW02 `nbr_raw_cos`는 0.5). 바꿀 것이 없으면 코드 변경 없음.

### 1.5 `.gitignore`와 arm 추가
- `.gitignore`에 `!*.yml`, `!results/hw_iislab4/**/*.log` 허용 규칙 추가(HW03-prep에서 `git add -f`로 넣은 파일들이 이후 정상 추적되게).
- `configs/hw.json`에 다음 arm 추가 (모두 `base: nbr_raw`, 즉 λ_neighbor 0.1, `selector: raw`).

| arm | hsan | pick |
|---|---|---|
| `cos_t005_none` | `{mode: none, logit: cos_hsan, tau_cos: 0.05}` | max |
| `cos_t005_p` | `{mode: p, apply: orig, beta: 1, space: gcn, logit: cos_hsan, tau_cos: 0.05}` | max |
| `cos_t005_pn` | `{mode: pn, apply: orig, beta: 1, space: gcn, logit: cos_hsan, tau_cos: 0.05}` | max |
| `cos_t010_none` / `_p` / `_pn` | 위와 같고 `tau_cos: 0.10` | max |
| `cos_t020_none` / `_p` / `_pn` | 위와 같고 `tau_cos: 0.20` | max |
| `nbr_raw_contrast` | null | contrast |
| `nbr_raw_random` | null | random |
| `nbr_bgrl_l01` | null (`base: nbr_bgrl`, `lambda_neighbor: 0.1`) | max |

(`nbr_bgrl_l01`은 §4 조건부 단계용. `nbr_raw`와 λ가 같고 선택 공간만 BGRL.)

### 1.6 회귀 확인과 smoke
- `naq_base`, `nbr_raw` seed 1212 두 데이터셋을 **새 하위 경로**에 다시 돌려 기존 `results/hw_iislab4/<arm>/<ds>/seed1212` 값과 비교: `HW_RESULTS_ROOT=results/hw_iislab4/_regress`. |Δ| ≤ 0.30. 어긋나면 **멈추고 보고**. (기존 결과를 덮어쓰지 않는다. `_regress/`는 커밋하되 집계에서 뺀다.)
- smoke(epochs 4, 스크래치 루트 `/tmp/hw_smoke`): `nbr_raw_contrast`, `nbr_raw_random`, `cos_t005_pn` 각 Cora_full 1회. 진단 키가 기록되는지 확인. 결과는 커밋하지 않는다.

커밋 `HW03a-a [iislab4]: pick (contrast/random), nbr diagnostics, collapse flag, arms, gitignore` → **즉시 push**. 사전 등록(§6)과 CLAUDE.md 개정 4, 이 지시서를 이 커밋에 포함한다.

## 2. HW03a-b: 1단계 격자

run 목록(`results/hw_iislab4/HW03a_stage1.txt`):
- `naq_base`, `nbr_raw` × 2 ds × seed 1217–1221 (1212–1216은 HW03-prep에 있음) = 20
- `cos_t{005,010,020}_{none,p,pn}` (9 arm) + `nbr_raw_contrast` + `nbr_raw_random` = 11 arm × 2 ds × 10 seed = 220
- 합계 240 run. 순서: seed 바깥, arm 안쪽.

끝나면 `python -m hw.summarize --ref nbr_raw --ref naq_base --ref cos_t005_none --ref cos_t010_none --ref cos_t020_none` (모두 `HW_RESULTS_ROOT=results/hw_iislab4`). 커밋 `HW03a-b [iislab4]: stage-1 grid (240 runs)` → push.

## 3. HW03a-c: 자동 2단계

1. **C\* 선택**: `cos_t*_p`, `cos_t*_pn` 6개 arm을 `paired_nbr_raw.md`의 "두 데이터셋 Δ 평균의 평균"으로 정렬, 1위를 C\*로 한다. 동률이면 양수 seed 합이 큰 쪽. C\*의 τ를 τ\*라 한다. 선택 근거 표를 보고서에 적는다.
2. arm 추가:
   - `C*_contrast`: C\*와 같고 `pick: contrast` (이름은 예: `cos_t010_pn_contrast`)
   - `cosnone_τ*_contrast`: `cos_t{τ*}_none`과 같고 `pick: contrast`
3. 2 arm × 2 ds × 10 seed = 40 run. `summarize --ref nbr_raw --ref nbr_raw_contrast --ref cos_t{τ*}_none --ref <C*>`.
4. 커밋 `HW03a-c [iislab4]: stage-2 (C*=<arm>, contrast variants)` → push.

## 4. HW03a-d: 조건부 BGRL 단계

1. **조건 확인**: `cache/bgrl/Cora_full.pt`와 `cache/bgrl/Amazon_clothing.pt`가 모두 있고, 각각 (a) `dataset` 키가 파일명과 같고, (b) `embeddings.shape[0]`이 노드 수(19,793 / 24,919)와 같고, (c) `node_ids`가 `arange(N)`과 같다.
   - 없으면 5분마다 다시 확인, **최대 60분** 기다린다. 그래도 없으면 이 단계를 건너뛰고 보고서에 "BGRL 파일 없음 — 건너뜀"이라 적는다. 조건 (a)–(c) 중 하나라도 어긋나면 건너뛰고 어긋난 값을 적는다.
   - 두 파일의 md5를 보고서에 적는다(seio 보고서 값과 대조용).
2. arm (모두 `selector: bgrl`, `bgrl_dir: cache/bgrl`):
   - `nbr_bgrl` (λ 1.0, = exp016)
   - `nbr_bgrl_l01` (λ 0.1)
   - `nbr_bgrl_l01_contrast` (λ 0.1, `pick: contrast`)
   - `C*_bgrl` (C\*와 같고 `selector: bgrl`)
   - `C*_contrast_bgrl` (`C*_contrast`와 같고 `selector: bgrl`)
3. 5 arm × 2 ds × 10 seed = 100 run. `summarize --ref nbr_raw --ref nbr_bgrl_l01`.
4. 커밋 `HW03a-d [iislab4]: BGRL stage (5 arms)` → push.

## 5. 게이트

- **G6**: 각 단계에서 run의 95% 이상이 `completed`(1회 재시도 후). 못 미치면 **멈추고 보고**.
- **G7 (기록만)**: 붕괴 run(best-epoch val ≤ 25%) 수를 arm별로 센다. 멈추지 않는다.
- **G8 (태그 조건)**: 회귀 확인 + G6(b·c 단계). 통과 시 `git tag hw03a && git push origin hw03a`. d 단계는 태그 조건이 아니다.

## 6. 사전 등록 (HW03a-a 커밋에 포함, 판정은 사람)

Δ = 같은 seed 짝지은 test 차이의 10 seed 평균. "양수 k/10".

| ID | 예측 | 기준 | 근거 | 확신도 |
|---|---|---|---|---|
| HWP014 | 작은 τ에서 cos 기준선이 `nbr_raw` 수준으로 올라온다 | τ ∈ {0.05, 0.1, 0.2} 중 하나에서 `cos_t*_none − nbr_raw` ≥ −0.30, 두 ds | HW02 τ=0.5에서 −5.7/−4.3; 작은 τ가 대조학습 표준 | 0.50 |
| HWP015 | HSAN 가중치는 같은 τ의 cos 기준선을 넘는다 | 최소 하나의 τ에서 `cos_t*_{p 또는 pn} − cos_t*_none` > 0, 양수 ≥ 7/10, 두 ds | HW02 cos 쌍 +1.13/+1.46 | 0.60 |
| HWP016 | 최고 HSAN-cos가 `nbr_raw`를 넘는다 | `C* − nbr_raw` ≥ +0.30, 양수 ≥ 7/10, 두 ds | — | 0.30 |
| HWP017 | 이득은 음성 쪽에서 온다 | 세 τ 중 2개 이상에서 `cos_t*_pn − cos_t*_p` > 0, 두 ds | 양성은 쉬운 쪽에 몰려 있음(HW02 진단) | 0.55 |
| HWP018 | `contrast`는 더 어렵고 덜 순수한 양성을 고른다 | `nbr_raw_contrast`의 `pos_nbr_cos_raw`가 `nbr_raw`보다 낮고, `nbr_eq_support`가 5%p 이상 낮다, 두 ds | 점수 정의 | 0.80 |
| HWP019 | hard positive 자체가 이득이다 | `nbr_raw_contrast − nbr_raw` > 0, 양수 ≥ 6/10, 두 ds | HL 트랙 "순도 × 다양성" | 0.35 |
| HWP020 | 무작위 이웃은 max보다 낮다 | `nbr_raw_random − nbr_raw` < 0, 두 ds | 순도 하락 | 0.60 |
| HWP021 | HSAN은 hard positive가 있을 때 더 이득이다 | `C*_contrast − C*` > 0, 두 ds | HSAN pulling 가중치가 올릴 대상이 생김 | 0.40 |
| HWP022 | BGRL 선택이 raw 선택보다 낫다 (λ 동일) | `nbr_bgrl_l01 − nbr_raw` > 0, 두 ds | 표 1 exp016 − exp015 +0.52/+0.49 (λ 다름) | 0.55 |
| HWP023 | 결과가 선택 임베딩과 무관하다 | sign(`C* − nbr_raw`) = sign(`C*_bgrl − nbr_bgrl_l01`), 두 ds | 석휘 주장 | 0.50 |

(HWP022·023은 d 단계를 건너뛰면 "판정 불가".)

## 7. 범위 밖

- `euclid_gradscale` arm, `apply: both`, β ≠ 1, `space: raw`의 cos arm, λ_neighbor 탐색, 클러스터링, hard negative, 다른 데이터셋, seed 추가, 원본 모듈·`choi_exps/` 수정, `results/hw/` 수정, BGRL 학습.

## 8. 보고 (`reports/HW03a_iislab4.md`, CLAUDE.md §7)

핵심 표:
1. arm × ds 10 seed 평균 ± 표준편차, 붕괴 수 (1·2·d 단계 전부)
2. `paired_nbr_raw` 전체 (Δ ± sd, 양수 k/10)
3. τ별 표: 행 τ, 열 {none − nbr_raw, p − none, pn − none, pn − p} × 2 ds
4. 이웃 진단 표: arm × ds의 `nbr_eq_support`, `nbr_eq_query`, `query_eq_support`, `pos_nbr_cos_raw`, `fallback_frac`, `multi_nbr_frac` (마지막 구간, 10 seed 평균)
5. C\* 선택 근거 표
6. d 단계: 조건 확인 결과, md5, `paired_nbr_bgrl_l01`
7. HWP014–023 대조
8. 소요 시간(단계별), HEAD
