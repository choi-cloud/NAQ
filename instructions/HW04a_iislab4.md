# HW04a [iislab4] — 2-hop 양성: support와 cos 최대 (`sup_cos`, Â²X · raw) + 1-hop 대조 (auto, 10 seed)

2026-11-08 · 작성: 이희태 (Claude 정리) · 서버 **iislab4** · 브랜치 `hl/hsan-weighting` · 선행 `e4faa1b` 이후 · 태그 `hw04a` · 사전 등록 HWP033–HWP040 · 선행 문서 `CLAUDE.md` 개정 5, `reports/HW03a_iislab4.md`, `reports/HW04m_iislab4.md`

- auto 모드. 멈추는 조건: §0 허용 외 미커밋 변경, §2.4 G-match 실패, §2.5 회귀 확인 실패, G6, 해결 못 한 오류.
- 결과 루트: **`HW_RESULTS_ROOT=results/hw_iislab4`**. `GPUS=0,1,2,3,4,5,6,7`. seed 1212–1221.
- **바꾸는 것은 이웃 선택 하나뿐이다.** 손실, `lambda_neighbor`(0.1), 학습 길이, 평가, 나머지 설정은 `nbr_raw`와 같다.
- 시간 예산 1시간 (구현 25 / 확인 10 / 실행 ~12 / 집계·보고 15).

## 0. 현황 (수정 없음, 보고서에 적고 계속)

`git fetch && git status -sb | head -1`, `git log --oneline -3`, `nvidia-smi --query-gpu=index,memory.used --format=csv`, `ls results/hw_iislab4/nbr_raw/*/ | head`. **작업 시작 HEAD 해시를 기록한다**(§2.5 불변 확인의 기준점). 허용되는 미커밋 변경은 미추적 `instructions/HW04a_iislab4.md`와 수정된 `CLAUDE.md`(개정 5)뿐이다(CLAUDE.md §2.1). 그 밖에 있으면 **멈추고 보고**. 먼저 `git pull --rebase`.

CLAUDE.md 개정 5를 다시 읽는다.

## 1. 비교 설계

| arm | 후보 | 고르는 기준 | 비교 |
|---|---|---|---|
| `nbr_raw` (기존, 10 seed 있음) | 유효 1-hop 이웃 | cos(이웃, **query**), raw | 기준선 |
| `nbr_h2_a2xsup` (새) | 최단거리 정확히 2인 노드 | cos(후보, **support**), Â²X | vs `nbr_raw` |
| `nbr_h2_rawsup` (새) | 최단거리 정확히 2인 노드 | cos(후보, **support**), raw | vs `nbr_raw`; vs `nbr_h2_a2xsup` = 공간 효과 |
| `nbr_h1_a2xsup` (새, 대조) | 유효 1-hop 이웃 (`nbr_raw`와 같은 집합) | cos(후보, **support**), Â²X | vs `nbr_raw` = 기준 변경 효과; `nbr_h2_a2xsup`와 = 거리 효과 |

참고 (HW04m 노드 macro 순도, Cora_full / Amazon_clothing): `hop2_top1_a2x` 0.6149 / 0.7091, `hop2_top1_raw` 0.5982 / 0.7183, `hop1_top1_a2x` 0.6577 / 0.6564.

## 2. HW04a-a: 구현

### 2.1 `pick: sup_cos` (`hw/neighbors.py`, 필요 시 `hw/run.py`의 준비 단계)
- 새 키: `pick_hop ∈ {1, 2}`, `pick_space ∈ {a2x, raw}`. `pick`이 `sup_cos`가 아니면 무시. 기존 `max`/`contrast`/`random` 경로는 한 줄도 바꾸지 않는다.
- 공간 (둘 다 L2 행 정규화):
  - `a2x`: Â²X. **`hw.structure.a2x_trainfree`를 import**해서 쓴다(HW04m과 같은 식).
  - `raw`: X. HW04m `hop*_top*_raw`와 같은 정규화.
- `pick_hop: 1`: 후보 = 기존 유효 이웃 집합 그대로(support의 1-hop, query 자신 제외, `max_neighbors` 적용). argmax cos(e_후보, e_support). 기존 `max` 경로와 같은 자리에서 계산해도 된다.
- `pick_hop: 2`: 후보 = 최단거리 정확히 2인 노드(`hw.structure`의 hop 집합 함수, 그래프는 HW04m의 `normalize_graph`). `max_neighbors`를 적용하지 않는다. 학습 전에 노드마다 **cos 상위 2개 인덱스 표**(없으면 −1)를 공간별로 한 번 계산해 두고, 선택 시 1위가 그 에피소드의 query 자신이면 2위, 둘 다 쓸 수 없으면 support 자신(fallback).
- 표 계산과 선택은 `torch.no_grad()`, 전역 RNG를 소비하지 않는다. 표 계산 위치는 CLAUDE.md §4.4 순서의 "(선택 임베딩·선택 표)" 자리.
- `max_neighbors` 값과, 1-hop 이웃 수가 그 값을 넘는 support 비율을 보고서에 적는다.

### 2.2 진단 추가 (`hw/run.py`, 측정용)
기존 `nbr_diag`(`nbr_eq_support`, `nbr_eq_query`, `query_eq_support`, `pos_nbr_cos_raw`, `fallback_frac`, `multi_nbr_frac`)에 다음을 더한다. 레이블이 필요 없는 항목은 레이블을 쓰지 않는다.
- `nbr_in_own_queries`: 고른 양성이 같은 에피소드에서 **그 support의 pseudo-query 10개** 중 하나인 비율
- `nbr_in_any_queries`: 고른 양성이 그 에피소드의 **어느 pseudo-query**(5 × 10개)든 하나인 비율
- `nbr_is_query`: 고른 양성이 지금 짝지어진 query 자신인 비율 (0이어야 한다)
- `pos_nbr_cos_a2x`: query와 고른 양성의 Â²X cos 평균
- `hw/summarize.py`의 `summary.csv`에 `nbr_in_own_queries`, `nbr_in_any_queries` 열 추가(마지막 구간 값).

### 2.3 arm 추가 (`configs/hw.json`)

| arm | 설정 |
|---|---|
| `nbr_h2_a2xsup` | `base: nbr_raw`, `pick: sup_cos`, `pick_hop: 2`, `pick_space: a2x` |
| `nbr_h2_rawsup` | `base: nbr_raw`, `pick: sup_cos`, `pick_hop: 2`, `pick_space: raw` |
| `nbr_h1_a2xsup` | `base: nbr_raw`, `pick: sup_cos`, `pick_hop: 1`, `pick_space: a2x` |

### 2.4 G-match (실패 시 **멈추고 보고**)
`pick_hop: 2` 표의 1위만으로(query 제외 없음) 노드 macro 순도를 측정 전용으로 계산해 HW04m과 비교한다. |Δ| ≤ 0.002.

| 공간 | Cora_full | Amazon_clothing |
|---|---|---|
| `a2x` (`hop2_top1_a2x`) | 0.6149 | 0.7091 |
| `raw` (`hop2_top1_raw`) | 0.5982 | 0.7183 |

2-hop이 없는 노드 비율도 HW04m coverage(0.9694 / 0.9576)와 맞는지 적는다. 이 확인은 스크래치 스크립트 또는 `hw/measure_positive.py`의 새 옵션으로 하고, 학습 경로에는 넣지 않는다.

### 2.5 회귀 확인, 불변 확인, smoke
- 회귀(CLAUDE.md §6-7): `naq_base`, `nbr_raw` seed 1212 두 ds → `results/hw_iislab4/_regress/`, 기존 값과 |Δ| ≤ 0.30. 어긋나면 **멈추고 보고**.
- 불변: `git diff <작업 시작 HEAD> -- hw/neighbors.py`에서 `max`/`contrast`/`random` 경로 줄이 바뀌지 않았음을 보고서에 적는다(diff 발췌).
- smoke(epochs 4, `/tmp/hw_smoke`, 커밋 안 함): 새 3 arm Cora_full 각 1회. 새 진단 키 기록과 `nbr_is_query` = 0 확인.

커밋 `HW04a-a [iislab4]: pick sup_cos (hop 1/2, a2x/raw), query-overlap diagnostics, arms` (CLAUDE.md 개정 5, 이 지시서, 사전 등록 포함) → pull --rebase → push.

## 3. HW04a-b: 실행

- run 목록 `results/hw_iislab4/HW04a_runs.txt`: 3 arm × 2 ds × seed 1212–1221 = 60 run. 순서: seed 바깥, arm 안쪽.
- `python -m hw.summarize --ref nbr_raw --ref naq_base --ref nbr_h1_a2xsup --ref nbr_h2_a2xsup` (`HW_RESULTS_ROOT=results/hw_iislab4`).
- 커밋 `HW04a-b [iislab4]: sup_cos h2 a2x/raw + h1 a2x (60 runs)` → pull --rebase → push.

## 4. 게이트

- **G-match** (§2.4), **회귀 확인** (§2.5): 실패 시 멈춤.
- **G6**: run의 95% 이상 `completed`(1회 재시도 후). 못 미치면 멈추고 보고.
- **G7 (기록만)**: arm별 붕괴 수.
- **태그**: G-match + 회귀 + G6 통과 → `git tag hw04a && git push origin hw04a`.

## 5. 사전 등록 (HW04a-a 커밋에 포함, 판정은 사람)

Δ = 같은 seed 짝지은 test 정확도 차이의 10 seed 평균. 진단값은 마지막 구간, 10 seed 평균. "두 ds" = 두 데이터셋 모두.

| ID | 예측 | 기준 | 근거 | 확신도 |
|---|---|---|---|---|
| HWP033 | 학습 중 순도가 측정값을 따른다 | `nbr_eq_support`: `nbr_h2_a2xsup` ≥ 0.60 / 0.69, `nbr_h2_rawsup` ≥ 0.58 / 0.70 (Cora / Amazon) | HW04m 0.615 / 0.709, 0.598 / 0.718 | 0.85 |
| HWP034 | support 기준 선택은 1-hop에서 손해가 아니다 | `nbr_h1_a2xsup − nbr_raw` ≥ 0, 두 ds | HW04m hop1_top1_a2x 0.658 / 0.656 vs `nbr_raw` 선택 0.628 / 0.661 | 0.50 |
| HWP035 | Â²X 2-hop 양성이 `nbr_raw`를 넘는다 | `nbr_h2_a2xsup − nbr_raw` > 0, 양수 ≥ 7/10, 두 ds | — | 0.35 |
| HWP036 | raw 2-hop 양성이 `nbr_raw`를 넘는다 | `nbr_h2_rawsup − nbr_raw` > 0, 양수 ≥ 7/10, 두 ds | — | 0.35 |
| HWP037 | 거리 효과가 있다 | `nbr_h2_a2xsup − nbr_h1_a2xsup` > 0, 두 ds | message passing 중복 가설 (SPGCL) | 0.40 |
| HWP038 | 2-hop 양성의 상당수가 이미 pseudo-query다 | 두 2-hop arm 모두 `nbr_in_own_queries` ≥ 0.20, 두 ds | NaQ-Feat query = 특징 최근접 | 0.50 |
| HWP039 | raw 선택이 query와 더 많이 겹친다 | `nbr_in_own_queries`: `nbr_h2_rawsup` > `nbr_h2_a2xsup`, 두 ds | NaQ-Feat query와 같은 raw 특징 공간 | 0.60 |
| HWP040 | 2-hop 양성 중 하나는 NaQ 기준선을 넘는다 | 두 2-hop arm 중 최소 하나가 `− naq_base` > 0, 양수 ≥ 7/10, 두 ds | `nbr_raw − naq_base` +0.3 내외 | 0.55 |

## 6. 범위 밖

- λ 변경, 1-hop raw-support arm, 3-hop, PPR·곡률·차수 조건 선택, validation 크기 변경, 새 HSAN arm, seed 추가, 다른 데이터셋, `results/hw/`·`results/measure/` 수정.

## 7. 보고 (`reports/HW04a_iislab4.md`, CLAUDE.md §7)

핵심 표 (값의 종류를 표마다 명시):
1. arm × ds 10 seed test 정확도 평균 ± 표준편차, 붕괴 수 (`naq_base`, `nbr_raw`, 새 3 arm)
2. 짝지은 Δ: 새 arm − `nbr_raw`, 새 arm − `naq_base`, `nbr_h2_a2xsup − nbr_h1_a2xsup`, `nbr_h2_rawsup − nbr_h2_a2xsup` (Δ ± sd, 양수 k/10)
3. 진단 표: arm × ds의 `nbr_eq_support`, `nbr_eq_query`, `query_eq_support`, `nbr_in_own_queries`, `nbr_in_any_queries`, `nbr_is_query`, `pos_nbr_cos_raw`, `pos_nbr_cos_a2x`, `fallback_frac` (`nbr_raw`는 기존 run 값, 새 키는 없으면 "—")
4. G-match 값(공간 2개 × ds), 회귀 확인 값, `max_neighbors`와 상한 초과 비율, 불변 diff 발췌
5. best epoch 분포 (arm × ds, 10 seed의 best epoch 값 목록)
6. HWP033–040 대조
7. 소요 시간, HEAD (작업 시작 HEAD 포함)
