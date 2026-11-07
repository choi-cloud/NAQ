# HW04a [iislab4] — 2-hop 양성: support와 cos 최대 (`sup_cos`, Â²X · raw) + 1-hop 대조

2026-11-07 · 서버 iislab4 · 브랜치 `hl/hsan-weighting` · 작업 시작 HEAD `e4faa1b` · 지시서 `instructions/HW04a_iislab4.md` · 사전 등록 HWP033–HWP040

## 1. 결론

1. test 정확도의 짝지은 Δ(새 arm − `nbr_raw`, 10 seed)는 −0.17 ~ +0.08 범위입니다. 같은 seed에서 양수인 seed는 2–5/10입니다. 새 arm − `naq_base`는 +0.15 ~ +0.35이고 양수 seed는 8–10/10입니다.
2. 학습 중 고른 양성의 노드 클래스 일치 비율(`nbr_eq_support`, Cora_full / Amazon_clothing)은 다음과 같습니다.
   - `nbr_h2_a2xsup` 0.6280 / 0.7228, `nbr_h2_rawsup` 0.6065 / 0.7396
   - `nbr_h1_a2xsup` 0.6566 / 0.6553
3. 고른 양성이 그 support의 pseudo-query 10개 안에 드는 비율(`nbr_in_own_queries`)은 다음과 같습니다.
   - `nbr_h2_rawsup` 0.5829 / 0.7459
   - `nbr_h2_a2xsup` 0.1931 / 0.4506
   - `nbr_h1_a2xsup` 0.3175 / 0.3252
   - 고른 양성이 짝지어진 query 자신인 비율(`nbr_is_query`)은 모든 arm에서 0입니다.

## 2. 핵심 표

### 2.1 test 정확도

- 값: test 정확도(%), 5-way 5-shot, test 500 task. best validation(50 task) 시점의 test 값입니다.
- 셀: 10 seed(1212–1221)의 평균 ± 표준편차(ddof=0). 붕괴는 best validation ≤ 25%인 run의 수입니다.
- 출처: `results/hw_iislab4/summary.md`

| arm | Cora_full | 붕괴 | Amazon_clothing | 붕괴 |
|---|---|---|---|---|
| `naq_base` | 77.85 ± 0.47 | 0 | 90.74 ± 0.21 | 0 |
| `nbr_raw` | 78.12 ± 0.35 | 0 | 91.07 ± 0.23 | 0 |
| `nbr_h1_a2xsup` | 78.20 ± 0.37 | 0 | 91.04 ± 0.16 | 0 |
| `nbr_h2_a2xsup` | 78.05 ± 0.29 | 0 | 90.89 ± 0.23 | 0 |
| `nbr_h2_rawsup` | 78.20 ± 0.48 | 0 | 90.96 ± 0.26 | 0 |

### 2.2 짝지은 Δ

- 값: 같은 seed끼리 짝지은 test 정확도 차이(%p).
- 셀: 평균 ± 표준편차(ddof=0) (양수 seed k/10).
- 출처: `results/hw_iislab4/paired_{nbr_raw,naq_base,nbr_h1_a2xsup,nbr_h2_a2xsup}.md`

| 비교 | Cora_full | Amazon_clothing |
|---|---|---|
| `nbr_h1_a2xsup − nbr_raw` | +0.07 ± 0.37 (3/10) | −0.03 ± 0.12 (3/10) |
| `nbr_h2_a2xsup − nbr_raw` | −0.08 ± 0.14 (3/10) | −0.17 ± 0.22 (2/10) |
| `nbr_h2_rawsup − nbr_raw` | +0.08 ± 0.34 (5/10) | −0.11 ± 0.10 (2/10) |
| `nbr_h1_a2xsup − naq_base` | +0.35 ± 0.26 (9/10) | +0.30 ± 0.17 (10/10) |
| `nbr_h2_a2xsup − naq_base` | +0.20 ± 0.30 (9/10) | +0.15 ± 0.11 (8/10) |
| `nbr_h2_rawsup − naq_base` | +0.35 ± 0.16 (10/10) | +0.22 ± 0.22 (9/10) |
| `nbr_h2_a2xsup − nbr_h1_a2xsup` | −0.15 ± 0.35 (3/10) | −0.14 ± 0.15 (1/10) |
| `nbr_h2_rawsup − nbr_h2_a2xsup` | +0.15 ± 0.36 (6/10) | +0.06 ± 0.18 (5/10) |

### 2.3 학습 중 진단

- 값: 마지막 평가 구간(episode 1921–2000)의 episode 평균을 10 seed에 걸쳐 평균한 값입니다.
- 쌍: 각 pseudo-query와 같은 pseudo-class인 support의 (query, support) 쌍입니다. 에피소드당 50쌍입니다.
- 쌍 비율로 계산하는 열:
  - `nbr_eq_support`: 고른 양성과 support의 GT 클래스가 같은 비율입니다(측정 전용 레이블).
  - `nbr_eq_query`: 고른 양성과 query의 GT 클래스가 같은 비율입니다(측정 전용 레이블).
  - `query_eq_support`: query와 support의 GT 클래스가 같은 비율입니다(측정 전용 레이블).
  - `own_q`(`nbr_in_own_queries`): 고른 양성이 그 support의 pseudo-query 10개 중 하나인 비율입니다.
  - `any_q`(`nbr_in_any_queries`): 고른 양성이 그 에피소드의 pseudo-query 50개 중 하나인 비율입니다.
  - `is_q`(`nbr_is_query`): 고른 양성이 짝지어진 query 자신인 비율입니다.
  - `fallback`: 쓸 수 있는 후보가 없어 support 자신을 쓴 비율입니다.
  - `multi`: 쓸 수 있는 후보가 2개 이상인 비율입니다.
    - hop 1은 유효 1-hop 이웃 수 기준입니다.
    - hop 2는 상위 2개 표 중 쓸 수 있는 항목 수 기준입니다.
- 쌍 평균으로 계산하는 열: `cos_raw`, `cos_a2x`는 query와 고른 양성 사이의 cos 평균입니다. 공간은 각각 L2(X), L2(Â²X)입니다.
- `nbr_raw`의 기존 10 seed run은 `nbr_diag`를 기록하기 전에 실행되어 모든 열이 "—"입니다.

| arm | ds | `nbr_eq_support` | `nbr_eq_query` | `query_eq_support` | own_q | any_q | is_q | cos_raw | cos_a2x | fallback | multi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `nbr_raw` | Cora_full | — | — | — | — | — | — | — | — | — | — |
| `nbr_raw` | Amazon_clothing | — | — | — | — | — | — | — | — | — | — |
| `nbr_h1_a2xsup` | Cora_full | 0.6566 | 0.3179 | 0.3520 | 0.3175 | 0.3192 | 0.0000 | 0.1228 | 0.5150 | 0.0042 | 0.8272 |
| `nbr_h1_a2xsup` | Amazon_clothing | 0.6553 | 0.5310 | 0.6022 | 0.3252 | 0.3262 | 0.0000 | 0.3510 | 0.6948 | 0.0073 | 0.7216 |
| `nbr_h2_a2xsup` | Cora_full | 0.6280 | 0.3083 | 0.3520 | 0.1931 | 0.1956 | 0.0000 | 0.1243 | 0.5216 | 0.0283 | 0.9070 |
| `nbr_h2_a2xsup` | Amazon_clothing | 0.7228 | 0.5599 | 0.6022 | 0.4506 | 0.4513 | 0.0000 | 0.4361 | 0.6951 | 0.0484 | 0.8452 |
| `nbr_h2_rawsup` | Cora_full | 0.6065 | 0.3118 | 0.3520 | 0.5829 | 0.5837 | 0.0000 | 0.1664 | 0.5096 | 0.0283 | 0.8424 |
| `nbr_h2_rawsup` | Amazon_clothing | 0.7396 | 0.5762 | 0.6022 | 0.7459 | 0.7460 | 0.0000 | 0.5210 | 0.6907 | 0.0484 | 0.7905 |

### 2.4 G-match, 회귀 확인, `max_neighbors`, 불변 diff

**G-match (§2.4)**
- 값: 노드 macro 순도입니다.
  - 각 노드의 hop-2 상위 2개 표에서 1위 노드의 GT 클래스가 그 노드와 같으면 1, 다르면 0으로 둡니다.
  - 이 값을 hop-2 집합이 비어 있지 않은 노드에 걸쳐 평균합니다. query 제외는 없습니다.
  - 표는 학습 경로 함수 `hw.neighbors.build_sup_cos`로 만들었습니다.
- coverage: hop-2 집합이 비어 있지 않은 노드의 비율입니다.
- 출처: `results/hw_iislab4/HW04a_gmatch.json`. 스크립트는 리포 밖 스크래치에 있습니다.

| 공간 | ds | HW04a | HW04m | Δ | coverage HW04a / HW04m |
|---|---|---|---|---|---|
| a2x | Cora_full | 0.6149 | 0.6149 | −0.0000 | 0.9694 / 0.9694 |
| raw | Cora_full | 0.5994 | 0.5982 | +0.0012 | 0.9694 / 0.9694 |
| a2x | Amazon_clothing | 0.7088 | 0.7091 | −0.0003 | 0.9576 / 0.9576 |
| raw | Amazon_clothing | 0.7195 | 0.7183 | +0.0012 | 0.9576 / 0.9576 |

- 판정: |Δ| ≤ 0.002를 4칸 모두 만족합니다. 통과입니다.
- hop-2 후보가 2개 이상인 노드 비율: Cora_full 0.9409, Amazon_clothing 0.9307.
- hop-2 집합 크기 중앙값: 38 / 47.
- 표 계산 시간은 run당 3.5–5.8 s입니다.

**회귀 확인 (§2.5)**
- 값: test 정확도(%), seed 1212입니다.
- 결과 위치: `results/hw_iislab4/_regress/HW04a/`

| arm | ds | 기존 | HW04a | Δ |
|---|---|---|---|---|
| `naq_base` | Cora_full | 77.96 | 77.80 | −0.16 |
| `naq_base` | Amazon_clothing | 90.60 | 90.64 | +0.04 |
| `nbr_raw` | Cora_full | 78.15 | 78.08 | −0.07 |
| `nbr_raw` | Amazon_clothing | 90.89 | 90.88 | −0.01 |

- 판정: 최대 |Δ|가 0.16으로 기준 0.30 이하입니다. 통과입니다.

**`max_neighbors`**
- `nbr_raw`의 `max_neighbors`는 0(상한 없음)이고 새 3 arm도 같습니다.
- 상한을 적용하지 않으므로 1-hop 이웃 수가 상한을 넘는 support 비율은 0입니다.
- 1-hop 이웃 수: Cora_full 중앙값 4, 최대 297. Amazon_clothing 중앙값 3, 최대 251. 고립 노드는 0입니다.

**불변 diff (`git diff e4faa1b -- hw/neighbors.py`)**
- 삭제된 줄은 0개입니다. `select_neighbors_pick` 안에 추가한 줄은 아래 2줄뿐입니다.
- 나머지 변경은 파일 끝에 덧붙인 새 함수입니다: `SupCosCandidates`, `sup_cos_space`, `hop2_top2_table`, `build_sup_cos`, `select_neighbors_sup_cos`.

```diff
@@ -92,6 +92,8 @@ def select_neighbors_pick(...)
     """
+    if pick == SUP_COS:  # HW04a: separate path; the max/contrast/random lines below are unchanged
+        return select_neighbors_sup_cos(neighbor_lookup, support_ids, query_ids, selection_out)
     if pick not in PICKS:
         raise ValueError(f"pick must be one of {PICKS}, got {pick!r}")
     if pick == "max" and selection_out is None:
```

### 2.5 best epoch 분포

- 값: best validation epoch입니다. 셀은 seed 1212 → 1221 순서입니다.

| arm | Cora_full | Amazon_clothing |
|---|---|---|
| `naq_base` | 1760,1920,1600,2000,2000,2000,2000,2000,1760,1920 | 1600,2000,2000,1920,1600,1440,1920,1760,2000,1920 |
| `nbr_raw` | 2000,1920,1440,1920,2000,2000,1600,1920,2000,1920 | 1280,1760,2000,2000,960,2000,1920,1600,1920,1920 |
| `nbr_h1_a2xsup` | 2000,1920,1760,2000,2000,2000,2000,2000,2000,1920 | 1440,1760,2000,1920,1440,2000,1920,1760,2000,1920 |
| `nbr_h2_a2xsup` | 2000,1920,1600,1920,2000,2000,1600,2000,1920,1920 | 1600,1920,2000,1920,1600,1440,1920,1600,2000,1920 |
| `nbr_h2_rawsup` | 2000,1600,1600,1920,2000,2000,2000,1760,1920,1920 | 1760,2000,2000,1440,1440,2000,1920,1600,1920,2000 |

## 3. 게이트·사전 등록 대조

| 게이트 | 값 | 판정 |
|---|---|---|
| G-match | 최대 \|Δ\| 0.0012 | 통과 |
| 회귀 확인 | 최대 \|Δ\| 0.16 | 통과 |
| G6 | 60/60 completed, 재시도 0 | 통과 |
| G7 (기록) | 새 3 arm 붕괴 0 | — |

태그 `hw04a`를 만들었습니다.

| ID | 값의 종류 | 기준 | 값 (Cora_full / Amazon_clothing) | 기준 충족 |
|---|---|---|---|---|
| HWP033 | 쌍 비율 `nbr_eq_support` | h2_a2x ≥ 0.60 / 0.69, h2_raw ≥ 0.58 / 0.70 | h2_a2x 0.6280 / 0.7228, h2_raw 0.6065 / 0.7396 | 충족 |
| HWP034 | 짝지은 Δ | `nbr_h1_a2xsup − nbr_raw` ≥ 0, 두 ds | +0.07 / −0.03 | 미충족 |
| HWP035 | 짝지은 Δ | `nbr_h2_a2xsup − nbr_raw` > 0, 양수 ≥ 7/10, 두 ds | −0.08 (3/10) / −0.17 (2/10) | 미충족 |
| HWP036 | 짝지은 Δ | `nbr_h2_rawsup − nbr_raw` > 0, 양수 ≥ 7/10, 두 ds | +0.08 (5/10) / −0.11 (2/10) | 미충족 |
| HWP037 | 짝지은 Δ | `nbr_h2_a2xsup − nbr_h1_a2xsup` > 0, 두 ds | −0.15 / −0.14 | 미충족 |
| HWP038 | 쌍 비율 `nbr_in_own_queries` | 두 2-hop arm 모두 ≥ 0.20, 두 ds | h2_a2x 0.1931 / 0.4506, h2_raw 0.5829 / 0.7459 | 미충족 (h2_a2x Cora_full 0.1931) |
| HWP039 | 쌍 비율 `nbr_in_own_queries` | h2_raw > h2_a2x, 두 ds | 0.5829 > 0.1931 / 0.7459 > 0.4506 | 충족 |
| HWP040 | 짝지은 Δ | 2-hop arm 하나 이상이 `− naq_base` > 0, 양수 ≥ 7/10, 두 ds | h2_raw +0.35 (10/10) / +0.22 (9/10) | 충족 |

판정은 사람이 합니다. 위 표의 "기준 충족" 열은 기준식을 적용한 결과입니다.

## 4. 수행 내용

**`78a267f` HW04a-a**
- `hw/neighbors.py`: `pick: sup_cos`를 추가했습니다(키 `pick_hop ∈ {1, 2}`, `pick_space ∈ {a2x, raw}`).
  - hop 1: 후보는 `_eligible`이 주는 유효 1-hop 이웃이고 `nbr_raw`와 같은 집합입니다. argmax cos(후보, support)로 고릅니다.
  - hop 2: `hw.structure.normalize_graph`와 `hop_sets_dense`로 최단거리 2인 노드 집합을 구합니다. 노드마다 cos 상위 2개 표를 학습 전에 GPU에서 계산합니다.
    - 선택 시 1위가 query 자신이면 2위를 씁니다.
    - 1위와 2위를 모두 쓸 수 없으면 support 자신을 씁니다.
  - a2x 공간은 `hw.structure.a2x_trainfree`(Â = `data.adj`)를 쓰고, a2x와 raw 공간 모두 L2 행 정규화합니다.
- `hw/run.py`:
  - 표 계산을 ProtoNet 생성 전("선택 표" 자리)에 둡니다.
  - 새 진단 4종(`nbr_in_own_queries`, `nbr_in_any_queries`, `nbr_is_query`, `pos_nbr_cos_a2x`)을 기록합니다.
  - `metrics.json`에 `sup_cos_table`(표 계산 시간, hop-2 coverage)을 남깁니다.
- `hw/summarize.py`: `summary.csv`에 `nbr_in_own_queries`, `nbr_in_any_queries` 열을 추가했습니다.
- `configs/hw.json`: 3 arm을 추가했습니다. 모두 `base: nbr_raw`입니다.
- G-match 결과(`HW04a_gmatch.json`), 회귀 run(`_regress/HW04a/`), CLAUDE.md 개정 5, 지시서를 함께 커밋했습니다.
- smoke(epochs 4, `/tmp/hw_smoke`, 커밋 안 함): 새 3 arm Cora_full 각 1회 completed. 새 진단 키가 기록되었고 `nbr_is_query`는 0입니다.

**`c30a9c1` HW04a-b**
- 60 run과 `summarize` 결과(`--ref nbr_raw --ref naq_base --ref nbr_h1_a2xsup --ref nbr_h2_a2xsup`)를 커밋했습니다.

**이 커밋 (HW04a-c)**
- 이 보고서입니다.

## 5. 판단한 사항

1. **후보 전달 방식**
   - `sup_cos` 후보(`SupCosCandidates`)는 `select_neighbors_pick`의 `neighbor_lookup` 인자 자리로 넘겼습니다.
   - `hw/losses.py`는 이 인자를 그대로 넘기기만 하므로 고치지 않았습니다.
2. **진단 공간 계산 범위**
   - `pos_nbr_cos_a2x`를 위한 L2(Â²X)는 `loss != original`인 모든 arm에서 계산합니다. 회귀 run의 `nbr_raw`도 포함됩니다.
   - 이 계산은 RNG를 소비하지 않습니다.
3. **회귀 결과 위치**
   - 회귀 run은 `_regress/HW04a/`에 썼습니다. HW03a가 커밋한 `_regress/` 파일을 덮어쓰지 않기 위해서입니다.
   - 비교 대상인 "기존 값"은 `results/hw_iislab4/{naq_base,nbr_raw}/<ds>/seed1212`입니다.
4. **hop 2의 `multi` 정의**
   - hop 2의 `valid_count`는 hop-2 집합 크기가 아니라 상위 2개 표 중 쓸 수 있는 항목 수(0–2)로 정했습니다.
   - 따라서 hop 2의 `multi`는 "1위와 2위가 모두 쓸 수 있는 비율"입니다.
   - `mean_support_neighbors_considered`에는 hop-2 집합 크기를 기록합니다.
5. **표준편차 기준**
   - 2.1·2.2의 표준편차는 `hw/summarize.py`와 같은 ddof=0입니다.
6. **`sup_cos`의 `max_neighbors` 제한**
   - `sup_cos`는 `max_neighbors == 0`일 때만 허용합니다. 아니면 오류가 납니다.

## 6. 제외·실패·붕괴 run

- 없습니다. 60/60 completed, 재시도 0, 붕괴 0입니다.

## 7. 소요 시간, 환경, HEAD

- 실행 큐: 60 run, 8 worker(GPUS=0–7), 15:47:59–15:58:02(10.1분).
- run당 시간:
  - Cora_full 58–64 s
  - Amazon_clothing 71–80 s
- 환경: iislab4, RTX 3090 × 8, conda `taskns`(python 3.9.25, torch 2.4.1+cu121).
- 작업 시작 HEAD: `e4faa1b`. 커밋: `78a267f`(a), `c30a9c1`(b), 보고서(c).
