# HW04m [iislab4] — 학습 없는 양성 후보 순도 측정 (hop · PPR · 차수 · 곡률)

2026-11-07 · 작성: 이희태 (Claude 정리) · 서버 **iislab4** · 브랜치 `hl/hsan-weighting` · 선행 `hw03a` 태그 및 seio HW03a 커밋 이후 · 태그 `hw04m` · 사전 등록 HWP026–HWP032 · 선행 문서 `CLAUDE.md` 개정 4, `reports/HW03a_iislab4.md`, `reports/HW03a_seio.md`

- auto 모드. 멈추는 조건: §2.4 G-toy 실패, §0 미커밋 변경, 구현 중 해결 못 한 오류.
- **학습 없음.** 모델을 학습하지 않는다. `hw/run.py`, `hw/neighbors.py`, `hw/losses.py`, `hw/weights.py`, `hw/evaluate.py`, `configs/hw.json`은 고치지 않는다.
- 결과 위치: **`results/measure/`** (새 루트). 학습이 없어 결정적인 측정이므로 서버별 결과 루트(CLAUDE.md §4.3)를 쓰지 않는다. `results/hw/`, `results/hw_iislab4/`에는 쓰지 않는다.
- GPU: PPR 계산에만 GPU 1장(`CUDA_VISIBLE_DEVICES=0`). 나머지는 CPU.
- 레이블: 전부 **측정 전용**(CLAUDE.md §5). 해당 코드에 `# measurement-only label use` 주석.
- 시간 예산 1시간 30분 (구현 40 / 확인 10 / 실행 ~15 / 보고 20).

## 0. 현황 (수정 없음, 보고서에 적고 계속)

`git fetch && git status -sb | head -1`(upstream = `origin/hl/hsan-weighting`), `git log --oneline -5`, `ls hw/`, `ls results/hw/purity/ 2>/dev/null`, `nvidia-smi --query-gpu=index,memory.used --format=csv`. 작업 트리에 미커밋 변경이 있으면 **멈추고 보고**. 먼저 `git pull --rebase`.

## 1. 드롭 파일

`~/wkspace/HW_drop/HW04m_iislab4.md` → `instructions/HW04m_iislab4.md`.

## 2. HW04m-a: 구현

### 2.1 새 파일 두 개 (기존 파일은 §2.3의 예외 외에 고치지 않는다)

- `hw/structure.py` — 그래프 구조량 계산 함수 (학습 코드에서 import하지 않는다)
- `hw/measure_positive.py` — 측정 실행 진입점: `python -m hw.measure_positive --dataset <ds> [--seed 1212] [--episodes 2000]`

### 2.2 공통 입력

- 그래프·feature·레이블은 `hw/run.py`와 같은 경로로 읽는다(seed → `load_data`). 구조 측정용 그래프는 **무향, 자기 루프 제거, 중복 edge 제거**로 정규화한 A를 쓴다. 노드 수, 무향 edge 수, 고립 노드 수를 보고서에 적는다.
- 집계 대상 support = **전체 노드**(NaQ는 anchor를 전체 노드에서 균일하게 뽑으므로, 노드 평균이 에피소드 기대값과 같다).
- 집계 방식: **macro**(노드마다 비율을 구하고 노드 평균, 해당 집합이 빈 노드는 제외) 를 주 값으로, **micro**(모든 (support, 후보) 쌍을 한데 모은 비율)를 보조 값으로 둘 다 적는다. 각 행에 `coverage`(집합이 비지 않은 노드 비율)와 집합 크기 중앙값을 붙인다.
- 우연 기준선: `chance = Σ_c p_c²` (p_c = 클래스 c의 노드 비율). 모든 표 맨 위에 적는다.
- 클래스별 개수는 `(후보 지시 희소행렬) @ (one-hot Y)`로 행 단위 계산한다. 밀집 N×N 행렬을 만들지 않는다(행을 청크로 나눠 처리).
- 특징 공간 두 가지 (특징 필터용):
  - `raw`: X를 L2 행 정규화
  - `a2x`: Â²X를 L2 행 정규화. **`hw/trainfree.py`의 Â²X 구성 함수를 import**해서 쓴다(정규화 방식이 HW03a와 같아야 함). 함수로 노출돼 있지 않으면 같은 식을 `hw/structure.py`에 옮기고, 같은 입력에서 두 결과가 같은지(최대 절대 차 < 1e-6) 확인해 보고서에 적는다.

### 2.3 측정 항목

**M1. hop별 순도**
- H1 = N(s), H2 = N²(s) \ (H1 ∪ {s}), H3 = N³(s) \ (H1 ∪ H2 ∪ {s}).
- 행: `hop1`, `hop2`, `hop3` 각각 purity(후보 GT == s의 GT), coverage, 크기 중앙값.
- 특징 필터 행: 각 Hk 안에서 cos(후보, s)로 정렬한 top-1, top-5의 순도. 공간은 `raw`, `a2x` 둘 다. (예: `hop2_top1_raw`, `hop2_top5_a2x`.) Hk 크기가 m보다 작으면 있는 만큼 쓴다.
- 내부 일치 확인: `hop1` micro 순도 == 전체 edge 동질도(양방향 edge 기준). 다르면 **멈추고 보고**.

**M2. PPR 구간별 순도**
- 행 정규화 전이 P = D⁻¹A, π_s = α·e_s + (1−α)·π_s·P. 소스 청크(예: 2048개)를 GPU에서 거듭제곱 반복, 최대 100회 또는 최대 변화 < 1e-6에서 멈춘다. α ∈ {0.15, 0.5}.
- 자기 자신은 제외하고 노드마다 PPR 상위 100개(인덱스·점수)를 저장: `cache/ppr/<ds>_a{015,050}_top100.pt` (git 밖, 이후 실험 재사용용). 고립 노드는 제외하고 수를 적는다.
- 행: 순위 구간 {1–5, 6–10, 11–20, 21–50, 51–100} × α의 순도. 각 구간에 그 후보가 hop 1 / hop 2 / hop ≥3인 비율도 함께 적는다(PPR이 hop과 다른 것을 고르는지 보기 위함).
- 특징 필터 행: PPR 상위 20 안에서 cos(후보, s) top-1, top-5 순도 (`raw`, `a2x`), α = 0.15만.

**M3. support 차수 분위별**
- 차수(정규화된 A 기준) 5분위. 동점 때문에 분위가 고르지 않으면 실제 경계와 bin별 노드 수를 적는다.
- bin마다: `hop1`, `hop2`, `hop2_top1_a2x`, `ppr015_1-10` 순도, coverage.
- **에피소드 조건 행 (기존 풀 재사용)**: `hw/purity.py`가 만드는 학습 에피소드 풀(seed 1212, 2000 에피소드)과 `hw/neighbors.py`의 선택(`selector: raw`, `pick: max`)을 써서, support 차수 bin별로 `query_eq_support`, `nbr_eq_support`, `nbr_eq_query`, `fallback_frac`.
  - `hw/purity.py`에서 풀 생성 부분이 함수로 노출돼 있지 않으면, **동작을 바꾸지 않는 범위에서만** 함수로 분리한다(이 지시서에서 허용하는 유일한 기존 파일 수정). 분리 후 `python -m hw.purity --dataset Cora_full --seed 1212`를 스크래치 경로에 다시 돌려 `results/hw/purity/HW03a_purity_Cora_full.json`의 값과 비교하고, 모든 값의 최대 절대 차를 보고서에 적는다. 차가 0.005를 넘으면 그 사실과 값을 적고 계속한다(멈추지 않는다). `results/hw/purity/`는 덮어쓰지 않는다.

**M4. edge 곡률 bin별**
- edge (u, v)마다 t = |N(u) ∩ N(v)| (삼각형 수), d_u, d_v.
- `forman`: 증강 Forman 곡률 F = 4 − d_u − d_v + 3t.
- `jl`: Ollivier-Ricci 곡률의 Jost–Liu 하한
  `κ_JL = −(1 − 1/d_u − 1/d_v − t/min(d_u,d_v))₊ − (1 − 1/d_u − 1/d_v − t/max(d_u,d_v))₊ + t/max(d_u,d_v)`
  (x₊ = max(x, 0)). 정확한 Ollivier-Ricci(최적수송)는 계산하지 않는다. 패키지 설치 금지.
- edge별 값 저장: `cache/curv/<ds>.pt` (edge_index, t, forman, jl).
- edge 표 (edge 순도 = y_u == y_v):
  - 부호 bin: `jl < 0`, `jl = 0`, `jl > 0` (부동소수 비교는 |x| < 1e-9를 0으로)
  - 5분위 bin: `jl`, `forman` 각각 (동점 처리는 M3와 같이)
  - 대조 bin: `t = 0`, `t ≥ 1` — 곡률이 삼각형 수 이상의 정보를 주는지 보기 위함
  - 대조 2: `t ≥ 1`인 edge만 모아 `jl` 5분위
- 노드 중심 행: support s의 1-hop 중 `jl > 0` edge로 이어진 이웃만의 순도, `jl ≤ 0`만의 순도, 각각 coverage.

### 2.4 G-toy (곡률 식 확인, 실패 시 **멈추고 보고**)

`hw/structure.py`의 곡률 함수에 작은 그래프를 넣어 확인한다(테스트는 `hw/measure_positive.py --selftest`로 실행, 결과를 보고서에 적는다).

| 그래프 · edge | t | d_u, d_v | 기대 `jl` | 기대 `forman` |
|---|---|---|---|---|
| 경로 P5의 가운데 edge | 0 | 2, 2 | 0 | 0 |
| 차수 3–3 트리 edge (두 노드가 각각 잎 2개를 더 가진 트리) | 0 | 3, 3 | −2/3 | −2 |
| 완전그래프 K4의 edge | 2 | 3, 3 | 2/3 | 4 |

±1e-9 안에서 일치해야 한다. 같은 selftest에서 hop 집합(P5에서 가운데 노드의 H1·H2 크기 2·2)과 PPR(행 합 ≈ 1, 자기 점수 ≥ α)도 확인한다.

### 2.5 커밋

`CLAUDE.md`는 두 군데만 고친다: §3 스키마에 `results/measure/` 한 줄과 `hw/structure.py`, `hw/measure_positive.py`; §8 등록부에 `HWP026–032 | HW04m_iislab4 | 배정`과 `HWP033– | 다음 | 미배정`. 다른 절은 건드리지 않는다.

커밋 `HW04m-a [iislab4]: train-free positive-candidate purity (hop/PPR/degree/curvature), selftest` (지시서, 사전 등록, CLAUDE.md 두 줄 포함) → `git pull --rebase` → `git push origin hl/hsan-weighting`.

학습 코드 경로를 바꾸지 않았으므로 CLAUDE.md §6-7의 학습 회귀 확인은 하지 않는다. 대신 `git diff --stat hw03a..HEAD -- hw/run.py hw/neighbors.py hw/losses.py hw/weights.py hw/evaluate.py configs/hw.json`이 비어 있음을 보고서에 적는다.

## 3. HW04m-b: 실행

- `python -m hw.measure_positive --dataset Cora_full`, `--dataset Amazon_clothing` (seed 1212, episodes 2000).
- 출력: `results/measure/HW04m_<ds>.json`, `results/measure/HW04m_<ds>.md`. 노드별 배열(차수, hop1/hop2 순도, 곡률 요약 등)은 `results/measure/HW04m_<ds>_nodes.npz`(10 MB 이하면 커밋, 넘으면 `cache/`로 두고 경로만 적는다).
- `.gitignore`가 `results/measure/`를 막으면 허용 규칙을 추가한다(`git add -f` 금지).
- 커밋 `HW04m-b [iislab4]: measurements for Cora_full, Amazon_clothing` → pull --rebase → push.

## 4. 게이트

- **G-toy** (§2.4): 실패 시 멈춤.
- **G-consist** (§2.3 M1 내부 일치): 실패 시 멈춤.
- **G-tag**: G-toy, G-consist 통과 + 두 데이터셋 출력 생성 → `git tag hw04m && git push origin hw04m`.

## 5. 사전 등록 (HW04m-a 커밋에 포함, 판정은 사람)

순도 = macro 값. "두 ds" = 두 데이터셋 모두.

| ID | 예측 | 기준 | 근거 | 확신도 |
|---|---|---|---|---|
| HWP026 | 2-hop 순도는 1-hop보다 낮다 | `hop1 − hop2` ≥ 10%p, 두 ds | 거리와 동질도의 일반적 감소 | 0.80 |
| HWP027 | 특징 필터가 2-hop 순도를 1-hop 수준으로 회복시킨다 | `hop2_top1_a2x` ≥ `hop1`, 두 ds | Â²X가 학습 arm 전부보다 높았음(HW03a) | 0.50 |
| HWP028 | PPR 상위 10은 1-hop 이상이다 | α=0.15 `ppr_1-5`, `ppr_6-10`의 노드 수 가중 평균 ≥ `hop1`, 두 ds | PPR은 공통 이웃이 많은 노드에 높은 점수 | 0.45 |
| HWP029 | 고차수 support의 1-hop 순도가 저차수보다 낮다 | 5분위 Q5 `hop1` < Q1 `hop1` − 5%p, 두 ds | GRADE: head 노드 이웃 정제 필요 | 0.45 |
| HWP030 | 양의 곡률 edge가 더 순수하다 | `jl > 0` edge 순도 − `jl < 0` edge 순도 ≥ 10%p, 두 ds | 양 곡률 = 커뮤니티 내부 | 0.70 |
| HWP031 | 곡률은 삼각형 수 이상의 정보를 준다 | `t ≥ 1` edge 안에서 `jl` 최상위 분위 − 최하위 분위 ≥ 5%p, 두 ds | — | 0.40 |
| HWP032 | NaQ query 순도는 저차수 support에서 더 낮다 | 에피소드 조건 행 Q1 `query_eq_support` < Q5 − 3%p, 두 ds | 저차수 노드의 feature 이웃이 덜 일관됨 | 0.35 |

## 6. 범위 밖

- 모델 학습, 새 arm, `configs/hw.json` 수정, 학습 코드 수정, 정확한 Ollivier-Ricci 계산, 패키지 설치, 다른 데이터셋, seed 추가, `results/hw/`·`results/hw_iislab4/` 수정, hard negative 측정(HW04 본편).

## 7. 보고 (`reports/HW04m_iislab4.md`, CLAUDE.md §7)

핵심 표 (두 ds 나란히, macro 주 값, micro 괄호):
1. 그래프 요약: 노드·edge·고립 노드 수, edge 동질도, chance
2. M1 hop 표: hop1/2/3 + 특징 필터 행 (순도, coverage, 크기 중앙값)
3. M2 PPR 표: α × 순위 구간 순도와 hop 구성비, 특징 필터 행
4. M3 차수 표: bin 경계·노드 수 × {hop1, hop2, hop2_top1_a2x, ppr015_1-10} + 에피소드 조건 행
5. M4 곡률 표: 부호 bin, jl·forman 5분위, t 대조, t≥1 안의 jl 5분위, 노드 중심 행
6. G-toy selftest 결과, G-consist, purity.py 분리 시 재현 차이
7. HWP026–032 대조
8. 소요 시간, HEAD, `git diff --stat` 확인 결과
