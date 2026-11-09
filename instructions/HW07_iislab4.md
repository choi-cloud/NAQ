# HW07 [iislab4] — cluster 효과 검증 (무작위 군집 대조 · k-means seed · anchor 군집 · m 100/200) + far 재확인 (m 200, far_strong) + 제외 집합 측정 (auto, 4000 episode, 140 run)

2026-11-09 · 작성: 이희태 (Claude 정리) · 서버 **iislab4** · 브랜치 `hl/hsan-weighting` · 선행 `17dd625`(HW06-c) 이후 · 태그 `hw07` · 사전 등록 HWP070–HWP078 · 선행 문서 `CLAUDE.md` 개정 9, `reports/HW06_iislab4.md`

- auto 모드. 멈추는 조건: §0 허용 외 미커밋 변경, §1.5 G-table 실패, §1.6 G-ident·회귀 확인 실패, G6, 해결 못 한 오류.
- 결과 루트 **`HW_RESULTS_ROOT=results/hw_iislab4`**, `GPUS=0,1,2,3,4,5,6,7`, seed 1212–1221. 모든 학습 arm은 **`valid_tasks: 500`, `episodes: 4000`, `geom_diag: true`**, base `naq_base`. HW06의 `e4k_*` arm과 같은 결과 루트·같은 프로토콜이므로 짝지은 비교에 그대로 쓴다.
- 시간 예산 2시간 30분 (구현 45 / 측정·확인 20 / 실행 ~45 / 집계·보고 30).

## 0. 현황 (수정 없음, 보고서에 적고 계속)

`git fetch && git status -sb | head -1`, `git log --oneline -3`, `nvidia-smi --query-gpu=index,memory.used --format=csv`, `ls cache/kmeans cache/rawcos`. **작업 시작 HEAD 기록.** 허용 미커밋 변경은 미추적 `instructions/HW07_iislab4.md`와 수정된 `CLAUDE.md`(개정 9)뿐. 그 밖에 있으면 **멈추고 보고**. `git pull --rebase`. CLAUDE.md 개정 9를 다시 읽는다.

기록: HW06 `StrategyNegatives`의 cluster·far 구현 위치와, q의 군집·far 제외 집합을 계산하는 함수 이름.

## 1. HW07-a: 구현

기존 `random`/`hard_*`/`far`/`band`/`cluster` 경로의 기본 동작은 한 줄도 바꾸지 않는다. 새 키는 기본값이면 HW06과 같은 결과를 내야 한다. 후보 제외(그 에피소드의 support 5개, pseudo-query 50개, q 자신)는 모든 모드에 공통이고, 한 query 안에서는 중복 없이 뽑는다.

### 1.1 `cluster_ref` (`hw/negatives.py`)
- `query`(기본): HW06 `cluster`와 같다. q의 군집 C(q)를 제외.
- `anchor`: q의 양성 support s⁺의 군집 C(s⁺)를 제외. C(s⁺) = C(q)인 비율을 진단 `anc_same_cluster`로 기록.
- `shuffled`: query마다 C(q) 밖에서 노드 u를 균등하게 하나 뽑아 C(u)를 제외. u를 노드 단위로 뽑으므로 제외 군집의 크기 분포가 "q가 속한 군집"의 크기 분포(노드 가중)와 같아진다. u 추출은 전용 Generator `seed + 61403`. 제외된 군집 크기의 평균을 진단 `excl_cluster_size`로 기록(세 `cluster_ref` 모두).

### 1.2 `kmeans_seed` (`hw/negatives.py`)
- 기본 20261108(= HW06 `cache/kmeans/<ds>_k100.pt`). 다른 값이면 같은 설정(`KMeans(n_clusters=k, n_init=4, random_state=kmeans_seed)`, Â²X L2 정규화)으로 계산해 `cache/kmeans/<ds>_k<k>_s<seed>.pt`에 저장·재사용.

### 1.3 `far_top` (`hw/negatives.py`)
- 기본 100(= HW06 `far`). far의 raw cos 제외 상위 개수. 1000이면 raw cos 상위 1000 표가 필요하다: CPU 청크로 계산해 `cache/rawcos/<ds>_top1000.pt`에 저장(앞 500열이 기존 `_top500`과 같은지 확인, §1.5).

### 1.4 진단 (`hw/run.py`, `nbr_diag`, 레이블 불필요)
모든 `extra_neg` arm에 다음을 추가한다(기준 군집은 k100·seed 20261108, 기준 far 집합은 2-hop ∪ raw cos 상위 100).
- `xneg_in_qcluster`: 추가 음성 중 C(q)에 속한 비율
- `xneg_in_farset`: 추가 음성 중 q의 기준 far 집합에 속한 비율
- `cluster_ref` arm: `anc_same_cluster`, `excl_cluster_size`

### 1.5 G-table (학습 전, `results/hw_iislab4/HW07_tables.json`, 실패 시 **멈추고 보고**)
두 데이터셋에서:
- **seed2 군집**: 크기 최소·중앙값·최대, 같은 군집 내 같은 클래스 비율(측정 전용), 기준 군집(seed 20261108)과의 ARI(`sklearn.metrics.adjusted_rand_score`).
- **far_strong**: 노드마다 (2-hop ∪ raw cos 상위 1000) 제외 후 남는 후보 수의 최소·중앙값. 최소 − 55 ≥ 1000이 아니면 멈춤. `_top1000`의 앞 500열이 `_top500`과 같은 노드 비율(동점 차이는 허용, 다른 노드는 순위 경계 동점인지 확인해 적는다).
- **shuffled**: 노드 가중 C(q) 크기 분포와, `shuffled` 규칙으로 1만 번 뽑은 C(u) 크기 분포의 평균·중앙값·q90. 두 평균의 상대 차이가 10%를 넘으면 멈추고 보고.

### 1.6 G-ident, 회귀, 불변, smoke
- G-ident: seed 1212, Cora_full, 학습 전 모델, 에피소드 3개에서 `extra_neg {cluster, 20, k 100}`에 새 키를 명시적 기본값(`cluster_ref query`, `kmeans_seed 20261108`)으로 준 경우와 HW06 방식(새 키 없음)이 **같은 음성 인덱스**를 뽑고 손실이 같다(|Δ| ≤ 1e-6). `far`도 `far_top 100` 명시 vs 없음으로 같은 확인. 실패 시 멈춤.
- 회귀: `naq_base`, `nbr_raw`(기본 설정) seed 1212 두 ds → `_regress/HW07/`, 기존 값과 |Δ| ≤ 0.30. 실패 시 멈춤.
- 불변: `git diff <작업 시작 HEAD> -- hw/losses.py hw/negatives.py hw/neighbors.py hw/run.py`의 삭제 줄 수와 기존 경로 발췌.
- smoke(episodes 4, `/tmp/hw_smoke`, 커밋 안 함): `e4k_clus_shuf_m20`, `e4k_clus_anc_m20`, `e4k_clus_ks2_m20`, `e4k_farS_m200` Cora_full 각 1회. 새 진단 키 기록 확인.

### 1.7 arm 추가 (`configs/hw.json`; 모두 `valid_tasks 500`, `episodes 4000`, `geom_diag true`, base `naq_base`)

| arm | `extra_neg` |
|---|---|
| `e4k_clus_shuf_m20` | `{cluster, 20, k 100, cluster_ref shuffled}` |
| `e4k_clus_ks2_m20` | `{cluster, 20, k 100, kmeans_seed 20261109}` |
| `e4k_clus_anc_m20` | `{cluster, 20, k 100, cluster_ref anchor}` |
| `e4k_clus_m100` | `{cluster, 100, k 100}` |
| `e4k_clus_m200` | `{cluster, 200, k 100}` |
| `e4k_far_m200` | `{far, 200}` |
| `e4k_farS_m200` | `{far, 200, far_top 1000}` |

7 arm × 2 ds × 10 seed = **140 run**.

커밋 `HW07-a [iislab4]: cluster_ref/kmeans_seed/far_top, xneg_in_qcluster/farset diagnostics, arms, G-table` (CLAUDE.md 개정 9, 지시서, 사전 등록, G-table·G-ident 결과 포함) → pull --rebase → push.

## 2. HW07-b: 실행

- `results/hw_iislab4/HW07_runs.txt`: 140 run, 순서 seed 바깥·arm 안쪽.
- 집계: `python -m hw.summarize --ref e4k_naq --ref e4k_rand_m20 --ref e4k_rand_m100 --ref e4k_rand_m200 --ref e4k_clus_m20`.
- 커밋 `HW07-b [iislab4]: 140 runs (cluster verification, far m200)` → pull --rebase → push.

## 3. HW07-c: 제외 집합 측정 (학습 없음, `hw/measure_positive.py --exclusion`)

레이블은 측정 전용. 기준 군집 = k100·seed 20261108, 기준 far 집합 F(q) = 2-hop ∪ raw cos 상위 100, 강한 far 집합 F_S(q) = 2-hop ∪ raw cos 상위 1000. N = 노드 수.

**(a) 노드 단위 (전체 노드 평균, 노드 macro)**
- |C(q)| − 1, |F(q)|, |F_S(q)|의 평균·중앙값·q90
- 각 집합 안에서 q와 같은 클래스 비율
- 겹침: |F(q) ∩ C(q)| / |C(q)|, |F(q) ∩ C(q)| / |F(q)|
- 무작위 음성 m개 중 각 집합에 떨어지는 **기대 개수** = m · |S(q)| / (N − 56), m ∈ {20, 100, 200}, S ∈ {C(q) \ {q}, F(q), F_S(q)}. 그리고 그중 같은 클래스(= 그 전략이 실제로 제거하는 FN)의 기대 개수.

**(b) 에피소드 단위 (`hw.purity.build_pool`, seed 1212 학습 풀, 2000 에피소드의 (q, s⁺) 쌍 10만 개)**
- C(s⁺) = C(q)인 비율
- C(s⁺) \ C(q)의 크기 평균, 그 안의 q와 같은 클래스 비율
- C(q) 안에 q와 같은 에피소드의 다른 pseudo-class support가 들어 있는 비율

출력 `results/measure/HW07_exclusion_<ds>.{json,md}`. 커밋 `HW07-c [iislab4]: exclusion-set measurement` → pull --rebase → push.

## 4. 게이트

- **G-table**(§1.5), **G-ident**·**회귀**(§1.6): 실패 시 멈춤.
- **G6**: run의 95% 이상 `completed`(1회 재시도 후). 못 미치면 멈추고 보고.
- **G7 (기록)**: arm별 붕괴·NaN 수.
- **태그**: 위 게이트 통과 → `git tag hw07 && git push origin hw07`.

## 5. 사전 등록 (HW07-a 커밋에 포함, 판정은 사람)

Δ = 같은 seed 짝지은 `test_acc` 차이의 10 seed 평균. rand20 = `e4k_rand_m20`, clus20 = `e4k_clus_m20` (모두 HW06).

| ID | 예측 | 기준 | 근거 | 확신도 |
|---|---|---|---|---|
| HWP070 | q와 무관한 군집을 빼면 무작위와 같다 | \|`clus_shuf_m20 − rand20`\| ≤ 0.15, Cora_full | 효과가 q 특이적이라면 | 0.55 |
| HWP071 | q의 군집을 빼는 것이 무관한 군집을 빼는 것보다 낫다 | `clus20 − clus_shuf_m20` > 0, 양수 ≥ 7/10, Cora_full | 위와 같음 | 0.55 |
| HWP072 | 다른 k-means 분할에서도 재현된다 | `clus_ks2_m20 − rand20` ≥ +0.20, Cora_full | 군집 우연이 아니라면 | 0.50 |
| HWP073 | 양성 support의 군집을 빼도 같은 효과다 | `clus_anc_m20 − rand20` ≥ +0.20, Cora_full | q와 s⁺는 특징 최근접이라 같은 군집일 가능성 높음 | 0.50 |
| HWP074 | 큰 m에서도 cluster가 무작위보다 높다 | `clus_m200 − rand_m200` > 0, Cora_full | m이 크면 무작위에 섞이는 같은 군집 노드도 늘어남 | 0.45 |
| HWP075 | 큰 m에서 far가 무작위보다 높다 | `far_m200 − rand_m200` > 0, 두 ds | 무작위 m 200의 기대 FN 약 4개 | 0.40 |
| HWP076 | 제외 범위를 넓히면 더 높다 | `farS_m200 − far_m200` > 0, 두 ds | — | 0.35 |
| HWP077 | Amazon에서는 cluster 변형 모두 무작위와 같다 | 같은 m의 무작위 대비 \|Δ\| ≤ 0.15, cluster 5 arm 모두, Amazon_clothing | HW06 +0.00 | 0.55 |
| HWP078 | 무작위 m 20이 q의 군집에 떨어뜨리는 음성은 query당 0.5개 이하다 | §3(a) 기대 개수, m 20, 두 ds | 군집 크기 중앙값 170–185 / N 2–2.5만 | 0.70 |

## 6. 범위 밖

- band 관련 arm(보류), k ≠ 100, raw 공간 군집, m 20의 far 재실행, 다른 `extra_neg` 모드, 큰 그림 ② 강도 조절, 전역 손실, episodes ≠ 4000, 기존 캐시 수정(새 캐시 추가는 허용), seed 추가, 다른 데이터셋, `results/hw/` 수정.

## 7. 보고 (`reports/HW07_iislab4.md`, CLAUDE.md §7)

값의 종류를 표마다 명시. 주 결과 = `test_acc`(500 기준).
1. 7 arm × ds 평균 ± sd, 붕괴·NaN 수. 비교용으로 HW06 `e4k_naq`, `e4k_rand_m20/m100/m200`, `e4k_clus_m20`, `e4k_far_m20` 행 포함
2. **cluster 검증 표 (m 20)**: {clus20, clus_shuf, clus_ks2, clus_anc} − rand20, − clus20 (Δ ± sd, 양수 k/10) + `xneg_eq_query`, `xneg_in_qcluster`, `excl_cluster_size`, `anc_same_cluster`
3. **큰 m 표**: m ∈ {20, 100, 200}에서 clus − rand, far − rand (같은 m), farS − far (m 200) + `xneg_eq_query`, `xneg_in_qcluster`, `xneg_in_farset`
4. **제외 집합 표** (§3 전부)
5. 기하 표: 7 arm × ds의 `norm_mean`, `uniformity`, `eff_rank`, `corr_offdiag_mean`
6. G-table, G-ident, 회귀, 불변 발췌
7. best epoch 분포 (arm × ds)
8. HWP070–078 대조
9. 소요 시간, HEAD(작업 시작 HEAD 포함)
