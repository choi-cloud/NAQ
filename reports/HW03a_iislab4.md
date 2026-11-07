# HW03a [iislab4] — cos HSAN τ 격자 + hard positive 선택 + 자동 2단계 + BGRL 단계

2026-11-07 · 서버 iislab4 (RTX 3090 × 8) · 브랜치 `hl/hsan-weighting` · 태그 `hw03a` (= `c4a8844`) · 사전 등록 HWP014–HWP023 · 결과 루트 `results/hw_iislab4/` · seed 1212–1221 (10개)

## 1. 결론

1. τ ∈ {0.05, 0.10, 0.20}에서 `cos_t*_*` 9개 arm과 2단계·d단계 cos arm이 모두 `nbr_raw`보다 낮다. 가장 작은 차이는 `cos_t020_none` −4.96 / −3.03, C\* = `cos_t020_pn` −6.21 / −2.96 (Amazon_clothing / Cora_full, 양수 0/10).
2. `nbr_raw_contrast − nbr_raw` +0.02 ± 0.22 (7/10) / +0.05 ± 0.33 (5/10), `nbr_raw_random − nbr_raw` +0.08 ± 0.16 (7/10) / +0.13 ± 0.35 (5/10), `nbr_bgrl_l01 − nbr_raw` −0.01 ± 0.17 (6/10) / −0.00 ± 0.22 (5/10). `nbr_bgrl`(λ 1.0) − `nbr_raw` −0.02 (6/10) / +0.76 (10/10).
3. 회귀 |Δ| ≤ 0.04, G6 b·c·d 100% (240/240, 40/40, 100/100), 붕괴 0. 사전 등록 충족 HWP017·HWP023, 미충족 HWP014·015·016·018·019·020·021·022.

## 2. 핵심 표

### 2.1 arm × ds, 10 seed 평균 ± 표준편차(ddof=0), 붕괴 수 c

test accuracy (%). `naq_base`·`nbr_raw`의 seed 1212–1216은 HW03-prep run(`1319e7d`), 1217–1221은 이번 1단계 run.

| arm | 단계 | Amazon_clothing | Cora_full |
|---|---|---|---|
| naq_base | 1 | 90.74 ± 0.21 (n=10, c=0) | 77.85 ± 0.47 (n=10, c=0) |
| nbr_raw | 1 | 91.07 ± 0.23 (n=10, c=0) | 78.12 ± 0.35 (n=10, c=0) |
| nbr_raw_contrast | 1 | 91.09 ± 0.19 (n=10, c=0) | 78.17 ± 0.48 (n=10, c=0) |
| nbr_raw_random | 1 | 91.14 ± 0.23 (n=10, c=0) | 78.25 ± 0.50 (n=10, c=0) |
| cos_t005_none | 1 | 81.18 ± 0.46 (n=10, c=0) | 72.18 ± 0.59 (n=10, c=0) |
| cos_t005_p | 1 | 71.09 ± 1.33 (n=10, c=0) | 62.26 ± 1.52 (n=10, c=0) |
| cos_t005_pn | 1 | 76.90 ± 0.53 (n=10, c=0) | 69.57 ± 0.61 (n=10, c=0) |
| cos_t010_none | 1 | 84.52 ± 0.34 (n=10, c=0) | 74.14 ± 0.62 (n=10, c=0) |
| cos_t010_p | 1 | 75.01 ± 0.76 (n=10, c=0) | 66.29 ± 0.75 (n=10, c=0) |
| cos_t010_pn | 1 | 82.38 ± 0.73 (n=10, c=0) | 74.14 ± 0.58 (n=10, c=0) |
| cos_t020_none | 1 | 86.11 ± 0.38 (n=10, c=0) | 75.09 ± 0.51 (n=10, c=0) |
| cos_t020_p | 1 | 78.26 ± 1.40 (n=10, c=0) | 69.51 ± 0.76 (n=10, c=0) |
| cos_t020_pn (C\*) | 1 | 84.86 ± 0.49 (n=10, c=0) | 75.16 ± 0.40 (n=10, c=0) |
| cos_t020_pn_contrast | 2 | 84.88 ± 0.47 (n=10, c=0) | 74.96 ± 0.48 (n=10, c=0) |
| cos_t020_none_contrast | 2 | 86.16 ± 0.50 (n=10, c=0) | 74.91 ± 0.54 (n=10, c=0) |
| nbr_bgrl | d | 91.04 ± 0.29 (n=10, c=0) | 78.88 ± 0.56 (n=10, c=0) |
| nbr_bgrl_l01 | d | 91.05 ± 0.30 (n=10, c=0) | 78.12 ± 0.44 (n=10, c=0) |
| nbr_bgrl_l01_contrast | d | 91.14 ± 0.24 (n=10, c=0) | 78.16 ± 0.40 (n=10, c=0) |
| cos_t020_pn_bgrl | d | 84.80 ± 0.60 (n=10, c=0) | 75.05 ± 0.72 (n=10, c=0) |
| cos_t020_pn_contrast_bgrl | d | 84.87 ± 0.49 (n=10, c=0) | 75.40 ± 0.43 (n=10, c=0) |

### 2.2 `paired_nbr_raw` (Δ = arm − nbr_raw, %p, 평균 ± 표준편차 (양수 k/10))

`results/hw_iislab4/paired_nbr_raw.md`에서 복사. seed별 Δ는 같은 파일.

| arm | Amazon_clothing | Cora_full | ds 평균의 평균 | 양수 합 |
|---|---|---|---|---|
| nbr_bgrl | −0.02 ± 0.24 (6/10) | +0.76 ± 0.44 (10/10) | +0.37 | 16 |
| nbr_raw_random | +0.08 ± 0.16 (7/10) | +0.13 ± 0.35 (5/10) | +0.10 | 12 |
| nbr_bgrl_l01_contrast | +0.07 ± 0.22 (7/10) | +0.04 ± 0.25 (8/10) | +0.06 | 15 |
| nbr_raw_contrast | +0.02 ± 0.22 (7/10) | +0.05 ± 0.33 (5/10) | +0.03 | 12 |
| nbr_bgrl_l01 | −0.01 ± 0.17 (6/10) | −0.00 ± 0.22 (5/10) | −0.01 | 11 |
| naq_base | −0.33 ± 0.22 (1/10) | −0.27 ± 0.33 (1/10) | −0.30 | 2 |
| cos_t020_none | −4.96 ± 0.46 (0/10) | −3.03 ± 0.64 (0/10) | −4.00 | 0 |
| cos_t020_none_contrast | −4.91 ± 0.57 (0/10) | −3.21 ± 0.56 (0/10) | −4.06 | 0 |
| cos_t020_pn_contrast_bgrl | −6.20 ± 0.61 (0/10) | −2.72 ± 0.26 (0/10) | −4.46 | 0 |
| cos_t020_pn | −6.21 ± 0.56 (0/10) | −2.96 ± 0.30 (0/10) | −4.59 | 0 |
| cos_t020_pn_bgrl | −6.26 ± 0.68 (0/10) | −3.07 ± 0.62 (0/10) | −4.67 | 0 |
| cos_t020_pn_contrast | −6.19 ± 0.49 (0/10) | −3.16 ± 0.49 (0/10) | −4.67 | 0 |
| cos_t010_none | −6.55 ± 0.39 (0/10) | −3.98 ± 0.58 (0/10) | −5.27 | 0 |
| cos_t010_pn | −8.68 ± 0.81 (0/10) | −3.99 ± 0.48 (0/10) | −6.34 | 0 |
| cos_t005_none | −9.89 ± 0.52 (0/10) | −5.94 ± 0.68 (0/10) | −7.91 | 0 |
| cos_t020_p | −12.80 ± 1.45 (0/10) | −8.61 ± 0.71 (0/10) | −10.71 | 0 |
| cos_t005_pn | −14.17 ± 0.45 (0/10) | −8.55 ± 0.67 (0/10) | −11.36 | 0 |
| cos_t010_p | −16.05 ± 0.63 (0/10) | −11.83 ± 0.90 (0/10) | −13.94 | 0 |
| cos_t005_p | −19.98 ± 1.29 (0/10) | −15.86 ± 1.52 (0/10) | −17.92 | 0 |

### 2.3 τ별 짝지은 Δ (%p, 평균 ± 표준편차 (양수 k/10))

| τ | ds | none − nbr_raw | p − none | pn − none | pn − p |
|---|---|---|---|---|---|
| 0.05 | Amazon_clothing | −9.89 ± 0.52 (0/10) | −10.09 ± 1.34 (0/10) | −4.28 ± 0.45 (0/10) | +5.81 ± 1.24 (10/10) |
| 0.05 | Cora_full | −5.94 ± 0.68 (0/10) | −9.92 ± 1.26 (0/10) | −2.61 ± 0.81 (0/10) | +7.31 ± 1.49 (10/10) |
| 0.10 | Amazon_clothing | −6.55 ± 0.39 (0/10) | −9.50 ± 0.94 (0/10) | −2.13 ± 0.78 (0/10) | +7.37 ± 0.99 (10/10) |
| 0.10 | Cora_full | −3.98 ± 0.58 (0/10) | −7.85 ± 0.66 (0/10) | −0.01 ± 0.66 (6/10) | +7.85 ± 0.81 (10/10) |
| 0.20 | Amazon_clothing | −4.96 ± 0.46 (0/10) | −7.84 ± 1.17 (0/10) | −1.25 ± 0.49 (0/10) | +6.60 ± 1.22 (10/10) |
| 0.20 | Cora_full | −3.03 ± 0.64 (0/10) | −5.58 ± 0.98 (0/10) | +0.07 ± 0.61 (5/10) | +5.64 ± 0.62 (10/10) |

참고(같은 결과 루트 안): `cos_t020_pn_contrast − cos_t020_pn` +0.02 ± 0.18 (5/10) / −0.19 ± 0.38 (2/10); `cos_t020_none_contrast − cos_t020_none` +0.05 ± 0.24 (5/10) / −0.18 ± 0.40 (3/10).

### 2.4 이웃 진단 (`epochs[-1].nbr_diag`, 마지막 구간 1841–2000 에피소드, seed 평균)

양성 = query × 자기 pseudo-class support. `query_eq_support`는 arm과 무관(같은 seed면 같은 에피소드).

| arm | ds | n | nbr_eq_support | nbr_eq_query | query_eq_support | pos_nbr_cos_raw | fallback_frac | multi_nbr_frac |
|---|---|---|---|---|---|---|---|---|
| naq_base | Amazon_clothing | 5 | — | — | 0.5956 | — | — | — |
| naq_base | Cora_full | 5 | — | — | 0.3557 | — | — | — |
| nbr_raw | Amazon_clothing | 5 | 0.6510 | 0.5522 | 0.5956 | 0.4467 | 0.0078 | 0.7217 |
| nbr_raw | Cora_full | 5 | 0.6251 | 0.3422 | 0.3557 | 0.1702 | 0.0042 | 0.8210 |
| max·raw 선택 (cos_t\*_\* 9 arm 공통) | Amazon_clothing | 10 | 0.6629 | 0.5577 | 0.6022 | 0.4455 | 0.0073 | 0.7216 |
| max·raw 선택 (cos_t\*_\* 9 arm 공통) | Cora_full | 10 | 0.6257 | 0.3400 | 0.3520 | 0.1732 | 0.0042 | 0.8272 |
| nbr_raw_contrast (= cos_t020_\*_contrast) | Amazon_clothing | 10 | 0.6370 | 0.5003 | 0.6022 | 0.2988 | 0.0073 | 0.7216 |
| nbr_raw_contrast (= cos_t020_\*_contrast) | Cora_full | 10 | 0.6305 | 0.2934 | 0.3520 | 0.1061 | 0.0042 | 0.8272 |
| nbr_raw_random | Amazon_clothing | 10 | 0.6233 | 0.5099 | 0.6022 | 0.3116 | 0.0073 | 0.7216 |
| nbr_raw_random | Cora_full | 10 | 0.5878 | 0.2912 | 0.3520 | 0.1095 | 0.0042 | 0.8272 |
| max·bgrl 선택 (nbr_bgrl, nbr_bgrl_l01, cos_t020_pn_bgrl 공통) | Amazon_clothing | 10 | 0.6399 | 0.5472 | 0.6022 | 0.3606 | 0.0073 | 0.7216 |
| max·bgrl 선택 (nbr_bgrl, nbr_bgrl_l01, cos_t020_pn_bgrl 공통) | Cora_full | 10 | 0.5963 | 0.3500 | 0.3520 | 0.1263 | 0.0042 | 0.8272 |
| contrast·bgrl 선택 (nbr_bgrl_l01_contrast, cos_t020_pn_contrast_bgrl 공통) | Amazon_clothing | 10 | 0.6006 | 0.4735 | 0.6022 | 0.2751 | 0.0073 | 0.7216 |
| contrast·bgrl 선택 (nbr_bgrl_l01_contrast, cos_t020_pn_contrast_bgrl 공통) | Cora_full | 10 | 0.5494 | 0.2411 | 0.3520 | 0.0978 | 0.0042 | 0.8272 |

- 선택 공간(raw 또는 BGRL)이 고정이고 같은 seed는 같은 에피소드를 쓰므로, 같은 (selector, pick, seed)인 arm들의 진단값은 손실·τ·HSAN 모드와 무관하게 같다. 확인: `nbr_raw` seed 1217 = `cos_t020_none` seed 1217, `nbr_raw` seed 1219 = `cos_t005_pn` seed 1219 (두 ds, 값 전부 일치). 그래서 공통 행으로 묶었다.
- `nbr_raw`·`naq_base`는 seed 1212–1216이 진단 추가 전 코드(HW03-prep) run이라 n=5(1217–1221)다.
- 같은 seed 1217–1221에서 `nbr_raw` 대 `nbr_raw_contrast`: `nbr_eq_support` 0.6510 → 0.6254 (Δ −2.55%p) / 0.6251 → 0.6262 (Δ +0.11%p); `pos_nbr_cos_raw` 0.4467 → 0.2994 / 0.1702 → 0.1043; `nbr_eq_query` 0.5522 → 0.4926 / 0.3422 → 0.2933 (Amazon_clothing / Cora_full).

### 2.5 C\* 선택 근거 (`paired_nbr_raw.md`, 1단계 종료 시점)

| arm − nbr_raw | Amazon_clothing | Cora_full | ds 평균의 평균 | 양수 합 | 순위 |
|---|---|---|---|---|---|
| cos_t020_pn | −6.21 ± 0.56 (0/10) | −2.96 ± 0.30 (0/10) | −4.59 | 0 | 1 |
| cos_t010_pn | −8.68 ± 0.81 (0/10) | −3.99 ± 0.48 (0/10) | −6.33 | 0 | 2 |
| cos_t020_p | −12.80 ± 1.45 (0/10) | −8.61 ± 0.71 (0/10) | −10.70 | 0 | 3 |
| cos_t005_pn | −14.17 ± 0.45 (0/10) | −8.55 ± 0.67 (0/10) | −11.36 | 0 | 4 |
| cos_t010_p | −16.05 ± 0.63 (0/10) | −11.83 ± 0.90 (0/10) | −13.94 | 0 | 5 |
| cos_t005_p | −19.98 ± 1.29 (0/10) | −15.86 ± 1.52 (0/10) | −17.92 | 0 | 6 |

C\* = `cos_t020_pn`, τ\* = 0.20. 동률 없음. (`paired_nbr_raw.md`의 "mean of ds means" 열은 반올림 차로 −6.34·−10.71로 적힌 행이 있다.) 2단계 arm: `cos_t020_pn_contrast`, `cos_t020_none_contrast`.

### 2.6 d 단계 (BGRL)

- 조건 확인 (12:24, §4 시작 시점에 파일 이미 있음, 대기 0분):

| 파일 | md5 | (a) `dataset` | (b) `embeddings.shape` | (c) `node_ids == arange(N)` | 기타 키 |
|---|---|---|---|---|---|
| `cache/bgrl/Cora_full.pt` | `c2ef0115b8f0e7732f206655f6eeee8b` | Cora_full ✓ | (19793, 256) float32 ✓ | ✓ | `embedding_type`, `epoch` |
| `cache/bgrl/Amazon_clothing.pt` | `bb1bd6d8fe8021dc4f2e93f269e53760` | Amazon_clothing ✓ | (24919, 256) float32 ✓ | ✓ | `embedding_type`, `epoch` |

  md5는 `reports/HW03a_seio.md`(92ac844)의 값과 같다. 파일 mtime 11:45.
- `paired_nbr_bgrl_l01` (`results/hw_iislab4/paired_nbr_bgrl_l01.md`에서 발췌):

| arm − nbr_bgrl_l01 | Amazon_clothing | Cora_full | ds 평균의 평균 | 양수 합 |
|---|---|---|---|---|
| nbr_bgrl | −0.01 ± 0.26 (6/10) | +0.76 ± 0.46 (10/10) | +0.38 | 16 |
| nbr_bgrl_l01_contrast | +0.09 ± 0.24 (8/10) | +0.04 ± 0.20 (5/10) | +0.06 | 13 |
| cos_t020_pn_bgrl | −6.25 ± 0.70 (0/10) | −3.07 ± 0.72 (0/10) | −4.66 | 0 |
| cos_t020_pn_contrast_bgrl | −6.18 ± 0.65 (0/10) | −2.72 ± 0.31 (0/10) | −4.45 | 0 |

  참고: `cos_t020_pn_bgrl − cos_t020_pn` −0.06 ± 0.39 (4/10) / −0.11 ± 0.57 (6/10).

## 3. 게이트·사전 등록 대조표

| 게이트 | 기준 | 값 | 판정 |
|---|---|---|---|
| 회귀 확인 (§1.6) | `naq_base`·`nbr_raw` seed 1212, |Δ| ≤ 0.30 | Cora −0.04 / −0.03, Amazon +0.03 / −0.02 (`naq_base` / `nbr_raw`) | 통과 |
| smoke | 진단 키 기록 | `nbr_raw_contrast`, `nbr_raw_random`, `cos_t005_pn`(+`naq_base`) epochs 4, 6키 기록, `naq_base`는 `query_eq_support`만 | 통과 |
| G6 b | ≥ 95% completed | 240/240 (재시도 0) | 통과 |
| G6 c | ≥ 95% | 40/40 (재시도 0) | 통과 |
| G6 d | ≥ 95% | 100/100 (재시도 0) | 통과 |
| G7 | 붕괴 수 (기록) | 전 arm 0 (`summary.csv` `collapsed=1` 0행 / 400행) | 기록 |
| G8 | 회귀 + G6 b·c | 통과 | 태그 `hw03a` = `c4a8844` push |

| ID | 기준 | 값 (Amazon_clothing / Cora_full) | 판정 |
|---|---|---|---|
| HWP014 | 어떤 τ에서 `cos_t*_none − nbr_raw` ≥ −0.30, 두 ds | 최대 τ=0.20: −4.96 / −3.03 | 미충족 |
| HWP015 | 어떤 τ에서 `cos_t*_{p,pn} − cos_t*_none` > 0, 양수 ≥ 7/10, 두 ds | 최대 τ=0.20 pn: −1.25 (0/10) / +0.07 (5/10) | 미충족 |
| HWP016 | `C* − nbr_raw` ≥ +0.30, 양수 ≥ 7/10, 두 ds | −6.21 (0/10) / −2.96 (0/10) | 미충족 |
| HWP017 | 2개 이상의 τ에서 `pn − p` > 0, 두 ds | τ 0.05: +5.81 / +7.31; 0.10: +7.37 / +7.85; 0.20: +6.60 / +5.64 (모두 10/10) | 충족 (3/3) |
| HWP018 | contrast의 `pos_nbr_cos_raw` < nbr_raw, 그리고 `nbr_eq_support` 5%p 이상 낮음, 두 ds | `pos_nbr_cos_raw` 0.2994 < 0.4467 / 0.1043 < 0.1702; `nbr_eq_support` Δ −2.55%p / +0.11%p (seed 1217–1221) | 미충족 (`nbr_eq_support` 조건) |
| HWP019 | `nbr_raw_contrast − nbr_raw` > 0, 양수 ≥ 6/10, 두 ds | +0.02 (7/10) / +0.05 (5/10) | 미충족 (Cora 5/10) |
| HWP020 | `nbr_raw_random − nbr_raw` < 0, 두 ds | +0.08 / +0.13 | 미충족 |
| HWP021 | `C*_contrast − C*` > 0, 두 ds | +0.02 / −0.19 | 미충족 (Cora) |
| HWP022 | `nbr_bgrl_l01 − nbr_raw` > 0, 두 ds | −0.01 / −0.00 | 미충족 |
| HWP023 | sign(`C* − nbr_raw`) = sign(`C*_bgrl − nbr_bgrl_l01`), 두 ds | −6.21 vs −6.25 / −2.96 vs −3.07 | 충족 |

## 4. 수행 내용

- §0 현황: `git status -sb` → `## hl/hsan-weighting...origin/hl/hsan-weighting`. HEAD = origin = `d277ea2`. GPU 8장 memory.used 0 MiB. `cache/bgrl/`: `.gitkeep`만. 작업 트리: `CLAUDE.md` 수정(개정 4), `instructions/HW03a_iislab4.md` 미추적 — 둘 다 사용자 드롭 파일. `hw/` 미커밋 변경 없음.
- HW03a-a `f8a6604` (11:38, 즉시 push):
  - `hw/neighbors.py`: `select_neighbors_pick(…, pick, generator, selection_out)`, `_eligible`. `pick="max"`는 기존 `select_neighbors`를 그대로 호출(기존 함수 변경 없음). `contrast` = argmax [cos(e_n, e_support) − cos(e_n, e_query)], `random` = 유효 이웃 위 `torch.rand(generator=…)` argmax. 유효 이웃·fallback은 기존과 같다. 모두 `no_grad`.
  - `hw/losses.py`: `episode_loss`·`episode_loss_hsan`에 `pick`, `generator`, `selection_out` 인자(기본값이면 기존 경로), 선택 호출을 `select_neighbors_pick`으로.
  - `hw/run.py`: 설정 `pick`(기본 `max`) 검증, `random`이면 `torch.Generator(device)` `manual_seed(seed + 7919)`. `neighbor_diagnostics()`(측정 전용, `# measurement-only label use` 주석) → `epochs[].nbr_diag` 6키. `neighbor_contrastive.pick` 기록.
  - `hw/summarize.py`: `summary.csv`에 `collapsed`, `nbr_eq_support`, `nbr_eq_query`, `pos_nbr_cos_raw` 열; `summary.md` 셀 `[c=k]`(k>0일 때), runs 표에 collapsed 열; `_regress/` 집계 제외.
  - `configs/hw.json`: 12 arm(§1.5 표). `.gitignore`: `!*.yml`, `!results/hw_iislab4/**/*.log`.
  - §1.4 cos τ: `hsan.tau_cos`는 `losses._cos_logits(…, tau_cos)`에 그대로 쓰인다. 코드 변경 없음.
  - 회귀 `results/hw_iislab4/_regress/` (4 run, 95 s), smoke `/tmp/hw_smoke`(커밋 안 함). CLAUDE.md 개정 4, 지시서(사전 등록 §6) 포함.
- HW03a-b `0d9fbd7`: 1단계 240 run (`results/hw_iislab4/HW03a_stage1.txt`), 11:38:49–12:16:03.
- HW03a-c `c4a8844`: C\* 선택, arm 2개 추가, 2단계 40 run (`HW03a_stage2.txt`), 427 s. 태그 `hw03a` push.
- HW03a-d `c615c94`: BGRL 조건 확인, arm 3개 추가(`nbr_bgrl_l01_contrast`, `cos_t020_pn_bgrl`, `cos_t020_pn_contrast_bgrl`), 100 run (`HW03a_stageD.txt`), 12:24:36–12:40:42.
- 집계: `HW_RESULTS_ROOT=results/hw_iislab4 python -m hw.summarize --ref nbr_raw --ref naq_base --ref cos_t005_none --ref cos_t010_none --ref cos_t020_none --ref nbr_raw_contrast --ref cos_t020_pn --ref nbr_bgrl_l01`.
- 이 보고서: 커밋 HW03a-e.
- 그사이 seio 커밋(`92ac844`, `9b1c214`, `c34c8db`, `43a0d6e`, `9c28bae`, `7a05bf9`)을 `pull --rebase`로 받았다. 내용 충돌 없음. `92ac844`가 기존 `hw/*.py`·`configs/hw.json`을 바꾸지 않았음을 확인(새 파일 `hw/compat/exp003.py`만).
- `results/hw/` 아래 쓰기 없음.

## 5. 판단한 사항

1. 드롭 파일 위치: 사용자 메시지는 `~/wkspace/HW_drop/CLAUDE.md`·`HW03a_iislab4.md`를 지목했으나 `HW_drop/`에는 없고, 리포 루트 `CLAUDE.md`(개정 4)와 `instructions/HW03a_iislab4.md`에 이미 놓여 있었다. 이 두 파일을 그대로 썼다(§1.1 복사 생략).
2. 시작 전 `git pull --rebase`: HEAD가 이미 origin(`d277ea2`)과 같고, 작업 트리의 드롭 파일 때문에 pull이 거부되어 생략했다. HW03a-a 커밋 직전 pull도 staged 상태라 거부됐고, push는 fast-forward로 성공(원격 신규 커밋 없음). 이후 커밋은 `git pull --rebase --autostash` 후 커밋·push.
3. `pick="max"` 경로 보존: 기존 `select_neighbors`를 한 줄도 바꾸지 않고, 새 함수가 `max`일 때 그것을 호출한다. 진단용 유효 이웃 수는 별도 helper(`_eligible`)로 다시 계산한다(선택에는 쓰지 않음).
4. `contrast`·`random`·진단 helper는 `max_neighbors == 0`에서만 지원(그 외 ValueError). 기존 `max_neighbors > 0` 부분 표본은 전역 RNG를 쓰고, 모든 arm이 0이다.
5. `pos_nbr_cos_raw`와 `nbr_eq_*`는 fallback(유효 이웃 없음 → support 자신)도 "고른 이웃"으로 포함해 계산했다. fallback_frac 0.0042–0.0078.
6. `summary.md`의 `[c=k]`는 k > 0일 때만 붙인다(이번 결과는 전부 k=0).
7. run 목록 순서: seed 바깥, arm 가운데, ds 가장 안쪽.
8. 2단계·d단계 arm 설정: `base`가 한 단계만 허용되므로 C\* 계열은 `base: nbr_raw` + hsan 블록 전체 + `pick`/`selector`/`bgrl_dir`을 적었다. `nbr_bgrl_l01_contrast`는 `base: nbr_bgrl` + λ 0.1 + `pick: contrast`. `load_config`로 각 arm의 최종 설정이 C\*·`nbr_bgrl_l01`과 지정한 키만 다른지 확인했다.
9. `nbr_raw`·`naq_base`의 10 seed 중 1212–1216은 HW03-prep run(이번 진단 키 없음)을 그대로 썼다(지시서 §2 목록대로). 진단 표의 `nbr_raw` n=5는 이 때문이다.
10. 태그 `hw03a`는 G8 조건(회귀 + G6 b·c) 충족 직후 `c4a8844`(HW03a-c)에 달았다. d 단계·보고서 커밋은 태그 뒤에 있다.
11. BGRL 대기: §4 시작(12:24) 시점에 두 파일이 이미 있어 대기하지 않았다.

## 6. 제외·실패·붕괴 run

없음. 회귀 4, 1단계 240, 2단계 40, d단계 100 run 모두 attempt 1에서 `completed`. 붕괴(best-epoch val ≤ 25%) 0.

## 7. 소요 시간, 환경, HEAD

| 단계 | 시각 (UTC) | 소요 |
|---|---|---|
| 구현 + smoke | — 11:36 | 시작 시각 미기록 |
| 회귀 (4 run, GPU 4장) | 11:36:47–11:38:22 | 95 s |
| HW03a-a push | 11:38:32 | |
| 1단계 (240 run, GPU 8장) | 11:38:49–12:16:03 | 37 min 14 s |
| 2단계 (40 run) | 12:16:54–12:24:01 | 427 s |
| BGRL 대기 | 12:24 | 0 |
| d단계 (100 run) | 12:24:36–12:40:42 | 16 min 6 s |

- run당 `elapsed_seconds`(GPU 8장 동시, GPU당 1 run): `naq_base` 52–54 / 65–68 s, `nbr_*`(raw·bgrl) 55–61 / 71–76 s, `cos_*` 60–67 / 73–81 s (Cora_full / Amazon_clothing).
- 환경: iislab4, conda `taskns`(python 3.9.25, torch 2.4.1+cu121, torch_geometric 2.6.1, numpy 1.26.4, sklearn 1.5.2), 드라이버 580.178.04.
- HEAD: 이 보고서 커밋(HW03a-e). 직전 `c615c94`.
