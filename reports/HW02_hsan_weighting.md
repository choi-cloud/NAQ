# HW02 — HSAN식 hard-sample 가중치를 `nbr_raw` 위에 얹기

2026-11-07 · 브랜치 `hl/hsan-weighting` · 지시서 `instructions/HW02_hsan_weighting.md` · CLAUDE.md 개정 3 · 선행 태그 `hw01`(b7124c3)

## 1. 결론

- `nbr_raw`(Δ 기준)와 짝지어 비교했다. HSAN arm 10개 모두 두 데이터셋에서 Δ 평균이 0 이하다. 이 중 Δ 평균이 가장 작은 arm은 `hsan_p_orig_b1_raw`로, Amazon_clothing −1.49 (양수 0/5), Cora_full −0.30 (양수 2/5)이다. 1단계 euclid arm 5개의 Δ 평균 범위는 −2.16 ~ −42.73이다.
- `nbr_raw − naq_base`는 Amazon_clothing +0.20 (양수 5/5), Cora_full +0.35 (양수 4/5)이다. cos 쌍 `hsan_pn_orig_b2_cos − nbr_raw_cos`는 Amazon_clothing +1.13 (4/5), Cora_full +1.46 (5/5)이다. 반면 `nbr_raw_cos − nbr_raw`는 Amazon_clothing −5.74, Cora_full −4.32이다.
- 회귀 확인과 G3·G4·G5를 모두 통과해 태그 `hw02`를 달았다. 사전 등록 중 HWP007·008은 기준을 충족했고, HWP009–013은 충족하지 못했다.

## 2. 핵심 표

### 2.1 arm × ds, 5 seed 평균 ± 표준편차 (test %, `results/hw/summary.md`)

| arm | 단계 | Amazon_clothing | Cora_full |
|---|---|---|---|
| naq_base | 1 | 90.80 ± 0.20 | 77.59 ± 0.70 |
| nbr_raw | 1 | 91.00 ± 0.20 | 77.94 ± 0.61 |
| hsan_p_orig_b2 | 1 | 80.81 ± 1.23 | 72.86 ± 0.98 |
| hsan_p_both_b2 | 1 | 56.19 ± 1.57 | 38.65 ± 1.66 |
| hsan_pn_orig_b2 | 1 | 88.84 ± 0.25 | 42.74 ± 21.49 |
| hsan_pn_both_b2 | 1 | 48.27 ± 11.14 | 39.63 ± 15.81 |
| hsan_p_orig_b2_raw | 1 | 86.18 ± 0.80 | 75.44 ± 0.43 |
| hsan_p_orig_b1_raw | 2 | 89.51 ± 0.30 | 77.64 ± 0.64 |
| hsan_p_orig_b3_raw | 2 | 83.50 ± 0.87 | 73.83 ± 0.58 |
| hsan_p_orig_b1 | 2 | 85.65 ± 1.19 | 76.98 ± 0.60 |
| hsan_p_orig_b3 | 2 | 79.04 ± 0.99 | 69.68 ± 1.54 |
| nbr_raw_cos | 2 | 85.26 ± 0.86 | 73.62 ± 0.98 |
| hsan_pn_orig_b2_cos | 2 | 86.39 ± 0.40 | 75.08 ± 0.52 |

모든 셀은 n = 5다(seed 1212–1216).

### 2.2 짝지은 Δ vs `nbr_raw` (`results/hw/paired_nbr_raw.md`)

| arm | Amazon_clothing | Cora_full | 두 ds 평균 |
|---|---|---|---|
| naq_base | −0.20 ± 0.09 (0/5) | −0.35 ± 0.79 (1/5) | −0.28 |
| hsan_p_orig_b1_raw | −1.49 ± 0.16 (0/5) | −0.30 ± 0.78 (2/5) | −0.90 |
| hsan_p_orig_b1 | −5.35 ± 1.18 (0/5) | −0.96 ± 1.08 (1/5) | −3.15 |
| hsan_p_orig_b2_raw | −4.82 ± 0.78 (0/5) | −2.50 ± 0.95 (0/5) | −3.66 |
| hsan_pn_orig_b2_cos | −4.62 ± 0.39 (0/5) | −2.86 ± 0.15 (0/5) | −3.74 |
| nbr_raw_cos | −5.74 ± 0.99 (0/5) | −4.32 ± 0.50 (0/5) | −5.03 |
| hsan_p_orig_b3_raw | −7.50 ± 0.83 (0/5) | −4.10 ± 0.50 (0/5) | −5.80 |
| hsan_p_orig_b2 | −10.19 ± 1.09 (0/5) | −5.08 ± 0.60 (0/5) | −7.64 |
| hsan_p_orig_b3 | −11.96 ± 0.95 (0/5) | −8.26 ± 1.33 (0/5) | −10.11 |
| hsan_pn_orig_b2 | −2.16 ± 0.42 (0/5) | −35.20 ± 21.28 (0/5) | −18.68 |
| hsan_p_both_b2 | −34.81 ± 1.47 (0/5) | −39.29 ± 1.31 (0/5) | −37.05 |
| hsan_pn_both_b2 | −42.73 ± 10.99 (0/5) | −38.31 ± 15.67 (0/5) | −40.52 |

seed별 Δ는 `results/hw/paired_nbr_raw.md`의 "per-seed Δ"에 있다. `hsan_pn_orig_b2`의 Cora_full seed별 Δ는 −52.33, −7.83, −54.27, −10.55, −51.03이다.

### 2.3 짝지은 Δ vs `naq_base` (`results/hw/paired_naq_base.md`)

| arm | Amazon_clothing | Cora_full | 두 ds 평균 |
|---|---|---|---|
| nbr_raw | +0.20 ± 0.09 (5/5) | +0.35 ± 0.79 (4/5) | +0.28 |
| hsan_p_orig_b1_raw | −1.28 ± 0.14 (0/5) | +0.05 ± 0.22 (3/5) | −0.62 |
| 그 밖의 hsan arm | `paired_naq_base.md` 참조 | | |

### 2.4 cos 쌍: 짝지은 Δ vs `nbr_raw_cos` (`results/hw/paired_nbr_raw_cos.md`)

| arm | Amazon_clothing | Cora_full | 두 ds 평균 |
|---|---|---|---|
| nbr_raw | +5.74 ± 0.99 (5/5) | +4.32 ± 0.50 (5/5) | +5.03 |
| hsan_pn_orig_b2_cos | +1.13 ± 1.22 (4/5) | +1.46 ± 0.60 (5/5) | +1.30 |

### 2.5 2단계 선택 근거 (1단계 `paired_nbr_raw.md`, hsan arm 5개)

| 순위 | arm | Amazon_clothing Δ | Cora_full Δ | 두 ds Δ 평균의 평균 | 양수 seed 합 | 선택 |
|---|---|---|---|---|---|---|
| 1 | hsan_p_orig_b2_raw | −4.82 | −2.50 | −3.66 | 0 | ✓ |
| 2 | hsan_p_orig_b2 | −10.19 | −5.08 | −7.64 | 0 | ✓ |
| 3 | hsan_pn_orig_b2 | −2.16 | −35.20 | −18.68 | 0 | |
| 4 | hsan_p_both_b2 | −34.81 | −39.29 | −37.05 | 0 | |
| 5 | hsan_pn_both_b2 | −42.73 | −38.31 | −40.52 | 0 | |

동률은 없었다. 2단계 arm 이름은 `b2`를 `b1`·`b3`으로 바꾼 것이다: `hsan_p_orig_b1_raw`, `hsan_p_orig_b3_raw`, `hsan_p_orig_b1`, `hsan_p_orig_b3`.

### 2.6 가중치 진단 (마지막 구간 epoch 1921–2000, 5 seed 평균)

| arm | ds | 항 | pos_weight_mean | neg_weight_mean | pos_weight_lt05_frac | pos_s_mean | neg_s_mean | best_epoch (seed 1212…1216) |
|---|---|---|---|---|---|---|---|---|
| hsan_p_orig_b2 | Cora_full | orig | 0.066 | 1.000 | 0.695 | 0.869 | 0.612 | 480,480,1600,480,960 |
| hsan_p_orig_b2 | Amazon_clothing | orig | 0.022 | 1.000 | 0.904 | 0.938 | 0.560 | 160,800,640,800,960 |
| hsan_p_both_b2 | Cora_full | orig | 0.016 | 1.000 | 0.930 | 0.999 | 0.996 | 160 ×5 |
| hsan_p_both_b2 | Cora_full | nbr | 0.017 | 1.000 | 0.928 | 0.999 | 0.996 | |
| hsan_p_both_b2 | Amazon_clothing | orig | 0.005 | 1.000 | 0.981 | 1.000 | 0.996 | 160 ×5 |
| hsan_p_both_b2 | Amazon_clothing | nbr | 0.005 | 1.000 | 0.980 | 1.000 | 0.997 | |
| hsan_pn_orig_b2 | Cora_full | orig | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 160,320,160,160,160 |
| hsan_pn_orig_b2 | Amazon_clothing | orig | 0.062 | 0.079 | 0.721 | 0.848 | 0.277 | 2000,2000,2000,1760,2000 |
| hsan_pn_both_b2 | Cora_full | orig | 0.412 | 0.327 | 0.541 | 0.600 | 0.600 | 160 ×5 |
| hsan_pn_both_b2 | Cora_full | nbr | 0.413 | 0.333 | 0.537 | 0.600 | 0.600 | |
| hsan_pn_both_b2 | Amazon_clothing | orig | 0.997 | 0.000 | 0.003 | 0.003 | 0.000 | 160 ×5 |
| hsan_pn_both_b2 | Amazon_clothing | nbr | 0.998 | 0.000 | 0.002 | 0.002 | 0.000 | |
| hsan_p_orig_b2_raw | Cora_full | orig | 0.336 | 1.000 | 0.110 | 0.250 | 0.053 | 480,960,1760,1120,640 |
| hsan_p_orig_b2_raw | Amazon_clothing | orig | 0.229 | 1.000 | 0.266 | 0.582 | 0.032 | 1120,800,1760,800,960 |
| hsan_p_orig_b1_raw | Cora_full | orig | 0.536 | 1.000 | 0.033 | 0.250 | 0.053 | 1760,960,2000,1760,1440 |
| hsan_p_orig_b1_raw | Amazon_clothing | orig | 0.410 | 1.000 | 0.171 | 0.582 | 0.032 | 1760,1600,2000,1440,1600 |
| hsan_p_orig_b3_raw | Cora_full | orig | 0.225 | 1.000 | 0.235 | 0.250 | 0.053 | 480,640,2000,480,640 |
| hsan_p_orig_b3_raw | Amazon_clothing | orig | 0.136 | 1.000 | 0.355 | 0.582 | 0.032 | 480,1120,640,800,800 |
| hsan_p_orig_b1 | Cora_full | orig | 0.293 | 1.000 | 0.149 | 0.764 | 0.468 | 1600,1920,1600,1280,800 |
| hsan_p_orig_b1 | Amazon_clothing | orig | 0.120 | 1.000 | 0.486 | 0.901 | 0.439 | 1920,1120,2000,800,960 |
| hsan_p_orig_b3 | Cora_full | orig | 0.028 | 1.000 | 0.871 | 0.891 | 0.655 | 480,480,1600,480,640 |
| hsan_p_orig_b3 | Amazon_clothing | orig | 0.009 | 1.000 | 0.963 | 0.945 | 0.586 | 160,800,640,480,800 |
| hsan_pn_orig_b2_cos | Cora_full | orig | 0.140 | 0.105 | 0.426 | 0.712 | 0.323 | 1280,1760,1440,2000,1920 |
| hsan_pn_orig_b2_cos | Amazon_clothing | orig | 0.066 | 0.044 | 0.739 | 0.843 | 0.135 | 1920,1920,2000,1920,1600 |

- `apply: orig`인 arm은 nbr 항을 변조하지 않으므로 그 항의 진단이 없다. `nbr_raw_cos`(가중치 없음)도 진단이 없다.
- `nbr_raw`의 best_epoch(seed 1212…1216)는 Cora_full 2000,1920,2000,1920,2000, Amazon_clothing 1440,1760,2000,1920,1600이다(`summary.md` runs 표).
- 손실 크기 (seed 1212, `train_original_loss_mean`, 1번째·5번째·9번째·13번째 평가 구간):

| run | 구간 1 | 구간 5 | 구간 9 | 구간 13 |
|---|---|---|---|---|
| nbr_raw Cora_full | 1.28 | 0.80 | 0.72 | 0.67 |
| hsan_p_orig_b2 Cora_full | 1.94 | 33.94 | 269.39 | 2068.33 |
| hsan_p_both_b2 Cora_full | 27.76 | 7.89e5 | 7.81e6 | 3.70e7 |

  - HSAN 전체 run의 구간별 `train_loss_mean` 최댓값은 4.60e7이다(hsan_p_both_b2 Cora_full seed 1212). 모든 값은 유한하다.
- validation 정확도가 20.0–22.7%(5-way 무작위 수준)로 끝난 run이 있다.
  - 해당 run: `hsan_pn_orig_b2` Cora_full seed 1212·1214·1216, `hsan_pn_both_b2`의 다수 seed.
  - 이 run들의 마지막 구간 `train_original_loss_mean`은 1.609(= ln 5)이고, pos/neg_s_mean은 0.000이다.

## 3. 게이트·사전 등록 대조표

| ID | 기준 | 측정값 | 통과 여부 |
|---|---|---|---|
| 회귀 (§1.5) | seed 1212 `naq_base`·`nbr_raw` \|Δ vs HW01\| ≤ 0.30 | naq_base 77.84/90.60 (+0.06/−0.07), nbr_raw 78.16/90.69 (+0.06/−0.17) | 통과 |
| G3 | 1단계 70 run completed, 제외 ≤ 4 | 70/70 completed, 재시도 0, 제외 0 | 통과 |
| G4 | NaN/Inf 손실 run ≤ 2 | 0 (1·2단계 130 run) | 통과 |
| G5 | G3·G4·회귀 통과 | — | 통과 → 태그 `hw02` |
| HWP007 | `nbr_raw − naq_base` 두 ds Δ ≥ +0.15, 양수 ≥ 3/5 | Amazon +0.20 (5/5), Cora +0.35 (4/5) | 기준 충족 |
| HWP008 | `hsan_p_both_b2 − hsan_p_orig_b2` < 0, 두 ds | Amazon −24.62, Cora −34.21 | 기준 충족 |
| HWP009 | `hsan_p_orig_b2 − nbr_raw` 두 ds Δ ≥ +0.20, 양수 ≥ 3/5 | Amazon −10.19 (0/5), Cora −5.08 (0/5) | 기준 미충족 |
| HWP010 | `hsan_pn_orig_b2 − hsan_p_orig_b2` > 0, 두 ds | Amazon +8.03, Cora −30.12 | 기준 미충족 (Cora) |
| HWP011 | `hsan_p_orig_b2 − hsan_p_orig_b2_raw` > 0, 두 ds | Amazon −5.37, Cora −2.58 | 기준 미충족 |
| HWP012 | `b2 − b1` > 0, 두 ds (상위 2 arm 평균) | raw: Amazon −3.33, Cora −2.20; gcn: Amazon −4.84, Cora −4.12; 평균 Amazon −4.09, Cora −3.16 | 기준 미충족 |
| HWP013 | sign(`hsan_pn_orig_b2_cos − nbr_raw_cos`) = sign(`hsan_pn_orig_b2 − nbr_raw`), 두 ds | cos: +1.13 / +1.46; euclid: −2.16 / −35.20 | 기준 미충족 (부호 반대, 두 ds) |

- arm 간 차이(HWP008·010·011·012)는 짝지은 Δ(vs `nbr_raw`)의 평균끼리 뺀 값이다. 이는 같은 seed끼리 직접 뺀 값의 평균과 같다.
- 지시서 §6 해석 규칙의 조건은 다음과 같다. 판정은 사람이 한다.
  - HWP007: 충족.
  - HWP009·010: 둘 다 미충족. euclid HSAN arm 9개 전부의 Δ(vs `nbr_raw`) 평균은 두 데이터셋에서 ≤ 0이다.
  - HWP008: 충족.
  - HWP011: 미충족. `space: raw`가 `gcn`보다 test가 높다.

## 4. 수행 내용

### §0 현황 (2026-11-07 08:02 UTC, 수정 없음)
- `git status --short`: 비어 있었다. HEAD는 `b7124c3 HW01-d`, 태그는 `hw01`과 다른 트랙 태그들이다.
- `ls results/hw/`: `naq_base nbr_raw repro summary.csv summary.md`.
- GPU: RTX 4080 SUPER, 1337 MiB 사용 중, 40% util. 다른 사용자(`schoi`)의 `DiffCDR/entry.py`(pid 2475286)가 실행 중이었고, 이후 작업 내내 같은 GPU를 함께 썼다.

### HW02-a (`3a9d13e`)
- 드롭 파일 복사: CLAUDE.md 개정 3 → 루트, 지시서 → `instructions/`. 복사 후 다시 읽었다.
- `hw/weights.py`: `hsan_weights(h_q, h_c, P, mode, beta)`와 `apply_weights(l, M, logit)`.
  - min-max는 항별 Q×C 행렬 전체에 적용한다. `torch.no_grad`이며 RNG를 소비하지 않는다(단위 확인: `torch.get_rng_state` 불변).
  - 단위 확인: `euclid_gradscale`에서 값은 같고 gradient는 M×g였다. `mode p`에서 음성 M은 1이었다.
- `hw/losses.py`: `episode_loss` 맨 앞에 `if hsan is not None: return episode_loss_hsan(...)` 분기만 추가했다. `hsan is None` 경로의 본문은 바꾸지 않았다.
- `hw/config.py`: `base` 한 단계와 `hsan` 블록 검증을 추가했다. 허용 값 밖이면 ValueError를 낸다.
- `hw/run.py`: 다음을 추가했다.
  - 진단 `epochs[].hsan_diag`(구간 평균)
  - 첫 5 에피소드의 M 통계를 `train.log`에 기록
  - 손실이 유한하지 않으면 즉시 `status: "nan"`으로 끝내고 exit 3. 직전 5 에피소드 진단을 저장한다.
- `hw/run_queue.sh`: flock 큐, 작업자 N개, 1회 재시도. completed는 건너뛰고 nan은 재시도하지 않는다.
- `hw/summarize.py`: `--ref`(반복 가능) → `paired_<ref>.md`, `summary.csv`에 `orig_/nbr_pos_weight_mean` 열을 추가했다.
- `configs/hw.json`: 1단계 hsan arm 5개, `nbr_raw_cos`, `hsan_pn_orig_b2_cos`를 추가했다.
- 4 에피소드 smoke(임시 설정, 결과는 스크래치에만): 7개 arm 모두 completed였다. euclid arm들의 episode-1 loss(1.4649)는 `nbr_raw`와 같다.
- 회귀 확인은 §3 표와 같다.

### HW02-b (`2f76cfa`)
- `results/hw/HW02_stage1.txt`: 70 run. 순서는 seed 바깥, arm 안쪽, ds 가장 안쪽이다.
- 회귀 확인의 seed 1212 `naq_base`·`nbr_raw` 4 run은 큐가 completed로 건너뛰어 재사용했다.

### HW02-c (이 커밋)
- 2단계 arm 4개를 `configs/hw.json`에 추가했다(§2.5).
- `results/hw/HW02_stage2.txt`: 60 run(β 40 + cos 쌍 20). 지시서의 시간 규칙(3시간 초과 시 cos 쌍 생략)은 적용되지 않았다.
- `python -m hw.summarize --ref nbr_raw --ref naq_base --ref nbr_raw_cos`를 실행했다.

## 5. 판단한 사항

1. **`nbr_raw_cos`의 구현 경로**: 지시서는 `nbr_raw_cos`를 "같은 경로로" 만들라고 한다. 그래서 `hsan` 블록에 `mode: "none"`(가중치 없음, `logit: cos_hsan` 전용)을 추가했다. `mode` 허용 값이 `{p, pn}`에서 `{p, pn, none}`으로 넓어졌고, 이는 CLAUDE.md §4.2의 허용 값 목록과 다르다. `none`은 `cos_hsan`과만 함께 쓸 수 있다.
2. **cos logit의 범위**: `cos_hsan`에서는 두 항(orig·nbr)의 logit을 모두 `cos/τ_cos`로 바꾸고, M은 `apply`가 가리키는 항에만 곱했다. `nbr_raw_cos`도 두 항 모두 cos logit이다. 학습 정확도 진단과 평가는 기존 유클리드 그대로다.
3. **`base` 병합**: base arm의 `note` 키는 상속하지 않는다.
4. **NaN 검사**: 모든 arm에 적용했다. `loss.item()`으로 값을 확인할 뿐 수치 경로는 바꾸지 않는다. 회귀 확인으로 `naq_base`·`nbr_raw`가 HW01과 |Δ| ≤ 0.17임을 확인했다.
5. **회귀 run 덮어쓰기**: `naq_base`·`nbr_raw` seed 1212의 HW01 결과 파일은 회귀 확인 run으로 덮어썼다. HW01 판본은 태그 `hw01`에 남아 있다.
6. **2단계 β 변형의 이름**: `hsan_p_orig_b2_raw`에서 나온 β 변형은 지시서 규칙("`b2`를 `b1`/`b3`로")대로 `hsan_p_orig_b1_raw`·`hsan_p_orig_b3_raw`로 지었다.
7. **평가 에피소드**: 모든 seed에서 valid/test 에피소드는 같은 캐시 파일(`save/downstream_tasks/<ds>/5way_5shot.pkl`)에서 읽는다. 작업 후 md5도 변화가 없다(Cora_full `cfef6a69…`, Amazon_clothing `ca3edaca…`). 따라서 seed가 바꾸는 것은 학습 쪽(모델 초기화, train pool 등)이다.
8. **CLAUDE.md와의 차이**: §4.2의 json 예시에는 arm `note` 키가 없지만 기존 `note`를 유지했다. 동작에는 영향이 없다.
9. **작업 트리의 외부 변경**: `choi_exps/legacy_naq_exp/exp004.py`·`exp004.sh`가 2026-11-07 09:01:54에 이 작업 밖에서 바뀌었다.
   - 새 내용의 docstring은 "exp004: BGRL pretraining and pool-purity comparison against baselines"이고, `.sh`는 `OUT_DIR=out/exp004`, `EPOCHS=400` 등이다.
   - 이 두 파일은 되돌리지도 커밋하지도 않았다(`choi_exps/` 동결, 범위 밖). 미커밋 변경으로 작업 트리에 남아 있다.

## 6. 제외·실패한 run

- 없다. 1·2단계 130 run(재사용 4개 포함) 모두 `status == "completed"`이고, 재시도와 NaN은 0이다.
- validation이 무작위 수준(20%대)으로 끝난 run도 completed이므로 집계에 포함했다(§2.6).

## 7. 소요 시간, 환경, HEAD

- 작업 시각: 2026-11-07 08:02–10:00 UTC(약 2시간). 시간 예산은 4시간이었다.
- 구간별 소요:
  - 구현과 smoke: 08:02–08:05, 08:08–08:09
  - 회귀 확인: 08:05–08:08 (2 병렬)
  - 1단계: 08:09:33–09:06:14 (56분 41초, 66 run 실행)
  - 2단계: 09:07:15–09:51:34 (44분 19초, 60 run)
- run당 소요(큐 기준, 작업자 2개 병렬, 다른 사용자 GPU 작업과 공유)는 다음과 같다. `metrics.json`의 `elapsed_seconds`는 평균 93.6 s(51.4–132.8 s, n = 130)다.

| 구간 | 평균 | 범위 |
|---|---|---|
| 1단계 | 102 s | 71–134 s |
| 2단계 | 88 s | 53–117 s |

- 병렬: 작업자 2개로 끝까지 진행했다. OOM은 없었다(1단계 시작 시 GPU 메모리 10.2 GB / 16 GB).
- 환경: RTX 4080 SUPER, conda `taskns`(torch 2.4.1+cu121).
- HEAD: 이 보고서를 포함한 HW02-c 커밋. 태그 `hw02`.
