# HW02 — HSAN식 hard-sample 가중치를 `nbr_raw` 위에 얹기 (auto, 5 seed, 2단계)

2026-11-07 · 작성: 이희태 (Claude 정리) · 브랜치 `hl/hsan-weighting` · 선행 태그 `hw01`(b7124c3) · 태그 `hw02` · 사전 등록 HWP007–HWP013 · 선행 문서 `CLAUDE.md` 개정 3, `reports/HW01_restructure_and_repro.md`

- auto 모드. "멈추고 보고"는 §1 회귀 확인 실패, §5 G4 실패, 구현 중 해결 못 한 오류뿐이다.
- 시간 예산 **4시간** (구현 40 / 회귀 5 / 1단계 실행 ~70 / 2단계 실행 ~60 / 집계·보고 20). run당 약 1분, 작업자 2개 기준. 3시간을 넘기면 2단계의 cos 쌍(§4-3)을 빼고 그 사실을 적는다.
- 목적: 석휘 아이디어 2 — 아이디어 1(이웃 양성 확장, `nbr_raw`)로 늘어난 양성·음성 쌍에 HSAN(Liu et al., AAAI'23)의 hard-sample 가중치를 적용했을 때, `nbr_raw` 대비 test 정확도가 오르는지를 5 seed 짝지은 비교로 잰다.
- 이 지시서에는 **클러스터링이 없다.** HSAN의 pseudo-relation Q는 에피소드의 pseudo-label(같은 pseudo-class = 1, 아니면 0)로 대신한다. high-confidence 집합 H는 쓰지 않는다(모든 쌍을 변조). 클러스터 기반 게이팅은 HW03.

## 0. 현황 (수정·커밋 없이, 보고서 §4에 적고 계속)

`git status --short`, `git log --oneline -3`, `git tag`, `ls results/hw/`, `nvidia-smi` 요약. HEAD가 b7124c3이고 작업 트리가 깨끗한지. 다르면 적고 계속한다(커밋되지 않은 변경이 `hw/`에 있으면 **멈추고 보고**).

## 1. HW02-a: 구현 + 회귀 확인

### 1.1 드롭 파일
`~/wkspace/HW_drop/CLAUDE.md`(개정 3) → 루트, `~/wkspace/HW_drop/HW02_hsan_weighting.md` → `instructions/`. CLAUDE.md를 다시 읽는다.

### 1.2 `hw/weights.py` — HSAN 가중치 (새 파일)

표기. 에피소드 하나에서 query 집합 Q(|Q| = N·q_query = 50), 후보 집합 C(|C| = N = 5; `orig` 항은 support 5개, `nbr` 항은 선택된 이웃 5개). 러너의 기존 logit은 `ℓ[q,c] = −‖z_q − z_c‖² / τ` (z = 학습 중 GCN 임베딩). positive mask `P[q,c] = (y_q == y_c)`.

1. **hardness 유사도** `s[q,c] = cos(h_q, h_c)`. `h`는 `space`에 따라
   - `gcn`: 현재 GCN 임베딩 `z`를 **detach**한 것 (HSAN이 학습 중 임베딩을 쓰는 것에 대응)
   - `raw`: raw feature L2 정규화 (대조군, 학습과 무관한 고정 공간)
2. **min-max 정규화** `s̃ = (s − min s) / (max s − min s + 1e-12)`, 범위는 **그 항의 Q×C 행렬 전체**(HSAN의 `Norm`). 두 항(`orig`, `nbr`)은 각각 따로 정규화한다.
3. **가중치** `M[q,c] = |P[q,c] − s̃[q,c]|^β` (HSAN Eq. 9, H 조건 없음). 즉
   - positive: `(1 − s̃)^β` — 유사도 낮은(어려운) 양성이 큰 가중치
   - negative: `s̃^β` — 유사도 높은(어려운) 음성이 큰 가중치
   - `mode == "p"`이면 negative 가중치를 1로 둔다(pulling만 변조). `mode == "pn"`이면 둘 다.
4. **logit에 적용** (`logit` 옵션)
   - `euclid_gradscale` (1단계 기본): 값은 그대로, gradient만 M배. 구현: `ℓ' = M·ℓ + (1 − M)·ℓ.detach()`. 손실 값과 확률은 기존과 같고 각 쌍의 gradient가 M배가 된다. HSAN Eq. 10에서 M이 하는 일 중 "gradient를 M배로"만 가져온 형태다. (기존 logit이 음의 제곱거리라 `M·ℓ`을 값으로 쓰면 M→0이 "거리 0"이 되어 뜻이 어긋난다.)
   - `cos_hsan` (2단계): HSAN Eq. 10 그대로. logit을 `ℓ = cos(z_q, z_c)/τ_cos`로 바꾸고 `ℓ' = M·ℓ`. `τ_cos`는 설정값(기본 0.5, `nbr_raw_cos` 대조군과 같은 값).
5. 가중 손실: 기존 형태 유지 — `−[logsumexp_{c∈P} ℓ' − log|P|] + logsumexp_{c∈C} ℓ'`.
6. `apply`: `orig` → `orig` 항만 변조, `nbr` 항은 기존 그대로. `both` → 두 항 모두.
7. RNG를 소비하지 않는다. 가중치 계산은 `torch.no_grad()` 안에서.
8. 진단(에피소드별로 계산, `eval_every` 구간 평균을 `epochs[]`에 기록): 항별로 `pos_weight_mean`, `neg_weight_mean`, `pos_weight_lt05_frac`(M<0.05인 양성 비율), `pos_s_mean`(정규화 전 양성 cos 평균), `neg_s_mean`.

### 1.3 `hw/losses.py`, `hw/run.py`, `hw/config.py`
- `config.py`: `base` 키 지원(한 단계), `hsan` 블록 검증(허용 값 밖이면 ValueError).
- `losses.py`: `hsan is None`이면 **기존 코드 경로를 한 줄도 바꾸지 않는다.** `hsan`이 있으면 `weights.py`를 호출해 ℓ'을 만든 뒤 기존 손실 식에 넣는다. `cos_hsan`은 두 항의 logit 정의 자체를 바꾸므로 `nbr_raw_cos` 대조군(가중치 없음, cos logit)도 같은 경로로 만든다.
- `run.py`: 진단 기록. `train.log`에 첫 5 에피소드의 M 통계를 찍는다.
- `hw/run_queue.sh`: 인자로 run 목록 파일(`arm,dataset,seed` 한 줄씩). 작업자 2개, `flock`, 실패 시 1회 재시도, 이미 `metrics.json`의 `status=="completed"`인 run은 건너뛴다.
- `hw/summarize.py`: `--ref` 옵션과 `paired_<ref>.md` (CLAUDE.md §4.3). 진단 열(`pos_weight_mean` 최종 구간 값)을 `summary.csv`에 추가.

### 1.4 `configs/hw.json` arm 추가 (전부 `base: nbr_raw`, `space: gcn`, `logit: euclid_gradscale`이 기본)

| arm | mode | apply | beta | space | logit |
|---|---|---|---|---|---|
| `hsan_p_orig_b2` | p | orig | 2 | gcn | euclid_gradscale |
| `hsan_p_both_b2` | p | both | 2 | gcn | euclid_gradscale |
| `hsan_pn_orig_b2` | pn | orig | 2 | gcn | euclid_gradscale |
| `hsan_pn_both_b2` | pn | both | 2 | gcn | euclid_gradscale |
| `hsan_p_orig_b2_raw` | p | orig | 2 | raw | euclid_gradscale |
| `hsan_*_b1`, `hsan_*_b3` | 2단계에서 상위 2 arm의 β 변형 | | 1, 3 | | |
| `nbr_raw_cos` | — | — | — | — | cos (가중치 없음, τ_cos 0.5) |
| `hsan_pn_orig_b2_cos` | pn | orig | 2 | gcn | cos_hsan |

2단계 arm 4개는 §4-2의 규칙으로 Agent가 이름을 정해 추가한다.

### 1.5 회귀 확인 (CLAUDE.md §6-7)
`naq_base`·`nbr_raw`를 seed 1212로 다시 돌려 HW01 값(77.78/90.67, 78.10/90.86)과 |Δ| ≤ 0.30. 어긋나면 **멈추고 보고**. (이 두 run은 1단계의 seed 1212 run으로 재사용한다.)

커밋 `HW02-a: hsan weights, queue, paired summary, arms` → push. 사전 등록(§6)은 이 커밋에 포함된다.

## 2. 가중치가 들어가는 자리 (설계 메모, 구현 참조용)

```
orig 항:  query q  vs  support s_c (c=1..5)       P[q,c] = [y_q == y_c]
nbr  항:  query q  vs  이웃 n_c (c=1..5)           P 동일
M = |P − Norm(cos(h_q, h_c))|^β      (항별 Q×C 행렬에서 Norm)
mode p : M[¬P] := 1
euclid_gradscale: ℓ' = M·ℓ + (1−M)·ℓ.detach()
cos_hsan        : ℓ' = M · cos(z_q,z_c)/τ_cos
```

아이디어 1이 고른 이웃 n_c는 query와 raw cos가 가장 큰 노드이므로 `nbr` 항의 양성은 쉬운 양성이 되기 쉽다. `apply: both`는 그 양성의 gradient를 줄인다. `apply: orig`은 그 항을 건드리지 않는다. 두 설정의 차이가 HWP008이 재는 것이다.

## 3. HW02-b: 1단계 실행

run 목록 (`results/hw/HW02_stage1.txt`): arm ∈ {`naq_base`, `nbr_raw`, `hsan_p_orig_b2`, `hsan_p_both_b2`, `hsan_pn_orig_b2`, `hsan_pn_both_b2`, `hsan_p_orig_b2_raw`} × ds ∈ {Cora_full, Amazon_clothing} × seed ∈ {1212, 1213, 1214, 1215, 1216} = 70 run. 순서: seed 바깥, arm 안쪽(한 seed의 전 arm이 먼저 끝나게).

끝나면 `python -m hw.summarize --ref nbr_raw`와 `--ref naq_base`. 커밋 `HW02-b: stage-1 runs (7 arms × 2 ds × 5 seeds)` → push.

## 4. HW02-c: 2단계 실행

1. **선택 규칙**: 1단계 hsan arm 5개를 `paired_nbr_raw.md`의 "두 데이터셋 Δ 평균의 평균"으로 정렬해 상위 2개를 고른다. 동률이면 양수 seed 수 합이 큰 쪽. 선택 근거 표를 보고서에 적는다.
2. 상위 2 arm 각각에 β ∈ {1, 3} 변형 arm을 만든다(이름은 `b2`를 `b1`/`b3`로). 4 arm × 2 ds × 5 seed = 40 run.
3. cos 쌍: `nbr_raw_cos`, `hsan_pn_orig_b2_cos` × 2 ds × 5 seed = 20 run. (시간 규칙에 걸리면 생략.)
4. `python -m hw.summarize --ref nbr_raw`(cos arm의 ref는 `nbr_raw_cos`로 한 표 더). 커밋 `HW02-c: stage-2 runs (beta sweep, cos pair)` → push.

## 5. 게이트

- **G3**: 1단계 70 run 모두 `status == "completed"`. 실패 run이 있으면 1회 재시도 후에도 실패하면 제외하고 적는다. 제외가 4개를 넘으면 **멈추고 보고**.
- **G4**: 어떤 run에서도 손실이 NaN/Inf가 되지 않는다. NaN이 나오면 그 run을 즉시 끝내고 `status: "nan"`으로 기록한다. NaN run이 2개를 넘으면 **멈추고 보고**(arm·ds·seed·첫 NaN 에피소드·직전 5 에피소드의 M 통계).
- **G5 (태그 조건)**: G3·G4 통과 + 회귀 확인 통과. 성능 조건은 없다(판정은 사람).

## 6. 사전 등록 (HW02-a 커밋에 포함, 판정은 사람)

Δ는 같은 seed끼리 짝지은 test 차이의 5 seed 평균. "양수 k/5"는 Δ>0인 seed 수.

| ID | 예측 | 기준 | 근거 | 확신도 |
|---|---|---|---|---|
| HWP007 | 아이디어 1의 이득이 5 seed에서 유지된다 | `nbr_raw − naq_base`: 두 데이터셋 모두 Δ ≥ +0.15, 양수 ≥ 3/5 | HW01 seed 1212 +0.24/+0.23, 표 1 +1.28/+0.24 | 0.60 |
| HWP008 | `nbr` 항까지 변조하면 `orig`만 변조할 때보다 낮다 | `hsan_p_both_b2 − hsan_p_orig_b2` < 0, 두 데이터셋 | 이웃 양성은 raw cos 최대로 뽑은 쉬운 양성 → gradient 축소 | 0.55 |
| HWP009 | pulling 변조만으로 `nbr_raw`를 넘는다 | `hsan_p_orig_b2 − nbr_raw`: 두 데이터셋 Δ ≥ +0.20, 양수 ≥ 3/5 | HSAN 논문 주장; 단 Q가 에피소드 pseudo-label이라 노이즈 포함 | 0.40 |
| HWP010 | pushing까지 변조하면 pulling만보다 높다 | `hsan_pn_orig_b2 − hsan_p_orig_b2` > 0, 두 데이터셋 | hard negative 가중은 HSAN의 주효과 | 0.45 |
| HWP011 | hardness 공간은 학습 중 임베딩이 raw보다 낫다 | `hsan_p_orig_b2 − hsan_p_orig_b2_raw` > 0, 두 데이터셋 | HSAN은 학습 중 임베딩 사용 | 0.50 |
| HWP012 | β=2가 β=1보다 낫다 (상위 2 arm 평균) | `b2 − b1` > 0, 두 데이터셋 | HSAN β ∈ [1,5] 권장 | 0.50 |
| HWP013 | cos 원식도 euclid gradscale과 같은 방향 | `hsan_pn_orig_b2_cos − nbr_raw_cos`의 부호 = `hsan_pn_orig_b2 − nbr_raw`의 부호, 두 데이터셋 | 같은 가중치, 다른 적용 방식 | 0.50 |

해석 규칙 (사전 명시, 판정은 사람)
- HWP007 실패 → 아이디어 1 자체가 이 서버·5 seed에서 흔들리는 것이므로, HSAN 결과와 무관하게 석휘와 공유하고 HW03 설계를 재검토한다.
- HWP009·010 모두 실패이고 모든 hsan arm Δ ≤ 0 → HSAN 가중치(에피소드 Q, 게이팅 없음)는 이 구성에서 닫고, HW03은 클러스터 게이팅 판으로 간다.
- HWP008 통과 → 이후 arm은 `apply: orig` 고정.
- HWP011 실패(raw가 더 좋음) → HW03의 hardness 공간은 raw 또는 X_smooth로.

## 7. 범위 밖

- 클러스터링(K-means), high-confidence 집합 H, 학습 중 재클러스터링, BGRL(`nbr_bgrl`), exp016 재현, λ_neighbor·τ·lr·epoch 변경, 다른 데이터셋, seed 추가, 원본 모듈·`choi_exps/` 수정.
- 진단값의 해석. 보고서에는 값만.

## 8. 완료 기준과 보고

- [ ] HW02-a 커밋·push (`hw/weights.py`, `hw/run_queue.sh`, summarize `--ref`, `configs/hw.json` arm, 사전 등록 포함), 회귀 확인 표
- [ ] HW02-b 커밋·push (70 run, `summary.*`, `paired_nbr_raw.md`, `paired_naq_base.md`)
- [ ] HW02-c 커밋·push (2단계 run, 갱신된 summary·paired, cos ref 표)
- [ ] G3·G4·G5, 통과 시 태그 `hw02`
- [ ] `reports/HW02_hsan_weighting.md` — CLAUDE.md §7. 핵심 표: (1) arm × ds 5-seed 평균 ± 표준편차 (1·2단계 전부); (2) `paired_nbr_raw` — Δ 평균, 표준편차, 양수 k/5; (3) `paired_naq_base` 같은 형식; (4) 2단계 선택 근거 표; (5) 가중치 진단 — arm·ds별 마지막 구간의 `pos_weight_mean`, `neg_weight_mean`, `pos_weight_lt05_frac`, `pos_s_mean`, `neg_s_mean`(항별); (6) HWP007–013 대조; (7) 제외·NaN run; (8) run당 시간과 총 시간.
