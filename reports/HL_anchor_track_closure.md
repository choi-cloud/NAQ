# Anchor 트랙(스토리 A·B) 종결 문서

작성 2026-10-25 · 결정자 이희태 · 근거 실험 HL01, HL02 (`choi-cloud/NAQ` 브랜치 `hl/anchor`, 태그 `hl01-p0`, `hl02-oracle`, `hl02-embed`)

---

## 1. 결정

**스토리 A(RAS, anchor 신뢰도 스케줄링)와 스토리 B(HNC, hard-negative co-anchoring)를 모두 종결한다.** 결합안(A+B)과 B의 분기 C(ECS, 충돌 없는 배치)도 함께 종결한다.

- 근거: 사전에 Choi와 합의한 go/no-go 규칙 — "GT를 아는 oracle로 anchor를 골랐을 때의 성능 상한이 1%p에 못 미치면 A·B 모두 추진하지 않는다".
- 측정값: max(순도 최고 분위, `hard:4`, `ab`) − `uniform` = **+0.19%p** (Amazon-Clothing, 5-way 1-shot, 5 seed).
- 레이블 없는 방법은 oracle 상한보다 낮을 수밖에 없으므로, 규칙을 바꾸거나 데이터셋을 늘려 1%p를 찾지 않는다.

## 2. 무엇을 검증했나

| 스토리 | 핵심 질문 | oracle 정책 |
|---|---|---|
| A | 어느 노드를 pseudo-class 대표(anchor)로 세우는가 | GT 순도 5분위별 anchor 풀 (`quintile:1`–`5`) |
| B | 어떤 anchor들을 한 에피소드에 함께 놓는가 | feature는 가깝고 필터 탈락 + GT 다른 후보를 co-anchor로 (`hard:2`, `hard:4`), GT 무시판 (`hard_blind:4`) |
| B 분기 C | 같은 클래스 anchor를 피하면 좋아지는가 | GT가 모두 다른 5 anchor (`gt_distinct`) |
| A+B | 둘을 합치면 | 고순도 a₁ + `hard:4` 규칙 (`ab`) |

공통 조건: query는 B_Choi(ÂX·Â²X 코사인 둘 다 τ=0.3 이상 → X 순위 정중앙 10개), 학습·평가는 NaQ 원본 ProtoNet, 고정 valid 50 / test 500, seed 1212–1216.

## 3. 결정의 근거가 된 결과

### 3.1 기준선 (HL01, 3 seed)

| | 1-shot test | 5-shot test | query GT 순도 |
|---|---|---|---|
| NaQ | 80.99 ± 0.41 | 90.78 ± 0.34 | 0.60 |
| B_Choi (τ=0.3) | 82.81 ± 0.38 | 90.18 ± 0.28 | 0.42 |
| 차이 | **+1.82** | **−0.61** | −0.18 |

### 3.2 oracle 사다리 (HL02, 5 seed, uniform = 82.70 ± 0.29)

| 정책 | 1-shot Δ | 5-shot Δ | 비고 |
|---|---|---|---|
| `quintile:1` (최저 순도) | −5.35 | −2.43 | |
| `quintile:2` | −5.29 | −3.23 | |
| `quintile:3` | −3.60 | −1.44 | |
| `quintile:4` | −0.14 | +0.52 | |
| `quintile:5` (최고 순도) | −0.86 | +0.44 | |
| `gt_distinct` | +0.25 | +0.24 | Choi CoraFull 결과 재현 |
| `hard:2` | −0.35 | −0.04 | k_eff 1.28 |
| `hard:4` | −0.58 | +0.40 | k_eff 2.18 |
| `hard_blind:4` | **−2.40** | −0.45 | 에피소드 내 same-GT 쌍 19.4% |
| `ab` | **+0.19** | +0.39 | 최고값 |

### 3.3 B 학습 전 점검 (HL02, B_Choi seed 1212 체크포인트)

| 측정 | 값 | 의미 |
|---|---|---|
| 임베딩 거리 중앙값: 탈락 후보 / 통과 후보 / uniform | 0.35 / 0.04 / 0.78 | 탈락 후보는 임베딩에서도 가깝다 |
| negative 확률 합 평균: uniform / `hard:4` / `hard_blind:4` | 0.17 / 0.32 / 0.42 | 끌어오면 negative가 실제로 어려워진다 |
| uniform에서 negative 확률 합 < 0.05인 query | 49.9% | 절반의 query는 negative에서 거의 배우지 않는다 |

## 4. 해석 — 왜 안 됐나

**A.** 순도 낮은 anchor는 크게 해롭지만(−5%p), 순도 높은 anchor만 골라도 uniform을 넘지 못한다. uniform은 저순도 anchor를 일정 비율 섞고도 q4와 같은 성능을 낸다 — 나쁜 anchor가 일부 섞여도 anchor 다양성이 유지되면 손해가 없다. "더 좋은 대표를 고른다"에는 올라갈 여지가 없다.
(단서: 분위 arm은 anchor 풀을 1/5로 줄이므로 커버리지 손실이 섞여 있다. 상방이 0에 가까워 결론은 바뀌지 않는다.)

**B.** 메커니즘은 설계대로 작동했다 — 탈락 후보는 임베딩에서 가깝고, 끌어오면 negative softmax 확률이 2.4배 오른다. 그런데 거짓 음성을 GT로 완벽히 걸러도(`hard:4`) 정확도는 −0.58이다. **negative를 어렵게 만드는 것 자체가 이 세팅에서는 novel 클래스 일반화로 이어지지 않는다.** 가드 없이 하면(`hard_blind:4`) 거짓 음성 때문에 −2.40으로 확실히 해롭다.

**분기 C.** uniform에서 같은 클래스 anchor 쌍은 2% 수준으로 드물고, 이를 GT로 없애도 +0.25에 그친다(석휘님 CoraFull 결과와 같다).

## 5. 종결 범위 — 닫힌 것과 열린 것

**닫힌 것**
- Amazon-Clothing, B_Choi query 아래에서 anchor의 **선택(marginal)**과 **조합(joint)**을 바꾸는 접근.
- "anchor 순도를 높이면 좋다", "negative를 어렵게 하면 좋다", "같은 클래스 anchor를 피하면 좋다"의 세 가설.

**닫히지 않은 것 (이번 실험이 말하지 않음)**
- 다른 데이터셋(dblp, Electronics)에서의 anchor 효과 — 재개 근거가 생기기 전에는 돌리지 않는다.
- query 쪽 자유도(석휘님 트랙), 손실·표현 쪽 자유도.
- B_Choi의 5-shot 손실의 원인.

## 6. 다음 스토리로 가져갈 사실

1. **B_Choi의 이득은 1-shot 전용이다.** NaQ 대비 1-shot +1.82, 5-shot −0.61. 석휘님 방법을 논문화할 때 반드시 설명해야 할 지점이다.
2. **B_Choi는 query 순도를 낮추면서 1-shot을 올렸다**(0.60 → 0.42, 선택 query의 평균 X 순위 ≈ 750위). positive 쪽에서도 "순도"보다 "선택 위치·난이도"가 레버라는 석휘님 19번 문서의 결론과 일치한다.
3. **저순도 anchor의 하방은 크다**(−5%p). 그러나 uniform에 섞인 정도로는 문제가 되지 않는다.
4. **feature는 가깝고 구조가 다른 쌍은 같은 클래스일 확률이 무작위의 10배**다(21.8% vs 2.0%). 이런 쌍을 negative로 쓰는 설계는 가드가 필수다.
5. **uniform 에피소드에서 query의 절반은 negative에서 거의 배우지 않는다**(Σp_neg < 0.05). 그러나 이것을 고쳐도 정확도는 오르지 않았다.
6. 필터 통과 수는 degree와 상관이 있다(ρ = 0.35). 필터 기반 신뢰도 점수를 만들면 degree 교란을 통제해야 한다.

## 7. 실험 운영상 알아둘 것

- **seed가 달라도 평가 에피소드는 같다.** NaQ 원본 `downstream_task_generator`가 `save/downstream_tasks/{dataset}/5way_{shot}shot.pkl` 캐시를 seed와 무관하게 재사용한다. HL01·HL02의 seed 간 분산은 학습 쪽만 반영한다. 정책 간 쌍비교에는 유리하지만, 결과는 "평가 에피소드 500개 한 벌" 위의 수치다.
- exp049의 query 선택은 mining 배치 구성에 따라 동률 노드에서 미세하게 달라질 수 있다(부동소수 1e-7 수준).
- 같은 설정 재학습 시 1-shot test가 최대 0.06–0.09%p 흔들린다.
- run당 약 50초, GPU 메모리 8.2GB(RTX 4080 SUPER 1장에서 동시 실행 1개).

## 8. 사전 등록 대조 (기계적 대조, 최종 판정은 이희태)

| 번호 | 예측 | 계산값 | 대조 |
|---|---|---|---|
| HLP001 | smoke NaQ 1-shot이 80.66 ± 0.5 안 | Δ = −0.02 | 맞음 |
| HLP002 | τ에 대해 비단조, 최적 0.3 | 76.25 / 76.60 / 74.50 | 맞음 |
| HLP003 | B_Choi − NaQ ≥ +0.5 (1-shot) | +1.82, 3 seed 모두 양수 | 맞음 |
| HLP004 | 5-shot 차이 < 1-shot 차이 | +1.82 vs −0.61 | 맞음 |
| HLP005 | B_Choi query 순도 > NaQ | 0.42 vs 0.60 | **틀림** |
| HLP006 | fallback 없는 커버리지 ≥ 0.8 | 0.965 | 맞음 |
| HLP007 | pi_bchoi ≈ 0.42, pi_naq ≈ 0.60 | 0.417 / 0.597 | 맞음 |
| HLP008 | ρ(통과 수, degree) ≥ 0.3 | 0.349 | 맞음 |
| HLP009 | ρ(통과 수, 순도) < 0 | −0.167 | 맞음 |
| HLP010 | 탈락 same-GT < 통과의 절반, > 무작위 | 0.218 / 0.574 / 0.020 | 맞음 |
| HLP011 | anchor GT 모두 다름 ≥ 0.75 | 0.815 | 맞음 |
| HLP012 | query 겹침 에피소드 ≥ 0.1 | 0.041 | **틀림** |
| HLP013 | 5분위 곡선 폭 ≥ 1.0 | 5.20 | 맞음 |
| HLP014 | q5 − uniform ≤ +2.0, 비단조 | −0.86, q4 > q5 | 맞음 |
| HLP015 | gt_distinct − uniform ±0.4 안 | +0.25 | 맞음 |
| HLP016 | hard:4 − uniform ≥ +1.0 (B go) | −0.58 | **틀림** |
| HLP017 | hard:4 − hard_blind:4 ≥ +0.5 | +1.81 | 맞음 |
| HLP018 | hard:2가 uniform과 hard:4 사이 | 82.12 < 82.35 < 82.70 | 맞음 |
| HLP019 | ab ≥ max(q5, hard:4) − 0.3 | 82.89 ≥ 81.82 | 맞음 |
| HLP020 | 모든 정책에서 Δ_5shot < Δ_1shot | 다수 정책에서 반대 | **틀림** |
| HLP021 | 탈락 쌍 거리 < uniform 쌍 거리 | 0.35 < 0.78 | 맞음 |
| HLP022 | uniform Σp_neg 중앙값 < 0.10 | 0.050 | 맞음 |
| HLP023 | hard_blind:4 Σp_neg > uniform × 1.5 | × 2.43 | 맞음 |

틀린 넷 중 결정에 영향을 준 것은 **HLP016(B의 go 기준)** 하나다. 나머지 셋은 진단·부수 예측이다.

## 9. 산출물 보관 위치

**설계 문서 (보관, 더 이상 갱신하지 않음)**
- `story_A_RAS_sketch_v2.md`, `story_B_HNC_sketch_v1.md`
- `GFSL_storyAB_review_request_for_ChatGPT_2026-10-23.md` 및 ChatGPT 회신 두 건
- `choi_meeting_questions_1024.md`, `story_A_overview_v2.png`

**리포 (`choi-cloud/NAQ`, 브랜치 `hl/anchor`)**
- 보고서: `reports/HL01_env_smoke_p0.md`, `reports/HL02_diag_oracle.md`, 표 `reports/HL0{1,2}_tables/*.csv`
- 코드: `hl/hl000_run.py`, `hl/hl000b_run_oracle.py`, `hl/hl_common.py`, `hl/hl002_diag_nodes.py`, `hl/hl003_oracle_train.py`, `hl/hl004_embed_check.py`
- 결과·체크포인트(서버, git 미추적): `results/hl/p0/`, `results/hl/hl02/`
- 재사용 가능한 자산: 전체 노드 진단표 `results/hl/hl02/diag/node_table_tau0.3.csv`(노드별 degree·필터 통과 수·B_Choi/NaQ query 순도), anchor 정책을 바꿔 끼울 수 있는 학습 스크립트 `hl003`.
