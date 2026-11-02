# 스토리 D(순도·거리 분리 query 선택) 종결 문서

작성 2026-11-02 · 결정자 이희태 · 근거 실험 HL06–HL10 (`choi-cloud/NAQ` 브랜치 `hl/anchor`, 태그 `hl06-diag`·`hl06-train`·`hl07`·`hl08-base`·`hl08`·`hl09-base`·`hl09`·`hl10`, 야간 시퀀스 2026-10-25 23:35 → 10-26 02:04 KST, 마감 보고 `reports/OVERNIGHT_2026-10-26.md`)
앞선 종결: 스토리 A·B — `reports/HL_anchor_track_closure.md` (HL01–HL02); 스토리 C — `reports/HL_storyC_closure.md` (HL03–HL05); ND 트랙(별도 브랜치 `nd/anchor-vote`) — `reports/ND_closure.md`(그 브랜치), 프로젝트 docs `claude/ND_closure_2026-11-02.md`
표기: [사실] = 측정값, [판정] = 사전 등록 대조, [해석] = 이희태 판단, [미확인] = 재지 않은 것.

---

## 1. 결정

**스토리 D — "판정기 선택을 Â²X 층 안으로 제한해 극단적 근접을 피하면, 같은 판정기의 top-10보다 1-shot이 오르고 B_Choi를 넘는다"(가설 H-D) — 를 종결한다.**

- 사전 등록 판정 규칙(HL06 §3): "HLP078 틀림 → H-D 기각. Clothing에서 레이블 없는 FP 억제 노선을 닫고 분석 문서를 쓴다." 측정값: `encb_strat5` − `encb_top10` = **+0.51**(기준 +2.0). 규칙대로 기각한다.
- 규칙에 딸린 단서: HLP070이 틀렸으므로(`o_combswap` − `comb_top10` = +1.09, 기준 ±1.0) 잔차 R1 해석에 "FP 종류" 단서를 붙인다(§5). HLP069는 맞았으므로(`o_topsame` 84.34 ≤ 84.49) 극단적 근접만으로 `comb_top10`의 손실 일부가 설명된다.
- 이 문서는 HL 트랙의 레이블 없는 query 선택 노선(스토리 C·D) 전체를 닫는다. 남는 것은 §6·§7.

## 2. 무엇을 검증했나

| 단계 | 질문 | 방법 | 결과 |
|---|---|---|---|
| HL06 §1 | 판정기 12개 × 거리 규칙 7개로 고른 query의 순도 | B4(X 순위 201–1000), anchor 2,000개, 채점 | strat5·ex20 순도 최고 0.35(`enc_n` ex20). **게이트 G3(0.45) 실패**. 새 판정기 4개는 `enc_b` 대비 ≤ +0.005 |
| HL06 §2 | 층 제한이 top-10보다 나은가(H-D) | 레이블 없는 6 arm × 5 seed(+ 여분 5 seed) | `encb_strat5` 80.20 vs `encb_top10` 79.69: **+0.51**. 10-seed 80.05 vs 79.61: +0.44 |
| HL07 §1 | (순도 × 구조적 거리) 2D oracle 격자 | 3 region × 4 순도 × 5 seed | 모든 region에서 순도에 단조 증가. **p=1.0: strat5 90.12 > far50 89.67 > t1 87.20**; p=0.3: strat5 84.45 > t1 82.68 > far50 79.49 |
| HL07 §2 | 변형·통제 | 7 arm × 5 seed | Choi 필터 + 거리 통제 ±0.2; FP 제거(`drop2`) −1.59; teacher 갱신(`r2`) −2.25 |
| HL08·HL09 | dblp·electronics 전이 | 기준선 + 진단 + oracle 4 + 레이블 없는 5 arm × 5 seed | oracle 레버 재현(dblp p1.0 +5.95, far−near +3.38; elec +4.74, +1.53). 레이블 없는 최고 dblp −2.71, elec −0.45 |
| HL10 | Clothing 층·판정기 변형 | 8 arm × 5 seed | strat3/10, ex10/40, nbr/clu/xidf 모두 B_Choi −1.7 ~ −5.2 |

공통 조건: anchor uniform, NaQ 원본 ProtoNet, 고정 valid 50 / test 500, seed 1212–1216(5), gpusystem(RTX 6000 Ada, 환경 `NAQ`). 기준선 B_Choi: Clothing 82.71 ± 0.31 / 89.91 ± 0.57, dblp 71.01 ± 0.30 / 80.94 ± 0.34, electronics 70.26 ± 0.53 / 82.93 ± 0.49.

## 3. 핵심 수치 한 표 (1-shot / 5-shot)

| 조건 | Clothing | dblp | electronics |
|---|---|---|---|
| NaQ (X top-10) | 80.65 / 90.52 | 68.51 / 81.96 | 68.93 / 83.68 |
| **B_Choi (필터 + middle)** | **82.71 / 89.91** | **71.01 / 80.94** | **70.26 / 82.93** |
| 레이블 없는 최고 arm | `knnppr_ex20` 81.93 / 90.01 | `encb_ex20` 68.30 / 79.26 | `choi_far` 69.81 / 82.33 |
| `encb_top10` | 79.69 / 89.11 | 68.21 / 81.17 | 67.72 / 81.47 |
| `encb_strat5` | 80.20 / 87.60 | 67.41 / 78.97 | 69.50 / 81.07 |
| Choi 필터 + 거리 통제 (`choi_far`) | 82.53 / 89.44 | 68.01 / 78.38 | 69.81 / 82.33 |
| oracle 순도 0.4 (`g1_p04`) | 83.92 / 91.12 | 73.37 / 83.54 | 70.71 / 83.26 |
| oracle 순도 1.0 (`g1_p10`) | 89.42 / 94.17 | 76.96 / 85.42 | 75.00 / 85.98 |
| oracle same-class near / far | 85.49 / 89.46 | 74.31 / 77.69 | 73.33 / 74.86 |
| oracle 순도 1.0 **strat5** (HL07) | **90.12 / 94.74** | — | — |

## 4. 확인된 사실 (다음 연구의 전제로 써도 됨)

**D1. H-D 기각.** 층 제한(strat5)은 top-10보다 Clothing +0.51, dblp −0.80, electronics +1.78. 세 데이터셋 모두 B_Choi 미달(−2.51 / −3.60 / −0.76). 10-seed로 늘려도 같다(HL06 여분).

**D2. 레이블 없는 query 선택은 어떤 형태로도 B_Choi를 넘지 못했다.** 판정기 12종 × 규칙 7종 × 변형(층 수, 제외 폭, teacher 갱신, FP 제거, 넓은 구간, 2-hop·군집 판정기) 전부. 가장 가까운 것이 Clothing `knnppr_ex20` −0.78(10-seed −0.90). 스토리 C의 C6을 더 넓은 범위에서 재확인.

**D3. [C2 정정] 레버는 "구조적으로 먼 positive"가 아니라 "구조적으로 다양한 positive"다.** 스토리 C 문서의 C2는 "먼 same-class가 가까운 것보다 +4.0"이었다. HL07 격자는 이를 바꾼다: 순도 1.0에서 strat5(다섯 층에서 고루) 90.12 > far50(먼 쪽만) 89.67 > t1(가장 가까운 층만) 87.20이고, 순도 0.3에서는 far50이 **가장 나쁘다**(79.49 < t1 82.68 < strat5 84.45). 먼 positive는 순도가 높을 때만 이롭고, 어느 순도에서나 가장 좋은 것은 층에 고루 퍼진 선택이다. HL05의 far(89.46)는 strat5에 못 미친다.

**D4. 순도는 어느 거리 영역에서든 단조 레버다.** HL07 세 region 모두 p 0.3 → 1.0에서 단조 증가(HLP085). 순도 0.3이라도 strat5면 B_Choi +1.74(HLP084). oracle 상한은 Clothing 1-shot 90.12 / 5-shot 94.74.

**D5. 판정기의 순도 벽.** B4에서 AUROC 0.74–0.80인 판정기로 10개를 고르면 strat5·ex20 순도 0.32–0.35, far50 0.24(HL06 P1). Choi 필터(0.42)에 못 미친다. 새 판정기(TF-IDF 특징, 임베딩 이웃 평균, 임베딩 kNN PPR, k-means 공동 소속)는 `enc_b`와 ±0.013 안(HLP063). dblp에서는 더 낮다(strat5 0.107, HLP102).

**D6. Choi 필터 위에 거리 통제를 더해도 얻는 것이 없다.** `choi_far` −0.18, `choi_strat` −0.14(Clothing); dblp −3.00; electronics −0.45. 필터가 이미 하는 일을 거리 규칙이 다시 하는 것이고, 필터 통과분을 줄이면 손해다.

**D7. FP를 정확히 빼도, teacher를 갱신해도 나빠진다.** `encb_strat5_drop2`(손실 상위 2개 제외, 제외분의 77.5%가 실제 FP) −1.59. `encb_strat5_r2`(teacher 1회 갱신) −2.25. [해석] 손실이 큰 query는 FP라도 학습 신호를 주고 있고, 자기 학습식 갱신은 편향을 키운다. C5("필터 통과 FP는 덜 해롭다")와 같은 방향.

**D8. oracle 레버는 세 데이터셋에서 재현되고, 판정기 실패도 재현된다.** 순도 1.0 − 0.4: Clothing +5.5, dblp +3.6, electronics +4.3. far − near: +4.0 / +3.4 / +1.5. 레이블 없는 최고 arm − B_Choi: −0.78 / −2.71 / −0.45. B_Choi − NaQ(1-shot): +2.06 / +2.50 / +1.33, 5-shot은 세 곳 모두 B_Choi가 낮다(−0.61 / −1.02 / −0.75). dblp의 B4 기본 순도는 0.047(Clothing 0.18)로 판정기가 일할 여지가 더 작다.

**D9. 층 수·제외 폭은 둔감하다.** strat3/5/10 ±0.5(HLP113), ex10/20/40 ±0.7(HL10). 규칙 세부 조정으로 바뀔 폭이 아니다.

## 5. 설명되지 않은 것

**잔차 R1(스토리 C에서 이월) — "FP 종류" 단서 추가.** HLP070이 틀렸다: `comb_top10`의 FP를 다른 B4 GT-diff로 무작위 교체한 `o_combswap`은 +1.09(79.41 → 80.50). 즉 `comb_top10` 손실의 일부(약 1%p)는 "어떤 FP인가"에서 온다 — 판정기가 고르는 FP는 무작위 FP보다 해롭다. 나머지(80.50 → near oracle 85.49)는 설명되지 않은 채 남고, HLP069(`o_topsame` 84.34 ≤ 84.49)는 극단적 근접이 그 일부임을 가리킨다. [미확인] 두 효과의 정확한 분해.

**잔차 R3(새로움).** 레이블 없는 arm의 5-shot이 1-shot보다 더 나쁘다(Clothing `encb_strat5` 5-shot −2.31, dblp −1.97, elec −1.86). C4는 "불순도가 5-shot을 깎는다"였는데, 판정기 arm의 순도(0.32)가 B_Choi(0.42)보다 낮으니 방향은 맞지만, 순도 0.18인 `rand10`의 5-shot(−1.09)보다도 나쁘다. [미확인] 순도 외의 요인(선택된 10개의 중복도 등).

**잔차 R4.** electronics만 HLP108(strat5 − top10 ≥ +1.5)을 통과했다(+1.78). 기제가 데이터셋에 따라 작동 여부가 갈리는 이유는 재지 않았다.

## 6. 종결 범위

**닫힌 것**
- 레이블 없는 판정기 + 거리 규칙으로 query를 고르는 모든 변형(C·D). 판정기 12종, 규칙 7종, 층 수·제외 폭 변형, teacher 갱신, FP 제거, Choi 필터 결합.
- (ND 트랙, 별도 종결) 이웃 클래스 분포·투표를 특징이나 same-class 점수로 쓰는 것.

**열린 것**
- 레버 자체는 확정되었다: **순도 × 구조적 다양성**(D3·D4), oracle 상한 Clothing 90.12. 이것을 레이블 없이 당기는 신호는 아직 없다. 시험한 신호가 모두 구조적 근접성 또는 특징 유사도를 재기 때문이다(C 문서 §6과 같은 결론).
- **학습 없는 Â²X 기준선(ND01 부산물).** 같은 test 500 에피소드에서 Â²X 프로토타입만으로 **85.10 / 92.72** — NaQ·B_Choi를 넘고, oracle 상한(90)과의 차이가 5%p다. [해석] 현재의 어떤 레이블 없는 에피소드도 "학습이 정보를 더한다"는 선을 못 넘는다. 다음 작업은 이 기준선을 통제(차원, 정규화)와 함께 확정하고, 학습이 Â²X 정보를 언제 잃는지(학습 곡선의 epoch 0부터)를 보는 것이다 — HL11.
- dblp·electronics의 같은 확인은 gpusystem 복구 뒤.

## 7. HL 트랙 전체 상태 (2026-11-02)

| 트랙 | 결론 | 근거 |
|---|---|---|
| 스토리 A (anchor 신뢰도) | 종결 | oracle 상한 ≤ 0 (HL02) |
| 스토리 B (hard co-anchoring) | 종결 | oracle-safe −0.58 (HL02) |
| 스토리 C (먼 positive 순도 회복) | 종결 | 게이트 실패, 학습 −1.84 (HL04) |
| 스토리 D (순도·거리 분리) | 종결 | HLP078 +0.51 < +2.0; 레이블 없는 arm 세 데이터셋 미달 (HL06–10) |
| ND (이웃 분포 패턴, 별도 브랜치) | 종결 | Â²X 위 이득 없음, 호모필 커버리지 1/3 (ND01–02) |
| 확정된 레버 | 순도(D4) × 구조적 다양성(D3) | oracle 상한 90.12 / 94.74 |
| 확정된 기준선 | 학습 없는 Â²X 85.10 / 92.72 (ND01) | 통제 미완 → HL11 |
| 미해결 | 레버를 레이블 없이 당기는 신호; 학습이 Â²X 정보를 잃는 기제 | — |

## 8. 사전 등록 대조 (HLP057–118, 기계적 대조; 계산값은 `reports/OVERNIGHT_2026-10-26.md` §3)

| 번호 | 예측(요약) | 계산값 | 대조 |
|---|---|---|---|
| HLP057 | `rand_far50` 순도 ≤ 0.10 | 0.088 | ✓ |
| HLP058 | `rand_ex20` 순도 ≤ 0.16 | 0.127 | ✓ |
| HLP059 | `enc_b` auroc_far50 ≥ 0.65 | 0.744 | ✓ |
| HLP060 | `enc_b` strat5 순도 ∈ [0.33, 0.45] | 0.324 | ✗ (근소) |
| HLP061 | `enc_b` far50 순도 ∈ [0.20, 0.33] | 0.242 | ✓ |
| HLP062 | 게이트 G3: 어느 판정기든 strat5/ex20 순도 ≥ 0.45 | 최고 0.350 | ✗ |
| HLP063 | 새 판정기 ≥ `enc_b` + 0.03 | 최고 +0.005 | ✗ |
| HLP064 | `xidf` ≥ `x` + 0.03 (far50) | +0.023 | ✗ |
| HLP065 | `comb_top10` mean_s2 ≥ 0.70 | 0.696 | ✗ (근소) |
| HLP066 | pair_s2 comb ≥ 0.60 & bchoi ≤ 0.40 | 0.759 / 0.607 | ✗ (bchoi) |
| HLP067 | comb hop2 ≥ 2 × bchoi | 0.588 vs 0.485 | ✓ |
| HLP068 | comb fp_s2 ≥ p04_choidiff | 0.665 vs 0.446 | ✓ |
| HLP069 | `o_topsame` ≤ 84.49 | 84.34 | ✓ |
| HLP070 | \|`o_combswap` − 79.41\| ≤ 1.0 | +1.09 | ✗ |
| HLP071 | `o_far50_p05` ≥ 85.0 | 84.15 | ✗ |
| HLP072 | `o_far50_p03` ≥ 83.21 | 79.49 | ✗ |
| HLP073 | `encb_strat5` − B_Choi ≥ +1.0 | −2.51 | ✗ |
| HLP074 | `encb_ex20` − B_Choi ≥ +1.0 | −2.69 | ✗ |
| HLP075 | `encb_far50` < `encb_strat5` | −1.36 | ✓ |
| HLP076 | `knnppr_strat5` ≥ `encb_strat5` | +1.01 | ✓ |
| HLP077 | 최고 arm 5-shot ≥ 89.91 | 90.01 | ✓ |
| **HLP078** | **핵심: `encb_strat5` − `encb_top10` ≥ +2.0** | **+0.51** | **✗** |
| HLP079 | `encb_top10` ≈ `encn_top10` ±1.0 | −0.25 | ✓ |
| HLP080 | `o_grid_t1_p10` ≤ 84.5 | 87.20 | ✗ |
| HLP081 | `o_grid_strat5_p10` ≥ 88.5 | 90.12 | ✓ |
| HLP082 | `o_grid_far50_p10` ≥ 89.0 | 89.67 | ✓ |
| HLP083 | p=0.5 strat5 − t1 ≥ +3.0 | +3.06 | ✓ |
| HLP084 | `o_grid_strat5_p03` ≥ 82.21 | 84.45 | ✓ |
| HLP085 | 세 region 모두 p에 단조 | 모두 True | ✓ |
| HLP086 | `rand_strat5` ≤ 81.34 | 79.49 | ✓ |
| HLP087 | `encb_strat5` − `rand_strat5` ≥ +2.0 | +0.71 | ✗ |
| HLP088 | `choi_far` − B_Choi ≥ +0.5 | −0.18 | ✗ |
| HLP089 | `choi_strat` − B_Choi ≥ +0.5 | −0.14 | ✗ |
| HLP090 | \|`encb_strat5_wide` − strat5\| ≤ 1.0 | +0.12 | ✓ |
| HLP091 | `encb_strat5_r2` ≥ strat5 | −2.25 | ✗ |
| HLP092 | `drop2` ≥ strat5 − 0.3 & 제외 FP ≥ 0.7 | −1.59 / 0.775 | ✗ (성능) |
| HLP093 | dblp B_Choi − NaQ > 0 | +2.50 | ✓ |
| HLP094 | dblp B4 기본 순도 ≤ 0.18 | 0.047 | ✓ |
| HLP095 | dblp `enc_b` auroc ≥ 0.70 | 0.783 | ✓ |
| HLP096 | dblp g1_p10 − g1_p04 ≥ +3.0 | +3.59 | ✓ |
| HLP097 | dblp far − near ≥ +2.0 | +3.38 | ✓ |
| HLP098 | dblp strat5 − top10 ≥ +1.5 | −0.80 | ✗ |
| HLP099 | dblp strat5 − B_Choi ≥ +0.5 | −3.60 | ✗ |
| HLP100 | dblp `choi_far` − B_Choi ≥ 0 | −3.00 | ✗ |
| HLP101 | dblp `rand_strat5` ≤ B_Choi | −3.04 | ✓ |
| HLP102 | dblp `enc_b` strat5 순도 ∈ [0.25, 0.45] | 0.107 | ✗ |
| HLP103 | elec B_Choi − NaQ > 0 | +1.33 | ✓ |
| HLP104 | elec B4 기본 순도 ≤ 0.18 | 0.130 | ✓ |
| HLP105 | elec `enc_b` auroc ≥ 0.70 | 0.745 | ✓ |
| HLP106 | elec g1_p10 − g1_p04 ≥ +3.0 | +4.29 | ✓ |
| HLP107 | elec far − near ≥ +2.0 | +1.53 | ✗ |
| HLP108 | elec strat5 − top10 ≥ +1.5 | +1.78 | ✓ |
| HLP109 | elec strat5 − B_Choi ≥ +0.5 | −0.76 | ✗ |
| HLP110 | elec `choi_far` − B_Choi ≥ 0 | −0.45 | ✗ |
| HLP111 | elec `rand_strat5` ≤ B_Choi | −2.81 | ✓ |
| HLP112 | elec `enc_b` strat5 순도 ∈ [0.25, 0.45] | 0.278 | ✓ |
| HLP113 | strat10·strat3 ≈ strat5 ±1.0 | +0.47 / −0.15 | ✓ |
| HLP114 | ex10 < ex20 | −0.72 | ✓ |
| HLP115 | ex40 ≤ ex20 + 0.5 | −0.72 | ✓ |
| HLP116 | `nbr_strat5` ≥ strat5 − 0.5 | +0.85 | ✓ |
| HLP117 | `xidf_strat5` ≤ strat5 − 1.0 | −2.70 | ✓ |
| HLP118 | `clu_strat5` ≈ strat5 ±1.5 | +0.82 | ✓ |

통과 38 / 미달 24. 미달의 분포: 레이블 없는 arm의 성능 예측(073·074·078·087–089·091·092·098–100·109·110) 14개, 판정기 순도 예측(060·062–066·102) 7개, oracle 영역 예측(071·072·080·107) 4개 — 070은 단서. 즉 **oracle 쪽 예측은 대체로 맞았고 레이블 없는 쪽 예측은 대체로 틀렸다.**

## 9. 산출물 보관 위치

- 보고서·표(커밋): `reports/HL06_purity_distance_split.md`–`HL10_clothing_fillers.md`, `reports/HL0{6,7,8,9}_tables/`, `reports/HL10_tables/`, `reports/OVERNIGHT_2026-10-26.md`, `reports/overnight_progress.md`.
- 코드(커밋): `hl/hl_pdd.py`, `hl/hl_pdd_diag.py`, `hl/hl010_diag_pdd.py`, `hl/hl011_pdd_train.py`, `hl/hl000f_run_hl06.py`, `hl/hl_hl0{6,7,8,10}_tables.py`.
- 비추적(**gpusystem에만**): `results/hl/hl06/`–`hl10/`, `results/hl/hl03/g0/anchors.npy`, dblp·electronics 고정 에피소드와 `save/downstream_tasks/{dblp,Amazon_eletronics}/*.pkl`(md5는 HL08·HL09 보고서 ①).
- 이 문서: `reports/HL_storyD_closure.md`, 커밋 `HL: story D closure doc`, 태그 `hl-storyD-closure`. 프로젝트 docs 사본 `claude/HL_storyD_closure_2026-11-02.md`.
