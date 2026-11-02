# ND 트랙 종결 문서 — 이웃 분포 패턴(투표 분포 d_v, inter-class profile)

작성 2026-11-02 · 이희태 · 브랜치 `nd/anchor-vote` · 종결 HEAD `99ad277`
근거: `reports/ND01_vote_gates.md`(HEAD `05cd2c3`), `reports/ND02_inter_profile.md`(HEAD `99ad277`), 표 `reports/ND01_tables/`, `reports/ND02_tables/`. 학습 기준선은 `hl/anchor`의 `reports/HL03_tables`(T0)·`HL01_tables`.
표기: [사실] = 측정값, [판정] = 사전 등록 대조, [해석] = 이희태 판단, [미확인] = 재지 않은 것.

## 0. 결론

ND 트랙은 **종결**한다(2026-11-02). 두 지시서(ND01·ND02, 모두 학습 없음, Amazon_clothing)로 확인한 것:

- 자기 그룹 성분을 포함한 이웃 투표 분포 d_v는 Â²X 프로토타입 위에 분류(meta-test)·노이즈 점수(B4 AUROC) 어느 쪽으로도 이득이 없다(ND01).
- 자기 그룹 성분을 뺀 inter-class 연결 패턴은 Clothing에 **존재하고** pseudo-task에서도 보이지만, 노드의 1/3에만 정의되고, 노이즈 판별로는 호모필 비율만도 못하며, Â²X에 더해지는 정보가 없다(ND02).
- 부산물로, 학습 없는 Â²X 프로토타입이 NaQ·B_Choi 학습 모델을 넘는다(ND01 관문 A). 이 사실은 ND와 별개로 `claude/share_choi_a2x_baseline_2026-11-02.md`에 정리해 Choi와 공유한다.

## 1. 트랙이 물었던 것

배경(2026-11-02): VISION·NaQ식 pseudo-task(anchor top-(K+M) 유사도 샘플링)는 같은 pseudo-label로 묶기엔 신뢰도 낮은 샘플이 섞이는 noisy-label 문제로 볼 수 있다(GAPNet의 IE noise에 대응). DCGC(2025)는 pseudo-label보다 이웃 분포가 더 안정적인 감독 신호임을 클러스터링에서 보였고, heterophily 문헌은 이웃 클래스 분포 패턴이 중심 노드 분류에 유용하다고 본다. 질문: 이 관찰을 episodic·비지도 설정으로 옮기면, pseudo-class 그룹의 식별이나 잘못 들어온 노드의 탐지에 쓸 수 있는가.

- ND01: d_v = normalize(Σ_{u∈N(v)} softmax(cos(f_u, anchors)/τ)) — (A) meta-test 두 번째 분류 기준, (B) B4 후보 same-class 점수.
- ND02: 자기 그룹 슬롯을 뺀 inter-profile — Q1 pseudo-label 식별, Q2 노이즈 query 탐지, Q3 같은 GT 쌍의 profile 유사성.

## 2. 확인된 사실

### N1. [사실] 학습 없는 Â²X 프로토타입이 학습 모델을 넘는다 (ND01 관문 A)
고정 test 500 에피소드, 분류기 동일(support 평균 프로토타입, 제곱 유클리드 최근접). `feat_a2x`(Â²X 행 정규화) **85.10 / 92.72**(1-shot / 5-shot); `feat_x` 72.35 / 87.61. 같은 에피소드에서 NaQ 80.65 / 90.52, B_Choi(τ=0.3, 5 seed) 82.71 / 89.91. [미확인] 차원(9,034 vs 256) 통제, 학습 임베딩 정규화 후 재평가, dblp·electronics.

### N2. [사실] d_v는 Â²X 위에 이득이 없다 (ND01 관문 A)
Â²X 위에 d_v를 결합한 변형 전부 `feat_a2x` ±0.5 안. valid로 고른 `comb_a2x_sel` −0.16(paired CI ±0.12, 승률 0.08). X 위에서는 `comb_x_sel` +7.74이나 이는 구조 유무 비교이고, 결과 80.09는 `feat_a2x` 85.10에 못 미친다. [판정] NDP001·002 문구상 통과, 기준선 선택(같은 f의 feat) 오류로 실질 미달 — 사전 등록 문구는 바꾸지 않고 여기 기록.

### N3. [사실] 투표(배정 후 집계)와 평균(집계 후 비교)은 같은 정보다 (ND01 관문 B)
B4 구간(X 순위 201–1000), 400 에피소드 × 5 anchor. AUROC_all: `a2x` 0.753, `enc_b` 0.775, `vote_x_0.1_1hop` 0.675(a2x 대비 −0.078, 1,992행 중 25%만 우세), X 계열 최고 `vote_x_0.2_a2` 0.703. f=`a2x`에서 `own_soft_a2x_0.2` 0.753 ≈ `vote_a2x_0.2_self` 0.754 ≈ `a2x` 0.753 — 구조가 든 특징 위에서는 anchor 경쟁(softmax)도 이웃 투표도 0.002 이내. [판정] NDP007 미달, NDP008 통과(X 위에서 이웃 집계가 +0.080 — 그러나 선형 집계 `ax` 0.728이 더 낫다).

### N4. [사실] Clothing은 inter-class 연결이 희박하다 (ND02 T1·T5)
주 설정(pseudo_k3, X 배정, 1-hop): 호모필 비율 h 평균 0.86, n_inter 중앙값 0. query의 **67%는 다른 그룹으로 가는 엣지가 없다**. inter-profile이 정의되는 support 23%, query 33%, 패턴이 있는 그룹 39%. Â²X로 배정하면 정의 비율 5%(h 0.94). [판정] NDP011 미달(0.33 < 0.5).

### N5. [사실] inter 패턴은 존재하고 pseudo-task에서도 보인다 (ND02 Q1)
inter 엣지가 있는 query만 보면 pseudo-label 식별 정확도 0.65–0.68(우연 0.20; n_inter=0 포함 전체 0.509). 엣지 섞기(degree 보존)에서 0.216. GT 그룹 oracle 에피소드에서 0.519(pseudo 0.509와 +0.01 차이). 변형: K=5 0.368, B_Choi query 0.555, 미배정 q30 0.593, 2-hop 가중 0.507, 분산 가중 0.501. inter 엣지 수가 많을수록 좋아지지 않는다(≥5: 0.648 < 1–2: 0.678). [판정] NDP012·013·020 통과, NDP021·022 미달. [해석] pseudo-label 품질이 패턴을 흐리는 것이 아니다 — Clothing의 inter 패턴이 원래 이 정도이고, 엣지 1–2개로 "어느 그룹과 닿아 있나"를 보는 수준 이상의 정보는 없다.

### N6. [사실] 노이즈 점수로는 호모필 비율보다 못하고 Â²X에 더해지지 않는다 (ND02 Q2)
주 설정, inter 엣지가 있는 query 2,991개, AUROC(에피소드 행별 평균): `s_own`(inter-profile 일치도) 0.583, `h`(호모필 비율만) 0.597, `x` 0.671, `a2x` 0.726, `comb`(s_own+a2x 순위 평균) 0.684. 변형 최고 `nbr_a2`(2-hop, 정의 비율 80%): s_own 0.629 < a2x 0.742, comb 0.705 < a2x. [판정] NDP017·018·019 모두 미달. [해석] 실용 가치를 결정하는 결과. "GT가 anchor와 다른 query"를 가려내는 데 inter-profile은 쓸 수 없고, 결합하면 오히려 깎인다.

### N7. [사실] 쌍 유사성의 신호는 약하고 일부는 정의의 기하학적 효과다 (ND02 Q3)
`cos_inter5` AUROC_all 0.707, 엣지 섞기 0.619(자기 슬롯을 0으로 두는 정의 때문에 같은 그룹 쌍의 코사인이 구조와 무관하게 높아지는 효과; 구조가 더한 몫 +0.09). 특징 유사도 분위별: q1–q3 0.46–0.50(신호 없음), q4(특징이 비슷한 쌍)만 0.678. 다른 그룹 쌍 0.427(<0.5; anchor GT 충돌 — 에피소드의 18% — 때 두 그룹이 서로의 슬롯에 질량을 몰아 반대 방향이 되는 효과). GT oracle에서는 충돌이 없어 q1–q3도 0.63–0.67. 비교: 자기 슬롯을 포함한 `cos_full5`(DCGC의 e_i) 0.908(섞기 0.531), `cos_a2x` 0.962(섞기 0.763). [판정] NDP014·015·016 문구상 통과 — NDP015는 "특징이 비슷한 쌍에서도"가 아니라 "에서만"이 맞다. [해석] 패턴은 있으나 자기 슬롯을 포함한 분포보다 훨씬 약하고, 그 분포도 Â²X 코사인이 넘는다.

### N8. [사실] pseudo-task 통계 (ND02 T0)
top-(K+M) pseudo 에피소드(K=3, M=10, 300개): support 순도 0.72(anchor 제외), query 순도 0.57(NaQ와 일치), anchor GT 충돌이 있는 에피소드 18%. B_Choi query 순도 0.41(HL 수치와 일치).

## 3. 종결 사유

사전 등록 분기(ND02 §12)로는 NDP020·012·014·016 통과("패턴이 보인다")와 NDP017 미달("종결")이 동시에 해당한다. 두 결과를 합친 판단:

- 패턴은 존재하지만 **정의되는 노드가 1/3**(N4)이고, 정의되는 노드에서도 **노이즈 판별 정보가 호모필 비율 이하**(N6)이며, **Â²X에 더해지는 정보가 없다**(N2·N3·N6·N7).
- 실용 용도 두 가지(두 번째 분류 기준, 노이즈 점수)가 모두 닫혔고, 남은 용도(pseudo-task query 선택 판정기로 학습)는 HL06–HL10에서 같은 계열 12개가 세 데이터셋 모두 실패한 자리라 사전 확률이 낮다.
- GFSL 벤치마크(Clothing·dblp·electronics·CoraFull)는 전부 호모필이라, 헤테로필에서 다를 가능성(N7 GT oracle의 전 구간 신호)은 11월 논문 범위에서 쓸 자리가 없다.

## 4. 재사용 가능한 것

- **N1**: 모든 제안 방법의 기준선. 표에 `Â²X (no training)`을 넣어야 하고, Clothing 1-shot 목표는 85.1 이상(oracle 상한 ≈ 90). Choi 공유 문서 `claude/share_choi_a2x_baseline_2026-11-02.md`.
- **N3**: "구조가 든 특징 위에서는 anchor 경쟁·이웃 집계가 추가 정보를 주지 않는다" — 비슷한 설계(soft assignment, 이웃 투표)가 다시 나올 때 반복 실험을 막는 근거.
- **N4**: Clothing의 h = 0.86, inter 엣지 없는 노드 2/3 — 이웃 클래스 분포 계열 아이디어를 Clothing에서 시험할 때의 상한.
- 코드: `nd/nd_vote.py`(대칭 A, 1hop/self/Â² 가중, CSR 행 선택 곱), `nd/nd003_episodes.py`(VISION식 순차 배제 pseudo 에피소드, GT 그룹 oracle 에피소드, degree 보존 엣지 재배선은 `nd004 --rewire`), `nd/nd004_profiles.py`(전역 배정, 이웃 분포, 가설별 슬롯 제거 분류, 쌍 AUROC 층화).

## 5. 미확인·한계

- [미확인] N1의 차원·정규화 통제, dblp·electronics 전이(구 서버 메모리 제약으로 Clothing만).
- [미확인] 헤테로필 데이터셋에서의 N5·N7.
- [한계] ND01 관문 A의 사전 등록 기준선(같은 f의 feat)이 구조 유무 비교로 변질됨(N2). 이후 사전 등록은 "구조를 이미 가진 기준선"을 명시한다.
- [한계] ND02의 전역 배정은 에피소드 밖 노드 2만 개를 5개 프로토타입에 강제 배정한다. `unassign=q30` 변형이 Q1을 +0.08 올렸으나(0.593) Q2는 바꾸지 않았다(s_own 0.589). 배정 잡음이 N6의 원인일 가능성은 남지만, 그 변형에서도 a2x를 넘지 못했다.
- [한계] `cos_inter5` 정의의 기하학적 효과(N7)는 설계 단계에서 예상하지 못했다. 엣지 섞기 대조군이 있어 분리할 수 있었다.
- ND02 ④: z1·z2의 프로세스 간 비결정성(소수 5자리, 원인 미확인), run 9개(gt 중복 계산), 가설 g≠own의 p에 own 슬롯이 남는 것(설계대로).

## 6. 산출물·커밋

- ND01: 태그 `nd01-setup` 1477483 · `nd01-gateA` c100341 · `nd01-gateB` f5eeca3 · report 05cd2c3. 표 `reports/ND01_tables/`(TA1–TA3, TB1–TB3, NDP_values).
- ND02: 태그 `nd02-setup` d7ea723 · `nd02-episodes` c8572e1 · `nd02-main` a378182 · `nd02-controls` d7340df · report 99ad277. 표 `reports/ND02_tables/`(T0–T5, NDP_values).
- 비추적(구 서버 seio): `results/nd/nd01/`, `results/nd/nd02/`.
- 이 문서: `reports/ND_closure.md`, 커밋 `ND: closure doc`, 태그 `nd-closure`. 브랜치 `nd/anchor-vote`는 삭제하지 않고 둔다.
