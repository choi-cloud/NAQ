# ND02 — inter-class 연결 패턴만으로 pseudo-class를 식별할 수 있는가 (학습 없음)

작성 2026-11-02 · 이희태 · 브랜치 `nd/anchor-vote`(HEAD `05cd2c3`에서 시작) · 서버 seio(RTX 4080 SUPER 1장) · 환경 `taskns`
데이터셋 Amazon_clothing만. **학습 없음.** 레이블은 §6(채점)과 §7-4(GT 그룹 oracle 에피소드 생성)에서만 읽는다. 모드: **auto**.

## 목적

ND01은 이웃 투표 분포 d_v(자기 그룹 성분 포함)가 Â²X 프로토타입 위에 아무것도 더하지 못함을 보였다. 그러나 원래 질문은 다른 것이었다: **pseudo-class 그룹의 노드들이 다른 그룹으로 가는 연결 분포(inter-class 패턴)를 공유하는가, 그 패턴만으로 그룹을 식별하거나 잘못 들어온 노드를 가려낼 수 있는가.** ND01은 자기 그룹 성분이 섞인 분포를 썼고, 호모필 그래프에서는 그 성분이 inter 성분을 묻는다. ND02는 자기 그룹 성분을 **빼고** 같은 질문을 다시 묻는다. 배경은 DCGC("같은 클래스 노드는 이웃 분포가 비슷하다")와 heterophily 문헌의 compatibility matrix.

핵심 질문 세 개:
- Q1 (식별) inter-profile만으로 query의 pseudo-label을 우연(20%)보다 잘 맞히는가. 엣지를 섞으면 사라지는가.
- Q2 (노이즈) 배정 그룹의 패턴과 어긋나는 정도가 "GT가 anchor와 다른 query"를 가려내는가. `a2x`보다 나은가, 더했을 때 개선되는가.
- Q3 (전제) 같은 GT 클래스 노드 쌍이 다른 클래스 쌍보다 inter-profile이 비슷한가 — 특히 특징이 비슷한 쌍에서도.

## 0. 시작

1. `git status` 깨끗한지, HEAD `05cd2c3`인지 확인. `CLAUDE.md`(개정 2)와 이 지시서를 한 커밋: `ND02: CLAUDE.md rev2, ND02 instruction`, 태그 `nd02-setup`, push.
2. 환경·GPU·pkl md5 확인은 ND01 §0과 같다(`results/nd/nd02/env_start.txt`). **설치 금지.** GPU 여유 10GB 미만이면 멈춤(ND01 실측 최대 9.9GB).
3. `mkdir -p results/nd/nd02/{episodes,profiles,rewire} reports/ND02_tables out/nd`.

## 1. 에피소드 생성 `nd/nd003_episodes.py` (레이블: §1-3 oracle 에피소드에서만)

### 1-1. pseudo 에피소드 (VISION·NaQ식) — 주 설정 `pseudo_k3`
- `rng = np.random.default_rng(1212)`. **E = 300** 에피소드, N = 5.
- anchor: 전체 노드에서 비복원 균등 5개(에피소드 안 중복 없음).
- 그룹: anchor a의 X 코사인 행(`data.Diffusion_matrix[a]`, 자기 자신 `-inf`)을 내림차순 정렬해 **이미 다른 그룹에 들어간 노드를 건너뛰며**(순차 배제, anchor 순서대로) 상위 K−1+M개를 취한다. support S_g = {a} ∪ 상위 K−1개, query Q_g = 그다음 M개. **K = 3, M = 10.** 다섯 그룹은 서로소.
- 저장 `results/nd/nd02/episodes/pseudo_k3.npz`: `anchors [E,5]`, `support [E,5,K]`, `query [E,5,M]`.
- 변형 `pseudo_k5`: K = 5(anchor + 상위 4), M = 10. 같은 rng seed로 anchor는 `pseudo_k3`와 동일하게.
- 변형 `pseudo_choi`: support는 `pseudo_k3`와 같고, query 10개는 B_Choi 방식(`hl_common.mine_queries_for`, τ = 0.3, 각 anchor 기준)으로 뽑는다. 다른 그룹과 겹치는 query는 뒤 anchor 쪽에서 제거하고 개수를 기록한다(보충하지 않음).

### 1-2. 기록
에피소드별 GT 순도(support·query 각각, anchor와 같은 GT 비율)와 anchor 쌍 GT 충돌 수를 `episodes/stats.csv`에. **이 값은 §6 채점 때 계산한다**(생성 코드는 레이블을 읽지 않는다).

### 1-3. GT 그룹 oracle 에피소드 `gt_k3` (레이블 사용)
전체 77 클래스에서 비복원 5개 → 클래스마다 노드 K=3 + M=10을 균등 추출(클래스 크기 13 미만이면 그 클래스는 다시 뽑는다). E = 300, 같은 rng. 저장 `episodes/gt_k3.npz`. 파일명에 `gt`가 들어가고 §7-4에서만 쓴다.

## 2. 전역 그룹 배정 `nd/nd004_profiles.py --stage assign`

에피소드 밖 노드에도 그룹을 붙여야 이웃이 어느 그룹인지 셀 수 있다.
- 프로토타입 P_g = S_g의 특징 평균을 행 정규화(특징 공간 `x` = z0 주 설정, 변형 `a2x` = z2; `hl_common.structural_scores`).
- 모든 노드 v: s_v = z_f[v] @ P.T (shape [n, 5]). `grp(v) = argmax`, `margin(v) = s_(1) − s_(2)`.
- **에피소드 노드(S ∪ Q)는 argmax가 아니라 자기 그룹 번호를 쓴다**(pseudo-label이 곧 그룹).
- 미배정 변형 `unassign=q30`: 에피소드 밖 노드 중 margin이 그 에피소드 안 하위 30%인 노드는 그룹 없음(−1)으로 두고 §3에서 세지 않는다. 주 설정은 `unassign=none`.
- 배정 결과는 에피소드마다 [n] int8로 메모리에만 두고 저장하지 않는다(300 × n은 작지만 불필요).

## 3. 노드별 inter-profile `--stage profile`

A = 대칭화한 이진 인접행렬, self-loop 없음(`nd_vote._sym_a`). 에피소드의 S ∪ Q 노드 v에 대해:
- `c_v ∈ R^5` = 이웃 u ∈ N(v) 가운데 grp(u) = g인 개수(미배정 이웃은 제외, 개수 `u_v` 기록). 변형 `nbr=a2`: 개수 대신 Â² 가중치 합(`nd_vote.build_weights`의 `a2`, 자기 자신 가중 제외).
- `h_v = c_v[own] / Σc_v` (호모필 비율; Σc_v = 0이면 NaN).
- `n_inter(v) = Σ_{g≠own} c_v[g]` (다른 그룹으로 가는 엣지 수).
- **inter-profile** `p_v ∈ R^4` = c_v에서 own 슬롯을 뺀 뒤 합 1로 정규화. `n_inter = 0`이면 정의되지 않음(NaN, 분석에서 제외, 비율 기록).
- `inter5_v ∈ R^5` = c_v에서 own 슬롯을 0으로 둔 뒤 합 1로 정규화(§6 쌍 비교용).
- `full5_v ∈ R^5` = c_v / Σc_v (own 포함, DCGC의 e_i에 해당).
- 저장 `profiles/<episodes>_<f>_<unassign>_<nbr>.npz`: `c [E,65,5]`, `h`, `n_inter`, `u` (65 = 5 × (K+M); K=5면 75).

## 4. 그룹 패턴 `--stage pattern` (같은 스크립트, 메모리에서 이어서)

- π_g ∈ R^4 = S_g의 support 중 p가 정의된 것들의 평균. σ_g ∈ R^4 = 같은 집합의 슬롯별 표준편차(정의된 support가 1개면 σ = 1로 둔다). 정의된 support가 0개면 그 그룹은 패턴 없음(그 에피소드의 Q1·Q2에서 그 그룹의 query는 제외, 비율 기록).
- 기록: 그룹당 정의된 support 수 분포, π_g의 엔트로피 평균(균등 분포 = log 4 ≈ 1.386; 이 값에 가까우면 패턴이 없다는 뜻), 에피소드 안 π_g끼리의 평균 코사인(그룹 패턴이 서로 구분되는가).

## 5. Q1·Q2 — query 분류와 노이즈 점수 `--stage classify`

query q(배정 그룹 own(q) = pseudo-label)에 대해, **가설별로** own 슬롯을 뺀다: 가설 g마다 `p_q^(g)` = c_q에서 슬롯 g를 뺀 4-벡터 정규화(c_q에서 g 외 합이 0이면 그 가설은 점수 −∞).
- 점수 `score(q, g)`:
  - 주 설정 `cos`: cos(p_q^(g), π_g).
  - 변형 `var`: −Σ_j (p_q^(g)[j] − π_g[j])² / (σ_g[j]² + 0.01).
- **Q1 식별**: 예측 = argmax_g score. `acc_pseudo` = 예측 == own(q) 비율(우연 0.20). 그룹에 패턴이 없거나 모든 가설이 −∞인 query는 제외하고 비율 기록. 에피소드별 정확도 300개를 저장해 평균·CI.
- **Q2 노이즈**: `s_own(q) = score(q, own(q))`. GT 레이블로 `same(q) = (GT(q) == GT(anchor_own))`. 다음 점수들의 AUROC(에피소드 행별 Mann-Whitney, `hl005.auroc_rows`, 양·음 둘 다 있는 에피소드만; pooled도):
  - `s_own` (inter-profile 일치도)
  - `h` (호모필 비율만 — 대조군 §7-2)
  - `a2x` = cos(z2[q], z2[anchor_own]) (대조군 §7-3)
  - `x` = cos(z0[q], z0[anchor_own]) (참고)
  - `comb` = `s_own`과 `a2x`의 에피소드 내 순위 평균
- 추가: Q1 예측이 own과 다른 query("패턴이 다른 그룹을 가리키는 query")의 GT 불일치율 vs 같은 query의 GT 불일치율.

## 6. Q3 — 쌍 비교 `--stage pairs` (레이블: 쌍 라벨)

에피소드 안 S ∪ Q 노드 65개의 모든 쌍(2,080쌍 × 300 에피소드). 쌍 (u, v)에 대해:
- 라벨 `same_gt = (GT(u) == GT(v))`.
- 점수: `cos_inter5`, `cos_full5`, `cos_x` = cos(z0), `cos_a2x` = cos(z2). profile이 정의되지 않은 노드가 포함된 쌍은 inter5·full5에서 제외.
- 전체 AUROC(pooled, 에피소드별 평균 둘 다). **층화**: 쌍을 `cos_x` 4분위로 나눠 분위별 AUROC — 최상위 분위("특징이 비슷한 쌍")에서 `cos_inter5`의 AUROC가 0.5를 넘는지가 핵심. 같은 pseudo-그룹 쌍 / 다른 그룹 쌍으로도 나눠 보고.

## 7. 대조군

### 7-1. 엣지 섞기 `--rewire`
degree 보존 무작위 재배선: 엣지 리스트에서 쌍 (a–b, c–d)를 뽑아 (a–d, c–b)로 바꾸는 swap을 10 × |E|회(self-loop·중복 엣지가 생기면 건너뜀, `rng(1212)`). networkx를 쓰지 않고 numpy로 구현한다(`taskns`에 없을 수 있음). 재배선 그래프의 degree 분포가 원본과 같은지 assert. 이 그래프로 §2–§6 전체를 주 설정(`pseudo_k3`, `x`, `none`, `1hop`, `cos`)에 대해 한 번 더 돈다. 결과 접미사 `_rewire`.

### 7-2. 호모필 비율만
§5 Q2의 `h` 행. Q1에서는 쓰지 않는다(분류 불가).

### 7-3. Â²X 기준선
§5 Q2의 `a2x` 행, §6의 `cos_a2x`, 그리고 `comb`.

### 7-4. GT 그룹 oracle
`gt_k3` 에피소드로 §2–§6 주 설정을 돈다(배정은 GT가 아니라 §2와 똑같이 특징 프로토타입 argmax로 — 레이블은 에피소드 구성에만 쓴다). 결과 접미사 `_gt`. **여기서 Q1·Q3 신호가 없으면 Clothing에는 inter 패턴이 없다는 뜻이고, 여기서는 있는데 pseudo에서 없으면 pseudo-label 품질 문제다.** 이 구분이 ND02의 가장 중요한 산출이다.

## 8. 커버리지 `--stage coverage`
query를 `n_inter` ∈ {0, 1–2, 3–4, ≥5} 구간으로 나눠 각 구간의 query 비율과 Q1 acc·Q2 AUROC(`s_own`, `a2x`)를 따로 낸다. 쌍 비교도 두 노드 모두 `n_inter ≥ 3`인 쌍으로 한정한 행을 추가한다.

## 9. 실행 범위

| 축 | 주 설정 | 변형 |
|---|---|---|
| 에피소드 | `pseudo_k3` | `pseudo_k5`, `pseudo_choi`, `gt_k3`(§7-4) |
| 특징 공간 f | `x` | `a2x` |
| 미배정 | `none` | `q30` |
| 이웃 | `1hop` | `a2` |
| 점수 | `cos` | `var` |
| 그래프 | 원본 | `rewire`(§7-1) |

변형은 **한 번에 한 축만** 바꾼다(주 설정 + 변형 7개 + rewire + gt = 10 run). 전부 학습 없는 전경 실행이고 run당 수 분이면 된다. 어떤 run이 실패하면 그 run만 빼고 ④에 적는다.

## 10. 표 `reports/ND02_tables/`
- **T1** `T1_pattern.csv`: run × {정의된 profile 비율(support·query), 그룹 패턴 있음 비율, π 엔트로피 평균, π 간 코사인 평균, h 평균, n_inter 평균·중앙값}.
- **T2** `T2_q1_identify.csv`: run × {acc_pseudo 평균, CI, 제외 query 비율, 예측≠own인 query의 GT 불일치율, 예측=own인 query의 GT 불일치율}. `_rewire`·`_gt` 행 포함.
- **T3** `T3_q2_noise.csv`: run × 점수(`s_own`, `h`, `a2x`, `x`, `comb`) × {auroc_rows 평균, n_episodes, auroc_pooled}.
- **T4** `T4_q3_pairs.csv`: run × 점수(`cos_inter5`, `cos_full5`, `cos_x`, `cos_a2x`) × {auroc_all, auroc by cos_x 4분위(4열), auroc same-group pairs, auroc diff-group pairs, n_pairs}.
- **T5** `T5_coverage.csv`: 주 설정의 n_inter 구간별 {query 비율, acc_pseudo, auroc s_own, auroc a2x} + 쌍 비교 `n_inter ≥ 3` 행.
- 에피소드 통계 `T0_episodes.csv`(§1-2).

## 11. 사전 등록 (Agent는 계산값만; 판정은 이희태) — 주 설정 기준

| 번호 | 예측 | 계산값 |
|---|---|---|
| NDP011 | query 중 inter-profile이 정의되는(n_inter ≥ 1) 비율 ≥ 0.5 | |
| NDP012 | **Q1 핵심**: `acc_pseudo` ≥ 0.30 (우연 0.20) | |
| NDP013 | `acc_pseudo` − `acc_pseudo_rewire` ≥ +0.05 | |
| NDP014 | **Q3 핵심**: `cos_inter5` auroc_all ≥ 0.55 | |
| NDP015 | `cos_inter5` auroc(cos_x 최상위 분위) ≥ 0.53 (특징이 비슷한 쌍에서도 신호) | |
| NDP016 | `cos_inter5` auroc_all − 같은 값(rewire) ≥ +0.03 | |
| NDP017 | **Q2**: `s_own` auroc_rows 평균 ≥ `h` + 0.02 (inter 성분이 호모필 비율보다 낫다) | |
| NDP018 | `s_own` auroc_rows 평균 ≥ `a2x` − 0.05 | |
| NDP019 | `comb` auroc_rows 평균 ≥ `a2x` + 0.01 | |
| NDP020 | **oracle**: `_gt` run의 `acc_pseudo` ≥ 0.35 그리고 `cos_inter5` auroc_all ≥ 0.60 (GT 그룹에서는 패턴이 보인다) | |
| NDP021 | `_gt` − `pseudo_k3`의 `acc_pseudo` 차이 ≥ +0.05 (pseudo-label 품질이 패턴을 흐린다) | |
| NDP022 | `n_inter ≥ 5` 구간의 `acc_pseudo` ≥ `1–2` 구간 + 0.05 | |

## 12. 판정 기준(참고, 판정은 이희태)
- NDP020 미달이면 다른 결과는 해석하지 않는다 — Clothing에는 확인할 패턴이 없다는 뜻이고, 헤테로필 데이터셋 추가를 검토한다.
- NDP020 통과 + NDP012·014·016 통과 → inter 패턴이 pseudo-task에서도 보인다. 다음은 Q2 용도(NDP019)에 따라 query 가중/배제 oracle 또는 분류 결합.
- NDP020 통과 + NDP012·014 미달 + NDP021 통과 → 패턴은 있으나 pseudo-label이 흐린다. 다음은 배정 품질(§2 변형, 반복 재배정) 쪽.
- NDP020 통과 + NDP017 미달 → inter 성분은 호모필 비율 이상의 정보가 없다. ND 종결.

## 13. 보고서 `reports/ND02_inter_profile.md`
CLAUDE.md §5 형식. ①~⑤ + T0~T5 전체 + 사전 등록 표. 레이블 사용 범위 한 줄. 커밋 단위: `ND02: episodes`(태그 `nd02-episodes`), `ND02: profiles and classification`(`nd02-main`), `ND02: rewire and gt controls`(`nd02-controls`), `ND02: report`. 끝나면 채팅에 T2 전체, T3(`s_own`·`h`·`a2x`·`comb` 행, 주 설정과 `_gt`), T4(`cos_inter5`·`cos_a2x` 행, 주 설정·`_rewire`·`_gt`), 사전 등록 표를 그대로 붙인다. 해석·제안 없음.
