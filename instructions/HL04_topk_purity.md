# HL04 — 운용 지점(anchor당 10개)에서의 순도 판정 (G0b) + 레이블 없는 먼-순수 query 학습 (G2) (auto)

작성 2026-10-25 · 이희태 · 브랜치 `hl/anchor` · 태그 `hl04-g0b` / `hl04-g2` · 서버 gpusystem, GPU 3번, 환경 `NAQ`
데이터셋 Amazon_clothing, τ = 0.3. 기준선 = HL03 새 서버 B_Choi uniform 82.71 / 89.91.

배경: HL03 G0는 "석휘님 필터와 같은 개수를 고르는" 채점(R1)에서 판정기 `comb`가 B4 순도를 0.36까지만 올려 게이트(0.55)에 실패했다. 그러나 (i) 판정기의 AUROC는 0.78–0.82로 분별력이 있고, (ii) 실제로 필요한 query는 anchor당 10개뿐이며 10개 선택 시 순도는 재지 않았고, (iii) G1에서 B4 순도 0.6이면 +4.0%p, 0.55 부근이면 약 +3%p가 기대된다. 이번 지시서는 **운용 지점(10개)에서의 순도**를 재고, 같은 큐에서 레이블 없는 학습 arm을 돌려 결과를 한 번에 해석한다.

레이블 사용 범위: G0b 채점, G2의 사후 진단(query GT 순도), 보조 oracle arm(§2-4)의 query 선택에만.

---

## 0. 시작

1. HEAD가 `hl03-g1`(`faba27c`) 또는 그 위의 문서 커밋인지, `git status` clean인지 확인. 아니면 **정지**.
2. 결과 루트 `results/hl/hl04/`, 표 `reports/HL04_tables/`.
3. 재사용: `hl/hl005_diag_distance_purity.py`의 점수 계산(동일 anchor 2,000개 `results/hl/hl03/g0/anchors.npy`, 동일 점수 정의), `hl/hl006_purity_oracle_train.py`(학습 뼈대), 체크포인트 둘, 노드표.

## 1. G0b — precision@k 채점 `hl/hl007_diag_topk_purity.py` (학습 없음, 태그 `hl04-g0b`)

hl005를 복사·수정한다(docstring에 출처). anchor·구간·점수(`x, ax, a2x, ppr, mv, enc_b, enc_n, comb`)는 HL03 G0와 **완전히 같게** 계산한다(`verify_against_exp049` 포함).

**1-1. 선택 규칙 (anchor별, 구간 안에서)**
| 규칙 | 정의 |
|---|---|
| `top:k` | 점수 상위 k개 (k ∈ {10, 20, 30}) |
| `mid:m` | 점수 상위 m개를 고른 뒤 그 안에서 X 순위 정중앙 10개 (m ∈ {30, 60}) — 석휘님의 "middle" 선택을 판정기 위에 얹은 것 |
| `choi` | 석휘님 필터 통과분에서 X 순위 정중앙 10개 (현재 B_Choi가 B4에서 하는 것과 같음; 통과분 < 10이면 제외하고 비율 기록) |
| `rand` | 구간에서 무작위 10개 (기본 순도 대조) |

**1-2. 채점**: 구간 B3·B4·B5 각각, 점수 × 규칙에 대해 anchor별 선택 query의 GT 순도 → 쌍 합산 순도(주)와 anchor 평균(보조). 추가로 선택 query의 평균 X 순위(구간 안에서 어느 쪽으로 쏠리는지)와, 선택 query 중 석휘님 필터 통과 비율.

**1-3. 출력**: `G0b_topk.csv`(band × score × rule × {purity_pooled, purity_anchor_mean, mean_x_rank, choi_pass_frac, n_anchors}), `G0b_summary.json`.

**게이트 G0b (판정은 이희태)**: B4에서 `top:10` 또는 `mid:30` 중 어느 점수든 쌍 합산 순도 ≥ 0.55. 그 점수·규칙을 "채택 판정기"라 부른다. 둘 이상이면 레이블 없이 결정 가능한 우선순위로 고른다: 정적(`a2x` > `mv` > `ax` > `ppr`) > 인코더(`enc_n` > `enc_b`) > `comb`. (정적이 통과하면 학습 중 갱신이 필요 없어 방법이 단순하다.)

커밋 `HL04: precision@k purity diagnostics`, 태그 `hl04-g0b`.

## 2. G2 — 레이블 없는 학습 arm `hl/hl008_distant_query_train.py` (학습, 태그 `hl04-g2`)

**G0b 결과와 무관하게 실행한다.** hl006을 복사해 만들고(출처 명시), anchor는 uniform(hl003과 같은 코드 경로), query 선택만 아래 규칙으로 바꾼다. **점수 계산에 레이블을 넘기지 않는다.** 각 anchor의 query는 한 번 정해 캐시(exp049와 같음). 구간 B4 = 그 anchor의 X 순위 201–1000.

| arm | query 10개 선택 (B4 안에서) | 성격 |
|---|---|---|
| `a2x_top10` | Â²X 코사인 상위 10 | 정적, 가장 단순 |
| `mv_top10` | `mv` 상위 10 | 정적 다중 뷰 |
| `mv_mid30` | `mv` 상위 30 → X 중앙 10 | 정적 + Choi식 middle |
| `encn_top10` | NaQ 체크포인트(`naq/baseline/seed_1212/best_1shot.pt`) 임베딩 코사인 상위 10 | 2단계 자기 학습(teacher 고정) |
| `comb_top10` | `comb`(mv + enc_n) 상위 10 | 결합 |
| `rand10` | B4 무작위 10 | 대조: 거리만 같고 판정기 없음 |
| `shuf_comb` | `comb` 점수를 **후보 간 무작위 치환** 후 상위 10 | 대조: 점수 분포만 같고 정보 없음 (결과적으로 rand10과 같아야 함) |

enc_n teacher는 seed와 무관하게 seed 1212 체크포인트 하나를 쓴다(진단 전용 고정 teacher; seed별 teacher는 후속).

**2-4. 보조 oracle arm (5 run, 레이블 사용)**: `p04_choidiff` — HL03 G1의 p=0.4와 같되, 틀린 query 6개를 B4 무작위가 아니라 **석휘님 필터 통과분 중 GT 다른 노드**에서 X 중앙으로 뽑는다. G1 p=0.4(83.92)와 B_Choi(82.71)의 +1.2 차이가 "구조적으로 비슷한 틀린 query가 더 해롭다"에서 오는지 보기 위함.

**실행**: 7 arm × seed 1212–1216 = 35 run + 보조 5 run = 40 run, 동시 3개. 러너는 `hl000c`를 복사해 `hl/hl000d_run_g2.py`. 결과 `results/hl/hl04/g2/<arm>/seed_<seed>/`. 각 run의 report.json에 query 사후 진단(GT 순도, 평균 X 순위, 석휘님 필터 통과 비율)을 기록.

**표 T7** `T7_g2.csv`: arm × {1-shot test 평균±std, 5-shot 평균±std, Δ1·Δ5 vs B_Choi(82.71/89.91), query GT 순도 평균, 평균 X 순위}. B_Choi와 NaQ(HL03 T0) 행을 참고로 포함.

커밋 `HL04: label-free distant-query training`, 태그 `hl04-g2`, push.

## 3. 사전 등록 (계산값만; 임계값은 편의 목표치)

| 번호 | 예측 | 계산값 |
|---|---|---|
| HLP036 | G0b B4: `rand`의 순도는 0.178 ± 0.02 (기본 순도 재현) | |
| HLP037 | G0b B4: `choi` 10개의 순도는 0.42 ± 0.04 (B_Choi 진단과 일치) | |
| HLP038 | G0b B4: `a2x` `top:10` 순도 ≥ 0.45 | |
| HLP039 | G0b B4: `enc_n` `top:10` 순도 ≥ 0.50 | |
| HLP040 | **게이트**: B4에서 `top:10` 또는 `mid:30`의 어느 점수가 순도 ≥ 0.55 | |
| HLP041 | G0b B4: 모든 점수에서 `top:10` 순도 > `top:30` 순도 > R1(HL03) 순도 (k가 작을수록 순도↑) | |
| HLP042 | G0b B4: `mid:30` 순도는 `top:30`과 ±0.03 안 (middle 선택이 순도를 깎지 않음) | |
| HLP043 | T7: `rand10` − B_Choi ≤ −1.0 (판정기 없이 거리만 두면 B_Choi보다 나쁨) | |
| HLP044 | T7: `shuf_comb`는 `rand10`과 ±0.5 안 | |
| HLP045 | T7: `encn_top10` − B_Choi ≥ +1.0 (1-shot) | |
| HLP046 | T7: `a2x_top10` − B_Choi ≥ +0.5 (1-shot) | |
| HLP047 | T7: 각 arm의 1-shot은 G1 곡선에서 그 arm의 실제 순도에 해당하는 값과 ±1.0 안 (순도가 성능을 예측함) | |
| HLP048 | T7: `encn_top10`·`comb_top10`의 5-shot Δ ≥ 0 (불순도를 줄이면 5-shot 손실이 사라짐) | |
| HLP049 | 보조: `p04_choidiff` − G1 p=0.4(83.92) ≤ −0.8 (구조적으로 비슷한 틀린 query가 더 해로움) | |

**판정 규칙(이희태 적용)**: HLP040 실패이고 T7의 최고 arm − B_Choi < +1.0 → 스토리 C 중단. HLP040 통과 또는 T7 최고 arm ≥ +1.0 → 채택 판정기로 C 본 설계(HL05: 동적 teacher, 다른 데이터셋).

## 4. 보고서 `reports/HL04_topk_purity.md`

CLAUDE.md §5 형식. 포함: G0b B4의 전체 표(점수 × 규칙)와 B3·B5의 `top:10` 행, T7, 보조 arm 결과, 사전 등록 표, 실패·경고, 산출물 경로. 끝나면 채팅에 G0b B4 표, T7, 사전 등록 표를 그대로 붙인다.
