# HL05 — 구조적으로 가까운 same-class vs 먼 same-class query (oracle, 10 run) (auto)

작성 2026-10-25 · 이희태 · 브랜치 `hl/anchor` · 태그 `hl05` · 서버 gpusystem, GPU 3번, 환경 `NAQ`
데이터셋 Amazon_clothing. 기준선 = HL03 B_Choi uniform 82.71 / 89.91, HL03 G1 p=1.0 89.42 / 94.17.

배경: HL04에서 레이블 없는 판정기로 B4(X 순위 201–1000) query 순도를 0.42 → 0.53으로 올렸는데 1-shot은 B_Choi보다 2–4%p **낮았다**(`rand10` 순도 0.18과 같은 수준). 같은 순도 0.40인 oracle 두 arm도 83.92 / 85.24로 갈렸다. 가설: **GCN 인코더에서 Â²X가 anchor와 비슷한 노드는 임베딩도 이미 anchor와 거의 같아, 그런 query는 클래스가 같아도 학습 신호를 거의 주지 않는다.** 유익한 positive는 같은 클래스이면서 구조적으로 먼 것이다. 이번 지시서는 순도(1.0)와 X 거리(B4)를 고정하고 **구조적 근접성만** 바꿔 이 가설을 직접 가른다.

이 지시서의 결과로 스토리 C의 종결 문서를 쓴다. 방법 제안은 하지 않는다.

레이블 사용 범위: oracle query 선택(same-GT 제한)과 진단 통계에만.

---

## 0. 시작
1. HEAD가 `ad5185c`(HL04: report) 또는 그 위의 문서 커밋, `git status` clean인지 확인. 아니면 **정지**.
2. 결과 루트 `results/hl/hl05/`, 표 `reports/HL05_tables/`.

## 1. 학습 스크립트 `hl/hl009_structural_oracle_train.py`

`hl/hl006_purity_oracle_train.py`를 복사(docstring에 출처). anchor는 uniform(hl003과 같은 코드 경로·난수). query 선택만 아래 함수로 바꾼다. 각 anchor의 query는 한 번 정해 캐시.

`structural_oracle_queries(anchor, mode)`, mode ∈ {`near`, `far`}:
- 후보 = anchor의 X 순위 B4(201–1000) 중 **GT가 anchor와 같은 노드**.
- **X 거리 통제를 위해 층화한다**: B4를 X 순위로 네 소구간 S1=[201,400], S2=[401,600], S3=[601,800], S4=[801,1000]으로 나누고, 소구간별 할당을 S1 3 / S2 3 / S3 2 / S4 2개(합 10)로 둔다.
- 각 소구간 안에서 same-GT 후보를 **Â²X 코사인**(exp049와 같은 z2)으로 정렬해, `near`는 상위부터, `far`는 하위부터 할당 개수만큼 뽑는다.
- 한 소구간의 same-GT가 할당보다 적으면 부족분을 인접 소구간(S1↔S2, S3↔S4, 그다음 S2↔S3)에서 같은 규칙으로 채운다. 그래도 10개가 안 되면 B5(1001–3000) same-GT에서 같은 규칙으로 채우고, 그래도 부족하면 B4 same-GT 중 아직 안 뽑힌 것을 X 중앙부터 채운다. 단계별 보충 개수를 기록.
- 두 mode는 같은 anchor에 대해 **같은 소구간 할당**을 쓴다(X 분포가 같도록).

학습·평가·report.json은 hl006 그대로. report.json `query_diagnostics`에 추가: 실제 순도, 선택 query의 평균 X 순위, 평균 Â²X 코사인, 보충 단계별 비율, 그리고 §2의 임베딩 진단값.

## 2. 학습 전 임베딩 진단 (같은 스크립트 안, 학습 시작 전 1회)

체크포인트 두 개(`results/hl/p0/Amazon_clothing/{ours/tau_0.3,naq/baseline}/seed_1212/best_1shot.pt`)로 각각 전체 임베딩을 계산하고, 이 run의 처음 500 에피소드(uniform anchor + 이 mode의 query)에 대해:
- anchor–query 임베딩 코사인 거리(1 − cos) 평균·중앙값
- `hl_common.proto_softmax`로 query별 p(자기 anchor)의 평균·중앙값, p(자기 anchor) ≥ 0.95인 query 비율
- query별 ProtoNet 손실(−log p(자기 anchor))의 평균 — 이 query가 학습 초기에 줄 신호의 크기

seed 1212 run에서만 계산하면 충분하다(anchor·query가 seed와 함께 바뀌므로 다른 seed는 생략 가능; 계산했다면 함께 기록).

## 3. 실행
arm `near`, `far` × seed 1212–1216 = **10 run**, 동시 3개. 러너는 `hl/hl000d_run_g2.py`를 복사해 `hl/hl000e_run_hl05.py`. 결과 `results/hl/hl05/<mode>/seed_<seed>/`.

**표 T8** `T8_structural.csv`: arm × {1-shot test 평균±std, 5-shot 평균±std, Δ vs B_Choi(82.71/89.91), Δ vs G1 p=1.0(89.42/94.17), 실제 순도, 평균 X 순위, 평균 Â²X 코사인, 보충 비율}. 참고 행: B_Choi, G1 p=1.0, HL04 `comb_top10`, `rand10`.
**표 T9** `T9_embed_diag.csv`: arm × 체크포인트 × {거리 평균·중앙값, p_own 평균·중앙값, p_own ≥ 0.95 비율, 손실 평균}.

커밋 `HL05: structural near/far same-class oracle`, 태그 `hl05`, push.

## 4. 사전 등록 (계산값만; 임계값은 편의 목표치)

| 번호 | 예측 | 계산값 |
|---|---|---|
| HLP050 | **핵심**: `far` − `near` ≥ +3.0 (1-shot, 5-seed 평균) | |
| HLP051 | `near` 1-shot ≤ B_Choi + 1.0 (= 83.71) — 구조적으로 가까운 same-class는 순도 1.0이어도 거의 무정보 | |
| HLP052 | `far` 1-shot ≥ G1 p=1.0 − 1.0 (= 88.42) | |
| HLP053 | `far` − `near` (5-shot) > 0 | |
| HLP054 | T9 (B_Choi ckpt): `near`의 anchor–query 거리 중앙값 < `far`의 절반 | |
| HLP055 | T9 (B_Choi ckpt): `near`의 p_own ≥ 0.95 비율 ≥ 0.5, `far`는 ≤ 0.2 | |
| HLP056 | T8: 두 arm의 평균 X 순위 차이 ≤ 50 (층화가 X 거리를 맞췄다) | |

**해석 규칙(이희태 적용)**: HLP050·HLP051 둘 다 맞으면 가설 확인 — "GCN 메타러너에서 정보를 주는 positive는 구조적으로 먼 same-class이고, 레이블 없는 구조 신호는 정의상 그 반대를 고른다"를 종결 문서의 확인된 결론으로 쓴다. 하나라도 틀리면 가설은 확인되지 않은 것으로 적는다. HLP056이 틀리면 결과 해석에 X 거리 교란 단서를 붙인다.

## 5. 보고서 `reports/HL05_structural_diversity.md`
CLAUDE.md §5 형식. 끝나면 채팅에 T8, T9, 사전 등록 표를 그대로 붙인다.
