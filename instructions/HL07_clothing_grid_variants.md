# HL07 — Clothing: 순도 × 거리 2D oracle 격자(50 run) + 레이블 없는 변형 7 arm(35 run) (night-auto, 야간 2단계)

작성 2026-10-25 · 이희태 · 브랜치 `hl/anchor` · 태그 `hl07` · GPU 2·3번 · 환경 `NAQ`
데이터셋 Amazon_clothing, τ = 0.3. 기준선 B_Choi 82.71 / 89.91. **목표 시각 03:15 KST**, 큐 `--deadline 03:00`.
전제: HL06의 `hl/hl_pdd.py`, `hl/hl011_pdd_train.py`, `hl/hl000f_run_hl06.py`. HL06이 건너뛰어졌거나 hl011이 없으면 이 단계도 건너뛰고 HL10으로(HL10도 같은 전제이므로 그때는 HL08로).

## 배경
HL06은 (순도, 거리) 평면의 몇 점만 찍는다. 이 단계는 **격자를 채워** 두 축의 효과를 분리된 수치로 남기고(oracle, 분석 재료), 레이블 없는 쪽에서는 HL06의 핵심 arm `encb_strat5`를 둘러싼 **통제·변형**을 돈다 — 무엇이 이득을 내는지(판정기인지 거리 통제인지), Choi 필터에 거리 통제만 더하면 어떻게 되는지, teacher를 한 번 갱신하면 어떻게 되는지.

레이블 사용 범위: §1 oracle arm의 query 선택과 모든 arm의 사후 진단에만.

## 0. 시작
HL06 끝의 HEAD 위에서 시작. `results/hl/hl07/`, `reports/HL07_tables/`. 새 스크립트 없음 — `hl011`에 arm을 추가하고(§1·§2), `hl_pdd`에 규칙을 추가한다. 추가 후 각 새 arm을 `--epoch 20`으로 smoke(실패 arm은 한 번 수정, 재실패 시 제외).

## 1. oracle 격자 `o_grid_<region>_p<pp>` (10 arm × 5 seed = 50 run)
B4(X 순위 201–1000) 안에서 anchor 기준 s2(Â²X 코사인) 순위로 정의한 영역:
| region | 후보 |
|---|---|
| `t1` | s2 순위 1–160 (가장 가까운 20%) |
| `strat5` | 다섯 층 T1–T5(각 160개) |
| `far50` | s2 순위 401–800 |

순도 p ∈ {0.3, 0.5, 0.7, 1.0}. 슬롯 10개 중 round(10p)개를 GT-same으로 둔다.
- `t1`, `far50`: 영역 안 GT-same에서 무작위 round(10p)개, GT-diff에서 무작위 나머지(rng = `default_rng(seed + 7)`, anchor 순서대로 소비).
- `strat5`: 층마다 슬롯 2개. 10개 슬롯 중 GT-same 슬롯 round(10p)개를 무작위로 고르고(같은 rng), 각 슬롯은 자기 층의 GT-same/GT-diff에서 무작위.
- 부족분: `t1`은 T2에서, `far50`은 s2 순위 161–400에서, `strat5`는 인접 층에서 같은 종류를 채우고, 그래도 부족하면 B5(1001–3000) GT-same, 마지막에 영역 GT-diff(`diff_fill`). 단계별 개수 기록.

arm 목록(HL06의 `o_far50_p03`·`o_far50_p05`·`o_topsame`는 이미 있으므로 제외): `o_grid_t1_p03`, `o_grid_t1_p05`, `o_grid_t1_p07`, `o_grid_t1_p10`, `o_grid_strat5_p03`, `o_grid_strat5_p05`, `o_grid_strat5_p07`, `o_grid_strat5_p10`, `o_grid_far50_p07`, `o_grid_far50_p10`.

## 2. 레이블 없는 변형 (7 arm × 5 seed = 35 run)
teacher = `enc_b`(seed 1212 B_Choi 체크포인트) — `encb_strat5_r2`만 예외.
| arm | 정의 | 묻는 것 |
|---|---|---|
| `rand_strat5` | 다섯 층에서 무작위 2개씩(판정기 없음) | 거리 통제만으로 얼마나 가는가(통제) |
| `choi_far` | exp049 필터 통과분(전체 노드, ÂX·Â²X ≥ τ) 중 **anchor 기준 s2 하위 절반** → X 순위 정중앙 10. 하위 절반 < 10이면 통과분 전체의 X 중앙 10, 통과분 < 10이면 exp049 relax 그대로 | Choi 방법에 거리 통제만 더하면 |
| `choi_strat` | 필터 통과분을 s2로 5분위 → 분위마다 X 순위 중앙 2개. 분위 크기 < 2면 인접 분위에서 채움, 통과분 < 10이면 relax | 같은 것의 층화형 |
| `encb_strat5_wide` | X 구간 51–1000(950개)에서 s2 5층 × enc_b 상위 2 | X 구간을 넓히면 |
| `encb_strat5_b45` | X 구간 201–3000에서 같은 규칙 | X 구간을 더 멀리 |
| `encb_strat5_r2` | teacher = HL06 `encb_strat5` seed 1212의 `best_1shot.pt`(없으면 `encb_ex20` seed 1212, 그것도 없으면 이 arm 제외). 규칙 `strat5` | teacher 1회 갱신 |
| `encb_strat5_drop2` | `strat5` 선택 + **손실 상위 2개 제외**: 400 에피소드 워밍업 뒤, 매 에피소드에서 anchor별 query 10개 중 현재 네트워크 기준 ProtoNet 손실(−log p(자기 anchor))이 가장 큰 2개를 그 에피소드 손실 평균에서 제외. 손실은 `meta.network`의 임베딩으로 `meta.ProtoNet.forward`와 같은 식(제곱 유클리드 → log_softmax)으로 hl011 안에서 계산한다(`hl_common.proto_softmax` 참고). `meta.py`는 수정하지 않는다. 제외된 query의 GT 순도(제외 FP 비율)를 report.json에 기록 | 손실 기반 FP 억제 |

`choi_far`·`choi_strat`는 `exp049.build_structural_queries`의 선택을 재현하는 `hl_common.verify_against_exp049`의 방식(z1·z2 임계, min(s1,s2) relax, X 정렬)을 그대로 쓰고 선택 규칙만 바꾼다. 필터 통과분 수와 relax 발생률을 기록.

## 3. 실행·표
- 큐 1: §1 50 run(`--deadline 03:00`). 큐 2: §2 35 run(같은 러너, 큐 1 뒤에 이어서; 둘을 한 큐로 묶어도 된다 — 단 **§2 arm을 큐 앞에** 두어 deadline에 걸려도 레이블 없는 결과가 먼저 남게 한다). 결과 `results/hl/hl07/train/Amazon_clothing/<arm>/seed_<seed>/`.
- **T11** `T11_grid.csv`: region × p → 1-shot 평균±std, 5-shot, 실제 순도, mean_s2, mean_s2rank, 보충 비율(HL06 `o_far50_p03/p05`, `o_topsame`, HL03 G1, HL05 near/far를 같은 표에 참고 행으로).
- **T12** `T12_variants.csv`: HL06 T10 형식, 참고 행에 HL06의 `encb_strat5`·`encb_top10`·`rand10`·B_Choi.
- 커밋 `HL07: purity-distance grid and label-free variants`, 태그 `hl07`, push. 보고서 `reports/HL07_clothing_grid_variants.md`(CLAUDE.md §5 형식). progress.md 한 줄, 채팅 한 줄.

## 4. 사전 등록 (계산값만)
| 번호 | 예측 | 계산값 |
|---|---|---|
| HLP080 | `o_grid_t1_p10` 1-shot ≤ 84.5 | |
| HLP081 | `o_grid_strat5_p10` ≥ 88.5 | |
| HLP082 | `o_grid_far50_p10` ≥ 89.0 | |
| HLP083 | p=0.5에서 `strat5` − `t1` ≥ +3.0 | |
| HLP084 | `o_grid_strat5_p03` ≥ B_Choi − 0.5 (= 82.21) | |
| HLP085 | 세 region 모두 1-shot이 p에 대해 단조 증가 | |
| HLP086 | `rand_strat5` ≤ HL04 `rand10` + 1.0 (= 81.34) | |
| HLP087 | HL06 `encb_strat5` − `rand_strat5` ≥ +2.0 (판정기의 기여) | |
| HLP088 | `choi_far` − B_Choi ≥ +0.5 | |
| HLP089 | `choi_strat` − B_Choi ≥ +0.5 | |
| HLP090 | \|`encb_strat5_wide` − HL06 `encb_strat5`\| ≤ 1.0 | |
| HLP091 | `encb_strat5_r2` ≥ HL06 `encb_strat5` | |
| HLP092 | `encb_strat5_drop2` ≥ HL06 `encb_strat5` − 0.3, 제외된 query의 FP 비율 ≥ 0.7 | |

판정은 아침에 이희태가 한다.
