# HL10 — Clothing 채움 실험: 층 수·제외 폭·판정기 교체 8 arm (night-auto, 야간 5단계 / 수시 채움)

작성 2026-10-25 · 이희태 · 브랜치 `hl/anchor` · 태그 `hl10` · GPU 2·3번 · 환경 `NAQ`
데이터셋 Amazon_clothing, τ = 0.3, teacher = `enc_b`(seed 1212). 기준선 B_Choi 82.71 / 89.91.
**역할**: 새 코드가 거의 없어 실패 위험이 가장 낮다. (i) HL08/HL09가 건너뛰어졐을 때 즉시, (ii) 다른 단계가 모두 끝나고 **07:15 이전**에 시간이 남을 때 돈다. 큐 `--deadline 07:15`. 두 번 이상 들어오면 완료 run을 건너뛰고 이어간다.
전제: HL06의 `hl_pdd`·`hl011`. 없으면 이 단계는 할 수 없다(progress.md에 기록하고 §마감으로).

## arm (8 × 5 seed = 40 run; 큐 순서 = 아래 순서)
`hl_pdd`에 규칙 `strat3`(s2 순위 3층 × 상위 3 + 마지막 층 4 → 10개: 층 T1'=1–267 3개, T2'=268–534 3개, T3'=535–800 4개), `strat10`(10층 × 1), `ex10`(s2 순위 81–800에서 상위 10), `ex40`(321–800에서 상위 10)을 추가.
| arm | 판정기 | 규칙 | 묻는 것 |
|---|---|---|---|
| `encb_strat10` | enc_b | strat10 | 층을 더 잘게 |
| `encb_strat3` | enc_b | strat3 | 층을 더 굵게 |
| `encb_ex10` | enc_b | ex10 | 극단 제외 폭 작게 |
| `encb_ex40` | enc_b | ex40 | 제외 폭 크게 |
| `nbr_strat5` | nbr_b | strat5 | 이웃 평균 판정기 |
| `xidf_strat5` | xidf | strat5 | 비구조 판정기 |
| `clu_strat5` | clu_b | strat5 | 군집 판정기 |
| `knnppr_far50` | knnppr_b | far50 | 전이적 판정기, 먼 절반 |

결과 `results/hl/hl10/train/Amazon_clothing/<arm>/seed_<seed>/`. **T17** `T17_fillers.csv`(HL06 T10 형식, 참고 행 HL06 `encb_strat5`·`encb_ex20`·`encb_far50`·`encb_top10`, B_Choi). 커밋 `HL10: clothing filler variants`, 태그 `hl10`, push. 보고서 `reports/HL10_clothing_fillers.md`. progress.md 한 줄, 채팅 한 줄.

## 사전 등록 (계산값만)
| 번호 | 예측 | 계산값 |
|---|---|---|
| HLP113 | `encb_strat10`·`encb_strat3`은 HL06 `encb_strat5`와 ±1.0 안 (층 수에 둔감) | |
| HLP114 | `encb_ex10` < HL06 `encb_ex20` (극단을 덜 빼면 낮다) | |
| HLP115 | `encb_ex40` ≤ HL06 `encb_ex20` + 0.5 | |
| HLP116 | `nbr_strat5` ≥ HL06 `encb_strat5` − 0.5 | |
| HLP117 | `xidf_strat5` ≤ HL06 `encb_strat5` − 1.0 (비구조 판정기는 순도가 낮다) | |
| HLP118 | `clu_strat5`는 HL06 `encb_strat5`와 ±1.5 안 | |
