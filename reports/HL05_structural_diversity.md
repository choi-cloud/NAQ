# HL05 — 구조적으로 가까운 same-class vs 먼 same-class query (oracle, 10 run)

작성 2026-10-25 · Agent · 지시서 `instructions/HL05_structural_diversity.md` (auto) · 브랜치 `hl/anchor`
서버 `gpusystem`, GPU 3번(`CUDA_VISIBLE_DEVICES=3`, `--device 0`), 환경 `NAQ`(추가 설치 없음). Amazon_clothing.

레이블 사용 범위: oracle query 선택(GT-same 제한, 부족 시 GT-diff 보충)과 진단 통계(순도, 임베딩 진단의 대상 query)에만.

## ① 실행 명령 · 환경 · commit

| 태그 | commit | 메시지 |
|---|---|---|
| (사전) | `4b686f7` | HL05: CLAUDE.md rev6, HL05 instruction |
| `hl05` | (이 보고서와 같은 커밋) | HL05: structural near/far same-class oracle |

§0: HEAD `4b686f7` → `ad5185c`(HL04: report). 추적 파일 변경 없음(④).

```bash
# 리포 루트, CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD, python = /home/h2tae/anaconda3/envs/NAQ/bin/python
python hl/hl000e_run_hl05.py --jobs 3        # near, far × seed 1212–1216 → results/hl/hl05/<mode>/seed_<seed>/
python hl/hl_hl05_tables.py t8 ; python hl/hl_hl05_tables.py t9 ; python hl/hl_hl05_tables.py prereg
```
새 스크립트: `hl/hl009_structural_oracle_train.py`(hl006 복사), `hl/hl000e_run_hl05.py`(hl000d 복사), `hl/hl_hl05_tables.py`.
run wall: 3개 동시 124.1–130.1 s, 마지막 단독 52.1 s. run 안 elapsed 110.7–115.5 s(단독 38.6 s; 임베딩 진단 포함). run당 `max_memory_reserved` 8.22 GB.

## ② 표

### T8 `T8_structural.csv` (참고 행은 HL03 T0·T6, HL04 T7에서 옮김)
| arm | n | 1-shot test | 5-shot test | Δ1 vs B_Choi | Δ5 vs B_Choi | Δ1 vs G1 p=1.0 | Δ5 vs G1 p=1.0 | 실제 순도 | 평균 X 순위 | 평균 Â²X | 보충 비율 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| near | 5 | 85.49 ± 0.59 | 92.60 ± 0.25 | +2.78 | +2.69 | -3.93 | -1.57 | 0.9916 | 574.9 | 0.5896 | 0.0503 |
| far | 5 | 89.46 ± 0.36 | 94.40 ± 0.24 | +6.75 | +4.49 | +0.04 | +0.23 | 0.9916 | 588.2 | 0.2665 | 0.0503 |
| B_Choi uniform (HL03 T0) | 5 | 82.71 ± 0.31 | 89.91 ± 0.57 | +0.00 | +0.00 | -6.71 | -4.26 | 0.42 | 750.4 |  |  |
| G1 p=1.0 (HL03 T6) | 5 | 89.42 ± 0.26 | 94.17 ± 0.20 | +6.71 | +4.26 | +0.00 | +0.00 | 0.9916 | 532.8 |  |  |
| HL04 comb_top10 | 5 | 79.41 ± 0.62 | 89.69 ± 0.28 | -3.30 | -0.22 | -10.01 | -4.48 | 0.5297 | 472.9 |  |  |
| HL04 rand10 | 5 | 80.34 ± 0.39 | 88.82 ± 0.27 | -2.37 | -1.09 | -9.08 | -5.35 | 0.1787 | 600.4 |  |  |

near·far의 순도 0.9916 < 1.0은 B4+B5 GT-same이 10개 미만인 anchor에서 B4 GT-diff로 채운 결과(`diff_fill`, 아래 표). 평균 Â²X는 z2(`hl_common.structural_scores`) 코사인.

### 보충 단계별 비율(전체 query 대비, seed별 `T8b_structural_per_seed.csv`)
| arm | seed | test_1shot | test_5shot | valid_1shot | supplement_query_frac | adjacent_query_frac | across_query_frac | b5_query_frac | b4_centre_query_frac | diff_fill_query_frac |
|---|---|---|---|---|---|---|---|---|---|---|
| near | 1212 | 85.41 | 92.86 | 81.1 | 0.04847000000000001 | 0.01571 | 0.00596 | 0.01849 | 0.00067 | 0.00764 |
| near | 1213 | 86.54 | 92.53 | 81.15 | 0.05286000000000002 | 0.017 | 0.00568 | 0.02033 | 0.00095 | 0.0089 |
| near | 1214 | 85.16 | 92.83 | 81.2 | 0.050449999999999995 | 0.01732 | 0.00573 | 0.01811 | 0.00109 | 0.0082 |
| near | 1215 | 85.14 | 92.25 | 81.35 | 0.050429999999999975 | 0.01666 | 0.00567 | 0.01798 | 0.00137 | 0.00875 |
| near | 1216 | 85.22 | 92.51 | 81.4 | 0.04945999999999995 | 0.01564 | 0.0059 | 0.01857 | 0.00084 | 0.00851 |
| far | 1212 | 89.96 | 94.38 | 84.75 | 0.04847000000000001 | 0.01571 | 0.00596 | 0.01849 | 0.00067 | 0.00764 |
| far | 1213 | 89.62 | 94.71 | 83.95 | 0.05286000000000002 | 0.017 | 0.00568 | 0.02033 | 0.00095 | 0.0089 |
| far | 1214 | 88.99 | 94.49 | 84.15 | 0.050449999999999995 | 0.01732 | 0.00573 | 0.01811 | 0.00109 | 0.0082 |
| far | 1215 | 89.33 | 94.06 | 83.45 | 0.050429999999999975 | 0.01666 | 0.00567 | 0.01798 | 0.00137 | 0.00875 |
| far | 1216 | 89.39 | 94.35 | 83.1 | 0.04945999999999995 | 0.01564 | 0.0059 | 0.01857 | 0.00084 | 0.00851 |

near와 far는 seed별 보충 단계 비율이 모두 같다(같은 anchor에 같은 할당).

### T9 `T9_embed_diag.csv` (seed 1212 run, 학습 전, 처음 500 에피소드, 고정 체크포인트)
| arm | checkpoint | n_episodes | n_queries | dist_mean | dist_median | p_own_mean | p_own_median | p_own_ge_095_frac | loss_mean |
|---|---|---|---|---|---|---|---|---|---|
| near | bchoi | 500 | 25000 | 0.1236 | 0.0534 | 0.7929 | 0.9329 | 0.4564 | 0.409 |
| near | naq | 500 | 25000 | 0.1484 | 0.0861 | 0.7865 | 0.9226 | 0.4284 | 0.4432 |
| far | bchoi | 500 | 25000 | 0.2878 | 0.2435 | 0.5931 | 0.6533 | 0.1805 | 0.955 |
| far | naq | 500 | 25000 | 0.3358 | 0.3111 | 0.5411 | 0.5766 | 0.1201 | 1.1462 |

`bchoi` = `results/hl/p0/Amazon_clothing/ours/tau_0.3/seed_1212/best_1shot.pt`, `naq` = `.../naq/baseline/seed_1212/best_1shot.pt`. 거리 = 1 − cos(anchor, query), p_own = `hl_common.proto_softmax`의 자기 anchor 확률(원시 임베딩, 제곱 유클리드), 손실 = −log p_own. 다섯 seed 모두 계산했고 `T9b_embed_diag_all_seeds.csv`에 있다.

## ③ 사전 등록 대조표 (`prereg_HL05.csv`, 계산값만)
| 번호 | 예측 | 계산값 |
|---|---|---|
| HLP050 | 핵심: far − near ≥ +3.0 (1-shot, 5-seed 평균) | 89.46 − 85.49 = +3.97 |
| HLP051 | near 1-shot ≤ B_Choi + 1.0 (= 83.71) | near 85.49 |
| HLP052 | far 1-shot ≥ G1 p=1.0 − 1.0 (= 88.42) | far 89.46 |
| HLP053 | far − near (5-shot) > 0 | 94.40 − 92.60 = +1.80 |
| HLP054 | T9 (B_Choi ckpt): near 거리 중앙값 < far의 절반 | near 0.0534 vs far/2 0.1217 (far 0.2435) |
| HLP055 | T9 (B_Choi ckpt): near p_own ≥ 0.95 비율 ≥ 0.5, far ≤ 0.2 | near 0.4564, far 0.1805 |
| HLP056 | T8: 두 arm의 평균 X 순위 차이 ≤ 50 | far 588.2 − near 574.9 = +13.3 |

## ④ 실패 · 경고 · 이상
- 10 run 모두 완료, 실패 없음. push 실패 없음. deprecation 경고 외 경고 없음.
- §0 `git status`: 추적 파일 변경은 없었고, 비추적은 `out/`, `results/HL04_chat_summary.txt`, `results/hl/hl02–hl04/`뿐이었다. 진행.
- 보충 규칙 구현(지시서 문구의 해석):
  - "인접 소구간"은 S1←S2, S2←S1, S3←S4, S4←S3. "그다음 S2↔S3"은 S1·S2의 남은 부족분을 S3에서, S3·S4의 남은 부족분을 S2에서 채우는 것으로 구현.
  - B5 보충도 Â²X 순서(near 상위, far 하위). "B4 same-GT 중 아직 안 뽑힌 것을 X 중앙부터"는 남은 B4 GT-same을 X 순위로 정렬해 중앙에서 뽑음.
  - 그래도 10개가 안 되는 경우는 지시서에 정의가 없어, G1(hl006)과 같이 **B4 GT-diff의 X 중앙**에서 채우고 `diff_fill`로 기록했다(전체 query의 0.76–0.89%). 이 때문에 실제 순도는 0.9916.
  - 소구간 안 동률은 X 순위 순.
- 임베딩 진단은 학습 query 풀의 처음 500 에피소드(이 mode의 query)로 계산했고, 모든 seed에서 계산했다(T9는 seed 1212).
- `hl009`은 `hl004_embed_check.embed`를 import해 임베딩을 계산했다(HL02 E2와 같은 방식: p_own은 원시 임베딩, 거리는 정규화 임베딩).
- 시작 전 `hl009`을 `--epoch 20`으로 near·far 두 번 scratchpad에서 돌려 동작을 확인했다(결과 디렉터리 밖, 표에 미포함).

## ⑤ 산출물 경로
- 표(커밋): `reports/HL05_tables/{T8_structural, T8b_structural_per_seed, T9_embed_diag, T9b_embed_diag_all_seeds, prereg_HL05}.csv`
- 결과(미추적): `results/hl/hl05/<mode>/seed_<seed>/`(report.json, best_{1,5}shot.pt), `results/hl/hl05/summary.csv`, `results/hl/hl05/logs/`
- 로그(미추적): `out/hl/hl05.log`
