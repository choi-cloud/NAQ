# HL07 — Clothing: 순도 × 거리 2D oracle 격자 + 레이블 없는 변형 7 arm

지시서 `instructions/HL07_clothing_grid_variants.md` · night-auto · Amazon_clothing, τ = 0.3 · 기준 B_Choi 82.71 / 89.91

레이블 사용 범위: §1 oracle 격자 arm(`o_grid_*`)의 query 선택, 모든 arm의 사후 query 진단, `encb_strat5_drop2`의 제외 query FP 비율 기록(학습 손실 계산·제외 선택에는 레이블을 쓰지 않음).

## ① 실행 명령·환경·commit

- 시작 HEAD `3921839`(HL06 끝). 환경 `NAQ`. GPU 2·3: 시작(00:03 KST) 둘 다 0 MiB(`results/hl/hl07/gpu_status_start.txt`). 85 run 분포 GPU 2 43 / GPU 3 42, run당 예약 메모리 최대 8.22 GB, wall 84–134초.
- 코드: 새 스크립트 없음. `hl/hl011_pdd_train.py`에 arm 추가(`rand_strat5`, `choi_far`, `choi_strat`, `encb_strat5_wide`(X 51–1000), `encb_strat5_b45`(X 201–3000), `encb_strat5_r2`, `encb_strat5_drop2`, `o_grid_<region>_p<pp>`), `hl/hl_pdd.py`에 `strata_of` 추가. 표 `hl/hl_hl07_tables.py`.
- smoke: 17 arm `--epoch 20 --seeds 1212 --gpus 2`(00:00–00:02, 모두 통과). `encb_strat5_drop2`는 epoch 20에서 워밍업(400)에 닿지 않아 `--epoch 420` 단독 smoke를 GPU 2에서 따로 돌림(통과, 제외 FP 비율 0.795). smoke 출력 삭제.
- 큐(00:03–00:31 KST): `python -u hl/hl000f_run_hl06.py --arms rand_strat5,choi_far,choi_strat,encb_strat5_wide,encb_strat5_b45,encb_strat5_r2,encb_strat5_drop2,o_grid_t1_p03,o_grid_t1_p05,o_grid_t1_p07,o_grid_t1_p10,o_grid_strat5_p03,o_grid_strat5_p05,o_grid_strat5_p07,o_grid_strat5_p10,o_grid_far50_p07,o_grid_far50_p10 --seeds 1212,1213,1214,1215,1216 --gpus 2,3 --jobs_per_gpu 3 --deadline 03:00 --output_root results/hl/hl07/train/Amazon_clothing` — §2 arm을 큐 앞에 둔 한 큐. 85/85 완료.
- `encb_strat5_r2` teacher: `results/hl/hl06/train/Amazon_clothing/encb_strat5/seed_1212/best_1shot.pt`(첫 후보가 있어 그것을 씀; 모든 seed 동일).
- 표: `PYTHONPATH=$PWD python hl/hl_hl07_tables.py {t11,t12,prereg}`. 커밋 `HL07: purity-distance grid and label-free variants`, 태그 `hl07`.

## ② 표

### T11 격자 (`T11_grid.csv`; HL06·HL03·HL05 행은 참고)

| region | p | arm | n_seeds | test_1shot_mean | test_1shot_std | test_5shot_mean | test_5shot_std | delta_1shot_vs_bchoi | purity | mean_s2 | mean_s2rank | supplement_query_frac | secondary_query_frac | b5_query_frac | diff_fill_query_frac | band_diff_fill_query_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t1 | 0.3 | o_grid_t1_p03 | 5 | 82.68 | 0.19 | 90.46 | 0.49 | -0.03 | 0.2987 | 0.4788 | 87.5947 | 0.0305 | 0.0171 | 0.0066 | 0.0013 | 0.0055 |
| t1 | 0.5 | o_grid_t1_p05 | 5 | 84.47 | 0.77 | 91.78 | 0.49 | 1.76 | 0.4971 | 0.4897 | 82.4645 | 0.0422 | 0.0217 | 0.0149 | 0.0029 | 0.0026 |
| t1 | 0.7 | o_grid_t1_p07 | 5 | 86.0 | 0.89 | 92.44 | 0.46 | 3.29 | 0.695 | 0.4988 | 78.1679 | 0.0606 | 0.029 | 0.0261 | 0.005 | 0.0005 |
| t1 | 1.0 | o_grid_t1_p10 | 5 | 87.2 | 0.75 | 93.12 | 0.38 | 4.49 | 0.9908 | 0.5096 | 73.3447 | 0.1024 | 0.0442 | 0.049 | 0.0092 |  |
| strat5 | 0.3 | o_grid_strat5_p03 | 5 | 84.45 | 0.23 | 91.15 | 0.3 | 1.74 | 0.2988 | 0.3069 | 385.3063 | 0.0628 | 0.0605 | 0.0012 | 0.0012 |  |
| strat5 | 0.5 | o_grid_strat5_p05 | 5 | 87.53 | 0.79 | 93.15 | 0.36 | 4.82 | 0.4973 | 0.3206 | 370.1242 | 0.1087 | 0.1029 | 0.0031 | 0.0027 |  |
| strat5 | 0.7 | o_grid_strat5_p07 | 5 | 89.01 | 0.74 | 94.1 | 0.23 | 6.3 | 0.6954 | 0.3354 | 353.1139 | 0.1606 | 0.1496 | 0.0064 | 0.0046 |  |
| strat5 | 1.0 | o_grid_strat5_p10 | 5 | 90.12 | 0.17 | 94.74 | 0.35 | 7.41 | 0.9916 | 0.3585 | 325.144 | 0.2474 | 0.2247 | 0.0143 | 0.0084 |  |
| far50 | 0.3 | HL06 o_far50_p03 | 5 | 79.49 | 0.83 | 88.1 | 0.99 | -3.22 | 0.2739 | 0.2094 | 578.1065 | 0.0644 |  |  | 0.0261 |  |
| far50 | 0.5 | HL06 o_far50_p05 | 5 | 84.15 | 0.31 | 91.33 | 0.37 | 1.44 | 0.4461 | 0.2206 | 556.945 | 0.1282 |  |  | 0.0539 |  |
| far50 | 0.7 | o_grid_far50_p07 | 5 | 88.68 | 0.22 | 93.96 | 0.29 | 5.97 | 0.6859 | 0.2555 | 531.3319 | 0.2033 | 0.1155 | 0.0737 | 0.0141 |  |
| far50 | 1.0 | o_grid_far50_p10 | 5 | 89.67 | 0.38 | 94.49 | 0.25 | 6.96 | 0.9735 | 0.2866 | 492.2181 | 0.3336 | 0.1838 | 0.1233 | 0.0265 |  |
| t1-top (s2 top same) | 1.0 | HL06 o_topsame | 5 | 84.34 | 0.68 | 92.28 | 0.28 | 1.63 | 0.9916 | 0.6209 | 42.486 | 0.0227 |  | 0.0143 | 0.0084 |  |
| reference |  | G1 p=0.4 (HL03 T6) | 5 | 83.92 | 0.4 | 91.12 | 0.24 | 1.21 | 0.3984 | 0.3257 | 367.4037 |  |  |  |  |  |
| reference |  | G1 p=1.0 (HL03 T6) | 5 | 89.42 | 0.26 | 94.17 | 0.2 | 6.71 | 0.9929 | 0.4225 | 234.7454 |  |  |  |  |  |
| reference |  | HL05 near (T8) | 5 | 85.49 | 0.59 | 92.6 | 0.25 | 2.78 | 0.9929 | 0.5901 | 64.0219 |  |  |  |  |  |
| reference |  | HL05 far (T8) | 5 | 89.46 | 0.36 | 94.4 | 0.24 | 6.75 | 0.9929 | 0.2662 | 471.514 |  |  |  |  |  |

### T12 레이블 없는 변형 (`T12_variants.csv`)

| arm | n_seeds | test_1shot_mean | test_1shot_std | test_5shot_mean | test_5shot_std | delta_1shot_vs_bchoi | delta_5shot_vs_bchoi | purity | mean_x_rank | mean_s2 | mean_s2rank | pair_s2 | hop2_frac | choi_pass_frac | n_passed_mean | relax_mean | half_fallback_mean | dropped_fp_frac | post_warmup_query_fp_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rand_strat5 | 5 | 79.49 | 0.93 | 88.13 | 0.48 | -3.22 | -1.78 | 0.1782 | 600.2242 | 0.2957 | 400.4463 | 0.2491 | 0.031 | 0.2385 |  |  |  |  |  |
| choi_far | 5 | 82.53 | 0.44 | 89.44 | 0.23 | -0.18 | -0.47 | 0.3641 | 1002.106 | 0.4736 | 232.1658 | 0.5551 | 0.1129 | 0.9814 | 520.5899 | 0.0351 | 0.0344 |  |  |
| choi_strat | 5 | 82.57 | 1.21 | 90.23 | 0.21 | -0.14 | 0.32 | 0.4274 | 881.0847 | 0.5686 | 198.4146 | 0.5562 | 0.251 | 0.9814 | 520.5899 | 0.0351 | 0.0 |  |  |
| encb_strat5_wide | 5 | 80.32 | 0.55 | 87.5 | 0.78 | -2.39 | -2.41 | 0.3487 | 510.0071 | 0.3867 | 434.3758 | 0.3169 | 0.1595 | 0.3544 |  |  |  |  |  |
| encb_strat5_b45 | 5 | 80.39 | 0.41 | 86.65 | 0.79 | -2.32 | -3.26 | 0.2538 | 1507.6492 | 0.3017 | 1278.6818 | 0.2255 | 0.1421 | 0.2204 |  |  |  |  |  |
| encb_strat5_r2 | 5 | 77.95 | 0.48 | 85.68 | 0.55 | -4.76 | -4.23 | 0.3196 | 591.2925 | 0.3559 | 373.2479 | 0.3141 | 0.133 | 0.3023 |  |  |  |  |  |
| encb_strat5_drop2 | 5 | 78.61 | 0.38 | 85.88 | 1.03 | -4.1 | -4.03 | 0.325 | 591.2658 | 0.3616 | 367.7269 | 0.3021 | 0.1374 | 0.3125 |  |  |  | 0.7754 | 0.6737 |
| HL06 encb_strat5 | 5 | 80.2 | 0.47 | 87.6 | 0.71 | -2.51 | -2.31 | 0.325 | 591.2699 | 0.3616 | 367.7264 | 0.3021 | 0.1374 | 0.3125 |  |  |  |  |  |
| HL06 encb_top10 | 5 | 79.69 | 0.63 | 89.11 | 0.45 | -3.02 | -0.8 | 0.5226 | 490.0973 | 0.6571 | 45.7677 | 0.7219 | 0.4915 | 0.7144 |  |  |  |  |  |
| HL04 rand10 (T7) | 5 | 80.34 | 0.39 | 88.82 | 0.27 | -2.37 | -1.09 | 0.1795 | 600.0812 | 0.2939 | 402.985 | 0.2577 | 0.0316 | 0.2345 |  |  |  |  |  |
| B_Choi uniform (HL03 T0) | 5 | 82.71 | 0.31 | 89.91 | 0.57 | 0.0 | 0.0 | 0.4328 | 790.0185 | 0.5613 | 167.234 | 0.6073 | 0.2426 | 0.9841 |  |  |  |  |  |

## ③ 사전 등록 대조표 (계산값만)

| id | prediction | computed |
|---|---|---|
| HLP080 | `o_grid_t1_p10` 1-shot ≤ 84.5 | 87.20 |
| HLP081 | `o_grid_strat5_p10` ≥ 88.5 | 90.12 |
| HLP082 | `o_grid_far50_p10` ≥ 89.0 | 89.67 |
| HLP083 | p=0.5에서 `strat5` − `t1` ≥ +3.0 | 87.53 − 84.47 = +3.06 |
| HLP084 | `o_grid_strat5_p03` ≥ B_Choi − 0.5 (= 82.21) | 84.45 |
| HLP085 | 세 region 모두 1-shot이 p에 대해 단조 증가 | t1 82.68/84.47/86.00/87.20 True; strat5 84.45/87.53/89.01/90.12 True; far50 79.49/84.15/88.68/89.67 True |
| HLP086 | `rand_strat5` ≤ HL04 `rand10` + 1.0 (= 81.34) | 79.49 |
| HLP087 | HL06 `encb_strat5` − `rand_strat5` ≥ +2.0 | 80.20 − 79.49 = +0.71 |
| HLP088 | `choi_far` − B_Choi ≥ +0.5 | 82.53 − 82.71 = -0.18 |
| HLP089 | `choi_strat` − B_Choi ≥ +0.5 | 82.57 − 82.71 = -0.14 |
| HLP090 | |`encb_strat5_wide` − HL06 `encb_strat5`| ≤ 1.0 | 80.32 − 80.20 = +0.12 |
| HLP091 | `encb_strat5_r2` ≥ HL06 `encb_strat5` | 77.95 − 80.20 = -2.25 |
| HLP092 | `encb_strat5_drop2` ≥ HL06 `encb_strat5` − 0.3, 제외된 query의 FP 비율 ≥ 0.7 | 78.61 − 80.20 = -1.59; 제외 FP 비율 0.7754 |

## ④ 실패·경고·이상

- 85 run 실패·timeout 없음. 경고는 `utils.py:83` deprecation뿐.
- 지시서에 정해지지 않아 정한 것:
  - `o_grid`: t1 영역에서 GT-diff가 T1·T2 모두에서 모자라는 anchor가 있었다(CPU 점검에서 T1 160개 중 GT-same 159–160개인 anchor). 지시서의 보충 순서(영역 GT-diff `diff_fill`)로도 모자라면 B4 전체 GT-diff에서 무작위로 채우고 `band_diff_fill`로 따로 기록했다(t1 p03 0.0055, p05 0.0026, p07 0.0005).
  - `o_grid` strat5의 "인접 층"은 층 번호 거리 순(같은 거리면 번호가 큰 층 먼저). B5 GT-same 보충은 무작위.
  - `o_grid`의 rng(`default_rng(seed + 7)`)는 슬롯 종류 선택과 각 슬롯의 무작위 추출에 같은 생성기를 anchor 순서대로 소비.
  - `choi_far`: 필터 통과분을 s2 내림차순(동률은 X 순위)으로 정렬해 뒤쪽 ⌊n/2⌋개를 "하위 절반"으로 썼다. X 순위 정렬·relax는 `hl_common.verify_against_exp049`와 같이 z0(행 정규화 X) 코사인. `half_fallback_mean` 0.0344, relax 0.0351(anchor 출현 기준).
  - `choi_strat`: 통과분을 s2 내림차순으로 `np.array_split` 5등분, 각 분위에서 X 중앙 2개. 2개가 안 되는 분위는 층 번호 거리 순으로 다른 분위의 남은 노드(X 중앙)로 채움.
  - `encb_strat5_drop2`: 401번째 에피소드부터 anchor별 10개 query 중 손실(−log p(자기 anchor), 현재 네트워크, 기울기 없이 순위만) 상위 2개를 평균에서 뺐다. 손실 식은 `meta.ProtoNet.forward`와 같고 `meta.py`는 고치지 않았다. 정확도(train_acc)는 10개 전체로 계산.
  - 사후 기술 통계의 `encb`·`pair_encb`는 모든 arm에서 고정 enc_b(seed 1212 B_Choi) 임베딩으로 계산(`encb_strat5_r2`도 같음).
- `encb_strat5_drop2`의 query 선택은 `encb_strat5`와 같다. 그래서 두 arm의 기술 통계가 소수 넷째 자리까지 같다.

## ⑤ 산출물

- 표: `reports/HL07_tables/` — `T11_grid.csv`, `T11b_grid_per_seed.csv`, `T12_variants.csv`, `T12b_variants_per_seed.csv`, `prereg_HL07.csv`.
- 결과(git 미추적): `results/hl/hl07/train/Amazon_clothing/<arm>/seed_<seed>/`, `summary.csv`, `logs/`, `results/hl/hl07/gpu_status_{start,end}.txt`.
- 로그: `out/hl/hl07_train.log`, `out/hl/hl07_smoke.log`.
