# HL10 — Clothing 채움 실험: 층 수·제외 폭·판정기 교체 8 arm

지시서 `instructions/HL10_clothing_fillers.md` · night-auto · Amazon_clothing, τ = 0.3, teacher enc_b(seed 1212) · 기준 B_Choi 82.71 / 89.91

레이블 사용 범위: 모든 arm은 레이블 없는 선택이며, 레이블은 사후 query 진단에만 쓴다.

## ① 실행 명령·환경·commit

- 시작 HEAD `180da97`(HL09 끝). GPU 2·3 시작(01:38 KST)·끝(01:52 KST) 모두 0 MiB(`results/hl/hl10/gpu_status_{start,end}.txt`).
- 코드: 새 스크립트 없음. `hl/hl_pdd.py`의 규칙 `strat3`·`strat10`·`ex10`·`ex40`과 `hl/hl011_pdd_train.py`의 arm 8개는 HL07 커밋(`7c01ef1`)에 이미 들어 있다. 표 `hl/hl_hl10_tables.py`.
- smoke: 8 arm `--epoch 20 --seeds 1212 --gpus 3 --jobs_per_gpu 1`(01:15–01:17 KST, HL09 큐와 같은 GPU에서 세 번째 작업), 모두 통과, 출력 삭제.
- 큐(01:38–01:52 KST, `out/hl/hl10_chain.sh`가 HL09 큐 종료 뒤 시작): `python -u hl/hl000f_run_hl06.py --arms encb_strat10,encb_strat3,encb_ex10,encb_ex40,nbr_strat5,xidf_strat5,clu_strat5,knnppr_far50 --seeds 1212,1213,1214,1215,1216 --gpus 2,3 --jobs_per_gpu 3 --deadline 07:15 --output_root results/hl/hl10/train/Amazon_clothing`. 40/40 완료. GPU 2 21 / GPU 3 19 run, wall 70–160초.
- 표: `PYTHONPATH=$PWD python hl/hl_hl10_tables.py {t17,prereg}`. 커밋 `HL10: clothing filler variants`, 태그 `hl10`.

## ② 표

### T17 (`T17_fillers.csv`; HL06 행·B_Choi는 참고)

| arm | n_seeds | test_1shot_mean | test_1shot_std | test_5shot_mean | test_5shot_std | delta_1shot_vs_bchoi | delta_5shot_vs_bchoi | purity | mean_x_rank | mean_s2 | mean_s2rank | pair_s2 | hop2_frac | choi_pass_frac | supplement_query_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| encb_strat10 | 5 | 80.67 | 0.33 | 87.27 | 1.07 | -2.04 | -2.64 | 0.2993 | 601.0807 | 0.3252 | 390.2715 | 0.2735 | 0.0833 | 0.267 | 0.0 |
| encb_strat3 | 5 | 80.05 | 0.66 | 87.47 | 0.81 | -2.66 | -2.44 | 0.3415 | 584.0591 | 0.3932 | 354.8274 | 0.3297 | 0.1896 | 0.3539 | 0.0 |
| encb_ex10 | 5 | 79.3 | 0.86 | 87.49 | 0.85 | -3.41 | -2.42 | 0.3874 | 547.8242 | 0.4105 | 182.5778 | 0.5687 | 0.0584 | 0.415 | 0.0 |
| encb_ex40 | 5 | 79.3 | 0.55 | 86.95 | 0.4 | -3.41 | -2.96 | 0.274 | 613.4696 | 0.2778 | 433.6154 | 0.4106 | 0.0018 | 0.2077 | 0.0 |
| nbr_strat5 | 5 | 81.05 | 0.44 | 88.39 | 0.32 | -1.66 | -1.52 | 0.3287 | 586.9751 | 0.354 | 374.8429 | 0.2755 | 0.1301 | 0.3075 | 0.0 |
| xidf_strat5 | 5 | 77.5 | 0.58 | 88.48 | 0.48 | -5.21 | -1.43 | 0.1952 | 407.5428 | 0.3182 | 392.9104 | 0.2563 | 0.0814 | 0.2807 | 0.0 |
| clu_strat5 | 5 | 81.02 | 0.41 | 88.31 | 0.79 | -1.69 | -1.6 | 0.3282 | 588.7221 | 0.3529 | 372.7208 | 0.2902 | 0.1213 | 0.3031 | 0.0 |
| knnppr_far50 | 5 | 79.45 | 0.51 | 88.39 | 0.39 | -3.26 | -1.52 | 0.229 | 618.0253 | 0.2328 | 540.986 | 0.2807 | 0.0015 | 0.1437 | 0.0 |
| HL06 encb_strat5 | 5 | 80.2 | 0.47 | 87.6 | 0.71 | -2.51 | -2.31 | 0.325 | 591.2699 | 0.3616 | 367.7264 | 0.3021 | 0.1374 | 0.3125 | 0.0 |
| HL06 encb_ex20 | 5 | 80.02 | 0.3 | 87.71 | 0.45 | -2.69 | -2.2 | 0.3356 | 571.4081 | 0.3502 | 273.3642 | 0.504 | 0.0129 | 0.3203 | 0.0 |
| HL06 encb_far50 | 5 | 78.84 | 0.45 | 86.2 | 0.42 | -3.87 | -3.71 | 0.2416 | 630.4538 | 0.2489 | 507.4297 | 0.3686 | 0.0007 | 0.1703 | 0.0 |
| HL06 encb_top10 | 5 | 79.69 | 0.63 | 89.11 | 0.45 | -3.02 | -0.8 | 0.5226 | 490.0973 | 0.6571 | 45.7677 | 0.7219 | 0.4915 | 0.7144 | 0.0 |
| B_Choi uniform (HL03 T0) | 5 | 82.71 | 0.31 | 89.91 | 0.57 | 0.0 | 0.0 | 0.4328 | 790.0185 | 0.5613 | 167.234 | 0.6073 | 0.2426 | 0.9841 |  |

## ③ 사전 등록 대조표 (계산값만)

| id | prediction | computed |
|---|---|---|
| HLP113 | `encb_strat10`·`encb_strat3`은 HL06 `encb_strat5`와 ±1.0 안 | 80.67 − 80.20 = +0.47; 80.05 − 80.20 = -0.15 |
| HLP114 | `encb_ex10` < HL06 `encb_ex20` | 79.30 − 80.02 = -0.72 |
| HLP115 | `encb_ex40` ≤ HL06 `encb_ex20` + 0.5 | 79.30 − 80.02 = -0.72 |
| HLP116 | `nbr_strat5` ≥ HL06 `encb_strat5` − 0.5 | 81.05 − 80.20 = +0.85 |
| HLP117 | `xidf_strat5` ≤ HL06 `encb_strat5` − 1.0 | 77.50 − 80.20 = -2.70 |
| HLP118 | `clu_strat5`는 HL06 `encb_strat5`와 ±1.5 안 | 81.02 − 80.20 = +0.82 |

## ④ 실패·경고·이상

- 40 run 실패·timeout 없음. 경고는 `utils.py:83` deprecation뿐.
- 예약 메모리가 8.22 GB를 넘은 run: `xidf_strat5` 9.9 GB(TF-IDF 가중 특징 행렬을 GPU에 따로 둔다).
- 규칙 경계(지시서에 m = 800 기준 수치만 있어 일반식으로 구현): `strat3` = s2 순위 경계 round(m/3), round(2m/3)(800에서 267, 533 — 지시서의 T2' 268–534와 한 칸 차이: T2' = 268–533, T3' = 534–800), `strat10` = round(i·m/10), `ex10` = s2 순위 > m//10, `ex40` = s2 순위 > 2m//5.
- `clu_strat5`는 run마다 sklearn KMeans(200 군집, n_init 1, random_state 0–4)를 다시 계산한다.

## ⑤ 산출물

- 표: `reports/HL10_tables/` — `T17_fillers.csv`, `T17b_fillers_per_seed.csv`, `prereg_HL10.csv`.
- 결과(git 미추적): `results/hl/hl10/train/Amazon_clothing/<arm>/seed_<seed>/`, `summary.csv`, `logs/`.
- 로그: `out/hl/hl10_train.log`, `out/hl/hl10_smoke.log`, `out/hl/hl10_chain.log`.
