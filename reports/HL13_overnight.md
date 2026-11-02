# HL13 — 야간 시퀀스: 사영 학습 · lr 스윕 · soft target · GCN 혼합 · oracle l2 · inductive PCA · 10-seed · 노름 진단

지시서 `instructions/HL13_overnight_mix_soft_oracle.md` · night-auto · Amazon_clothing · 서버 seio · 환경 `taskns`

레이블 사용 범위: base 40클래스 레이블은 `sup`·`mix`에, 모든 클래스 레이블은 `oracle_strat5`(oracle)에 사용했다. 고정 valid·test 에피소드 레이블은 meta-test 평가에, 고정 valid·test 에피소드의 클래스 id는 base에서 제외하는 데에, `query_diagnostics`와 `hl016`은 진단·채점에 읽었다. `pca64_base`는 base 40클래스 노드 집합만 썼고 레이블 값은 쓰지 않았다. 그 밖의 PCA에는 레이블을 쓰지 않았다.

## ① 실행 명령·환경·commit

- 시작 HEAD `2d85e4d`, 브랜치 `hl/anchor`. 커밋·태그: `3941837`(`hl13-setup`), `a882aa4`(`hl13-proj`), `ba4c56f`(`hl13-lr`), `8544e6f`(`hl13-soft`), `7ef80ed`(`hl13-mix`), `08093cf`(`hl13-oracle`), `0fbfd6b`(`hl13-pca`), `687e52f`(`hl13-seeds`), `983a690`(`hl13-diag`), `2756216`(단계 9, 태그 없음).
- 환경: seio, RTX 4080 SUPER 1장, torch 2.4.1+cu121, PyG 2.6.1, scipy 1.13.1, sklearn 1.5.2. 설치 없음. nvidia-smi 시작·끝 2 / 16376 MiB(`results/hl/hl13/env_{start,end}.txt`).
- md5: 1-shot `5f901727f81903cc12159d280be6915d`, 5-shot `ca3edaca02d541cbc0ee15a2122ef24a`. 시작과 큐 종료 후 모두 CLAUDE.md §3과 일치.
- 재사용: `results/hl/hl12/feat/pca64.npy`(`3d432fd1110e689f0e722056d23b2242`), `pca256.npy`(`eb97f813c2d6a5c2c76900164540be32`).
- 새 특징: `results/hl/hl13/feat/pca1024.npy`(`56f0823efbdacbef1b7ce4210177577c`, 설명 분산 0.9320, test 87.9150 / 92.6300), `pca64_base.npy`(`ef3ad61ab88c76e68878c23aee1dfe2d`, fit 노드 10,711, 설명 분산 0.5664), `pca64_components.npy`, `pca64_mean.npy`.
- 고정 에피소드 seed 1217–1221: `CUDA_VISIBLE_DEVICES= python choi_exps/exp026.py --dataset Amazon_clothing --k_shot_test <shot> --seed <seed> --json_path results/hl/hl13/feat/exp026_<shot>shot_seed<seed>.json`. shot·split별 sha256이 seed 1212와 모두 일치.
- 명령(리포 루트, `PYTHONPATH=$PWD`, `/home/h2tae/.conda/envs/taskns/bin/python`):
  - `python -u hl/hl015_train_hl13.py --prepare_feat`
  - `nohup bash out/hl/hl13_chain.sh > out/hl/hl13_chain.log 2>&1 &` (단계별 `hl/hl000j_run_hl13.py --stage N --deadline <KST>`, `hl/hl016_norm_diag.py`, `hl/hl_hl13_tables.py`, 커밋·태그·push)
- 150/150 run 완료, 실패 0, deadline으로 시작 못 한 run 0. peak allocated 최대 8430 MiB.

## progress.md 전체

```
2026-11-02 22:01:17 KST | 시작 §0 | 문서 커밋 hl13-setup(3941837), md5 일치, GPU 2/16376 MiB
2026-11-02 22:09:21 KST | smoke | 통과: sgc·pcainit·pca1024·soft·mix·oracle·pca_fit 12분기(epoch 20), oracle은 NaN 직렬화 수정 후 재통과; seed 1217–1221 에피소드 생성·sha 일치
2026-11-02 22:09:21 KST | 단계 1 사영 학습 | 시작 (deadline 2026-11-02T22:50)
2026-11-02 22:21:38 KST | 단계 1 사영 학습 | 끝: 30/30 run 완료, 실패 0
2026-11-02 22:21:40 KST | 단계 1 | 커밋 a882aa4, 태그 hl13-proj
2026-11-02 22:21:40 KST | 단계 2 lr 스윕 | 시작 (deadline 2026-11-02T23:35)
2026-11-02 22:33:13 KST | 단계 2 lr 스윕 | 끝: 30/30 run 완료, 실패 0
2026-11-02 22:33:14 KST | 단계 2 | 커밋 ba4c56f, 태그 hl13-lr
2026-11-02 22:33:14 KST | 단계 3 soft target | 시작 (deadline 2026-11-03T01:00)
2026-11-02 22:54:36 KST | 단계 3 soft target | 끝: 30/30 run 완료, 실패 0
2026-11-02 22:54:38 KST | 단계 3 | 커밋 8544e6f, 태그 hl13-soft
2026-11-02 22:54:38 KST | 단계 4 GCN 혼합 | 시작 (deadline 2026-11-03T01:45)
2026-11-02 23:07:10 KST | 단계 4 GCN 혼합 | 끝: 15/15 run 완료, 실패 0
2026-11-02 23:07:12 KST | 단계 4 | 커밋 7ef80ed, 태그 hl13-mix
2026-11-02 23:07:12 KST | 단계 5 oracle l2 | 시작 (deadline 2026-11-03T02:45)
2026-11-02 23:16:05 KST | 단계 5 oracle l2 | 끝: 15/15 run 완료, 실패 0
2026-11-02 23:16:07 KST | 단계 5 | 커밋 08093cf, 태그 hl13-oracle
2026-11-02 23:16:07 KST | 단계 6 inductive PCA | 시작 (deadline 2026-11-03T03:10)
2026-11-02 23:18:02 KST | 단계 6 inductive PCA | 끝: 5/5 run 완료, 실패 0
2026-11-02 23:18:04 KST | 단계 6 | 커밋 0fbfd6b, 태그 hl13-pca
2026-11-02 23:18:04 KST | 단계 7 10-seed | 시작 (deadline 2026-11-03T03:45)
2026-11-02 23:28:46 KST | 단계 7 10-seed | 끝: 15/15 run 완료, 실패 0
2026-11-02 23:28:47 KST | 단계 7 | 커밋 687e52f, 태그 hl13-seeds
2026-11-02 23:28:47 KST | 단계 8 노름 진단 | 시작
2026-11-02 23:28:53 KST | 단계 8 노름 진단 | 끝
2026-11-02 23:28:55 KST | 단계 8 | 커밋 983a690, 태그 hl13-diag
2026-11-02 23:28:55 KST | 단계 9 여분 | 시작 (deadline 2026-11-03T05:15)
2026-11-02 23:32:46 KST | 단계 9 여분 | 끝: 10/10 run 완료, 실패 0
2026-11-02 23:32:48 KST | 단계 9 | 커밋 2756216
2026-11-02 23:32:48 KST | 큐 끝
2026-11-02 23:34:23 KST | 마감 보고서 | 작성·push (예정 08:30보다 이르게; 큐가 23:32에 끝남)
```

## ② 표

### T30 (`T30_norm_diag.csv`)

| checkpoint | norm_q10 | norm_q50 | norm_q90 | norm_std_over_mean | spearman_norm_degree | spearman_norm_a2x_rownorm | support_minus_query_norm_mean | support_minus_query_norm_std | test_1shot_raw | test_1shot_l2 | raw_wrong_queries | norm_induced_error | between_over_within_all77 | between_over_within_test20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| encb | 1.4798 | 2.0246 | 2.7614 | 0.2611 | 0.4896 | 0.5629 | -0.0088 | 0.5425 | 83.2000 | 87.7450 | 3360 | 0.3848 | 0.1597 | 0.1112 |
| encn | 1.4302 | 2.0540 | 2.9254 | 0.3000 | 0.5217 | 0.6010 | -0.0011 | 0.6550 | 80.6350 | 87.8400 | 3873 | 0.4751 | 0.1959 | 0.1497 |
| gcn_cos_naq | 0.5011 | 0.7736 | 1.4115 | 0.4955 | 0.3575 | 0.9599 | -0.0024 | 0.4415 | 64.9850 | 83.4300 | 7003 | 0.6253 | 0.1413 | 0.1328 |
| gcn_cos_ours | 0.6943 | 1.1268 | 2.0141 | 0.4879 | 0.4237 | 0.9250 | -0.0029 | 0.6379 | 70.6100 | 87.3400 | 5878 | 0.6664 | 0.1454 | 0.1927 |
| gcn_raw_sup | 0.9002 | 1.4543 | 2.4427 | 0.4245 | 0.4912 | 0.6902 | -0.0036 | 0.6193 | 71.4800 | 83.3350 | 5704 | 0.5456 | 0.3931 | 0.4487 |

### T31 (`T31_projection.csv`)

| stage | arm | n_seeds | epoch0_test_l2_1shot | bestR_test_raw_1shot | bestL_test_l2_1shot | bestL_test_l2_std_1shot | bestL_epoch_mean_1shot | bestL_incl_epoch0_test_l2_1shot | trajmax_test_l2_1shot | trajmax_epoch_1shot | final_test_l2_1shot | epoch0_test_l2_5shot | bestR_test_raw_5shot | bestL_test_l2_5shot | bestL_test_l2_std_5shot | bestL_epoch_mean_5shot | bestL_incl_epoch0_test_l2_5shot | trajmax_test_l2_5shot | trajmax_epoch_5shot | final_test_l2_5shot |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | sgc64_rand_cos_naq_lr1e-3 | 5 | 75.0430 | 82.4910 | 82.4910 | 0.6204 | 1600.0 | 82.4910 | 82.5710 | 1760 | 82.0250 | 88.2380 | 91.5080 | 91.5080 | 0.3104 | 1760.0 | 91.5080 | 91.3810 | 1920 | 91.3190 |
| 1 | sgc64_rand_cos_sup_lr1e-3 | 5 | 75.0430 | 76.6650 | 76.6650 | 1.4601 | 736.0 | 76.6650 | 77.7550 | 160 | 74.4440 | 88.2380 | 87.0960 | 87.0960 | 0.6777 | 880.0 | 87.0960 | 88.6910 | 160 | 86.0970 |
| 1 | sgc256_rand_cos_naq_lr1e-3 | 5 | 81.4940 | 81.3350 | 81.3350 | 1.2822 | 1056.0 | 81.3350 | 82.7960 | 1760 | 82.1330 | 91.4160 | 91.7030 | 91.7030 | 0.2207 | 1520.0 | 91.7370 | 91.5410 | 1920 | 91.5090 |
| 1 | sgc64_pcainit_cos_naq_lr1e-3 | 5 | 89.0500 | 87.2330 | 87.2330 | 0.7382 | 224.0 | 89.0500 | 89.0500 | 0 | 83.3300 | 93.7250 | 92.8520 | 92.8520 | 0.6543 | 832.0 | 92.8940 | 93.7250 | 0 | 91.9230 |
| 1 | sgc64_pcainit_cos_sup_lr1e-3 | 5 | 89.0500 | 88.5440 | 88.5440 | 0.2659 | 160.0 | 88.8870 | 89.0500 | 0 | 82.6350 | 93.7250 | 93.3580 | 93.3580 | 0.5413 | 512.0 | 93.3580 | 93.7590 | 160 | 91.2540 |
| 1 | pca1024_lin64_cos_naq_lr1e-3 | 5 | 75.3430 | 81.8760 | 81.8760 | 0.5510 | 1440.0 | 81.8760 | 82.6010 | 1280 | 82.2150 | 88.2580 | 91.4230 | 91.4230 | 0.2310 | 1552.0 | 91.4230 | 91.4730 | 1440 | 91.4120 |
| 2 | pca64_lin_cos_naq_lr1e-3 | 5 | 89.0450 | 88.6150 | 88.5840 | 0.7705 | 592.0 | 89.0450 | 89.0450 | 0 | 87.2230 | 93.7150 | 93.3650 | 93.4760 | 0.1064 | 1024.0 | 93.4760 | 93.7150 | 0 | 93.3330 |
| 2 | pca64_lin_cos_naq_lr1e-2 | 5 | 89.0450 | 83.0640 | 84.4350 | 0.8839 | 1024.0 | 89.0450 | 89.0450 | 0 | 85.1910 | 93.7150 | 92.3750 | 92.5650 | 0.2153 | 1440.0 | 93.4860 | 93.7150 | 0 | 92.5620 |
| 2 | pca64_lin_cos_sup_lr1e-3 | 5 | 89.0450 | 89.2240 | 89.1030 | 0.1532 | 224.0 | 89.1120 | 89.1750 | 160 | 87.5490 | 93.7150 | 93.8020 | 93.8160 | 0.0664 | 1120.0 | 93.8160 | 93.9250 | 320 | 93.5070 |
| 2 | pca64_lin_cos_sup_lr1e-2 | 5 | 89.0450 | 85.7960 | 85.5550 | 1.4735 | 544.0 | 89.0450 | 89.0450 | 0 | 81.7380 | 93.7150 | 91.8650 | 92.1790 | 0.7038 | 544.0 | 93.1280 | 93.7150 | 0 | 90.7630 |
| 2 | sgc64_rand_cos_naq_lr1e-4 | 5 | 75.0430 | 81.4440 | 81.4440 | 0.5866 | 1552.0 | 81.4440 | 82.0180 | 1600 | 81.4950 | 88.2380 | 91.6480 | 91.6480 | 0.1509 | 1584.0 | 91.6480 | 91.6680 | 1920 | 91.5880 |
| 2 | sgc64_rand_cos_naq_lr1e-2 | 5 | 75.0430 | 81.9770 | 81.9770 | 0.7196 | 1584.0 | 81.9770 | 82.5650 | 1760 | 82.0590 | 88.2380 | 91.2000 | 91.2000 | 0.3724 | 1856.0 | 91.2000 | 91.3600 | 1920 | 91.3470 |
| ref | a2x_pca64_l2 (HL11 T18, no training) |  |  |  | 89.0450 |  |  |  |  |  |  |  |  | 93.7150 |  |  |  |  |  |  |
| ref | pca64_lin_cos_naq (HL12 T22, lr 1e-4) |  |  |  | 89.1480 |  |  |  |  |  |  |  |  | 93.7640 |  |  |  |  |  |  |

### T32 (`T32_soft.csv`)

| encoder_loss | episodes | A | arm | source | n_seeds | bestL_test_l2_1shot | bestL_test_l2_std_1shot | bestL_test_l2_5shot | t_struct_agreement |
|---|---|---|---|---|---|---|---|---|---|
| gcn raw | naq | 0 | gcn_raw_naq | T22 | 5 | 87.7170 | 0.4147 | 92.1330 |  |
| gcn raw | naq | 0.3 | gcn_raw_soft0.3_naq | hl015 | 5 | 87.7210 | 0.3675 | 92.3660 | 0.8805 |
| gcn raw | naq | 0.6 | gcn_raw_soft0.6_naq | hl015 | 5 | 87.7510 | 0.2856 | 92.2660 | 0.8805 |
| gcn raw | naq | 1.0 | gcn_raw_soft1.0_naq | hl015 | 5 | 87.3250 | 0.2877 | 91.9870 | 0.8805 |
| gcn raw | ours | 0 | gcn_raw_ours | T22 | 5 | 87.0930 | 0.4003 | 91.3220 |  |
| gcn raw | ours | 0.3 | gcn_raw_soft0.3_ours | hl015 | 5 | 87.0700 | 0.5331 | 91.4780 | 0.8369 |
| pca64_lin cos | naq | 0 | pca64_lin_cos_naq | T22 (lr 1e-4) | 5 | 89.1480 | 0.0440 | 93.7640 |  |
| pca64_lin cos | naq | 0 | pca64_lin_cos_naq_lr1e-3 | hl015 | 5 | 88.5840 | 0.7705 | 93.4760 |  |
| pca64_lin cos | naq | 0.3 | pca64_lin_cos_soft0.3_naq_lr1e-3 | hl015 | 5 | 88.1110 | 0.8988 | 93.4070 | 0.8805 |
| pca64_lin cos | naq | 1.0 | pca64_lin_cos_soft1.0_naq_lr1e-3 | hl015 | 5 | 87.7180 | 0.4441 | 93.3640 | 0.8805 |

### T33 (`T33_mix.csv`)

| R | arm | source | n_seeds | bestL_test_l2_1shot | bestL_test_l2_std_1shot | trajmax_test_l2_1shot | bestL_epoch_mean_1shot | bestL_test_l2_5shot | bestL_test_l2_std_5shot | trajmax_test_l2_5shot | bestL_epoch_mean_5shot |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | gcn_raw_naq | T22 | 5 | 87.7170 | 0.4147 | 87.7180 | 1216.0 | 92.1330 | 0.2830 | 92.5290 | 640.0 |
| 0.1 | gcn_raw_mix_r0.1 | hl015 | 5 | 87.4800 | 0.5744 | 88.0920 | 1184.0 | 92.2360 | 0.2268 | 92.5640 | 1024.0 |
| 0.25 | gcn_raw_mix_r0.25 | hl015 | 5 | 87.6370 | 0.4523 | 87.8000 | 1424.0 | 92.2680 | 0.2493 | 92.4840 | 1216.0 |
| 0.5 | gcn_raw_mix_r0.5 | hl015 | 5 | 86.9190 | 0.6443 | 87.2730 | 1536.0 | 92.0300 | 0.3505 | 92.0360 | 1696.0 |
| 1 | gcn_raw_sup | T22 | 5 | 81.1870 | 1.2630 | 82.5830 | 832.0 | 90.2830 | 0.2159 | 90.3500 | 192.0 |

### T34 (`T34_oracle_l2.csv`)

| arm | n_seeds | bestR_test_raw_1shot | bestL_test_l2_1shot | bestL_test_l2_std_1shot | bestR_test_raw_5shot | bestL_test_l2_5shot | query_gt_match |
|---|---|---|---|---|---|---|---|
| gcn_raw_oracle_strat5 | 5 | 90.1120 | 92.8860 | 0.3006 | 94.7870 | 95.3450 | 0.9916 |
| pca64_lin_cos_oracle_strat5_lr1e-3 | 5 | 91.3370 | 91.3880 | 0.1814 | 94.7150 | 94.6980 | 0.9916 |
| sgc64_rand_cos_oracle_strat5_lr1e-3 | 5 | 92.7780 | 92.7780 | 0.2237 | 95.9080 | 95.9080 | 0.9916 |
| HL07 o_grid_strat5_p10 (ref, gpusystem, raw) | 5 | 90.12 |  |  | 94.74 |  |  |

### T35 (`T35_pca_inductive.csv`)

| row | n_seeds | n_fit_nodes | explained_variance_ratio_sum | valid_1shot | test_1shot | test_1shot_ci95 | test_5shot | paired_delta_1shot_vs_a2x_pca64_l2 |
|---|---|---|---|---|---|---|---|---|
| pca64_base no training |  | 10711 | 0.5664 | 77.1000 | 83.3300 | 0.9655 | 91.2600 | -5.7150 ± 0.5969 |
| a2x_pca64_l2 (HL11, all nodes) |  | 24919 | 0.5344 | 81.3500 | 89.0450 | 0.8606 | 93.7150 |  |
| pca64_lin_cos_naq_pcabase_lr1e-3 (bestL) | 5 | 10711 |  |  | 83.0490 |  | 91.3470 |  |

### T36 (`T36_tenseed.csv`)

| arm | n_first5 | n_second5 | first5_1shot | second5_1shot | all_1shot | first5_5shot | second5_5shot | all_5shot |
|---|---|---|---|---|---|---|---|---|
| gcn_raw_naq | 5 | 5 | 87.7170 ± 0.4147 | 87.4890 ± 0.4097 | 87.6030 ± 0.4277 | 92.1330 ± 0.2830 | 92.0310 ± 0.2562 | 92.0820 ± 0.2747 |
| gcn_raw_ours | 5 | 5 | 87.0930 ± 0.4003 | 87.0410 ± 0.5010 | 87.0670 ± 0.4542 | 91.3220 ± 0.2111 | 91.5370 ± 0.2103 | 91.4295 ± 0.2365 |
| pca64_lin_cos_sup | 5 | 5 | 89.1710 ± 0.0471 | 89.2470 ± 0.0584 | 89.2090 ± 0.0653 | 93.9560 ± 0.0347 | 93.9230 ± 0.0417 | 93.9395 ± 0.0417 |
| sgc64_pcainit_cos_sup_lr1e-3 | 5 | 5 | 88.5440 ± 0.2659 | 88.2960 ± 0.6586 | 88.4200 ± 0.5173 | 93.3580 ± 0.5413 | 93.4920 ± 0.3289 | 93.4250 ± 0.4529 |
| pca64_lin_cos_soft0.3_naq_lr1e-3 | 5 | 5 | 88.1110 ± 0.8988 | 88.4750 ± 0.5429 | 88.2930 ± 0.7645 | 93.4070 ± 0.0926 | 93.4610 ± 0.0828 | 93.4340 ± 0.0919 |

## ③ 사전 등록 대조표 (계산값만)

| id | prediction | computed |
|---|---|---|
| HLP153 | **A 핵심**: `sgc64_rand_cos_naq`(lr 1e-3) ≥ 87.0 | 82.4910 |
| HLP154 | `sgc64_rand_cos_naq` ≤ `a2x_pca64_l2` − 0.5 | sgc64_rand_cos_naq_lr1e-3 82.4910 − a2x_pca64_l2 89.0450 = -6.5540 |
| HLP155 | `sgc64_pcainit_cos_naq` ≥ `a2x_pca64_l2` − 0.3 | sgc64_pcainit_cos_naq_lr1e-3 87.2330 − a2x_pca64_l2 89.0450 = -1.8120 |
| HLP156 | `sgc64_pcainit_cos_sup` ≤ `sgc64_pcainit_cos_naq` | sgc64_pcainit_cos_sup_lr1e-3 88.5440 − sgc64_pcainit_cos_naq_lr1e-3 87.2330 = +1.3110 |
| HLP157 | `sgc64_rand_cos_sup` ≤ `sgc64_rand_cos_naq` − 2.0 | sgc64_rand_cos_sup_lr1e-3 76.6650 − sgc64_rand_cos_naq_lr1e-3 82.4910 = -5.8260 |
| HLP158 | **B**: `pca64_lin_cos_naq_lr1e-3` − HL12 `pca64_lin_cos_naq`(89.15) ∈ [−0.3, +0.5] | pca64_lin_cos_naq_lr1e-3 88.5840 − HL12 pca64_lin_cos_naq 89.1480 = -0.5640 |
| HLP159 | `pca64_lin_cos_*_lr1e-2`가 epoch 0보다 낮아진다(≤ 89.05 − 0.5) | `pca64_lin_cos_naq_lr1e-2` 84.4350(− 89.05 = -4.6150; epoch 0 포함 89.0450); `pca64_lin_cos_sup_lr1e-2` 85.5550(− 89.05 = -3.4950; epoch 0 포함 89.0450) |
| HLP160 | **C 핵심**: `gcn_raw_soft0.3_naq` ≥ `gcn_raw_naq`(87.72) + 0.5 | gcn_raw_soft0.3_naq 87.7210 − gcn_raw_naq 87.7170 = +0.0040 |
| HLP161 | `gcn_raw_soft1.0_naq` ≥ `gcn_raw_naq` − 1.0 | gcn_raw_soft1.0_naq 87.3250 − gcn_raw_naq 87.7170 = -0.3920 |
| HLP162 | t_struct 동의도 평균(`naq`) ≥ 0.5, `ours`는 그보다 낮다 | naq 0.8805; ours 0.8369 |
| HLP163 | `gcn_raw_mix_r0.1` ≥ `gcn_raw_naq` − 0.3 | gcn_raw_mix_r0.1 87.4800 − gcn_raw_naq 87.7170 = -0.2370 |
| HLP164 | `gcn_raw_mix_r0.5` ≤ `gcn_raw_naq` − 2.0 | gcn_raw_mix_r0.5 86.9190 − gcn_raw_naq 87.7170 = -0.7980 |
| HLP165 | **D**: `gcn_raw_oracle_strat5` bestR_test_raw가 HL07 90.12와 ±0.5 안; bestL_test_l2 ≥ 91.5 | bestR_test_raw 90.1120(− 90.12 = -0.0080); bestL_test_l2 92.8860 |
| HLP166 | `pca64_lin_cos_oracle_strat5` ≥ 90.0 | 91.3880 |
| HLP167 | `pca64_base` 학습 없는 l2 ≥ 89.05 − 0.5 | 83.3300; − 89.05 = -5.7200 |
| HLP168 | 노름: `encn`·`encb`의 norm-induced error ≥ 0.3; `gcn_cos_*`는 ≤ 0.1 | `encb` 0.3848; `encn` 0.4751; `gcn_cos_naq` 0.6253; `gcn_cos_ours` 0.6664; `gcn_raw_sup` 0.5456 |

## ④ 실패·경고·이상·지시서에 없어 정한 것

1. **시작·종료 시각.** 지시서 기준 21:30 시작이었으나 실제 §0 시작은 21:58 KST, 체인 시작은 22:09 KST다. 큐는 23:32 KST에 끝났고 모든 단계가 deadline 전에 끝났다. 지시서 밖의 run을 추가하지 않아 이후 GPU는 비어 있다. 보고서는 지시서의 "08:15 이후"보다 일찍, 마감 08:30 이내에 push했다.
2. **`sgc64_pcainit`의 평균 보정.** `pca_meta.json`에는 성분 행렬이 없어 같은 절차(`random_state=1212`)로 PCA-64를 다시 fit해 `pca64_components.npy`·`pca64_mean.npy`로 저장했다. `bias=False`를 유지하려고 입력을 z2 − mean으로 두었다. HL12 `pca64.npy`와 행 단위 최대 절대 차이는 0.0369이고(`pca_meta_hl13.json`), epoch 0은 89.0500 / 93.7250이다(`a2x_pca64_l2` 89.0450 / 93.7150). epoch 0 assert(±0.05)는 모두 통과했다.
3. **`sgc*`·`pca1024_lin64`의 출력.** 지시서 정의대로 인코더 출력이 행 l2이므로 raw와 l2 평가가 같다. 이 arm들의 `bestR_test_raw`와 `bestL_test_l2`는 정의상 같은 값이다.
4. **arm 이름.** lr ≠ 1e-4 또는 `sgc*`이면 `_lr<값>`을 붙였다(`sgc64_rand_cos_naq_lr1e-4` 포함). 사전 등록의 `sgc64_rand_cos_naq`는 `_lr1e-3`, `sgc64_pcainit_*`는 `_lr1e-3`, `pca64_lin_cos_oracle_strat5`는 `_lr1e-3` run이다. inductive는 `pca64_lin_cos_naq_pcabase_lr1e-3`이다.
5. **단계 9 선택(지시서에 기준 없음).** "1-shot 최고"를 5-seed 평균 `bestL_test_l2`로 정했다. 단계 1은 `sgc64_pcainit_cos_sup_lr1e-3`, 단계 3은 `pca64_lin_cos_soft0.3_naq_lr1e-3`이 선택됐다. test 값으로 고른 것이다. valid 평균 기준이면 단계 1은 같고 단계 3은 `gcn_raw_soft0.3_ours`다(`results/hl/hl13/stage9_selection.json`). 단계 9에는 태그를 두지 않았다(지시서 태그 목록에 없음).
6. **epoch 0과 선택.** hl014와 같이 epoch 0은 체크포인트 후보가 아니다. 후보에 넣은 값은 `bestL_incl_epoch0_*` 열이다. HLP159 계산값에도 함께 적었다.
7. **T32의 A=0.** 지시서의 HL12 행(lr 1e-4)에 더해, `pca64_lin`은 soft arm과 lr이 같은 `pca64_lin_cos_naq_lr1e-3` 행을 함께 두었다.
8. **T36의 앞 5 seed.** `gcn_raw_naq`·`gcn_raw_ours`는 HL11 history의 post-hoc L, 뒤 5 seed(1217–1221)는 hl015의 체크포인트 L이다. `pca64_lin_cos_sup` 앞 5 seed는 HL12 값이다.
9. **t_struct 동의도.** T32의 값은 해당 arm의 에피소드별 평균을 seed 평균한 것이다. HLP162의 `naq`는 `gcn_raw_soft{0.3,0.6,1.0}_naq`, `ours`는 `gcn_raw_soft0.3_ours`로 계산했다.
10. **oracle 순도.** `oracle_strat5`의 `query_gt_match`는 0.9916이고, seed 1212에서 `fallback_occurrence_rate`는 0.5764다(hl011 정의: 1차 층에서 슬롯을 다 못 채운 anchor). oracle 진단의 NaN은 strict JSON 저장을 위해 `report.json`에서 null로 바꿨다(smoke 첫 시도에서 3분기 실패, 1회 수정 후 재통과).
11. **HL07 참고 행.** T34의 90.12 / 94.74는 gpusystem 측정값이다.
12. **동시 실행.** 슬롯 2개(gcn·lin)다. 두 번째 run은 GPU 여유 8600 MiB 이상일 때만 시작하도록 러너에 정했다(지시서에 없음; 측정 peak 8430 MiB 기준). 그래서 GCN과 선형이 항상 겹치지는 않았다. 10GB 조건은 우리 run이 없을 때만 적용했다.
13. **wall 값.** 러너가 run 시작 뒤 20초를 기다린 후 종료를 확인하므로 `wall_seconds.txt`의 하한이 약 23초다. 실제 시간은 `report.json`의 `elapsed_seconds`다.
14. **HLP168의 `gcn_raw_sup`.** 예측 문구에 없는 체크포인트지만 T30의 값을 함께 적었다.
15. **표 열 추가.** T31 `stage`·`bestR_test_raw_*`·`bestL_test_l2_std_*`·`bestL_epoch_mean_*`·`trajmax_epoch_*`, T32 `source`·`n_seeds`·`bestL_test_l2_std_1shot`, T33 `source`·`n_seeds`·`bestL_test_l2_std_*`, T34 `query_gt_match`·`bestL_test_l2_std_1shot`, T30 `test_1shot_raw`·`test_1shot_l2`·`raw_wrong_queries`.
16. **T30의 정의.** degree는 대칭화한 A(self-loop 제외)이고, 지표는 test 500 1-shot 에피소드 기준이다. between/within은 GT 클래스별 노름 평균의 분산 / 클래스 내 분산 평균이며 전체 77클래스와 test 20클래스를 따로 적었다.
17. **부작용.** `exp026.py`가 `results/Amazon_clothing/exp026/result.txt`를 덮어썼다(미추적, CLAUDE.md §9). `save/fixed_episodes/`에 seed 1217–1221 폴더가 추가됐다(지시서 §0-4). pkl은 변하지 않았다.
18. **남은 임시 파일.** smoke 출력은 세션 scratchpad의 `smoke13`·`tables_test` 폴더에 남아 있다. 일괄 삭제 명령이 안전 검사에 막혀, 표 smoke를 scratchpad로 돌리는 방식으로 바꿨고 삭제는 하지 않았다. `reports/HL13_tables/`에는 smoke 산출물이 없다.
19. push 실패 없음.

## ⑤ 산출물 경로

- 코드: `hl/hl015_train_hl13.py`, `hl/hl000j_run_hl13.py`, `hl/hl016_norm_diag.py`, `hl/hl_hl13_tables.py`
- 표: `reports/HL13_tables/{T30_norm_diag,T31_projection,T32_soft,T33_mix,T34_oracle_l2,T35_pca_inductive,T36_tenseed,prereg_HL13}.csv`, `reports/hl13_progress.md`
- 특징(미추적): `results/hl/hl13/feat/`
- 결과(미추적): `results/hl/hl13/curve/<arm>/seed_<seed>/`(`report.json`, `best_raw_{1,5}.pt`, `best_l2_{1,5}.pt`), `results/hl/hl13/curve/summary.csv`, `results/hl/hl13/stage9_selection.json`, `results/hl/hl13/progress.md`, `results/hl/hl13/env_{start,end}.txt`
- 로그(미추적): `out/hl/hl13_chain.sh`, `out/hl/hl13_chain.log`, `out/hl/hl13_stage<N>.log`, `out/hl/hl13_git.log`, `results/hl/hl13/curve/logs/`
