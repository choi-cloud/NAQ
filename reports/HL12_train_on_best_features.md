# HL12 — 가장 좋은 학습 없는 표현 위에서 학습이 무엇을 더하는가: 에피소드 출처 × 인코더 × 손실

지시서 `instructions/HL12_train_on_best_features.md` · auto · Amazon_clothing · 서버 seio · 환경 `taskns`

레이블 사용 범위: base 클래스 40개(④-1)의 레이블을 `sup`·`sup_strat` 에피소드 생성에 사용했다. 고정 valid·test 에피소드의 y_support·y_query는 meta-test 평가에, 고정 valid·test 에피소드의 클래스 id는 base에서 제외하는 데에만, `exp049.query_diagnostics`는 진단에만 레이블을 읽었다. PCA에는 쓰지 않았다.

## ① 실행 명령·환경·commit

- 커밋·태그: `c260db6`(`hl12-setup`), `341ad69`(`hl12-sup`), `9070491`(`hl12-cos`), `abf8135`(`hl12-pca`). 브랜치 `hl/anchor`.
- 환경: seio, RTX 4080 SUPER 1장, torch 2.4.1+cu121, PyG 2.6.1, scipy 1.13.1, sklearn 1.5.2. 설치 없음.
- nvidia-smi: 시작 2 / 16376 MiB(`results/hl/hl12/env_start.txt`), 끝 2 / 16376 MiB(`env_end.txt`). 각 run 시작 전 여유 16374 MiB. peak allocated: GCN sup 5186, GCN naq 4960, GCN ours 7571, pca naq 4960, pca ours 7559, pca sup 5173–5192 MiB.
- md5: 1-shot `5f901727f81903cc12159d280be6915d`, 5-shot `ca3edaca02d541cbc0ee15a2122ef24a`. CLAUDE.md §3과 일치.
- HL11 산출물: `results/hl/hl11/curve/{naq,ours}/seed_{1212..1216}/report.json` 10개, `results/hl/hl11/feat/` 존재. 재실행하지 않았다.
- PCA 저장 파일(`results/hl/hl12/feat/pca_meta.json`): 전체 노드 fit, 레이블 없음. HL11 per-episode 정확도와 1-shot·5-shot test 500 전부 일치(다른 에피소드 0개).

| 파일 | md5 | shape | 설명 분산 합 | test 1-shot | test 5-shot |
|---|---|---|---|---|---|
| `pca64.npy` | `3d432fd1110e689f0e722056d23b2242` | 24919 × 64 | 0.5344 | 89.0450 | 93.7150 |
| `pca256.npy` | `eb97f813c2d6a5c2c76900164540be32` | 24919 × 256 | 0.7547 | 88.5450 | 93.1400 |

- 명령(리포 루트, `PYTHONPATH=$PWD`, `/home/h2tae/.conda/envs/taskns/bin/python`):
  - `python -u hl/hl014_train_variants.py --prepare_feat`
  - `nohup python -u hl/hl000h_run_hl12.py --groups A > out/hl/hl12_A.log`
  - `nohup python -u hl/hl000h_run_hl12.py --groups C,B > out/hl/hl12_CB.log`
  - `nohup python -u hl/hl000h_run_hl12.py --groups D > out/hl/hl12_D.log`
  - `nohup python -u hl/hl000h_run_hl12.py --groups C > out/hl/hl12_C2.log` (실패 5 run 재실행, ④-2)
  - `python hl/hl_hl12_tables.py {tables,prereg}`
- wall: GCN sup 48.6–50.2초, GCN cos naq 49.0–49.8초, GCN cos ours 58.4–59.6초, pca naq 13.2초, pca ours 22.0–22.3초, pca sup·sup_strat 14.1–17.2초.
- 최종 57/57 run 완료(`results/hl/hl12/curve/summary.csv`).

## ② 표

### T22 (`T22_arms.csv`; 1-shot 주 지표 `bestL_test_l2_1shot`)

| arm | source | n_seeds | epoch0_test_l2_1shot | bestR_test_raw_1shot | bestR_test_l2_1shot | bestL_test_l2_1shot | bestL_test_l2_std_1shot | bestL_epoch_mean_1shot | trajmax_test_l2_1shot | trajmax_epoch_1shot | final_test_l2_1shot | bestL_incl_epoch0_test_l2_1shot | bestL_incl_epoch0_n_chosen_1shot | epoch0_test_l2_5shot | bestR_test_raw_5shot | bestR_test_l2_5shot | bestL_test_l2_5shot | bestL_test_l2_std_5shot | bestL_epoch_mean_5shot | trajmax_test_l2_5shot | trajmax_epoch_5shot | final_test_l2_5shot | bestL_incl_epoch0_test_l2_5shot | bestL_incl_epoch0_n_chosen_5shot |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gcn_raw_naq | HL11 hl013; L post-hoc | 5 | 72.5530 | 80.8120 | 87.5720 | 87.7170 | 0.4147 | 1216.0 | 87.7180 | 1120 | 87.5240 | 87.7170 | 0 | 87.9510 | 90.7180 | 92.5080 | 92.1330 | 0.2830 | 640.0 | 92.5290 | 1760 | 92.4710 | 92.1330 | 0 |
| gcn_raw_ours | HL11 hl013; L post-hoc | 5 | 72.5530 | 82.6320 | 87.3170 | 87.0930 | 0.4003 | 1504.0 | 87.1860 | 1600 | 87.1270 | 87.0930 | 0 | 87.9510 | 89.8980 | 91.5590 | 91.3220 | 0.2111 | 960.0 | 91.6730 | 2000 | 91.6730 | 91.3220 | 0 |
| gcn_raw_sup | hl014 | 5 | 72.5530 | 74.1040 | 80.1850 | 81.1870 | 1.2630 | 832.0 | 82.5830 | 160 | 79.5250 | 81.1870 | 0 | 87.9510 | 84.6380 | 88.2070 | 90.2830 | 0.2159 | 192.0 | 90.3500 | 160 | 87.7350 | 90.2830 | 0 |
| gcn_raw_sup_strat | hl014 | 5 | 72.5530 | 74.2470 | 80.0390 | 81.6870 | 0.9177 | 576.0 | 82.4350 | 160 | 79.3850 | 81.6870 | 0 | 87.9510 | 84.5770 | 88.2200 | 90.1590 | 0.3846 | 192.0 | 90.2720 | 160 | 87.7280 | 90.1590 | 0 |
| gcn_cos_naq | hl014 | 5 | 72.5530 | 66.2440 | 84.2650 | 84.1690 | 0.5170 | 1312.0 | 84.6770 | 1280 | 84.3340 | 84.1690 | 0 | 87.9510 | 81.3930 | 92.1260 | 92.0220 | 0.2257 | 1600.0 | 92.0380 | 1920 | 91.9430 | 92.0220 | 0 |
| gcn_cos_ours | hl014 | 5 | 72.5530 | 70.3080 | 86.4740 | 86.5430 | 0.4242 | 1648.0 | 86.8420 | 1920 | 86.6760 | 86.5430 | 0 | 87.9510 | 82.6720 | 92.3370 | 92.1920 | 0.1366 | 1168.0 | 92.4440 | 1920 | 92.4000 | 92.1920 | 0 |
| gcn_cos_sup | hl014 | 5 | 72.5530 | 62.1350 | 77.8640 | 78.7500 | 0.6556 | 1120.0 | 80.5830 | 160 | 76.6950 | 78.7500 | 0 | 87.9510 | 77.3720 | 90.0110 | 90.0240 | 0.6026 | 736.0 | 90.5930 | 160 | 88.3670 | 90.0240 | 0 |
| pca64_lin_cos_naq | hl014 | 5 | 89.0450 | 89.0660 | 89.1020 | 89.1480 | 0.0440 | 896.0 | 89.2020 | 1440 | 89.1860 | 89.1480 | 0 | 93.7150 | 93.7860 | 93.7630 | 93.7640 | 0.0335 | 1376.0 | 93.7680 | 1280 | 93.7490 | 93.7640 | 0 |
| pca64_lin_cos_ours | hl014 | 5 | 89.0450 | 89.1210 | 89.1240 | 89.1430 | 0.0453 | 1312.0 | 89.1990 | 2000 | 89.1990 | 89.1430 | 0 | 93.7150 | 93.7420 | 93.7450 | 93.7620 | 0.0181 | 1184.0 | 93.7610 | 960 | 93.7340 | 93.7620 | 0 |
| pca64_lin_cos_sup | hl014 | 5 | 89.0450 | 89.2560 | 89.2240 | 89.1710 | 0.0471 | 544.0 | 89.3350 | 1760 | 89.3190 | 89.1270 | 2 | 93.7150 | 93.9080 | 93.9370 | 93.9560 | 0.0347 | 1648.0 | 94.0050 | 2000 | 94.0050 | 93.9560 | 0 |
| pca64_lin_cos_sup_strat | hl014 | 5 | 89.0450 | 89.3490 | 89.2910 | 89.1630 | 0.0633 | 544.0 | 89.3810 | 1760 | 89.3650 | 89.1160 | 2 | 93.7150 | 93.8900 | 93.9530 | 93.9710 | 0.0284 | 1760.0 | 94.0050 | 2000 | 94.0050 | 93.9710 | 0 |
| pca256_lin_cos_sup | hl014 | 5 | 88.5450 | 88.9180 | 88.9350 | 88.9670 | 0.0764 | 1504.0 | 88.9970 | 1440 | 88.9170 | 88.9670 | 0 | 93.1400 | 93.3750 | 93.3750 | 93.3780 | 0.0254 | 1952.0 | 93.3890 | 2000 | 93.3890 | 93.3780 | 0 |
| pca256_mlp_cos_sup | hl014 | 5 | 81.0950 | 83.0930 | 83.7600 | 83.7500 | 0.5581 | 1248.0 | 84.6060 | 480 | 82.8530 | 83.7500 | 0 | 91.0500 | 91.8160 | 91.9930 | 91.9880 | 0.3186 | 1120.0 | 92.1130 | 640 | 91.7640 | 91.9880 | 0 |
| a2x_l2 | HL11 T18 (no training) |  |  |  |  | 85.0950 |  |  |  |  |  |  |  |  |  |  | 92.7150 |  |  |  |  |  |  |  |
| a2x_pca64_l2 | HL11 T18 (no training) |  |  |  |  | 89.0450 |  |  |  |  |  |  |  |  |  |  | 93.7150 |  |  |  |  |  |  |  |
| a2x_pca256_l2 | HL11 T18 (no training) |  |  |  |  | 88.5450 |  |  |  |  |  |  |  |  |  |  | 93.1400 |  |  |  |  |  |  |  |
| encb_l2 | HL11 T18 (seed 1212 ckpt) |  |  |  |  | 87.7450 |  |  |  |  |  |  |  |  |  |  | 92.0800 |  |  |  |  |  |  |  |
| encn_l2 | HL11 T18 (seed 1212 ckpt) |  |  |  |  | 87.8400 |  |  |  |  |  |  |  |  |  |  | 92.5800 |  |  |  |  |  |  |  |

### T22b (`T22b_paired_vs_pca64.csv`; 규칙 L l2 test의 에피소드별 5 seed 평균 − `a2x_pca64_l2`. 지시서에 없는 보조 표, CLAUDE.md §3의 쌍 차이 규약)

| arm | n_seeds | baseline | paired_delta_1shot | paired_ci95_1shot | paired_win_frac_1shot | paired_delta_5shot | paired_ci95_5shot | paired_win_frac_5shot |
|---|---|---|---|---|---|---|---|---|
| gcn_raw_sup | 5 | a2x_pca64_l2 | -7.8580 | 0.6535 | 0.1100 | -3.4320 | 0.3108 | 0.1140 |
| gcn_raw_sup_strat | 5 | a2x_pca64_l2 | -7.3580 | 0.6523 | 0.1320 | -3.5560 | 0.3159 | 0.1080 |
| gcn_cos_naq | 5 | a2x_pca64_l2 | -4.8760 | 0.4689 | 0.1560 | -1.6930 | 0.2446 | 0.2120 |
| gcn_cos_ours | 5 | a2x_pca64_l2 | -2.5020 | 0.4275 | 0.2580 | -1.5230 | 0.2333 | 0.2100 |
| gcn_cos_sup | 5 | a2x_pca64_l2 | -10.2950 | 0.7136 | 0.0720 | -3.6910 | 0.3470 | 0.1280 |
| pca64_lin_cos_naq | 5 | a2x_pca64_l2 | 0.1030 | 0.1046 | 0.2280 | 0.0490 | 0.0817 | 0.1180 |
| pca64_lin_cos_ours | 5 | a2x_pca64_l2 | 0.0980 | 0.1037 | 0.2380 | 0.0470 | 0.0621 | 0.1040 |
| pca64_lin_cos_sup | 5 | a2x_pca64_l2 | 0.1260 | 0.0947 | 0.2100 | 0.2410 | 0.1123 | 0.1960 |
| pca64_lin_cos_sup_strat | 5 | a2x_pca64_l2 | 0.1180 | 0.0943 | 0.2260 | 0.2560 | 0.1176 | 0.2060 |
| pca256_lin_cos_sup | 5 | a2x_pca64_l2 | -0.0780 | 0.3045 | 0.3980 | -0.3370 | 0.2117 | 0.2100 |
| pca256_mlp_cos_sup | 5 | a2x_pca64_l2 | -5.2950 | 0.5234 | 0.1720 | -1.7270 | 0.2615 | 0.1860 |

### T23 (`T23_curves_l2.csv`; std는 `T23b_curves_l2_std.csv`)

| arm | epoch | n_seeds | valid_1shot_l2 | test_1shot_l2 | test_5shot_l2 |
|---|---|---|---|---|---|
| gcn_raw_naq | 0 | 5 | 65.6700 | 72.5530 | 87.9510 |
| gcn_raw_naq | 160 | 5 | 77.8100 | 85.8760 | 91.6450 |
| gcn_raw_naq | 320 | 5 | 79.0200 | 86.8610 | 91.7460 |
| gcn_raw_naq | 480 | 5 | 79.7600 | 87.1180 | 91.8800 |
| gcn_raw_naq | 640 | 5 | 79.6300 | 87.3710 | 92.0930 |
| gcn_raw_naq | 800 | 5 | 79.3400 | 87.6940 | 92.2190 |
| gcn_raw_naq | 960 | 5 | 79.2000 | 87.7110 | 92.2100 |
| gcn_raw_naq | 1120 | 5 | 79.4500 | 87.7180 | 92.3330 |
| gcn_raw_naq | 1280 | 5 | 79.3100 | 87.6780 | 92.4210 |
| gcn_raw_naq | 1440 | 5 | 79.9200 | 87.5370 | 92.5110 |
| gcn_raw_naq | 1600 | 5 | 79.4300 | 87.4950 | 92.5190 |
| gcn_raw_naq | 1760 | 5 | 79.3000 | 87.4770 | 92.5290 |
| gcn_raw_naq | 1920 | 5 | 79.3400 | 87.4060 | 92.4810 |
| gcn_raw_naq | 2000 | 5 | 79.2600 | 87.5240 | 92.4710 |
| gcn_raw_ours | 0 | 5 | 65.6700 | 72.5530 | 87.9510 |
| gcn_raw_ours | 160 | 5 | 78.1300 | 85.4290 | 91.1110 |
| gcn_raw_ours | 320 | 5 | 79.5200 | 86.3310 | 91.1450 |
| gcn_raw_ours | 480 | 5 | 80.5600 | 86.6450 | 91.3140 |
| gcn_raw_ours | 640 | 5 | 80.2700 | 86.7380 | 91.2910 |
| gcn_raw_ours | 800 | 5 | 80.1100 | 86.8530 | 91.4070 |
| gcn_raw_ours | 960 | 5 | 80.2700 | 86.9610 | 91.4030 |
| gcn_raw_ours | 1120 | 5 | 80.3000 | 86.9440 | 91.3520 |
| gcn_raw_ours | 1280 | 5 | 80.3700 | 87.0560 | 91.4730 |
| gcn_raw_ours | 1440 | 5 | 80.5900 | 86.9460 | 91.5010 |
| gcn_raw_ours | 1600 | 5 | 80.5000 | 87.1860 | 91.6230 |
| gcn_raw_ours | 1760 | 5 | 80.5500 | 87.0710 | 91.5160 |
| gcn_raw_ours | 1920 | 5 | 80.2900 | 87.1560 | 91.6450 |
| gcn_raw_ours | 2000 | 5 | 80.1100 | 87.1270 | 91.6730 |
| gcn_raw_sup | 0 | 5 | 65.6700 | 72.5530 | 87.9510 |
| gcn_raw_sup | 160 | 5 | 75.1700 | 82.5830 | 90.3500 |
| gcn_raw_sup | 320 | 5 | 76.0400 | 82.1940 | 89.6910 |
| gcn_raw_sup | 480 | 5 | 76.2500 | 81.7780 | 89.3250 |
| gcn_raw_sup | 640 | 5 | 75.8200 | 81.3920 | 89.1260 |
| gcn_raw_sup | 800 | 5 | 75.7200 | 80.8380 | 88.7900 |
| gcn_raw_sup | 960 | 5 | 75.6900 | 80.7240 | 88.7050 |
| gcn_raw_sup | 1120 | 5 | 75.5100 | 80.6750 | 88.7130 |
| gcn_raw_sup | 1280 | 5 | 75.4900 | 80.4190 | 88.6030 |
| gcn_raw_sup | 1440 | 5 | 75.6000 | 80.3120 | 88.4130 |
| gcn_raw_sup | 1600 | 5 | 75.7000 | 80.0340 | 88.3920 |
| gcn_raw_sup | 1760 | 5 | 75.6200 | 80.0330 | 88.2500 |
| gcn_raw_sup | 1920 | 5 | 75.2100 | 79.5830 | 87.9290 |
| gcn_raw_sup | 2000 | 5 | 75.1800 | 79.5250 | 87.7350 |
| gcn_raw_sup_strat | 0 | 5 | 65.6700 | 72.5530 | 87.9510 |
| gcn_raw_sup_strat | 160 | 5 | 75.6400 | 82.4350 | 90.2720 |
| gcn_raw_sup_strat | 320 | 5 | 76.0100 | 82.2890 | 89.7580 |
| gcn_raw_sup_strat | 480 | 5 | 76.2300 | 81.8490 | 89.4440 |
| gcn_raw_sup_strat | 640 | 5 | 75.8200 | 81.2350 | 89.0770 |
| gcn_raw_sup_strat | 800 | 5 | 75.9700 | 80.9390 | 88.9010 |
| gcn_raw_sup_strat | 960 | 5 | 75.9800 | 80.8420 | 88.8250 |
| gcn_raw_sup_strat | 1120 | 5 | 75.5700 | 80.5840 | 88.6380 |
| gcn_raw_sup_strat | 1280 | 5 | 75.9100 | 80.3160 | 88.5180 |
| gcn_raw_sup_strat | 1440 | 5 | 75.6800 | 80.1590 | 88.3190 |
| gcn_raw_sup_strat | 1600 | 5 | 75.3300 | 79.9720 | 88.3820 |
| gcn_raw_sup_strat | 1760 | 5 | 75.3600 | 79.9300 | 88.1680 |
| gcn_raw_sup_strat | 1920 | 5 | 74.9800 | 79.6130 | 87.9220 |
| gcn_raw_sup_strat | 2000 | 5 | 74.9600 | 79.3850 | 87.7280 |
| gcn_cos_naq | 0 | 5 | 65.6700 | 72.5530 | 87.9510 |
| gcn_cos_naq | 160 | 5 | 71.9900 | 81.6320 | 90.4630 |
| gcn_cos_naq | 320 | 5 | 73.6900 | 82.8330 | 90.9990 |
| gcn_cos_naq | 480 | 5 | 73.8700 | 83.2570 | 91.2960 |
| gcn_cos_naq | 640 | 5 | 75.1500 | 84.0490 | 91.6110 |
| gcn_cos_naq | 800 | 5 | 75.6200 | 84.4400 | 91.6800 |
| gcn_cos_naq | 960 | 5 | 75.6000 | 83.8300 | 91.5970 |
| gcn_cos_naq | 1120 | 5 | 75.8400 | 84.3640 | 91.8400 |
| gcn_cos_naq | 1280 | 5 | 75.4600 | 84.6770 | 91.8750 |
| gcn_cos_naq | 1440 | 5 | 75.0400 | 84.5770 | 92.0010 |
| gcn_cos_naq | 1600 | 5 | 75.5800 | 84.5130 | 91.9710 |
| gcn_cos_naq | 1760 | 5 | 76.1500 | 84.4110 | 91.9520 |
| gcn_cos_naq | 1920 | 5 | 76.0300 | 84.5100 | 92.0380 |
| gcn_cos_naq | 2000 | 5 | 75.7800 | 84.3340 | 91.9430 |
| gcn_cos_ours | 0 | 5 | 65.6700 | 72.5530 | 87.9510 |
| gcn_cos_ours | 160 | 5 | 75.3700 | 83.0780 | 91.2290 |
| gcn_cos_ours | 320 | 5 | 77.2800 | 84.5990 | 91.6210 |
| gcn_cos_ours | 480 | 5 | 77.9400 | 85.0100 | 91.9160 |
| gcn_cos_ours | 640 | 5 | 78.6200 | 85.8500 | 92.1180 |
| gcn_cos_ours | 800 | 5 | 78.6700 | 86.1220 | 92.2660 |
| gcn_cos_ours | 960 | 5 | 79.2500 | 86.1130 | 92.2020 |
| gcn_cos_ours | 1120 | 5 | 79.2900 | 86.3010 | 92.1920 |
| gcn_cos_ours | 1280 | 5 | 79.2400 | 86.2450 | 92.2240 |
| gcn_cos_ours | 1440 | 5 | 79.1300 | 86.4100 | 92.2830 |
| gcn_cos_ours | 1600 | 5 | 79.6200 | 86.6470 | 92.4000 |
| gcn_cos_ours | 1760 | 5 | 79.7100 | 86.5420 | 92.2500 |
| gcn_cos_ours | 1920 | 5 | 79.5100 | 86.8420 | 92.4440 |
| gcn_cos_ours | 2000 | 5 | 79.4500 | 86.6760 | 92.4000 |
| gcn_cos_sup | 0 | 5 | 65.6700 | 72.5530 | 87.9510 |
| gcn_cos_sup | 160 | 5 | 74.4800 | 80.5830 | 90.5930 |
| gcn_cos_sup | 320 | 5 | 74.4800 | 80.4670 | 90.5430 |
| gcn_cos_sup | 480 | 5 | 74.2900 | 80.4960 | 90.3810 |
| gcn_cos_sup | 640 | 5 | 73.7500 | 79.6130 | 90.2380 |
| gcn_cos_sup | 800 | 5 | 74.1000 | 79.0230 | 89.7530 |
| gcn_cos_sup | 960 | 5 | 74.6400 | 78.8660 | 89.5680 |
| gcn_cos_sup | 1120 | 5 | 74.3200 | 78.6950 | 89.5840 |
| gcn_cos_sup | 1280 | 5 | 74.7100 | 78.0440 | 89.2880 |
| gcn_cos_sup | 1440 | 5 | 73.8900 | 78.1000 | 88.9480 |
| gcn_cos_sup | 1600 | 5 | 73.6300 | 77.2690 | 88.6980 |
| gcn_cos_sup | 1760 | 5 | 73.3900 | 77.5220 | 88.8980 |
| gcn_cos_sup | 1920 | 5 | 74.2300 | 77.4610 | 88.6050 |
| gcn_cos_sup | 2000 | 5 | 73.8900 | 76.6950 | 88.3670 |
| pca64_lin_cos_naq | 0 | 5 | 81.3500 | 89.0450 | 93.7150 |
| pca64_lin_cos_naq | 160 | 5 | 81.3900 | 89.0950 | 93.7540 |
| pca64_lin_cos_naq | 320 | 5 | 81.3400 | 89.0810 | 93.7570 |
| pca64_lin_cos_naq | 480 | 5 | 81.3300 | 89.0660 | 93.7510 |
| pca64_lin_cos_naq | 640 | 5 | 81.3700 | 89.0960 | 93.7560 |
| pca64_lin_cos_naq | 800 | 5 | 81.2900 | 89.1150 | 93.7480 |
| pca64_lin_cos_naq | 960 | 5 | 81.3200 | 89.1210 | 93.7520 |
| pca64_lin_cos_naq | 1120 | 5 | 81.3100 | 89.1260 | 93.7600 |
| pca64_lin_cos_naq | 1280 | 5 | 81.3900 | 89.1580 | 93.7680 |
| pca64_lin_cos_naq | 1440 | 5 | 81.3600 | 89.2020 | 93.7570 |
| pca64_lin_cos_naq | 1600 | 5 | 81.2900 | 89.2010 | 93.7570 |
| pca64_lin_cos_naq | 1760 | 5 | 81.3200 | 89.1860 | 93.7410 |
| pca64_lin_cos_naq | 1920 | 5 | 81.2300 | 89.2000 | 93.7460 |
| pca64_lin_cos_naq | 2000 | 5 | 81.2800 | 89.1860 | 93.7490 |
| pca64_lin_cos_ours | 0 | 5 | 81.3500 | 89.0450 | 93.7150 |
| pca64_lin_cos_ours | 160 | 5 | 81.3200 | 89.0910 | 93.7290 |
| pca64_lin_cos_ours | 320 | 5 | 81.3700 | 89.1190 | 93.7230 |
| pca64_lin_cos_ours | 480 | 5 | 81.5000 | 89.0950 | 93.7340 |
| pca64_lin_cos_ours | 640 | 5 | 81.5700 | 89.0990 | 93.7420 |
| pca64_lin_cos_ours | 800 | 5 | 81.5400 | 89.1010 | 93.7440 |
| pca64_lin_cos_ours | 960 | 5 | 81.5700 | 89.1010 | 93.7610 |
| pca64_lin_cos_ours | 1120 | 5 | 81.5700 | 89.1020 | 93.7570 |
| pca64_lin_cos_ours | 1280 | 5 | 81.5400 | 89.1220 | 93.7460 |
| pca64_lin_cos_ours | 1440 | 5 | 81.5400 | 89.1230 | 93.7440 |
| pca64_lin_cos_ours | 1600 | 5 | 81.5500 | 89.1700 | 93.7420 |
| pca64_lin_cos_ours | 1760 | 5 | 81.5700 | 89.1890 | 93.7380 |
| pca64_lin_cos_ours | 1920 | 5 | 81.5000 | 89.1940 | 93.7390 |
| pca64_lin_cos_ours | 2000 | 5 | 81.4700 | 89.1990 | 93.7340 |
| pca64_lin_cos_sup | 0 | 5 | 81.3500 | 89.0450 | 93.7150 |
| pca64_lin_cos_sup | 160 | 5 | 81.3900 | 89.1150 | 93.7920 |
| pca64_lin_cos_sup | 320 | 5 | 81.3700 | 89.1450 | 93.8220 |
| pca64_lin_cos_sup | 480 | 5 | 81.3400 | 89.1580 | 93.8200 |
| pca64_lin_cos_sup | 640 | 5 | 81.2700 | 89.1710 | 93.8320 |
| pca64_lin_cos_sup | 800 | 5 | 81.3300 | 89.2140 | 93.8530 |
| pca64_lin_cos_sup | 960 | 5 | 81.2800 | 89.2550 | 93.8680 |
| pca64_lin_cos_sup | 1120 | 5 | 81.2400 | 89.2780 | 93.8850 |
| pca64_lin_cos_sup | 1280 | 5 | 81.1900 | 89.2780 | 93.9080 |
| pca64_lin_cos_sup | 1440 | 5 | 81.1000 | 89.2930 | 93.9310 |
| pca64_lin_cos_sup | 1600 | 5 | 81.0800 | 89.3240 | 93.9670 |
| pca64_lin_cos_sup | 1760 | 5 | 80.9900 | 89.3350 | 93.9730 |
| pca64_lin_cos_sup | 1920 | 5 | 80.9500 | 89.3200 | 93.9950 |
| pca64_lin_cos_sup | 2000 | 5 | 80.9700 | 89.3190 | 94.0050 |
| pca64_lin_cos_sup_strat | 0 | 5 | 81.3500 | 89.0450 | 93.7150 |
| pca64_lin_cos_sup_strat | 160 | 5 | 81.3300 | 89.1270 | 93.7800 |
| pca64_lin_cos_sup_strat | 320 | 5 | 81.2400 | 89.1670 | 93.8100 |
| pca64_lin_cos_sup_strat | 480 | 5 | 81.1900 | 89.1620 | 93.8330 |
| pca64_lin_cos_sup_strat | 640 | 5 | 81.2100 | 89.1670 | 93.8410 |
| pca64_lin_cos_sup_strat | 800 | 5 | 81.3100 | 89.1960 | 93.8500 |
| pca64_lin_cos_sup_strat | 960 | 5 | 81.2800 | 89.2410 | 93.8640 |
| pca64_lin_cos_sup_strat | 1120 | 5 | 81.1400 | 89.2650 | 93.9030 |
| pca64_lin_cos_sup_strat | 1280 | 5 | 81.1500 | 89.2980 | 93.9290 |
| pca64_lin_cos_sup_strat | 1440 | 5 | 81.0600 | 89.3300 | 93.9370 |
| pca64_lin_cos_sup_strat | 1600 | 5 | 81.0000 | 89.3480 | 93.9560 |
| pca64_lin_cos_sup_strat | 1760 | 5 | 80.8600 | 89.3810 | 93.9820 |
| pca64_lin_cos_sup_strat | 1920 | 5 | 80.9300 | 89.3750 | 93.9970 |
| pca64_lin_cos_sup_strat | 2000 | 5 | 80.9500 | 89.3650 | 94.0050 |
| pca256_lin_cos_sup | 0 | 5 | 80.0500 | 88.5450 | 93.1400 |
| pca256_lin_cos_sup | 160 | 5 | 80.3200 | 88.6680 | 93.1890 |
| pca256_lin_cos_sup | 320 | 5 | 80.4800 | 88.7490 | 93.2280 |
| pca256_lin_cos_sup | 480 | 5 | 80.8200 | 88.8660 | 93.2570 |
| pca256_lin_cos_sup | 640 | 5 | 80.9800 | 88.9220 | 93.2710 |
| pca256_lin_cos_sup | 800 | 5 | 81.0300 | 88.9770 | 93.2890 |
| pca256_lin_cos_sup | 960 | 5 | 81.1100 | 88.9660 | 93.2860 |
| pca256_lin_cos_sup | 1120 | 5 | 81.0900 | 88.9490 | 93.3110 |
| pca256_lin_cos_sup | 1280 | 5 | 81.0100 | 88.9770 | 93.3260 |
| pca256_lin_cos_sup | 1440 | 5 | 81.0400 | 88.9970 | 93.3380 |
| pca256_lin_cos_sup | 1600 | 5 | 81.0300 | 88.9670 | 93.3630 |
| pca256_lin_cos_sup | 1760 | 5 | 80.9500 | 88.9760 | 93.3710 |
| pca256_lin_cos_sup | 1920 | 5 | 80.9800 | 88.9400 | 93.3770 |
| pca256_lin_cos_sup | 2000 | 5 | 81.1000 | 88.9170 | 93.3890 |
| pca256_mlp_cos_sup | 0 | 5 | 73.6700 | 81.0950 | 91.0500 |
| pca256_mlp_cos_sup | 160 | 5 | 76.9800 | 83.8870 | 91.7830 |
| pca256_mlp_cos_sup | 320 | 5 | 77.6700 | 84.4450 | 91.9790 |
| pca256_mlp_cos_sup | 480 | 5 | 77.1200 | 84.6060 | 92.0940 |
| pca256_mlp_cos_sup | 640 | 5 | 77.2900 | 84.4730 | 92.1130 |
| pca256_mlp_cos_sup | 800 | 5 | 77.8400 | 84.3300 | 92.1020 |
| pca256_mlp_cos_sup | 960 | 5 | 77.9100 | 84.2660 | 92.1130 |
| pca256_mlp_cos_sup | 1120 | 5 | 77.8700 | 84.0900 | 92.0750 |
| pca256_mlp_cos_sup | 1280 | 5 | 77.7000 | 83.8430 | 92.0030 |
| pca256_mlp_cos_sup | 1440 | 5 | 77.4500 | 83.5620 | 91.9430 |
| pca256_mlp_cos_sup | 1600 | 5 | 77.4800 | 83.2910 | 91.8950 |
| pca256_mlp_cos_sup | 1760 | 5 | 77.6200 | 83.2640 | 91.8700 |
| pca256_mlp_cos_sup | 1920 | 5 | 77.8200 | 83.2430 | 91.8300 |
| pca256_mlp_cos_sup | 2000 | 5 | 77.7100 | 82.8530 | 91.7640 |
| pca64_lin_cos_sup_s5 | 0 | 1 | 81.3500 | 89.0450 | 93.7150 |
| pca64_lin_cos_sup_s5 | 160 | 1 | 81.5000 | 89.1350 | 93.7900 |
| pca64_lin_cos_sup_s5 | 320 | 1 | 81.4000 | 89.1600 | 93.8000 |
| pca64_lin_cos_sup_s5 | 480 | 1 | 81.4000 | 89.1900 | 93.8050 |
| pca64_lin_cos_sup_s5 | 640 | 1 | 81.5000 | 89.2500 | 93.8150 |
| pca64_lin_cos_sup_s5 | 800 | 1 | 81.6000 | 89.3150 | 93.8300 |
| pca64_lin_cos_sup_s5 | 960 | 1 | 81.6500 | 89.3300 | 93.8650 |
| pca64_lin_cos_sup_s5 | 1120 | 1 | 81.2500 | 89.3150 | 93.8950 |
| pca64_lin_cos_sup_s5 | 1280 | 1 | 81.4000 | 89.3650 | 93.9100 |
| pca64_lin_cos_sup_s5 | 1440 | 1 | 81.3500 | 89.3650 | 93.9500 |
| pca64_lin_cos_sup_s5 | 1600 | 1 | 81.4000 | 89.4100 | 93.9850 |
| pca64_lin_cos_sup_s5 | 1760 | 1 | 81.4000 | 89.4400 | 93.9750 |
| pca64_lin_cos_sup_s5 | 1920 | 1 | 81.4000 | 89.4650 | 94.0000 |
| pca64_lin_cos_sup_s5 | 2000 | 1 | 81.4500 | 89.4700 | 94.0100 |
| pca64_lin_cos_sup_s20 | 0 | 1 | 81.3500 | 89.0450 | 93.7150 |
| pca64_lin_cos_sup_s20 | 160 | 1 | 81.3000 | 89.1250 | 93.7900 |
| pca64_lin_cos_sup_s20 | 320 | 1 | 81.2500 | 89.1200 | 93.8000 |
| pca64_lin_cos_sup_s20 | 480 | 1 | 81.2500 | 89.1400 | 93.8050 |
| pca64_lin_cos_sup_s20 | 640 | 1 | 81.1000 | 89.1450 | 93.8050 |
| pca64_lin_cos_sup_s20 | 800 | 1 | 81.2000 | 89.1850 | 93.7950 |
| pca64_lin_cos_sup_s20 | 960 | 1 | 81.3500 | 89.1950 | 93.8250 |
| pca64_lin_cos_sup_s20 | 1120 | 1 | 81.1500 | 89.1750 | 93.8400 |
| pca64_lin_cos_sup_s20 | 1280 | 1 | 81.2000 | 89.2050 | 93.8650 |
| pca64_lin_cos_sup_s20 | 1440 | 1 | 81.0000 | 89.2250 | 93.8950 |
| pca64_lin_cos_sup_s20 | 1600 | 1 | 80.7000 | 89.2550 | 93.9200 |
| pca64_lin_cos_sup_s20 | 1760 | 1 | 80.8500 | 89.3200 | 93.9500 |
| pca64_lin_cos_sup_s20 | 1920 | 1 | 80.7500 | 89.2950 | 93.9550 |
| pca64_lin_cos_sup_s20 | 2000 | 1 | 80.7500 | 89.3250 | 93.9800 |

### T24 (`T24_scale.csv`)

| arm | seed | s | bestL_test_l2_1shot | bestL_epoch_1shot | trajmax_test_l2_1shot | trajmax_epoch_1shot | bestL_test_l2_5shot | bestL_epoch_5shot | trajmax_test_l2_5shot | trajmax_epoch_5shot |
|---|---|---|---|---|---|---|---|---|---|---|
| pca64_lin_cos_sup | 1212 | 5 | 89.3300 | 960 | 89.4700 | 2000 | 94.0100 | 2000 | 94.0100 | 2000 |
| pca64_lin_cos_sup | 1212 | 10 | 89.1200 | 160 | 89.3700 | 2000 | 93.9350 | 1760 | 93.9950 | 2000 |
| pca64_lin_cos_sup | 1212 | 20 | 89.1950 | 960 | 89.3250 | 2000 | 93.9500 | 1760 | 93.9800 | 2000 |

### T25 (`T25_episodes.csv`)

| episodes | seed | n_episodes | n_base_classes | class_size_min | class_size_median | class_size_max | n_classes_sampled | classes_sampled_min | classes_sampled_max | drawn_with_replacement | query_a2x_cos_mean | stratum1_a2x_cos_mean | stratum2_a2x_cos_mean | stratum3_a2x_cos_mean | stratum4_a2x_cos_mean | stratum5_a2x_cos_mean | episode_sha256_prefix |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sup | 1212 | 2000 | 40 | 101 | 201.0 | 923 | 40 | 222 | 283 | 0 | 0.4428 |  |  |  |  |  | 86018d9d456b |
| sup | 1213 | 2000 | 40 | 101 | 201.0 | 923 | 40 | 224 | 286 | 0 | 0.4417 |  |  |  |  |  | ad14f7170fcf |
| sup | 1214 | 2000 | 40 | 101 | 201.0 | 923 | 40 | 225 | 279 | 0 | 0.4444 |  |  |  |  |  | dbc8ad70f0e4 |
| sup | 1215 | 2000 | 40 | 101 | 201.0 | 923 | 40 | 209 | 285 | 0 | 0.4449 |  |  |  |  |  | 899d0cdcf7ca |
| sup | 1216 | 2000 | 40 | 101 | 201.0 | 923 | 40 | 218 | 288 | 0 | 0.4448 |  |  |  |  |  | 36c591960825 |
| sup_strat | 1212 | 2000 | 40 | 101 | 201.0 | 923 | 40 | 222 | 283 | 0 | 0.4421 | 0.6523 | 0.5086 | 0.4356 | 0.3647 | 0.2491 | c659dfa0c321 |
| sup_strat | 1213 | 2000 | 40 | 101 | 201.0 | 923 | 40 | 224 | 286 | 0 | 0.4398 | 0.6479 | 0.5065 | 0.4334 | 0.3627 | 0.2485 | 6cd994d665f5 |
| sup_strat | 1214 | 2000 | 40 | 101 | 201.0 | 923 | 40 | 225 | 279 | 0 | 0.4435 | 0.6508 | 0.5105 | 0.4375 | 0.3673 | 0.2514 | 880b09b3607b |
| sup_strat | 1215 | 2000 | 40 | 101 | 201.0 | 923 | 40 | 209 | 285 | 0 | 0.4442 | 0.6517 | 0.5103 | 0.4376 | 0.3687 | 0.2526 | 8bef00548a9e |
| sup_strat | 1216 | 2000 | 40 | 101 | 201.0 | 923 | 40 | 218 | 288 | 0 | 0.4438 | 0.6517 | 0.5099 | 0.4378 | 0.3679 | 0.2517 | 4d46c1271882 |

## ③ 사전 등록 대조표 (계산값만)

| id | prediction | computed |
|---|---|---|
| HLP131 | **Q1**: `gcn_raw_sup` ≥ `gcn_raw_ours` + 1.0 | 81.1870 − 87.0930(post-hoc L) = -5.9060 |
| HLP132 | `gcn_raw_sup_strat` ≥ `gcn_raw_sup` + 0.5 | 81.6870 − 81.1870 = +0.5000 |
| HLP133 | **Q3**: `gcn_cos_naq` ≥ `gcn_raw_naq`(post-hoc L) − 0.3 | 84.1690 − 87.7170(post-hoc L) = -3.5480 |
| HLP134 | `gcn_cos_sup` ≥ 89.0 | 78.7500 |
| HLP135 | **Q2 핵심**: `pca64_lin_cos_sup` ≥ 89.05 + 0.5 | 89.1710; − 89.05 = +0.1210. epoch 0을 선택 후보에 넣으면 89.1270 |
| HLP136 | `pca64_lin_cos_naq` ≥ 89.05 − 0.5 | 89.1480; − 89.05 = +0.0980. epoch 0을 선택 후보에 넣으면 89.1480 |
| HLP137 | `pca64_lin_cos_ours` − `pca64_lin_cos_naq` ∈ [−0.5, +0.5] | 89.1430 − 89.1480 = -0.0050 |
| HLP138 | `pca64_lin_cos_sup_strat` ≥ `pca64_lin_cos_sup` + 0.3 | 89.1630 − 89.1710 = -0.0080 |
| HLP139 | `pca256_mlp_cos_sup` ≤ `pca256_lin_cos_sup` + 0.5 | 83.7500 − 88.9670 = -5.2170 |
| HLP140 | `pca*_lin` 모든 arm의 epoch0_test_l2가 HL11 PCA 값과 ±0.05 안 | `pca64_lin_cos_naq` 89.0450(− 89.05 = -0.0050); `pca64_lin_cos_ours` 89.0450(− 89.05 = -0.0050); `pca64_lin_cos_sup` 89.0450(− 89.05 = -0.0050); `pca64_lin_cos_sup_strat` 89.0450(− 89.05 = -0.0050); `pca256_lin_cos_sup` 88.5450(− 88.55 = -0.0050) |
| HLP141 | 5-shot: `pca64_lin_cos_sup` ≥ 93.72 | 93.9560; − 93.72 = +0.2360 |
| HLP142 | scale s=5·20의 `bestL_test_l2`가 s=10과 ±1.0 안 | seed 1212. s=10 89.1200; s=5 89.3300(+0.2100); s=20 89.1950(+0.0750) |

## ④ 실패·경고·이상·지시서에 없어 정한 것

1. **base 클래스는 57개가 아니라 40개로 고정했다(지시서·CLAUDE.md §4와 다름).** `load_data`는 train.mat의 57개를 seed에 따라 `class_list_train` 40개와 `class_list_valid` 17개로 나눈다(`data.py` `valid_num_dic`). 고정 valid pkl의 클래스는 17개, test는 20개로 서로 겹치지 않는다. seed별 `class_list_train`과 고정 valid 클래스의 겹침은 1212: 0, 1213: 10, 1214: 12, 1215: 12, 1216: 11개이고, test와의 겹침은 모두 0개다. 지시서의 `data.class_list_train`을 그대로 쓰면 seed 1213–1216에서, 57개를 쓰면 모든 seed에서 고정 valid 클래스 레이블이 학습에 들어가 §4에 어긋난다. 그래서 모든 seed에서 (train.mat 57개) − (고정 valid 17개) = 40개(seed 1212의 `class_list_train`과 같은 집합)를 `supervised_task_generator`에 넘겼다. 57개·seed별 40개로는 돌리지 않았다. 이 영향을 받는 arm은 `*_sup`, `*_sup_strat`이다.
2. **run 실패 5건 → 재실행.** `pca64_lin_cos_sup_strat` 5 seed가 첫 시도에서 `AssertionError: episode pool differs`로 실패했다(`results/hl/hl12/curve/logs_failed_first_attempt/`). 원인은 hl014에 넣은 에피소드 해시 동일성 assert다. `sup_strat`은 Â²X 코사인으로 정렬하는데 값이 프로세스마다 미세하게 달라 같은 seed에서도 풀이 달라졌다. assert를 기록(`episode_pool_matches_first`)으로 바꾸고(`abf8135`) 5 run을 재실행해 모두 완료했다. 학습이 시작되기 전에 실패했으므로 학습 코드와 다른 run의 결과는 바뀌지 않았다.
3. **`sup_strat` 풀의 프로세스 간 차이.** seed 1212로 두 프로세스에서 풀만 생성해 비교했다. support는 전부 같고 `sup`과도 같다. query 슬롯 3.84%가 다르고, 72.8%의 에피소드에 차이가 1개 이상 있다. T25의 `sup_strat` 행은 `gcn_raw_sup_strat`의 풀이다. `pca64_lin_cos_sup_strat`은 같은 절차의 다른 풀이다. `ours`도 seed 1212·1214·1215에서 run 간 풀 해시가 다르다(차이 크기는 재지 않음). `naq`·`sup`은 같은 seed면 run 간 해시가 같다.
4. **epoch 0과 체크포인트 선택.** hl013과 같이 epoch 0은 선택 후보에서 뺐다. 지시서에 명시가 없어 정했다. 항등 초기화 arm은 epoch 0이 곧 학습 없는 값이므로, 후보에 넣은 값을 `bestL_incl_epoch0_*`와 `bestL_incl_epoch0_n_chosen_*` 열(history에서 사후 계산)에 함께 두었다. 동률은 앞 epoch를 택한다. 1-shot에서 epoch 0이 선택되는 경우는 `pca64_lin_cos_sup`·`pca64_lin_cos_sup_strat` 각 2 seed다.
5. **선택 규칙의 shot.** 지시서의 "raw/l2 valid 1-shot 최대 → `best_*_{shot}.pt`"는 shot별로 그 shot의 valid로 고르는 것으로 정했다(§3, HL11과 같음). 5-shot 값은 5-shot valid로 고른 체크포인트의 값이다.
6. **post-hoc.** `gcn_raw_naq`·`gcn_raw_ours`의 `bestL_*`는 HL11 history에서 l2 valid 최대(epoch 0 제외, 첫 동률)인 epoch의 test_l2다. `bestR_test_l2`도 해당 epoch의 history 값이다. T22 `source` 열에 표시했다. T22b에는 에피소드별 값이 없어 이 두 arm이 없다.
7. **trajmax 정의.** HL11과 같이 seed 평균 궤적의 최대이고 epoch 0을 포함한다. test를 보고 뽑은 값이다.
8. **epoch 0 assert.** `pca*_lin`은 1-shot l2 test를 HL11 T18 값과 비교해 ±0.05 assert했고 모두 통과했다. 5-shot은 assert하지 않았다(`epoch0_test_l2_5shot` 열).
9. **GCN epoch 0.** 모든 `gcn_*` arm이 72.5530 / 87.9510으로 같다(같은 초기화).
10. **큐·커밋 순서.** 지시서 큐 A → C → B → D를 따랐다. 커밋 조건("B 끝나면", "C·D 끝나면")대로라 태그 순서는 `hl12-sup` → `hl12-cos` → `hl12-pca`다. `hl12-cos` 시점의 표에는 `pca64_lin_cos_sup_strat`·D 행이 없다.
11. **파일명.** 지시서의 `episodes_<episodes>_seed.json`은 `episodes_<episodes>_seed<seed>.json`으로 저장했다. 저장은 `sup`·`sup_strat` 각 seed의 첫 run만 했다. assert 변경 뒤의 run은 통계를 `report.json`의 `episode_stats`에도 둔다.
12. **`scale`.** s=5·20은 seed 1212 run이고, T24의 s=10은 `pca64_lin_cos_sup` seed 1212 run이다. T23에는 s=5·20이 `n_seeds` 1로 들어 있다.
13. **표 열 추가.** T22 `source`·`bestL_test_l2_std_*`·`trajmax_epoch_*`·`bestL_incl_epoch0_*`, 참고 행의 5-shot, T23·T23b `n_seeds`, T24 `*_epoch_*`, T25 `n_classes_sampled`·`classes_sampled_*`·`query_a2x_cos_mean`(`sup` 포함)·`episode_sha256_prefix`.
14. **smoke·동등성 확인(지시서에 없음).** 분기 6종을 `--epoch 4`로, `gcn_raw_naq`를 `--epoch 160`으로 scratchpad에 돌린 뒤 출력을 삭제했다. 후자는 hl013 seed 1212와 epoch 0이 일치했고, epoch 160에서 test 1-shot 71.65 vs 71.66, l2 86.49 vs 86.48이었다.
15. **CLAUDE.md §8의 "train 57 / valid·test 20"** 은 `load_data`의 구성(train 40 / valid 17 / test 20)과 다르다. CLAUDE.md는 고치지 않았다.
16. push 실패 없음.

## ⑤ 산출물 경로

- 코드: `hl/hl014_train_variants.py`, `hl/hl000h_run_hl12.py`, `hl/hl_hl12_tables.py`
- 표: `reports/HL12_tables/{T22_arms,T22b_paired_vs_pca64,T23_curves_l2,T23b_curves_l2_std,T24_scale,T25_episodes,prereg_HL12}.csv`
- 특징(미추적): `results/hl/hl12/feat/{pca64.npy,pca256.npy,pca_meta.json}`
- 결과(미추적): `results/hl/hl12/curve/<arm>/seed_<seed>/`(`report.json`, `best_raw_{1,5}.pt`, `best_l2_{1,5}.pt`), `results/hl/hl12/curve/episodes_*_seed*.json`, `results/hl/hl12/curve/summary.csv`, `results/hl/hl12/env_{start,end}.txt`
- 로그(미추적): `out/hl/hl12_{A,CB,D,C2}.log`, `results/hl/hl12/curve/logs/`, `results/hl/hl12/curve/logs_failed_first_attempt/`
