# ND02 — inter-class 연결 패턴만으로 pseudo-class를 식별할 수 있는가 (학습 없음)

지시서 `instructions/ND02_inter_profile.md` · 브랜치 `nd/anchor-vote` · 서버 seio · 모드 auto · 데이터셋 Amazon_clothing · seed 1212

레이블 사용 범위: `gt_k3` 에피소드 구성(`gt_episodes`), Q2의 same(q)·T2의 GT 불일치율(`grade_q1_q2`), Q3 쌍 라벨(`grade_pairs`), 에피소드 통계(`grade_episode_stats`)에서만 `H.labels_np`로 읽었다. 배정·profile·패턴·점수 계산에는 넣지 않았고, `gt_k3`의 배정도 특징 프로토타입 argmax다.

## ① 실행 명령·환경·commit

```
export PYTHONPATH=$PWD; PY=/home/h2tae/.conda/envs/taskns/bin/python
$PY nd/nd003_episodes.py
$PY nd/nd004_profiles.py --stage stats
$PY nd/nd004_profiles.py --run main
$PY nd/nd004_profiles.py --run pseudo_k5 --episodes pseudo_k5
$PY nd/nd004_profiles.py --run pseudo_choi --episodes pseudo_choi
$PY nd/nd004_profiles.py --run f_a2x --f a2x
$PY nd/nd004_profiles.py --run unassign_q30 --unassign q30
$PY nd/nd004_profiles.py --run nbr_a2 --nbr a2
$PY nd/nd004_profiles.py --run score_var --score var
$PY nd/nd004_profiles.py --run main_rewire --rewire
$PY nd/nd004_profiles.py --run main_gt --episodes gt_k3
$PY nd/nd_nd02_tables.py --ndp
```

- 태그: `nd02-setup` d7ea723, `nd02-episodes` c8572e1, `nd02-main` a378182, `nd02-controls` d7340df. 시작 HEAD 05cd2c3.
- 시작 시 기록(`results/nd/nd02/env_start.txt`):

```
2026-11-02T06:22:49+00:00
seio
d7ea723df5013b82174657f529acf783b326ac04
?? results/Amazon_clothing/
?? results/ND01_chat_output.txt
?? results/hl/hl02/
?? results/hl/p0/Amazon_clothing/
?? results/nd/
2.4.1+cu121 2.6.1 1.13.1 1.5.2
index, memory.used [MiB], memory.total [MiB]
0, 2 MiB, 16376 MiB
5f901727f81903cc12159d280be6915d  save/downstream_tasks/Amazon_clothing/5way_1shot.pkl
ca3edaca02d541cbc0ee15a2122ef24a  save/downstream_tasks/Amazon_clothing/5way_5shot.pkl
```

- pkl md5는 CLAUDE.md §2의 값과 같다. GPU 최대 할당 7.37 GB(run마다 같음).

## ② 표

### T0 에피소드 통계 (`T0_episodes.csv`)

| episodes | n_episodes | support_purity_mean | query_purity_mean | anchor_gt_conflicts_mean | episodes_with_conflict_frac | n_query_mean |
|---|---|---|---|---|---|---|
| pseudo_k3 | 300 | 0.7233 | 0.5692 | 0.1933 | 0.1800 | 50.0000 |
| pseudo_k5 | 300 | 0.6782 | 0.5439 | 0.1933 | 0.1800 | 50.0000 |
| pseudo_choi | 300 | 0.7233 | 0.4132 | 0.1933 | 0.1800 | 49.6167 |
| gt_k3 | 300 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 50.0000 |

### T1 패턴 (`T1_pattern.csv`)

| run | defined_support_frac | defined_query_frac | group_pattern_frac | pi_entropy_mean | pi_cos_mean | h_mean | n_inter_mean | n_inter_median |
|---|---|---|---|---|---|---|---|---|
| main | 0.2284 | 0.3308 | 0.3913 | 0.2003 | 0.2651 | 0.8605 | 1.0146 | 0.0000 |
| pseudo_k5 | 0.2419 | 0.3333 | 0.5313 | 0.2431 | 0.2692 | 0.8622 | 0.9961 | 0.0000 |
| pseudo_choi | 0.2244 | 0.4214 | 0.3867 | 0.1939 | 0.2633 | 0.8408 | 1.6517 | 0.0000 |
| f_a2x | 0.0336 | 0.1190 | 0.0713 | 0.0065 | 0.1123 | 0.9386 | 0.4926 | 0.0000 |
| unassign_q30 | 0.1871 | 0.2736 | 0.3160 | 0.1612 | 0.2278 | 0.8822 | 0.7743 | 0.0000 |
| nbr_a2 | 0.7444 | 0.7999 | 0.8733 | 0.5836 | 0.3520 | 0.8329 | 0.1316 | 0.0491 |
| score_var | 0.2284 | 0.3308 | 0.3913 | 0.2003 | 0.2651 | 0.8605 | 1.0146 | 0.0000 |
| main_rewire | 0.9353 | 0.9456 | 0.9973 | 0.9918 | 0.5456 | 0.2013 | 6.5547 | 3.0000 |
| main_gt | 0.1938 | 0.3896 | 0.4127 | 0.2153 | 0.2538 | 0.8072 | 1.1153 | 0.0000 |

### T2 Q1 식별 (`T2_q1_identify.csv`)

| run | acc_pseudo | ci95 | n_episodes | excluded_query_frac | gt_mismatch_pred_ne_own | gt_mismatch_pred_eq_own |
|---|---|---|---|---|---|---|
| main | 0.5090 | 0.0412 | 268 | 0.6364 | 0.5178 | 0.4778 |
| pseudo_k5 | 0.3677 | 0.0328 | 290 | 0.4823 | 0.5310 | 0.5079 |
| pseudo_choi | 0.5553 | 0.0396 | 273 | 0.6358 | 0.6188 | 0.6471 |
| f_a2x | 0.6831 | 0.0891 | 65 | 0.9501 | 0.5493 | 0.5217 |
| unassign_q30 | 0.5934 | 0.0455 | 241 | 0.7278 | 0.5313 | 0.4898 |
| nbr_a2 | 0.5070 | 0.0172 | 300 | 0.1267 | 0.5057 | 0.3730 |
| score_var | 0.5007 | 0.0422 | 268 | 0.6364 | 0.5135 | 0.4843 |
| main_rewire | 0.2157 | 0.0073 | 300 | 0.0027 | 0.4370 | 0.4068 |
| main_gt | 0.5192 | 0.0397 | 285 | 0.6139 | 0.0000 | 0.0000 |

### T3 Q2 노이즈 (`T3_q2_noise.csv`)

| run | score | auroc_rows | n_episodes | auroc_pooled | n_queries |
|---|---|---|---|---|---|
| main | s_own | 0.5826 | 210 | 0.6464 | 2991 |
| main | h | 0.5971 | 210 | 0.6210 | 2991 |
| main | a2x | 0.7255 | 210 | 0.7890 | 2991 |
| main | x | 0.6713 | 210 | 0.6965 | 2991 |
| main | comb | 0.6836 | 210 | 0.6524 | 2991 |
| pseudo_k5 | s_own | 0.5713 | 235 | 0.6330 | 3804 |
| pseudo_k5 | h | 0.5938 | 235 | 0.6009 | 3804 |
| pseudo_k5 | a2x | 0.7181 | 235 | 0.7737 | 3804 |
| pseudo_k5 | x | 0.6592 | 235 | 0.6727 | 3804 |
| pseudo_k5 | comb | 0.6675 | 235 | 0.6290 | 3804 |
| pseudo_choi | s_own | 0.5390 | 210 | 0.5297 | 3388 |
| pseudo_choi | h | 0.5523 | 210 | 0.5717 | 3388 |
| pseudo_choi | a2x | 0.6569 | 210 | 0.6606 | 3388 |
| pseudo_choi | x | 0.5328 | 210 | 0.5556 | 3388 |
| pseudo_choi | comb | 0.6255 | 210 | 0.5435 | 3388 |
| f_a2x | s_own | 0.5775 | 35 | 0.6398 | 440 |
| f_a2x | h | 0.5983 | 35 | 0.6822 | 440 |
| f_a2x | a2x | 0.7565 | 35 | 0.8193 | 440 |
| f_a2x | x | 0.6769 | 35 | 0.7323 | 440 |
| f_a2x | comb | 0.7484 | 35 | 0.6315 | 440 |
| unassign_q30 | s_own | 0.5891 | 172 | 0.6514 | 2292 |
| unassign_q30 | h | 0.5754 | 172 | 0.6143 | 2292 |
| unassign_q30 | a2x | 0.7256 | 172 | 0.8053 | 2292 |
| unassign_q30 | x | 0.6903 | 172 | 0.7019 | 2292 |
| unassign_q30 | comb | 0.6920 | 172 | 0.6389 | 2292 |
| nbr_a2 | s_own | 0.6290 | 298 | 0.6482 | 11163 |
| nbr_a2 | h | 0.6237 | 298 | 0.6375 | 11163 |
| nbr_a2 | a2x | 0.7421 | 298 | 0.7609 | 11163 |
| nbr_a2 | x | 0.6666 | 298 | 0.6812 | 11163 |
| nbr_a2 | comb | 0.7052 | 298 | 0.6793 | 11163 |
| score_var | s_own | 0.5722 | 210 | 0.6331 | 2991 |
| score_var | h | 0.5971 | 210 | 0.6210 | 2991 |
| score_var | a2x | 0.7251 | 210 | 0.7890 | 2991 |
| score_var | x | 0.6713 | 210 | 0.6965 | 2991 |
| score_var | comb | 0.6785 | 210 | 0.6486 | 2991 |
| main_rewire | s_own | 0.5352 | 299 | 0.5378 | 14160 |
| main_rewire | h | 0.5073 | 299 | 0.5130 | 14160 |
| main_rewire | a2x | 0.6274 | 299 | 0.6434 | 14160 |
| main_rewire | x | 0.6675 | 299 | 0.6827 | 14160 |
| main_rewire | comb | 0.5976 | 299 | 0.5931 | 14160 |
| main_gt | s_own |  | 0 |  | 3466 |
| main_gt | h |  | 0 |  | 3466 |
| main_gt | a2x |  | 0 |  | 3466 |
| main_gt | x |  | 0 |  | 3466 |
| main_gt | comb |  | 0 |  | 3466 |

### T4 Q3 쌍 비교 (`T4_q3_pairs.csv`)

| run | score | auroc_all | auroc_episode_mean | auroc_cos_x_q1 | auroc_cos_x_q2 | auroc_cos_x_q3 | auroc_cos_x_q4 | auroc_same_group | auroc_diff_group | n_pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| main | cos_inter5 | 0.7072 | 0.7091 | 0.4595 | 0.4951 | 0.4673 | 0.6779 | 0.6887 | 0.4267 | 68386 |
| main | cos_full5 | 0.9080 | 0.9102 | 0.6407 | 0.6806 | 0.7295 | 0.7281 | 0.6415 | 0.7746 | 624000 |
| main | cos_x | 0.9536 | 0.9527 | 0.5000 | 0.5461 | 0.6917 | 0.7890 | 0.7351 | 0.8918 | 624000 |
| main | cos_a2x | 0.9623 | 0.9611 | 0.6969 | 0.7790 | 0.8613 | 0.8256 | 0.7870 | 0.9281 | 624000 |
| pseudo_k5 | cos_inter5 | 0.7049 | 0.7088 | 0.5246 | 0.4625 | 0.4740 | 0.6672 | 0.6824 | 0.4254 | 89402 |
| pseudo_k5 | cos_full5 | 0.9082 | 0.9101 | 0.6655 | 0.6824 | 0.7403 | 0.7223 | 0.6382 | 0.7853 | 832500 |
| pseudo_k5 | cos_x | 0.9516 | 0.9502 | 0.5000 | 0.5240 | 0.7028 | 0.7821 | 0.7333 | 0.8915 | 832500 |
| pseudo_k5 | cos_a2x | 0.9612 | 0.9599 | 0.6932 | 0.7126 | 0.8620 | 0.8244 | 0.7897 | 0.9268 | 832500 |
| pseudo_choi | cos_inter5 | 0.6716 | 0.6890 | 0.5647 | 0.5709 | 0.6176 | 0.6196 | 0.6033 | 0.4123 | 95736 |
| pseudo_choi | cos_full5 | 0.9006 | 0.9027 | 0.7305 | 0.7484 | 0.8091 | 0.7098 | 0.6140 | 0.8018 | 616734 |
| pseudo_choi | cos_x | 0.9241 | 0.9237 | 0.5000 | 0.6123 | 0.6958 | 0.7135 | 0.6724 | 0.8939 | 616734 |
| pseudo_choi | cos_a2x | 0.9457 | 0.9454 | 0.7793 | 0.8035 | 0.9035 | 0.7826 | 0.7306 | 0.9293 | 616734 |
| f_a2x | cos_inter5 | 0.6769 | 0.6890 | 0.6101 | 0.5352 | 0.4539 | 0.6176 | 0.6634 | 0.4637 | 10260 |
| f_a2x | cos_full5 | 0.9083 | 0.9143 | 0.6132 | 0.6652 | 0.6654 | 0.6998 | 0.5761 | 0.7136 | 624000 |
| f_a2x | cos_x | 0.9536 | 0.9527 | 0.5000 | 0.5461 | 0.6917 | 0.7890 | 0.7351 | 0.8918 | 624000 |
| f_a2x | cos_a2x | 0.9623 | 0.9611 | 0.6969 | 0.7790 | 0.8613 | 0.8256 | 0.7870 | 0.9281 | 624000 |
| unassign_q30 | cos_inter5 | 0.7159 | 0.7330 | 0.4645 | 0.4998 | 0.4677 | 0.6831 | 0.6866 | 0.4327 | 48798 |
| unassign_q30 | cos_full5 | 0.9130 | 0.9149 | 0.6362 | 0.6641 | 0.7222 | 0.7243 | 0.6237 | 0.7794 | 602093 |
| unassign_q30 | cos_x | 0.9536 | 0.9527 | 0.5000 | 0.5461 | 0.6917 | 0.7890 | 0.7351 | 0.8918 | 624000 |
| unassign_q30 | cos_a2x | 0.9623 | 0.9611 | 0.6969 | 0.7790 | 0.8613 | 0.8256 | 0.7870 | 0.9281 | 624000 |
| nbr_a2 | cos_inter5 | 0.7693 | 0.7789 | 0.4063 | 0.4534 | 0.3987 | 0.7218 | 0.6861 | 0.3287 | 393305 |
| nbr_a2 | cos_full5 | 0.9442 | 0.9437 | 0.7072 | 0.7971 | 0.8113 | 0.7894 | 0.7270 | 0.8721 | 624000 |
| nbr_a2 | cos_x | 0.9536 | 0.9527 | 0.5000 | 0.5461 | 0.6917 | 0.7890 | 0.7351 | 0.8918 | 624000 |
| nbr_a2 | cos_a2x | 0.9623 | 0.9611 | 0.6969 | 0.7790 | 0.8613 | 0.8256 | 0.7869 | 0.9281 | 624000 |
| score_var | cos_inter5 | 0.7072 | 0.7091 | 0.4595 | 0.4951 | 0.4673 | 0.6779 | 0.6887 | 0.4267 | 68386 |
| score_var | cos_full5 | 0.9080 | 0.9102 | 0.6407 | 0.6806 | 0.7295 | 0.7281 | 0.6415 | 0.7746 | 624000 |
| score_var | cos_x | 0.9536 | 0.9527 | 0.5000 | 0.5461 | 0.6917 | 0.7890 | 0.7351 | 0.8918 | 624000 |
| score_var | cos_a2x | 0.9623 | 0.9611 | 0.6969 | 0.7790 | 0.8613 | 0.8256 | 0.7870 | 0.9281 | 624000 |
| main_rewire | cos_inter5 | 0.6188 | 0.6251 | 0.5256 | 0.5231 | 0.5126 | 0.5733 | 0.5354 | 0.5110 | 555449 |
| main_rewire | cos_full5 | 0.5311 | 0.5322 | 0.5117 | 0.5358 | 0.4858 | 0.5312 | 0.5405 | 0.5063 | 624000 |
| main_rewire | cos_x | 0.9536 | 0.9527 | 0.5000 | 0.5461 | 0.6917 | 0.7890 | 0.7351 | 0.8918 | 624000 |
| main_rewire | cos_a2x | 0.7632 | 0.7750 | 0.5087 | 0.5705 | 0.4719 | 0.6780 | 0.6704 | 0.5940 | 624000 |
| main_gt | cos_inter5 | 0.7027 | 0.6912 | 0.6285 | 0.6486 | 0.6678 | 0.7704 |  |  | 80820 |
| main_gt | cos_full5 | 0.8984 | 0.8943 | 0.7977 | 0.8124 | 0.8553 | 0.8909 |  |  | 624000 |
| main_gt | cos_x | 0.8383 | 0.8406 | 0.5000 | 0.5519 | 0.6011 | 0.7861 |  |  | 624000 |
| main_gt | cos_a2x | 0.8890 | 0.8943 | 0.6813 | 0.6768 | 0.7921 | 0.8629 |  |  | 624000 |

### T5 커버리지, 주 설정 (`T5_coverage.csv`)

| row | query_frac | acc_pseudo | auroc_s_own | auroc_a2x | n_q1 | n_ep_s_own | n_ep_a2x | pair_cos_inter5 | pair_cos_full5 | pair_cos_x | pair_cos_a2x | n_pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n_inter 0 | 0.6692 | 0.0000 |  | 0.6951 | 2463 | 0 | 201 |  |  |  |  |  |
| n_inter 1-2 | 0.2293 | 0.6783 | 0.5801 | 0.7031 | 1983 | 182 | 182 |  |  |  |  |  |
| n_inter 3-4 | 0.0474 | 0.6541 | 0.6897 | 0.6617 | 477 | 56 | 56 |  |  |  |  |  |
| n_inter >=5 | 0.0541 | 0.6478 | 0.5683 | 0.7189 | 531 | 56 | 56 |  |  |  |  |  |
| pairs n_inter>=3 (both nodes) |  |  |  |  |  |  |  | 0.7154 | 0.8553 | 0.8739 | 0.9333 | 7608 |

## ③ 사전 등록 대조표 (계산값만)

| 번호 | 예측 | 계산값 |
|---|---|---|
| NDP011 | query 중 inter-profile이 정의되는(n_inter ≥ 1) 비율 ≥ 0.5 | n_inter>=1 query frac = 0.3308 |
| NDP012 | **Q1 핵심**: `acc_pseudo` ≥ 0.30 (우연 0.20) | acc_pseudo = 0.5090 (CI 0.0412) |
| NDP013 | `acc_pseudo` − `acc_pseudo_rewire` ≥ +0.05 | 0.5090 - rewire 0.2157 = +0.2933 |
| NDP014 | **Q3 핵심**: `cos_inter5` auroc_all ≥ 0.55 | cos_inter5 auroc_all (pooled) = 0.7072; episode mean = 0.7091 |
| NDP015 | `cos_inter5` auroc(cos_x 최상위 분위) ≥ 0.53 (특징이 비슷한 쌍에서도 신호) | cos_inter5 auroc cos_x q4 = 0.6779 |
| NDP016 | `cos_inter5` auroc_all − 같은 값(rewire) ≥ +0.03 | 0.7072 - rewire 0.6188 = +0.0884 |
| NDP017 | **Q2**: `s_own` auroc_rows 평균 ≥ `h` + 0.02 (inter 성분이 호모필 비율보다 낫다) | s_own 0.5826 - h 0.5971 = -0.0146 |
| NDP018 | `s_own` auroc_rows 평균 ≥ `a2x` − 0.05 | s_own 0.5826 - a2x 0.7255 = -0.1429 |
| NDP019 | `comb` auroc_rows 평균 ≥ `a2x` + 0.01 | comb 0.6836 - a2x 0.7255 = -0.0419 |
| NDP020 | **oracle**: `_gt` run의 `acc_pseudo` ≥ 0.35 그리고 `cos_inter5` auroc_all ≥ 0.60 (GT 그룹에서는 패턴이 보인다) | _gt acc_pseudo = 0.5192; _gt cos_inter5 auroc_all = 0.7027 |
| NDP021 | `_gt` − `pseudo_k3`의 `acc_pseudo` 차이 ≥ +0.05 (pseudo-label 품질이 패턴을 흐린다) | _gt 0.5192 - main 0.5090 = +0.0102 |
| NDP022 | `n_inter ≥ 5` 구간의 `acc_pseudo` ≥ `1–2` 구간 + 0.05 | n_inter>=5 acc 0.6478 - n_inter 1-2 acc 0.6783 = -0.0304 |

## ④ 실패·경고·이상·지시서에 없어 정한 것

### 실행
- 실패한 run·정지 사유 없음. push 실패 없음.
- run 수: 지시서 §9는 "주 설정 + 변형 7개 + rewire + gt = 10 run"이나, 표의 축에서 나오는 서로 다른 run은 9개다(주 설정, `pseudo_k5`, `pseudo_choi`, `f_a2x`, `unassign_q30`, `nbr_a2`, `score_var`, `main_rewire`, `main_gt`). 변형 7개에 `gt_k3`가 들어 있어 gt가 두 번 세어진 것으로 보고 9 run을 돌렸다.
- CLAUDE.md §2와 지시서는 GPU 여유 10GB, §7(d)는 6GB로 적혀 있다. 10GB를 적용했다. 시작 시 사용량 2 MiB / 16376 MiB.
- 지시서 머리말의 레이블 범위("§6과 §7-4")는 Q2의 same(§5)·에피소드 통계(§1-2)·gt 생성(§1-3)도 포함하는 채팅 지시 범위로 적용했다.

### 재현성
- `H.structural_scores`의 z1·z2는 프로세스마다 바이트가 달랐다(3회 md5 모두 다름; z0는 같음). 이 때문에 값이 run마다 조금 달라진다. 주 설정 5회 관측: `a2x` auroc_rows 0.72533(1회), 0.72552(4회); `comb` 0.68358–0.68367; `s_own`·`h`·`x`·`acc_pseudo`·`cos_inter5`·`cos_x`는 소수 5자리까지 같다. 표의 값은 `summary/main.json`의 실행 값이다. z1·z2 차이의 원인은 확인하지 않았다.

### 에피소드
- 순차 배제는 에피소드의 anchor 5개를 처음부터 후보에서 뺐다(서로소 조건을 지키기 위해).
- `pseudo_k5`의 anchor는 `pseudo_k3`와 같다. `gt_k3`는 새 `default_rng(1212)`를 썼고, 그룹의 첫 support를 anchor로 둔다.
- `pseudo_choi`: 뽑힌 query 중 제거 — 자기 그룹 support와 겹침 85, 다른 그룹 support와 겹침 13, 앞 그룹 query와 겹침 17. 남은 query 14885 / 15000. 자기 그룹 support와 겹치는 경우는 지시서에 없어 제거했다. 다른 그룹 support와 겹치면 그 support가 뒤 anchor의 것이어도 query 쪽을 제거했다(support 고정). mining fallback은 anchor 슬롯 기준 47회.
- `gt_k3`: 13개 미만 클래스 0개, 다시 뽑은 횟수 0회.
- T0의 support 순도는 anchor 자신을 뺀 K−1개 기준이다. `gt_k3`의 `anchor_gt_conflicts`는 구성상 0이다.

### 정의·계산
- `main_rewire`: `data.edge_index`를 재배선 그래프로 바꾸고 `structural_scores`를 다시 계산했다. `a2x`·`cos_a2x`는 재배선 그래프의 z2이고, 에피소드와 z0(`x`, `cos_x`)는 원본과 같다. 시도 916800회, 수락 905757회, 원본에 남은 엣지 509 / 91680. degree 일치 assert 통과. swap마다 두 번째 엣지 방향을 무작위로 뒤집었다.
- Q1: §5대로 "그룹 패턴 없음"과 "모든 가설 −∞"만 제외했다. 그래서 `n_inter = 0`이고 Σc > 0인 query는 Q1에 남고 own 가설이 −∞라 항상 오답이다(T5의 `n_inter 0` 행 acc 0.0000, 2463개). T2의 `gt_mismatch_pred_ne_own`에도 이 query가 들어 있다. 주 설정 제외 내역: 패턴 없음 9130, 모든 가설 −∞ 416.
- Q1: 패턴 없는 그룹 g는 다른 그룹 query에서도 가설 g의 점수를 −∞로 뒀다.
- Q1 정의상 가설 g ≠ own(q)의 `p_q^(g)`에는 own 슬롯이 남아 있다(가설별로 슬롯 g만 뺌).
- Q2: §3의 "n_inter = 0이면 분석에서 제외"를 따라 `n_inter ≥ 1`이고 그룹 패턴이 있는 query만 썼고, 다섯 점수 모두 같은 query 집합이다(주 설정 2991개). 참고로 `n_inter = 0`을 −∞로 포함한 `s_own` auroc_rows는 0.5139(에피소드 247개, `summary/main.json`).
- `main_gt`: 모든 query의 GT가 anchor와 같아 음성이 없다. T3의 `main_gt` 행은 n_episodes 0, AUROC 빈칸이고 T2의 GT 불일치율은 0이다. T4의 same-group·diff-group 열도 빈칸이다(같은 그룹 쌍은 전부 same_gt, 다른 그룹 쌍은 전부 diff_gt).
- `cos_inter5` 정의: `inter5`는 own 슬롯이 0이므로 같은 그룹 쌍은 0인 슬롯이 같고 다른 그룹 쌍은 다르다(나머지 슬롯이 균등이면 코사인 1 대 0.75). `main_gt`에서는 same_gt와 같은 그룹 여부가 일치한다. 재배선 그래프에서 `cos_inter5`는 auroc_all 0.6188, 같은 그룹 쌍 0.5354, 다른 그룹 쌍 0.5110이다. `gt_k3` × rewire 조합은 지시서 범위 밖이라 돌리지 않았다.
- `comb`: 같은 Q2 query 집합 안에서 에피소드별 순위 평균. pooled는 그 순위를 그대로 합쳤다.
- T4 `auroc_all`은 전체 쌍 pooled이고, 지시서 §6의 "둘 다"에 따라 `auroc_episode_mean` 열을 추가했다. 분위·same/diff-group 열은 pooled다.
- `cos_x` 분위 경계는 run마다 전체 쌍에서 구했다. `cos_x = 0`인 쌍이 많아 주 설정 분위 크기가 0.3914, 0.1087, 0.2500, 0.2500로 균등하지 않다(경계 0.0000, 0.0224, 0.1015; 경계와 같은 값은 아래 분위). `cos_inter5`·`cos_full5`는 정의되지 않은 노드가 든 쌍을 뺐고, 분위 경계는 뺀 뒤 다시 구하지 않았다. `pseudo_choi`의 빈 query 슬롯이 든 쌍은 모든 점수에서 뺐다.
- π 간 코사인(T1)은 π_g에 슬롯 g = 0을 넣은 5차원에서 슬롯을 맞춰 계산했다. T1의 h·n_inter는 S ∪ Q 노드 전체 기준이고, `nbr_a2`의 n_inter는 개수가 아니라 Â² 가중치 합이다.
- T5 acc는 구간 안 query pooled이고, AUROC는 에피소드 행별 평균이다(`n_ep_*` 열). 주 설정 그룹당 정의된 support 수(0/1/2/3개): 913 / 292 / 149 / 146.
- `vote_rows`는 c_v 계산에 재사용했다(마지막 열 = 미배정). 1hop에서 빈 행이 없음을 assert했다.
- 첫 주 설정 실행 후 `cos_x` 분위 코드를 고쳤다(경계값 처리; 1분위가 비던 문제). 표는 수정 후 실행 값이다.

## ⑤ 산출물

- 코드: `nd/nd003_episodes.py`, `nd/nd004_profiles.py`, `nd/nd_nd02_tables.py`
- 표: `reports/ND02_tables/T0_episodes.csv`, `T1_pattern.csv`, `T2_q1_identify.csv`, `T3_q2_noise.csv`, `T4_q3_pairs.csv`, `T5_coverage.csv`, `NDP_values.csv`
- 결과(git 미추적): `results/nd/nd02/env_start.txt`, `episodes/{pseudo_k3,pseudo_k5,pseudo_choi,gt_k3}.npz`, `episodes/generation.json`, `episodes/stats.csv`, `profiles/*.npz`, `rewire/edge_index.npy`, `summary/<run>.json`
- 로그(git 미추적): `out/nd/nd003.log`, `out/nd/nd004_*.log`
