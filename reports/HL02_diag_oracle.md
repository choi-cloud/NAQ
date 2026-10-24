# HL02 보고서 — 학습 없는 진단, anchor oracle 사다리, B 학습 전 점검

지시서: `instructions/HL02_diag_oracle.md` · 브랜치 `hl/anchor` · 태그 `hl02-diag` / `hl02-oracle` / `hl02-embed` · 2026-10-25
데이터셋 Amazon_clothing, τ = 0.3.

레이블 사용 범위: GT 레이블은 `hl/hl002_diag_nodes.py`(진단 통계), `hl/hl003_oracle_train.py`(oracle anchor 선택 + `anchor_diagnostics`), `hl/hl004_embed_check.py`(`hard:4` oracle anchor 선택)에서만 읽었다(`hl_common.labels_np`). query 선택(`exp049.build_structural_queries`)에는 레이블이 들어가지 않는다.

## ① 실행 명령·환경·commit

- 환경: `taskns`(python 3.9.25, torch 2.4.1+cu121, PyG 2.6.1). 설치 없음. GPU RTX 4080 SUPER, `--device 0`.
- commit: `186653d` HL02: node/pair/episode diagnostics (`hl02-diag`) · `a014749` HL02: anchor oracle ladder (`hl02-oracle`) · `1bc35cf` HL02: embedding pre-checks (`hl02-embed`) · 이 보고서 커밋. 시작 HEAD `127f213`(`hl01-p0`).
- 명령(리포 루트, `conda activate taskns`):
```bash
# §0 고정 에피소드 (seed 1215, 1216 × shot 1, 5)
PYTHONPATH=$PWD python -u choi_exps/exp026.py --dataset Amazon_clothing --n_way 5 --k_shot_test <shot> --q_query_test 8 \
  --num_valid_tasks 50 --num_downstream_tasks 500 --seed <seed> --device 0 \
  --output_dir save/fixed_episodes/Amazon_clothing/5way_<shot>shot_8query_seed<seed> \
  --json_path results/hl/hl02/episodes_Amazon_clothing_<shot>shot_seed<seed>.json
# §2
PYTHONPATH=$PWD nohup python -u hl/hl002_diag_nodes.py > out/hl/HL02_diag.log 2>&1 & echo $! > out/hl/HL02_diag.pid
# §3 (55 run, --jobs 1)
PYTHONPATH=$PWD nohup python -u hl/hl000b_run_oracle.py --jobs 1 > out/hl/HL02_oracle.log 2>&1 & echo $! > out/hl/HL02_oracle.pid
# §4
PYTHONPATH=$PWD nohup python -u hl/hl004_embed_check.py > out/hl/HL02_embed.log 2>&1 & echo $! > out/hl/HL02_embed.pid
# 표
python hl/hl005_hl02_tables.py
```

## ② 표

정확도 단위 %. 모든 표는 스크립트가 쓴 `reports/HL02_tables/*.csv`를 그대로 옮긴 것이다.

### 시간
- 고정 에피소드 생성(`results/hl/hl02/episode_generation_times.txt`): seed 1215 1-shot 4.1s, 5-shot 4.1s; seed 1216 1-shot 4.0s, 5-shot 4.1s.
- `D0_timing.csv` (hl002):
```
item,seconds
load,2.42
mine_all_nodes_seconds,26.82
topL_filter_seconds,1.07
total_seconds,31.18
```
전체 24,919 노드 mining 26.82s.
- oracle 55 run wall 합계 2840.1s(`results/hl/hl02/oracle/summary.csv`의 `wall_seconds` 합). 정책별 run당 wall: quintile:* 48.0s, uniform·gt_distinct·hard:2 52.0–52.1s, hard:4 52.0–66.1s, ab 52.0–78.1s, hard_blind:4 52.1–90.1s (러너가 2초 간격으로 종료를 확인하므로 해상도 약 2s).

### D1 — 노드 통계 (`D1_node_stats.csv`)
```
stat,p10,p25,p50,p75,p90,mean
raw_retained,30.0,106.0,301.0,729.0,1439.2000000000007,521.4160279304948
H50,0.0,1.0,6.0,24.0,40.0,13.294313575986195
pass50,10.0,26.0,44.0,49.0,50.0,36.705686424013805
pi_bchoi,0.0,0.1,0.3,0.8,1.0,0.41718367510734783
pi_naq,0.1,0.3,0.7,1.0,1.0,0.59719892451543
degree,1.0,1.0,3.0,7.0,17.0,7.358240699867571
spearman:raw_retained~degree,,,,,,0.3494446277315423
spearman:raw_retained~pi_bchoi,,,,,,-0.1667185035355541
spearman:pass50~pi_bchoi,,,,,,0.031636866289253204
spearman:degree~pi_bchoi,,,,,,-0.017343234180008448
spearman:H50~pi_bchoi,,,,,,-0.031636866289253204
spearman:raw_retained~H50,,,,,,-0.6599703754958002
spearman:raw_retained~pi_bchoi|degree,,,,,,-0.17149364165204312
fallback_rate,,,,,,0.03535454873791083
```

### D1b — `pi_bchoi` 5분위 (`D1b_pi_bchoi_quintiles.csv`)
```
quintile,n_nodes,pi_bchoi_min,pi_bchoi_max,pi_bchoi_mean,degree_mean,raw_retained_mean
1,4984,0.0,0.0,0.0,7.006821829855538,590.0692215088283
2,4984,0.0,0.2,0.119161316211878,8.36376404494382,643.8172150882825
3,4984,0.2,0.5,0.34616773675762436,8.153290529695024,674.8485152487962
4,4984,0.5,0.8,0.6648475120385232,6.814606741573034,420.0449438202247
5,4983,0.8,1.0,0.9558498896247241,6.452538631346578,278.2514549468192
```

### D2 — 쌍 통계 (`D2_pair_stats.csv`)
```
set,n_pairs,same_gt_rate
topL_accepted,914669,0.5738075741060428
topL_rejected,331281,0.2175343590486626
uniform_random,100000,0.02029
topL_rejected_rank1-10,34413,0.23209252317438178
topL_rejected_rank11-30,129087,0.21765940799615763
topL_rejected_rank31-50,167781,0.2144521727728408
```

### D3 — 에피소드 충돌 통계 (`D3_episode_stats.csv`)
```
stat,value
anchor_gt_all_distinct,0.815
anchor_gt_exactly_one_pair_same,0.175
anchor_gt_two_or_more_pairs_same,0.01
query_sets_overlap_between_anchors,0.041
anchor_in_other_anchor_queries,0.0095
anchor_pair_mean_x_cosine,0.02978295087814331
anchor_pair_mean_a2x_cosine,0.09757895022630692
n_episodes,2000
```

### T4 — oracle 사다리, 5 seed (`T4_oracle.csv`; seed별 값은 `T4b_oracle_per_seed.csv`)
```
policy,n_seeds,test_1shot_mean,test_1shot_std,test_5shot_mean,test_5shot_std,valid_1shot_mean,delta_1shot_vs_uniform,delta_5shot_vs_uniform,same_gt_pair_rate,anchor_pi_bchoi_mean,k_eff_mean
uniform,5,82.7,0.29,89.91,0.59,77.06,0.0,0.0,0.0196,0.42,
quintile:1,5,77.35,0.91,87.48,0.41,72.6,-5.35,-2.43,0.023,0.0,
quintile:2,5,77.41,0.33,86.68,0.45,74.11,-5.29,-3.23,0.0217,0.1191,
quintile:3,5,79.1,0.86,88.47,0.38,74.43,-3.6,-1.44,0.0363,0.3455,
quintile:4,5,82.56,0.27,90.43,0.65,75.87,-0.14,0.52,0.0354,0.6638,
quintile:5,5,81.84,0.35,90.35,0.17,73.06,-0.86,0.44,0.0518,0.9557,
gt_distinct,5,82.95,0.77,90.15,0.35,76.91,0.25,0.24,0.0,0.4164,
hard:2,5,82.35,0.85,89.87,0.42,76.84,-0.35,-0.04,0.0,0.3874,1.279
hard:4,5,82.12,0.42,90.31,0.35,76.0,-0.58,0.4,0.0,0.3619,2.184
hard_blind:4,5,80.31,0.69,89.46,0.3,75.13,-2.4,-0.45,0.194,0.3889,2.582
ab,5,82.89,0.41,90.3,0.47,76.29,0.19,0.39,0.0,0.4858,1.891
```
std는 표본 표준편차(n−1). 모든 정책에서 `anchor_diagnostics.uniform_fallback_rate` = 0.0(재추출 상한 20회 초과 없음).

### T5 — 5분위 곡선 (`T5_quintile_curve.csv`)
```
policy,test_1shot_mean,test_1shot_std,n_seeds
uniform,82.7,0.29,5
quintile:1,77.35,0.91,5
quintile:2,77.41,0.33,5
quintile:3,79.1,0.86,5
quintile:4,82.56,0.27,5
quintile:5,81.84,0.35,5
```

### 일관성 점검 — uniform vs P0 B_Choi (`prereg_HL02.csv` 행 `consistency`)
```
consistency,uniform 5-seed mean=82.70; P0 B_Choi 3-seed mean=82.81; diff=-0.11
```
|diff| = 0.11 ≤ 0.6. seed별(1-shot test, `query_gt_match`, best epoch — 왼쪽 hl003 uniform, 오른쪽 P0): 1212 83.14 / 83.20, 0.41924 / 0.41924, 1920 / 1920; 1213 82.725 / 82.77, 0.42127 / 0.42127, 2000 / 2000; 1214 82.48 / 82.45, 0.41482 / 0.41482, 1920 / 1920.

### E1 — 임베딩 쌍 거리 1 − cos (`E1_pair_dist.csv`)
```
checkpoint,ckpt_epoch,set,n,mean,median,p25,p75
ours_tau0.3,1920,rejected,26505,0.39301617387510024,0.3520166873931885,0.17625081539154053,0.5910085439682007
ours_tau0.3,1920,accepted,73495,0.06918497532595988,0.04084402322769165,0.014552652835845947,0.08566027879714966
ours_tau0.3,1920,uniform,100000,0.7109086472678184,0.7776181399822235,0.6218761205673218,0.8647292703390121
naq,1600,rejected,26505,0.39664361396390885,0.3695034980773926,0.22301030158996582,0.5524103045463562
naq,1600,accepted,73495,0.10056379680220258,0.06384831666946411,0.018771737813949585,0.14829638600349426
naq,1600,uniform,100000,0.691090501999855,0.739658385515213,0.6032829135656357,0.8263539969921112
```

### E2 — negative 난이도 (`E2_neg_softmax.csv`; 500 에피소드 × 50 query = 25,000 query)
```
checkpoint,policy,n_queries,p_own_mean,p_own_median,p_neg_sum_mean,p_neg_sum_median,p_neg_max_mean,p_neg_max_median,margin_mean,margin_median,frac_p_neg_sum_lt_0.05
ours_tau0.3,uniform,25000,0.829135537147522,0.9498305320739746,0.17086447775363922,0.05016949400305748,0.13855041563510895,0.033916376531124115,3.3259730339050293,3.331608772277832,0.49936
ours_tau0.3,hard_blind:4,25000,0.5845854878425598,0.5566584467887878,0.4154144823551178,0.44334155321121216,0.2572360038757324,0.24782498180866241,1.4677814245224,0.6615867614746094,0.23544
ours_tau0.3,hard:4,25000,0.6819007992744446,0.7966105937957764,0.3180992603302002,0.20338934659957886,0.2176356017589569,0.13737386465072632,2.0708136558532715,1.742497444152832,0.3072
naq,uniform,25000,0.7770926356315613,0.9095891714096069,0.2229074239730835,0.09041084349155426,0.17160281538963318,0.05959988012909889,2.7156500816345215,2.7213103771209717,0.3896
naq,hard_blind:4,25000,0.557648777961731,0.5365614891052246,0.44235125184059143,0.4634385108947754,0.2775532901287079,0.254108726978302,1.1175940036773682,0.6221320629119873,0.19276
naq,hard:4,25000,0.6439874768257141,0.7397683262825012,0.3560124933719635,0.26023170351982117,0.24140650033950806,0.16791188716888428,1.6305731534957886,1.4620401859283447,0.241
```

## ③ 사전 등록 대조표 (계산값만, `reports/HL02_tables/prereg_HL02.csv`)

| 번호 | 예측 | 계산값 |
|---|---|---|
| HLP007 | `pi_bchoi` 전체 평균은 0.42 ± 0.03, `pi_naq`는 0.60 ± 0.03 (HL01 T3와 일치) | pi_bchoi mean = 0.4172; pi_naq mean = 0.5972 |
| HLP008 | ρ(`raw_retained`, `degree`) ≥ 0.3 — 통과 수는 degree에 묶여 있다 | ρ = 0.3494 |
| HLP009 | ρ(`raw_retained`, `pi_bchoi`) < 0 — 생존자가 많을수록 middle이 멀어져 순도가 낮다 | ρ = −0.1667 (degree 통제 편상관 −0.1715) |
| HLP010 | X 상위 50 중 rejected의 same-GT 비율 < accepted의 절반; 그러나 균등 쌍 비율보다는 높다 | rejected = 0.2175; accepted = 0.5738 (절반 0.2869); uniform = 0.0203 |
| HLP011 | 2-4: anchor GT 모두 다름 ≥ 0.75, 한 쌍 겹침 ≤ 0.22 (Choi CoraFull과 같은 양상) | 모두 다름 = 0.8150; 정확히 한 쌍 = 0.1750 |
| HLP012 | 2-4: query 겹침 에피소드 비율 ≥ 0.1 (코드가 막지 않으므로 드물지 않다) | 0.0410 |
| HLP013 | T5: 5분위 곡선의 max − min ≥ 1.0%p (평평하지 않다) | q1..q5 = 77.35, 77.41, 79.10, 82.56, 81.84; max − min = 5.20 |
| HLP014 | T5: q=5(최고 순도) − uniform ≤ +2.0 이고, 곡선은 단조 증가가 **아니다** (HL01에서 순도↓가 1-shot↑였으므로) | q5 − uniform = −0.86 (uniform 82.70); 곡선 q1..q5 = 77.35, 77.41, 79.10, 82.56, 81.84 |
| HLP015 | T4: `gt_distinct` − `uniform` 은 ±0.4 안 (Choi CoraFull 재현) | +0.25 |
| HLP016 | T4: `hard:4` − `uniform` ≥ +1.0 (B go 기준) | −0.58 |
| HLP017 | T4: `hard:4` − `hard_blind:4` ≥ +0.5 (hard 샘플링 아래서는 거짓 음성 비용이 보인다) | +1.81 |
| HLP018 | T4: `hard:2`는 `uniform`과 `hard:4` 사이 | uniform = 82.70; hard:2 = 82.35; hard:4 = 82.12 |
| HLP019 | T4: `ab` ≥ max(`quintile:5`, `hard:4`) − 0.3 | ab = 82.89; max(quintile:5, hard:4) − 0.3 = 81.82 |
| HLP020 | T4: 모든 oracle 정책에서 Δ_5shot < Δ_1shot | (Δ1, Δ5) quintile:1 (−5.35, −2.43); quintile:2 (−5.29, −3.23); quintile:3 (−3.60, −1.44); quintile:4 (−0.14, +0.52); quintile:5 (−0.86, +0.44); gt_distinct (+0.25, +0.24); hard:2 (−0.35, −0.04); hard:4 (−0.58, +0.40); hard_blind:4 (−2.40, −0.45); ab (+0.19, +0.39) |
| HLP021 | 4-1 (ours ckpt): rejected 쌍 거리 중앙값 < 균등 쌍 거리 중앙값 (B 전제) | rejected 중앙값 = 0.3520; uniform 중앙값 = 0.7776 |
| HLP022 | 4-2 (ours ckpt, uniform): Σp(negative) 중앙값 < 0.10 | 0.0502 |
| HLP023 | 4-2: `hard_blind:4`의 Σp(negative) 평균 > `uniform`의 1.5배 (k가 hardness를 올린다) | ours ckpt: hard_blind:4 = 0.4154, uniform = 0.1709, 비 = 2.431 (naq ckpt 비 = 1.984) |

판정 규칙 입력값(`prereg_HL02.csv` 행 `decision_input`): max(quintile 최고 = 82.56[q4], hard:4 = 82.12, ab = 82.89) − uniform = **0.19**.

## ④ 실패·경고·이상

- **고정 에피소드가 seed와 무관하게 동일**: seed 1212–1216의 `save/fixed_episodes/Amazon_clothing/5way_{1,5}shot_8query_seed*/{valid,test}_episodes.json`은 shot별로 sha256이 모두 같다(1-shot test `0e7baca74817f0c9…`, valid `64e99dc7e84f864c…`; 5-shot test `0dd19325135a0ef9…`, valid `643b950fc64231f9…`). 원인: `choi_exps/exp026.py`가 부르는 `task_generator.downstream_task_generator`가 `save/downstream_tasks/Amazon_clothing/5way_{1,5}shot.pkl`이 있으면 seed와 상관없이 그 파일을 반환한다(pkl은 HL01 seed 1212 첫 생성 때 만들어짐, 2026-10-24 06:03). HL01의 seed 1213·1214와 HL02의 5 seed 모두 같은 valid/test 에피소드를 쓰고, seed 사이 차이는 학습 쪽(초기화·anchor·pseudo-label)뿐이다. 지시서 §0-2("HL01 방식 그대로")대로 생성했고 pkl은 건드리지 않았다.
- **§0-1**: 시작 시 `git status`에는 이희태가 넣은 `CLAUDE.md`(수정)·`instructions/HL02_diag_oracle.md`(미추적)와 CLAUDE.md §9에 적힌 실행 부산물(`results/Amazon_clothing/`, run 디렉터리의 `wall_seconds.txt`)만 있었다. HEAD는 `hl01-p0`(`127f213`). clean으로 보고 진행했다. CLAUDE.md·지시서는 `hl02-diag` 커밋에 포함했다.
- **exp049 동일성 assert**: 처음에는 노드별 단일 행 matmul로 점수를 계산해 비교했고, node 6718에서 `selected_ids`가 달랐다. 배치 matmul과 단일 행의 X 코사인 차이는 최대 1.19e-7이었다. exp049와 같은 형태(정렬된 unique id를 한 배치로 GEMM)로 바꾼 뒤 통과했다(`selected_ids` 20개 일치, `mean_ax_cosine` 최대 오차 5.96e-8; `diag_summary.json`의 `identity_check`). 따라서 exp049의 query 선택은 mining 배치 구성에 따라 일부 노드에서 달라질 수 있다. hl002 노드 표(전체 노드 mining)와 학습 run의 mining은 노드에 따라 선택이 다를 수 있다.
- **동시 실행**: hl003 테스트 run의 `max_cuda_memory_reserved_gb` = 8.22 → 2개 동시 실행은 16GB를 넘을 수 있어 `--jobs 1`로 순차 실행했다.
- **테스트 run**: 그리드 전에 `hard:4` seed 1212를 `results/hl/hl02/oracle_test/`에 한 번 돌렸다(1-shot test 81.595). 같은 설정의 그리드 run은 81.51이다. 표에는 그리드 값만 쓴다. uniform과 P0 B_Choi도 같은 anchor·query(`query_gt_match` 동일)에서 1-shot test가 seed별로 −0.06 / −0.045 / +0.03 다르다.
- **정책 디렉터리 이름**: `:` 대신 `_`(`hard:4` → `hard_4`, `quintile:3` → `quintile_3`).
- **5분위 정의**: `pi_bchoi`는 0.1 단위 값이라 동률이 많다. 동률은 `default_rng(1212).permutation`으로 끊고 순위 기준 동일 크기(4984×4 + 4983)로 나눴다. q1은 전부 0.0이고, 경계값 0.0·0.2·0.5·0.8이 이웃 분위에 함께 나타난다(D1b).
- **degree 정의**: 대칭화한 `edge_index`의 이웃 수, self-loop 제외(exp049 필터가 쓰는 그래프). `data.degree`(원본 방향 행합)가 아니다.
- **D3 query**: 전체 노드 mining의 B_Choi query로 계산했다. anchor는 seed 1212에서 exp049와 같은 RNG 순서(`load_data` 직후 `spt_sampling` 2000회)로 뽑았다.
- **k_eff < k**: `hard:4`의 k_eff 평균 2.184, `hard:2` 1.279, `ab` 1.891, `hard_blind:4` 2.582. 탈락 후보(X 상위 50 중 rejected; D1의 H50 중앙값 6)가 부족한 a₁에서는 부족분을 지시서 규칙대로 균등 추출로 채웠다.
- **hl003의 uniform**: exp049와 같은 `proto_task_generator` 호출로 anchor를 만든다(같은 RNG 상태).
- **E2 uniform 에피소드**: `random.seed(1212)` 뒤 `spt_sampling` 500회. hard 계열은 같은 seed로 `AnchorSampler`(hl003)를 썼다.
- **hl004 소요**: 약 66s(로그 파일 시각 기준, 따로 측정하지 않음).
- **summary 사본**: oracle `summary.csv`를 `reports/HL02_tables/`에 복사해 커밋했다가, 채팅 규칙("summary.csv는 results/hl/hl02/에만")에 따라 push 전에 그 커밋을 amend해 뺐다.
- 실패한 run은 없다(55/55 완료).

## ⑤ 산출물 경로

- 스크립트: `hl/hl_common.py`, `hl/hl002_diag_nodes.py`, `hl/hl003_oracle_train.py`, `hl/hl000b_run_oracle.py`, `hl/hl004_embed_check.py`, `hl/hl005_hl02_tables.py`
- 표: `reports/HL02_tables/{D0_timing,D1_node_stats,D1b_pi_bchoi_quintiles,D2_pair_stats,D3_episode_stats,T4_oracle,T4b_oracle_per_seed,T5_quintile_curve,E1_pair_dist,E2_neg_softmax,prereg_HL02}.csv`
- 진단 산출물: `results/hl/hl02/diag/node_table_tau0.3.csv`, `pi_bchoi_quintiles.csv`, `bchoi_selected_ids.npy`, `diag_summary.json`
- oracle: `results/hl/hl02/oracle/summary.csv`; run별 `results/hl/hl02/oracle/<policy>/seed_<seed>/{report.json,best_1shot.pt,best_5shot.pt}`(policy 디렉터리: uniform, quintile_1…quintile_5, gt_distinct, hard_2, hard_4, hard_blind_4, ab; seed 1212–1216); run 로그 `results/hl/hl02/oracle/logs/`
- 테스트 run: `results/hl/hl02/oracle_test/hard_4/seed_1212/`
- 고정 에피소드(신규): `save/fixed_episodes/Amazon_clothing/5way_{1,5}shot_8query_seed{1215,1216}/`, 메타 `results/hl/hl02/episodes_Amazon_clothing_{1,5}shot_seed{1215,1216}.json`, 로그 `results/hl/hl02/logs/`
- 로그: `out/hl/HL02_{diag,oracle_test,oracle,embed}.log` (+ `.pid`)
