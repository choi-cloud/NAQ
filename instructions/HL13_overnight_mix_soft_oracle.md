# HL13 — 야간 시퀀스: 사영 학습(SGC) · lr 스윕 · soft target · GCN 혼합 · oracle l2 · inductive PCA · 10-seed · 노름 진단 (Clothing, night-auto)

작성 2026-11-02(HL12 결과 반영) · 이희태 · 브랜치 `hl/anchor`(HL12 report 커밋 `2d85e4d` 위) · 서버 seio(RTX 4080 SUPER 1장) · 환경 `taskns`
**night-auto** 2026-11-02 21:30 → 11-03 08:30 KST(CLAUDE.md §7-1). Clothing만. 학습 run 약 170개(GCN 1분, 선형 수 초~20초). 단계는 서로 독립.

## 배경 (HL12가 바꾼 것)

- `gcn_raw_naq` 87.7(l2) > `gcn_raw_ours` 87.1 ≫ `gcn_raw_sup` 81.2 (base 40클래스 순도 1.0 에피소드; epoch 160 정점 뒤 하락). **base 레이블 pure 에피소드는 레이블 없는 pseudo-episode보다 novel 전이가 나쁘다.**
- `pca64_lin_cos_*` 전부 89.14–89.17(시작 89.05, 쌍 Δ +0.1, CI ±0.1). lr 1e-4·항등 초기화에서 가중치가 거의 안 움직였을 가능성.
- `gcn_cos_naq` 84.2 < `gcn_raw_naq` 87.7. cosine 손실은 GCN에 해롭다(s=10). 단 `gcn_cos_ours` 86.5 > `gcn_cos_naq` 84.2.
- `pca256_mlp` 83.8 — 무작위 초기화 비선형층은 PCA 정보를 망가뜨린다.

HL13의 질문: **학습이 PCA-64(89.05)를 넘을 수 있는가.** (A) 사영을 처음부터 에피소드로 학습하면 PCA가 하는 일을 배우는가, (B) pca64_lin이 안 움직인 것이 lr 때문인가, (C) GCN(raw)에 soft target·소량 base 혼합이 더하는가, (D) oracle 상한은 l2에서 몇 점인가.

레이블 사용 범위: base(train.mat 57 − 고정 valid 17 = **40클래스**, HL12 ④-1과 동일 집합) 레이블은 `sup`·`mix` 에피소드에. **oracle 에피소드(§2-4)에는 모든 클래스 레이블**(oracle 전용). valid·test 레이블은 그 밖에는 meta-test 평가에서만. PCA는 레이블 없이(§2-6 inductive 변형은 base 40클래스 노드 *집합*만 쓴다).

## 0. 시작 (목표 21:45)

1. HEAD가 `2d85e4d`(HL12: report)인지 확인. 아니면 **멈춤**(①). `CLAUDE.md`(개정 11)와 이 지시서를 한 커밋 `HL13: CLAUDE.md rev11, HL13 instruction`, 태그 `hl13-setup`, push.
2. `nvidia-smi`(`results/hl/hl13/env_start.txt`). 여유 10GB 미만이면 30분 간격 세 번 뒤 멈춤(②). pkl md5 확인.
3. `results/hl/hl13/progress.md` 생성. `mkdir -p results/hl/hl13/{curve,feat,diag} reports/HL13_tables`.
4. 고정 에피소드 seed 1217–1221이 없으면 `choi_exps/exp026.py`로 생성(CPU, 큐와 병행; pkl 불변).
5. `results/hl/hl12/feat/pca64.npy`(md5 `3d432fd1…`)·`pca256.npy` 존재 확인 — 재사용.

## 1. 학습 스크립트 `hl/hl015_train_hl13.py` (`hl014` 복사 + 아래 추가; `hl014`는 수정 금지) · 러너 `hl/hl000j_run_hl13.py` (`hl000h` 복사 + `--deadline HH:MM`(KST) + gcn/lin 2슬롯)

### 1-1. 인코더 추가 `--encoder`
| 값 | 정의 |
|---|---|
| `sgc64_rand` | 입력 Â²X_l2 **9,034차원 그대로**(z2, `structural_scores`; GPU float32 0.9GB) → `nn.Linear(9034, 64, bias=False)` **PyTorch 기본 무작위 초기화** → 행 l2. "PCA를 쓰지 않고 사영을 에피소드로 배우는가" |
| `sgc256_rand` | 같은 것, 출력 256 |
| `sgc64_pcainit` | 같은 구조, weight를 **PCA-64 성분 행렬**(`pca_meta`의 components, 평균 보정 포함)로 초기화 → epoch 0 = 89.05여야 함(assert ±0.05). 거기서 더 배우는가 |
| `pca1024_lin64` | 입력 PCA-1024(새로 계산, `results/hl/hl13/feat/pca1024.npy`) → `Linear(1024, 64)` 무작위 초기화 → l2 |
`pca64_lin`·`pca256_lin`·`gcn`은 hl014 그대로.

### 1-2. lr `--lr` (기본 1e-4 = 프로토콜) — **선형 인코더에만** {1e-3, 1e-2} 허용. GCN은 1e-4 고정. arm 이름 끝에 `_lr1e-3` 등.

### 1-3. 손실 추가 `--loss {raw_soft<A>, cos_soft<A>}`
목표가 soft인 cross-entropy. query q, anchor(support) 5개에 대해
- `t_struct(q) = softmax( cos(z2_q, z2_{a_1..5}) / 0.1 )` — z2 = Â²X 행 정규화(학습 전 1회, 고정, 레이블 없음)
- `t(q) = (1 − A)·onehot(pseudo-label) + A·t_struct(q)`, A ∈ {0.3, 0.6, 1.0}
- 손실 = −Σ_k t_k · log_softmax(logits)_k. `raw_soft`는 raw 거리 logits(meta.ProtoNet 식), `cos_soft`는 cosine logits(s=10).
에피소드별 `t_struct`의 pseudo-label 슬롯 평균(구조가 pseudo-label에 동의하는 정도)을 기록.

### 1-4. 에피소드 추가 `--episodes`
| 값 | 정의 |
|---|---|
| `mix_r<R>` | 2000개 중 round(2000·R)개 `sup`(base 40클래스, HL12 정의), 나머지 `naq`; 순서 `default_rng(seed)`로 섞음. R ∈ {0.1, 0.25, 0.5}. 출처 플래그 기록 |
| `oracle_strat5` | HL07 `o_grid_strat5_p10` 정의 — anchor 5개 균등(전체 노드), 각 anchor의 B4(X 코사인 순위 201–1000) 안 **GT-same**을 Â²X 순위 5층에서 2개씩(부족 시 HL07 보충). `hl011_pdd_train`의 함수 import. 모든 클래스 레이블 — oracle |

### 1-5. PCA 변형 `--pca_fit {all, base}` — `base`: base 40클래스 노드 집합으로만 fit(레이블 값은 안 씀) 후 전체 변환 → `feat/pca64_base.npy`. 학습 없는 l2 test를 먼저 잰다(`feat/pca_base_eval.json`).

### 1-6. 평가·선택: hl014와 동일(raw·l2, R·L, epoch 0; `bestL_incl_epoch0` 열 포함). 주 지표 `bestL_test_l2` 1-shot.

## 2. 단계와 큐 (KST; 모두 seed 1212–1216)

| # | 단계 | arm | run | 목표 | deadline |
|---|---|---|---|---|---|
| 1 | **사영 학습**(A) | `sgc64_rand_cos_naq`, `sgc64_rand_cos_sup`, `sgc256_rand_cos_naq`, `sgc64_pcainit_cos_naq`, `sgc64_pcainit_cos_sup`, `pca1024_lin64_cos_naq` — 모두 lr 1e-3 | 30 | 23:00 | 22:50 |
| 2 | **lr 스윕**(B) | `pca64_lin_cos_naq`, `pca64_lin_cos_sup` × lr {1e-3, 1e-2}; `sgc64_rand_cos_naq` × lr {1e-4, 1e-2} | 30 | 23:45 | 23:35 |
| 3 | **soft target**(C) | `gcn_raw_soft{0.3,0.6,1.0}_naq`, `gcn_raw_soft0.3_ours`, `pca64_lin_cos_soft{0.3,1.0}_naq`(lr 1e-3) | 30 | 01:15 | 01:00 |
| 4 | **GCN 혼합**(C) | `gcn_raw_mix_r{0.1,0.25,0.5}` | 15 | 02:00 | 01:45 |
| 5 | **oracle l2**(D) | `gcn_raw_oracle_strat5`, `pca64_lin_cos_oracle_strat5`(lr 1e-3), `sgc64_rand_cos_oracle_strat5`(lr 1e-3) | 15 | 03:00 | 02:45 |
| 6 | **inductive PCA** | `pca64_base` 학습 없는 평가 + `pca64_lin_cos_naq --pca_fit base`(lr 1e-3) | 5 | 03:20 | 03:10 |
| 7 | **10-seed** | `gcn_raw_naq`, `gcn_raw_ours`, `pca64_lin_cos_sup` × seed 1217–1221 | 15 | 04:00 | 03:45 |
| 8 | **노름 진단**(학습 없음) `hl/hl016_norm_diag.py` | §3 | — | 04:30 | — |
| 9 | **여분** | 단계 1·3의 1-shot 최고 arm 각 1개 × seed 1217–1221 | 10 | 05:30 | 05:15 |
| — | 마감 | 08:15 이후 새 run 금지, 08:30 보고서 push | | | |

러너 슬롯: gcn 1 + lin 1 동시. 단계 끝마다 표·커밋·태그(`hl13-proj`, `hl13-lr`, `hl13-soft`, `hl13-mix`, `hl13-oracle`, `hl13-pca`, `hl13-seeds`, `hl13-diag`)·push·progress.md 한 줄·채팅 한 줄. smoke(epoch 20)는 단계 1 시작 전 분기 전부(sgc·pcainit·soft·mix·oracle·pca_fit) 한 번에.

## 3. 노름 진단 `hl/hl016_norm_diag.py` (학습 없음, 레이블은 채점에만)
체크포인트: `encb`(B_Choi seed 1212 p0)·`encn`(NaQ seed 1212 p0)·HL12 `gcn_cos_naq`·`gcn_cos_ours`·`gcn_raw_sup` seed 1212 `best_l2_1.pt`. 각 전체 노드 raw 임베딩:
- 노름 분위(10/50/90)·std/mean, 노름–degree Spearman, 노름–Â²X 행 노름(정규화 전) Spearman.
- test 500에서 support 노름 − 같은 클래스 query 노름 평균의 분포(std), raw 오답 중 l2에서는 정답인 query 비율(norm-induced error).
- 클래스 간 평균 노름 분산 / 클래스 내 분산.
표 **T30** `T30_norm_diag.csv`. 커밋 `HL13: norm diagnostics`, 태그 `hl13-diag`.

## 4. 표 `reports/HL13_tables/` (`hl/hl_hl13_tables.py`)
- **T31** `T31_projection.csv`: 단계 1·2 arm × {epoch0_test_l2, bestL_test_l2, bestL_incl_epoch0, trajmax, final, 1·5-shot}; 참고 행 `a2x_pca64_l2` 89.05, HL12 `pca64_lin_cos_naq` 89.15.
- **T32** `T32_soft.csv`: A ∈ {0(= HL12 raw/cos), 0.3, 0.6, 1.0} × 인코더·에피소드 × {bestL_test_l2 1·5-shot, t_struct 동의도}.
- **T33** `T33_mix.csv`: R ∈ {0(= gcn_raw_naq), 0.1, 0.25, 0.5, 1(= gcn_raw_sup)} × {bestL_test_l2, trajmax, bestL_epoch, 1·5-shot}.
- **T34** `T34_oracle_l2.csv`: oracle 3 arm × {bestR_test_raw, bestL_test_l2, 1·5-shot}; 참고 HL07 raw 90.12 / 94.74.
- **T35** `T35_pca_inductive.csv`, **T36** `T36_tenseed.csv`(5-seed vs 10-seed), T30.

## 5. 사전 등록 (계산값만; 1-shot `bestL_test_l2`, 5 seed)

| 번호 | 예측 | 계산값 |
|---|---|---|
| HLP153 | **A 핵심**: `sgc64_rand_cos_naq`(lr 1e-3) ≥ 87.0 (무작위 사영을 레이블 없는 에피소드로 배워 GCN 수준에 닿는다) | |
| HLP154 | `sgc64_rand_cos_naq` ≤ `a2x_pca64_l2` − 0.5 (그러나 PCA-64에는 못 미친다) | |
| HLP155 | `sgc64_pcainit_cos_naq` ≥ `a2x_pca64_l2` − 0.3 (PCA 초기화에서 떨어지지 않는다) | |
| HLP156 | `sgc64_pcainit_cos_sup` ≤ `sgc64_pcainit_cos_naq` (base 에피소드가 novel 전이를 깎는다 — 사영에서도) | |
| HLP157 | `sgc64_rand_cos_sup` ≤ `sgc64_rand_cos_naq` − 2.0 | |
| HLP158 | **B**: `pca64_lin_cos_naq_lr1e-3` − HL12 `pca64_lin_cos_naq`(89.15) ∈ [−0.3, +0.5] (lr를 올려도 PCA-64 위에서는 거의 안 움직인다) | |
| HLP159 | `pca64_lin_cos_*_lr1e-2`가 epoch 0보다 낮아진다(≤ 89.05 − 0.5; 큰 lr는 PCA를 망가뜨린다) | |
| HLP160 | **C 핵심**: `gcn_raw_soft0.3_naq` ≥ `gcn_raw_naq`(87.72) + 0.5 | |
| HLP161 | `gcn_raw_soft1.0_naq` ≥ `gcn_raw_naq` − 1.0 | |
| HLP162 | t_struct 동의도 평균(`naq`) ≥ 0.5, `ours`는 그보다 낮다 | |
| HLP163 | `gcn_raw_mix_r0.1` ≥ `gcn_raw_naq` − 0.3 (소량 base 혼합은 해치지 않는다) | |
| HLP164 | `gcn_raw_mix_r0.5` ≤ `gcn_raw_naq` − 2.0 (절반이면 깎인다) | |
| HLP165 | **D**: `gcn_raw_oracle_strat5` bestR_test_raw가 HL07 90.12와 ±0.5 안; bestL_test_l2 ≥ 91.5 | |
| HLP166 | `pca64_lin_cos_oracle_strat5` ≥ 90.0 (oracle 에피소드는 PCA-64 위에서도 더한다 — base `sup`과 달리) | |
| HLP167 | `pca64_base` 학습 없는 l2 ≥ 89.05 − 0.5 | |
| HLP168 | 노름: `encn`·`encb`의 norm-induced error ≥ 0.3; `gcn_cos_*`는 ≤ 0.1 | |

## 6. 마감 보고서 `reports/HL13_overnight.md` (08:15 이후, 08:30 push)
CLAUDE.md §5 형식. progress.md 전체, T30–T36 전체, 사전 등록 표(계산값만; 미실행은 "미실행"), ④, ⑤. 레이블 범위 한 줄. 커밋 `HL13: overnight report`, push. 채팅에 progress.md + T31 + T32 + T33 + T34 + 사전 등록 표. 해석·제안 없음.
