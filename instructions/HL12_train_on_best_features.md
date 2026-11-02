# HL12 — 가장 좋은 학습 없는 표현 위에서 학습이 무엇을 더하는가: 에피소드 출처 × 인코더 × 손실 (Clothing)

작성 2026-11-02 · 이희태 · 브랜치 `hl/anchor`(HL11 report 커밋 `463bd8c` 위) · 서버 seio(RTX 4080 SUPER 1장) · 환경 `taskns`
데이터셋 **Amazon_clothing만**. 모드: **auto**. 학습 run 57개(동시 1개).

## 배경 (HL11이 바꾼 것)

- 학습 임베딩을 **L2 정규화해서 평가하면** NaQ 87.84, B_Choi 87.75(1-shot)로 Â²X_l2 85.10을 넘는다. exp027의 비정규화 제곱 유클리드 평가(83.2 / 80.6)는 ReLU 출력의 노름 편차에 지배된다. 학습 곡선(l2)은 epoch 0 72.6 → 87.7. **학습은 더한다.**
- 그러나 **PCA-64(Â²X)_l2 = 89.05 / 93.72**가 학습 모델(l2)보다 +1.3 높다. 저분산 방향 제거의 효과이고(무작위 사영 256은 81.8), 학습 없는 상한이 oracle 학습 90.1(raw 평가)에 붙었다.
- B_Choi의 NaQ 대비 +2(raw)는 l2 평가에서 사라진다(87.75 vs 87.84).

세 질문:
- Q1 (M0) **base 레이블**로 만든 순도 1.0 에피소드로 학습하면 레이블 없는 에피소드보다 얼마나 높은가. 구조 층화(strat5식) positive를 더하면 더 오르는가.
- Q2 **PCA-64(Â²X) 입력 + 항등 초기화 선형층**(시작점 89.05)에 각 에피소드 출처로 학습하면 올라가는가, 내려가는가.
- Q3 GCN을 **cosine 손실**(정규화 후 거리)로 학습하면 raw 손실보다 l2 평가가 나아지는가.

레이블 사용 범위: **train(base) 클래스 57개의 레이블은 `sup`·`sup_strat` 에피소드 생성에 쓴다**(허용 — novel 클래스 레이블만 금지). valid·test 클래스 노드의 레이블은 meta-test 평가에서만. PCA는 전체 노드에 레이블 없이 fit(HL11과 동일).

## 0. 시작

1. `git status` 깨끗한지, HEAD가 `463bd8c`(HL11 report) 위의 문서 커밋인지 확인. 아니면 `CLAUDE.md`(개정 10)와 이 지시서를 한 커밋 `HL12: CLAUDE.md rev10, HL12 instruction`, 태그 `hl12-setup`, push.
2. 환경·GPU·pkl md5 확인은 HL11 §0과 같다(`results/hl/hl12/env_start.txt`). 설치 금지. GPU 여유 10GB 미만이면 멈춤.
3. HL11 산출물 존재 확인: `results/hl/hl11/curve/{naq,ours}/seed_{1212..1216}/report.json`(10개), `results/hl/hl11/feat/` — 이 run들은 **다시 돌리지 않고** 표에 재사용한다.
4. `mkdir -p results/hl/hl12/{feat,curve} reports/HL12_tables out/hl`.

## 1. 학습 스크립트 `hl/hl014_train_variants.py` (`hl013_curve.py`를 복사해 확장; 출처 docstring)

### 1-1. 에피소드 출처 `--episodes`
| 값 | 정의 | 레이블 |
|---|---|---|
| `naq` | hl013과 동일(`proto_task_generator`, unsup) | 없음 |
| `ours` | hl013과 동일(B_Choi, τ 0.3) | 없음 |
| `sup` | `task_generator.supervised_task_generator(data.id_by_class, data.class_list_train, labels, 5, 1, 10)` × 2000 (base 클래스 57개에서 5-way 1-shot, query 10/클래스 균등) | base만 |
| `sup_strat` | 클래스·support는 `sup`과 같게 뽑되, query 10개는 같은 클래스의 나머지 노드를 support 기준 **Â²X 코사인 내림차순**으로 정렬해 5등분한 뒤 각 층에서 2개 균등 추출(HL07 strat5의 레이블 있는 판). 나머지 노드가 10개 이하면 전부 쓰고 부족분은 복원 추출로 채우며 횟수 기록 | base만 |

`sup`·`sup_strat`의 rng는 `seed_everything(seed)` 뒤의 `random`·`default_rng(seed)`를 쓴다. 에피소드 2000개는 hl013처럼 학습 전에 일괄 생성한다. 생성된 에피소드의 통계(클래스 크기 분포, `sup_strat`의 query Â²X 코사인 평균·층별 평균, 복원 추출 횟수)를 `results/hl/hl12/curve/episodes_<episodes>_seed.json`에.

### 1-2. 인코더 `--encoder`
| 값 | 정의 |
|---|---|
| `gcn` | hl013과 동일(`GNNEncoder`, config [GCN 9034→512, GCN 512→256]; 출력은 마지막 ReLU 뒤) |
| `pca64_lin` | 입력 = PCA-64(Â²X_l2)_l2 (HL11 `a2x_pca64_l2`와 같은 절차: `structural_scores`의 z2 → `PCA(64, randomized, random_state=1212)` fit on all nodes → 변환 → 행 l2; **한 번 계산해 `results/hl/hl12/feat/pca64.npy`에 저장하고 모든 run이 읽는다**). 모델 = `nn.Linear(64, 64, bias=True)`, weight를 **항등**으로, bias를 0으로 초기화 |
| `pca256_lin` | 같은 절차로 PCA-256, `nn.Linear(256, 256)` 항등 초기화, `pca256.npy` |
| `pca256_mlp` | PCA-256 입력, `Linear(256,256) → ReLU → Linear(256,256)`, PyTorch 기본 초기화 |

`gcn`은 `data.features`·`data.adj`를 쓰고, `pca*`는 저장된 특징 행렬을 입력으로 쓰며 adj를 쓰지 않는다. optimizer·lr(1e-4)·epoch 2000·eval_interval 160은 전부 hl013과 같다.

### 1-3. 손실 `--loss`
| 값 | 정의 |
|---|---|
| `raw` | `meta.ProtoNet.forward` 식 그대로(제곱 유클리드 → log_softmax → NLL) |
| `cos` | 임베딩을 행 l2 정규화한 뒤 제곱 유클리드에 **scale s(기본 10)** 를 곱해 log_softmax → NLL. 즉 logits = −s·‖ẑ_q − p̂_k‖², p̂_k = 정규화된 support 평균(1-shot이면 support 자체) |

`meta.py`는 고치지 않는다. `hl014`에 `ProtoNetV`(ProtoNet 복사 + encoder·loss 분기)를 둔다.

### 1-4. 평가(hl013과 동일 + 선택 규칙 둘)
epoch 0과 매 160 epoch에 valid 50·test 500을 **raw 임베딩과 l2 임베딩 둘 다**로 1-shot·5-shot 평가해 `history`에 기록(`exp027.episode_accuracy`, l2는 `F.normalize` 후). `pca*_lin`의 epoch 0 l2 값이 HL11 `a2x_pca{64,256}_l2`(89.05 / 88.55)와 ±0.05 안인지 assert(항등 초기화 확인).
체크포인트 선택을 **두 규칙**으로 따로 둔다: (R) raw valid 1-shot 최대 → `best_raw_{shot}.pt`, (L) l2 valid 1-shot 최대 → `best_l2_{shot}.pt`. report.json의 `evaluation`에 두 규칙의 test(raw·l2 평가 각각)를 모두 넣는다. HL11의 hl013 run은 규칙 (R)만 있으므로, 그 run들의 (L) 값은 `history`에서 **l2 valid 최대 epoch의 test_l2**로 사후 계산한다(체크포인트 재저장 없음; 표에 `post-hoc`로 표시).

## 2. 실행 범위 (arm = `<encoder>_<loss>_<episodes>`, seed 1212–1216)

| 그룹 | arm | 새 run | 비고 |
|---|---|---|---|
| A (M0: 에피소드 출처, GCN raw) | `gcn_raw_naq`, `gcn_raw_ours` | 0 | HL11 hl013 재사용 |
| | `gcn_raw_sup`, `gcn_raw_sup_strat` | 10 | |
| B (GCN cosine) | `gcn_cos_naq`, `gcn_cos_ours`, `gcn_cos_sup` | 15 | |
| C (PCA 입력) | `pca64_lin_cos_naq`, `pca64_lin_cos_ours`, `pca64_lin_cos_sup`, `pca64_lin_cos_sup_strat` | 20 | |
| | `pca256_lin_cos_sup`, `pca256_mlp_cos_sup` | 10 | |
| D (scale 민감도) | `pca64_lin_cos_sup` s ∈ {5, 20}, seed 1212만 | 2 | |

합계 57 run. 러너는 `hl000g_run_hl11.py`를 복사한 `hl000h_run_hl12.py`(순차, 재시작 가능, summary.csv → `results/hl/hl12/curve/`). 큐 순서: A → C → B → D (M0와 PCA 결과가 먼저 나오게). `pca*` run은 GCN보다 훨씬 빠르다(수 초/run 예상). 예상 총 40–50분.

커밋: A 끝나면 `HL12: supervised episodes`(태그 `hl12-sup`), C·D 끝나면 `HL12: PCA-input training`(`hl12-pca`), B 끝나면 `HL12: cosine-loss GCN`(`hl12-cos`). 각각 push.

## 3. 표 `reports/HL12_tables/` (`hl/hl_hl12_tables.py`)

- **T22** `T22_arms.csv`: arm × {n_seeds, epoch0_test_l2, bestR_test_raw, bestR_test_l2, bestL_test_l2, bestL_epoch_mean, trajmax_test_l2, final_test_l2} 1-shot, 같은 열 5-shot. 참고 행: `a2x_l2` 85.10 / 92.72, `a2x_pca64_l2` 89.05 / 93.72, `a2x_pca256_l2` 88.55 / 93.14, HL11 `encb_l2` 87.75, `encn_l2` 87.84. **1-shot 주 지표는 `bestL_test_l2`**(l2 valid로 고른 체크포인트의 l2 test).
- **T23** `T23_curves_l2.csv`: arm × epoch × {valid_1shot_l2, test_1shot_l2, test_5shot_l2}(5 seed 평균; std는 `T23b`).
- **T24** `T24_scale.csv`: `pca64_lin_cos_sup` seed 1212, s ∈ {5, 10, 20} × {bestL_test_l2 1-shot·5-shot, trajmax}.
- **T25** `T25_episodes.csv`: `sup`·`sup_strat`의 에피소드 통계(§1-1).

## 4. 사전 등록 (Agent는 계산값만; 판정은 이희태). 1-shot `bestL_test_l2` 기준, 5 seed 평균

| 번호 | 예측 | 계산값 |
|---|---|---|
| HLP131 | **Q1**: `gcn_raw_sup` ≥ `gcn_raw_ours` + 1.0 (base 레이블 순도 1.0 에피소드가 레이블 없는 것보다 높다) | |
| HLP132 | `gcn_raw_sup_strat` ≥ `gcn_raw_sup` + 0.5 (다양성 레버가 진짜 레이블에서도 작동) | |
| HLP133 | **Q3**: `gcn_cos_naq` ≥ `gcn_raw_naq`(post-hoc L) − 0.3 (cosine 학습이 l2 평가를 해치지 않는다) | |
| HLP134 | `gcn_cos_sup` ≥ 89.0 | |
| HLP135 | **Q2 핵심**: `pca64_lin_cos_sup` ≥ 89.05 + 0.5 (PCA-64 위에서 순도 1.0 에피소드 학습이 더한다) | |
| HLP136 | `pca64_lin_cos_naq` ≥ 89.05 − 0.5 (레이블 없는 에피소드가 PCA-64를 망가뜨리지 않는다) | |
| HLP137 | `pca64_lin_cos_ours` − `pca64_lin_cos_naq` ∈ [−0.5, +0.5] | |
| HLP138 | `pca64_lin_cos_sup_strat` ≥ `pca64_lin_cos_sup` + 0.3 | |
| HLP139 | `pca256_mlp_cos_sup` ≤ `pca256_lin_cos_sup` + 0.5 (비선형이 더하지 않는다) | |
| HLP140 | `pca*_lin` 모든 arm의 epoch0_test_l2가 HL11 PCA 값과 ±0.05 안 (항등 초기화 확인) | |
| HLP141 | 5-shot: `pca64_lin_cos_sup` ≥ 93.72 | |
| HLP142 | scale s=5·20의 `bestL_test_l2`가 s=10과 ±1.0 안 | |

## 5. 판정 기준(참고, 판정은 이희태)
- HLP135 통과 → "PCA-64 Â²X + base 레이블 에피소드 + cosine 선형층"이 현재 최고점. 다음은 레이블 없는 에피소드를 섞어 novel 쪽 커버리지를 더할 때(M3) 떨어지지 않는지(HLP136·137이 그 예비 답).
- HLP135 미달·HLP136 통과 → 학습이 PCA-64 위에서 아무것도 더하지 못함. 학습 목표(cosine ProtoNet)가 아니라 다른 목표(M2 soft target, 대조 손실)를 다음에 본다.
- HLP131 미달(base 레이블 에피소드가 레이블 없는 것과 같음) → 에피소드 순도가 레버가 아니라는 뜻이라, D4(순도 레버)는 oracle에서만 성립하는 것 — 그 경우 novel 클래스 pure 에피소드와 base 클래스 pure 에피소드의 차이(전이 문제)가 다음 질문.
- HLP133 미달 → cosine 손실이 학습을 불안정하게 함. s 민감도(T24)를 먼저 본다.

## 6. 보고서 `reports/HL12_train_on_best_features.md`
CLAUDE.md §5 형식. 포함: 환경·md5, PCA 저장 파일 md5·설명 분산, T22 전체, T23(epoch 0·160·…·2000), T24, T25, 사전 등록 표(계산값만), ④, ⑤. 레이블 사용 범위 한 줄(base 레이블 사용 명시). 커밋 `HL12: report`, push. 끝나면 채팅에 T22 전체, T24, 사전 등록 표를 그대로 붙인다. 해석·제안 없음.
