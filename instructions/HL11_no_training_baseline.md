# HL11 — 학습 없는 Â²X 기준선 확정(차원·정규화 통제) + 학습 곡선(epoch 0부터) (Clothing)

작성 2026-11-02 · 이희태 · 브랜치 `hl/anchor`(HEAD `f86f817` 위의 문서 커밋에서 시작) · 서버 seio(RTX 4080 SUPER 1장) · 환경 `taskns`
데이터셋 **Amazon_clothing만**. 모드: **auto**. §1–§3은 학습 없음, §4는 학습 run 10개(동시 1개).

## 배경

ND01(브랜치 `nd/anchor-vote`)에서 학습 없는 Â²X 프로토타입이 같은 test 500 에피소드에서 **85.10 / 92.72**(1-shot / 5-shot)를 냈다. 학습된 NaQ는 80.65 / 90.52, B_Choi는 82.71 / 89.91이다. 이 사실이 맞다면 11월 논문의 모든 제안 방법은 85.1을 넘어야 하고, 비지도 meta-training은 Â²X가 가진 정보를 *잃고* 있다는 뜻이다. ND01은 Â²X를 행 정규화해 9,034차원 그대로 썼고 학습 임베딩은 256차원·비정규화였다. 이 지시서는 (1) 그 차이가 결과를 설명하는지 통제하고, (2) 학습 없는 상한이 어디까지인지(ÂX·Â³X·결합) 재며, (3) 학습이 정보를 **언제** 잃는지 — 무작위 초기화(epoch 0)에서 시작하는 곡선으로 본다.

세 질문:
- Q1 차원·정규화를 맞춰도 학습 없는 Â²X가 학습 모델을 넘는가.
- Q2 학습 없는 특징의 최고는 무엇인가(ÂX / Â²X / Â³X / 결합).
- Q3 학습 곡선: 무작위 초기화 GCN(epoch 0)은 몇 점이고, 학습이 그것을 올리는가 내리는가. valid 기준 best 시점과 test 궤적이 어긋나는가.

레이블 사용 범위: §1–§4의 meta-test 평가(valid·test 에피소드의 y_support·y_query)에서만. 특징 변환(PCA·사영)과 학습에는 쓰지 않는다.

## 0. 시작

1. `git status` 깨끗한지, HEAD가 문서 커밋(`HL: story D closure doc, CLAUDE.md rev9, HL11 instruction`)인지 확인. 아니면 그 커밋을 먼저 만든다(`CLAUDE.md`, `reports/HL_storyD_closure.md`, 이 지시서), 태그 `hl-storyD-closure`, push.
2. `conda activate taskns`. `python -c "import torch, torch_geometric, scipy, sklearn; print(torch.__version__, sklearn.__version__)"` 기록. **설치 금지.**
3. `nvidia-smi` 기록(`results/hl/hl11/env_start.txt`). 여유 10GB 미만이면 멈춤.
4. `md5sum save/downstream_tasks/Amazon_clothing/5way_{1,5}shot.pkl` — CLAUDE.md §3 값과 다르면 멈춤(§7(e)).
5. `ls save/fixed_episodes/Amazon_clothing/5way_{1,5}shot_8query_seed1212/{valid,test}_episodes.json`, `ls results/hl/p0/Amazon_clothing/{ours/tau_0.3,naq/baseline}/seed_1212/best_1shot.pt` — 넷 다 있어야 한다. 없으면 멈춤.
6. `mkdir -p results/hl/hl11/{feat,curve} reports/HL11_tables out/hl`.

## 1. 평가 공용 `hl/hl012_feat_eval.py` (학습 없음, 태그 `hl11-feat`)

### 1-1. 에피소드와 평가 함수
- 에피소드는 `exp049.load_splits(args)`(seed 1212, `q_query_test` 8)로 읽는다 — 학습 run이 쓰는 것과 같은 JSON. **정합성 검사**: `pickle.load`로 읽은 pkl의 `test_pool`과 JSON test 500개가 노드 집합·레이블까지 같은지 assert(ND01은 pkl을 직접 썼다). 다르면 멈추고 ④에 적는다.
- 정확도는 `exp027.episode_accuracy(embeddings, task, 5, shot)`를 그대로 호출한다. 에피소드별 정확도를 저장한다(`results/hl/hl11/feat/<arm>_<shot>shot_<pool>.npy`).
- 데이터 로드는 `hl_common.load_everything`을 **복사해** `add_diffusion`을 빼고 `hl012`에 둔다(X 코사인 dense 행렬 불필요). 출처를 docstring에 적는다.

### 1-2. 학습 없는 특징 arm (모두 전체 노드 [n, d] 행렬; `l2` = 행 L2 정규화, `raw` = 안 함)
Â = `hl_common.structural_scores`가 쓰는 대칭 정규화 행렬(그 함수의 h1·h2 계산을 복사해 h3까지 확장; z0·z1·z2는 그 함수의 반환값과 같은지 assert).

| arm | 정의 |
|---|---|
| `x_l2`, `x_raw` | X |
| `ax_l2`, `ax_raw` | ÂX |
| `a2x_l2`, `a2x_raw` | Â²X (`a2x_l2` = ND01의 `feat_a2x`, **85.10 / 92.72 재현 확인**) |
| `a3x_l2`, `a3x_raw` | Â³X |
| `cat_ax_a2x_l2` | [ÂX_l2 ‖ Â²X_l2] (각각 정규화 후 연결) |
| `sgc_mean_l2` | (X + ÂX + Â²X)/3 후 l2 (APPNP/SGC식 혼합의 가장 단순한 형태) |

### 1-3. 차원 통제 (Â²X_l2를 입력으로; 레이블 없음)
| arm | 정의 |
|---|---|
| `a2x_pca256_l2` | sklearn `PCA(n_components=256, svd_solver='randomized', random_state=1212)`를 전체 노드 Â²X_l2(CPU, float32)에 fit → 변환 → l2 |
| `a2x_pca256_raw` | 같은 것, 변환 후 정규화 없음 |
| `a2x_rp256_l2_s{0,1,2}` | 가우시안 무작위 사영 R ∈ R^{9034×256}, R_ij ~ N(0, 1/256), `default_rng(1212 + s)`, s = 0, 1, 2 → l2 |
| `a2x_pca64_l2`, `a2x_pca1024_l2` | 차원 민감도(64, 1024) |

PCA 1024는 메모리를 보고 안 되면 빼고 ④에 적는다(24,919 × 9,034 float32 ≈ 0.9GB, CPU).

### 1-4. 학습 임베딩 재평가 (정규화 통제)
체크포인트 둘(`enc_b` = B_Choi seed 1212, `enc_n` = NaQ seed 1212, 둘 다 `best_1shot.pt`). `hl005.encoder_embeddings`를 복사해 `F.normalize`를 선택 인자로 둔 판을 `hl012`에 만든다(`net.eval()`, `net(data.features, adj=data.adj)`).
| arm | 정의 |
|---|---|
| `encb_raw`, `encb_l2` | B_Choi 임베딩, 비정규화 / l2 |
| `encn_raw`, `encn_l2` | NaQ 임베딩 |
`encb_raw` 1-shot이 HL03 T0의 seed 1212 B_Choi(83.20)와 ±0.3 안인지 기록(같은 체크포인트·같은 에피소드라면 같아야 한다 — 다르면 체크포인트가 1-shot용이 아니거나 에피소드가 다른 것; ④에 적고 계속).

### 1-5. 표 `reports/HL11_tables/`
- **T18** `T18_feat.csv`: arm × {valid_1shot, test_1shot, test_1shot_ci95, valid_5shot, test_5shot, test_5shot_ci95, dim, paired_delta_1shot_vs_a2x_l2, paired_ci95, paired_win_frac, 같은 세 열 5-shot}. paired는 `a2x_l2` 기준 에피소드별 차이.
- **T19** `T19_dim_control.csv`: `a2x_l2`, `a2x_pca{64,256,1024}_l2`, `a2x_pca256_raw`, `a2x_rp256_l2_s0–2`(평균±std 행 추가), `encb_raw`, `encb_l2`, `encn_raw`, `encn_l2` — 1-shot·5-shot test와 `a2x_l2` 대비 paired Δ.

커밋 `HL11: no-training feature baselines`, 태그 `hl11-feat`, push.

## 2. (§1에 포함) 사전 등록 계산값은 §5 표에.

## 3. 선택 규칙(사전 고정)
"학습 없는 최고"는 **valid 50 1-shot**으로 고른 arm(`feat_best`)이고, 보고는 그 arm의 test로 한다. test에서 고르지 않는다. `a2x_l2`는 항상 기준 행으로 둔다.

## 4. 학습 곡선 `hl/hl013_curve.py` + 러너 (태그 `hl11-curve`)

### 4-1. 스크립트
`choi_exps/exp049.py: main`의 뼈대를 복사해(출처 docstring) 다음을 바꾼다. 나머지(에피소드 생성, `build_structural_queries`, 손실, optimizer, best 체크포인트 저장, report.json)는 **그대로**.
- **epoch 0 평가**: 학습 루프에 들어가기 전, 무작위 초기화 상태에서 valid·test를 1-shot·5-shot 모두 평가해 `history`의 첫 행(`epoch: 0`)으로 넣는다.
- **test 궤적**: 매 `eval_interval`(160)마다 valid뿐 아니라 **test 500도** 평가해 `history` 행에 `test_1shot`, `test_5shot`을 넣는다(체크포인트 선택은 종전대로 valid만 쓴다 — test는 기록용). 또한 같은 시점에 **행 정규화한 임베딩**으로도 valid·test를 평가해 `*_l2` 열을 둔다(`exp027.episode_accuracy`에 `F.normalize(embeddings, dim=1)`을 넘김).
- 출력 루트 `results/hl/hl11/curve/<method>/seed_<seed>/`. report.json에 `history`(epoch 0 포함, 열: epoch, train_loss, train_acc, valid_1shot, valid_5shot, test_1shot, test_5shot, valid_1shot_l2, valid_5shot_l2, test_1shot_l2, test_5shot_l2)와 종전 `evaluation`(best-valid 체크포인트의 test).
- 학습 자체는 바뀌지 않으므로 best-valid test는 HL03 T0(B_Choi 5 seed 82.71 ± 0.31)과 서버 차이 안에서 일치해야 한다. 기록.

### 4-2. 실행
`--method naq`와 `--method ours --tau 0.3` × seed 1212–1216 = **10 run, 동시 1개**(러너는 `hl/hl000_run.py`를 복사해 `hl000g_run_hl11.py`, 순차·재시작 가능, summary.csv를 `results/hl/hl11/curve/`에). 예상 10 × ~60초(test 평가 추가로 조금 늘어남) ≈ 15분.

### 4-3. 표
- **T20** `T20_curve.csv`: method × epoch(0, 160, …, 2000) × {valid_1shot 평균, test_1shot 평균, test_1shot_l2 평균, valid_5shot, test_5shot, test_5shot_l2}(5 seed 평균, std는 `T20b`).
- **T21** `T21_curve_summary.csv`: method × {epoch0 test_1shot, epoch0 test_1shot_l2, best-valid epoch(평균), best-valid test_1shot(= 종전 보고 방식), test_1shot 궤적 최대값과 그 epoch, 궤적 최종(2000) 값, 5-shot 같은 열}. 참고 행으로 `a2x_l2`(T18)와 `feat_best`.

커밋 `HL11: learning curves from epoch 0`, 태그 `hl11-curve`, push.

## 5. 사전 등록 (Agent는 계산값만; 판정은 이희태)

| 번호 | 예측 | 계산값 |
|---|---|---|
| HLP119 | `a2x_l2` test 1-shot이 ND01 값 85.10과 ±0.05 안 (재현) | |
| HLP120 | **Q1 핵심**: `a2x_pca256_l2` test 1-shot ≥ 84.0 (차원 256으로 줄여도 B_Choi 82.71을 넘는다) | |
| HLP121 | `a2x_rp256_l2` 3-seed 평균 ≥ 83.0 | |
| HLP122 | `encb_l2` − `encb_raw` (test 1-shot) ∈ [−0.5, +1.0] (정규화는 학습 임베딩을 크게 바꾸지 않는다) | |
| HLP123 | `a2x_raw` − `a2x_l2` ≤ −1.0 (정규화 없는 Â²X는 더 낮다 — 차수 효과) | |
| HLP124 | **Q2**: `feat_best`(valid 선택)는 `a2x_l2` 또는 `cat_ax_a2x_l2` 또는 `a3x_l2` 중 하나이고, test 1-shot이 `a2x_l2` + 1.5 이내 | |
| HLP125 | `ax_l2` < `a2x_l2` (test 1-shot) | |
| HLP126 | `a3x_l2` − `a2x_l2` ∈ [−1.0, +1.0] | |
| HLP127 | **Q3 핵심**: epoch 0 test 1-shot(무작위 GCN, `ours`·`naq` 평균) ≥ 78.0 (학습 전 GCN이 이미 X_l2 72.35보다 높다) | |
| HLP128 | `ours`의 test_1shot 궤적 최대값 ≤ `a2x_l2` − 1.0 (학습 중 어느 시점도 학습 없는 Â²X를 넘지 못한다) | |
| HLP129 | `ours` epoch 0 test 1-shot < best-valid test 1-shot (학습은 무작위 초기화보다는 올린다) | |
| HLP130 | `naq`·`ours` 모두 test_1shot_l2 궤적 최대값 − test_1shot 궤적 최대값 ∈ [−0.5, +1.0] | |

## 6. 판정 기준(참고, 판정은 이희태)
- HLP120·121 통과 → "차원이 아니라 학습 목표의 문제"로 확정. Choi 공유 문서 §3의 [미확인] 두 개가 닫힌다.
- HLP120 미달(PCA-256이 82.7 아래) → 85.1의 상당 부분이 고차원 희소 특징 효과. 기준선은 `a2x_pca256_l2`로 바꿔 적는다.
- HLP127·128 통과 → "학습은 무작위 초기화보다 올리지만 Â²X에는 못 미친다"가 곡선으로 확정. HLP127 미달(epoch 0이 낮음)이면 무작위 GCN이 Â²X 정보를 보존하지 못한다는 뜻이라, 손실이 학습 목표가 아니라 아키텍처(256차원 사영 + ReLU)에서 올 가능성을 다음 지시서에서 본다.

## 7. 보고서 `reports/HL11_no_training_baseline.md`
CLAUDE.md §5 형식. 포함: 환경·md5·nvidia-smi, pkl–JSON 정합성 검사 결과, T18 전체, T19, T20(epoch 0·160·320·…·2000 전부), T21, 사전 등록 표(계산값만), ④, ⑤. 레이블 사용 범위 한 줄. 커밋 `HL11: report`, push. 끝나면 채팅에 T19 전체, T21 전체, T18의 `a2x_l2`·`feat_best`·`cat_ax_a2x_l2`·`a3x_l2`·`encb_raw`·`encn_raw` 행, 사전 등록 표를 그대로 붙인다. 해석·제안 없음.
