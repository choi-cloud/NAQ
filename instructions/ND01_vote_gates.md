# ND01 — 이웃 투표 분포 d_v의 두 관문 (학습 없음): A. meta-test 분류, B. 후보 same-class AUROC

작성 2026-11-02 · 이희태 · 브랜치 `nd/anchor-vote`(HEAD `f86f817`에서 시작) · 서버 seio(RTX 4080 SUPER 1장) · 환경 `taskns`
데이터셋 Amazon_clothing만. **학습 없음.** 레이블은 §4(평가)와 §5(AUROC 채점)에서만 읽는다. 모드: **auto**.

## 목적

d_v(CLAUDE.md §0)가 쓸모 있는지 두 가지 용도로 각각 판정한다.
- **관문 A**: meta-test(실제 support가 anchor)에서 d_v를 특징 프로토타입과 결합하면 1-shot 정확도가 오르는가.
- **관문 B**: pseudo-episode 후보(B4 구간)가 anchor와 같은 클래스인지 가르는 레이블 없는 점수로서, d_v의 anchor 슬롯 값이 `ax`·`a2x`·`enc_b`보다 나은가. 특히 **이웃 투표(집계) 효과**와 **anchor 경쟁(softmax) 효과**를 분리한다.

두 관문 모두 같은 공용 모듈 `nd/nd_vote.py` 위에서 돈다.

## 0. 시작

1. `git status`가 깨끗한지, HEAD가 `f86f817`(또는 그 위의 ND01 문서 커밋)인지 확인. `CLAUDE.md`(개정 1)와 이 지시서를 한 커밋으로: `ND01: CLAUDE.md rev1, ND01 instruction`, 태그 `nd01-setup`, push.
2. `conda activate taskns`(또는 환경 python 전체 경로). `python -c "import torch, torch_geometric, scipy, sklearn; print(torch.__version__)"` 기록. **설치 금지.**
3. `nvidia-smi` 기록. 여유 6GB 미만이면 멈춤(CLAUDE.md §7(d)).
4. `md5sum save/downstream_tasks/Amazon_clothing/5way_1shot.pkl save/downstream_tasks/Amazon_clothing/5way_5shot.pkl` — CLAUDE.md §2의 값과 다르면 멈춤(§7(e)).
5. `ls results/hl/p0/Amazon_clothing/ours/tau_0.3/seed_1212/best_1shot.pt` — 없으면 enc_b 열을 비우고 ④에 적는다(정지 아님).
6. `mkdir -p nd results/nd/nd01 reports/ND01_tables out/nd`.

## 1. 공용 모듈 `nd/nd_vote.py`

docstring 첫 줄에 "formulas: CLAUDE.md §0; Â and z0/z1/z2 from hl/hl_common.structural_scores; adjacency as hl/hl005_diag_distance_purity.sym_adjacency"를 적는다. `hl_common`을 `H`로 import하고 `hl005_diag_distance_purity.sym_adjacency`를 가져온다.

### 1-1. 이웃 가중 행렬 (세 종류, 모두 torch sparse [n, n], 행 합 1)

| nbr | 정의 |
|---|---|
| `1hop` | A_sym = 대칭화한 A(자기 자신 **제외**), 행 정규화 D^-1 A |
| `self` | (A + I) 행 정규화 D^-1 (A + I) — 자기 배정도 한 표 |
| `a2` | Â² = (D^-1/2 (A+I) D^-1/2)² 의 행을 양수 그대로 쓰고 행 합이 1이 되게 다시 나눈 것(2-hop 가중) |

A는 `data.edge_index`에서 `structural_scores`와 같은 방식으로 대칭화한다. degree 0인 노드(자기 자신 제외 시)는 `1hop`에서 행이 비므로 그 노드의 d_v는 §1-2의 Q 행(자기 배정)으로 대체하고 개수를 `isolated_count`로 기록한다. `a2`는 sparse–sparse 곱 한 번(`torch.sparse.mm`)으로 만들고 nnz를 기록한다(Clothing에서 메모리 문제가 없어야 한다; 2GB를 넘으면 `a2`만 빼고 ④에 적는다).

### 1-2. 배정과 투표

```
assign(zf, P, tau, mode) -> Q [n, N]
  S = zf @ P.T                      # zf: 행 정규화 특징 [n, d], P: anchor 특징 [N, d] (행 정규화)
  mode == "soft": Q = softmax(S / tau, dim=1)
  mode == "hard": Q = one_hot(argmax S)
vote(Q, W) -> D [n, N] = W @ Q        # W: §1-1의 가중 행렬. D 행 합 1 (수치 오차 허용 1e-4)
```

필요한 행만 쓰므로 `vote_rows(Q, W, idx)`를 두어 `W.index_select(0, idx) @ Q`로 계산한다(sparse 행 선택이 안 되면 `W @ Q` 전체 후 gather — Clothing은 전체 계산도 가볍다).

특징 공간 f ∈ {`x` (z0), `a2x` (z2)}. anchor 특징 P는 §4·§5가 정한다. **f=`a2x`와 nbr=`a2`의 조합은 돌리지 않는다**(3-hop 이상이 섞여 해석이 어렵다).

### 1-3. 파라미터 격자(공통)

- tau ∈ {0.05, 0.1, 0.2} (soft) + `hard`
- nbr ∈ {`1hop`, `self`, `a2`}
- f ∈ {`x`, `a2x`} (단 f=`a2x`일 때 nbr은 `1hop`·`self`만)
→ arm 이름 `vote_<f>_<tau|hard>_<nbr>`, 총 4 × (3 + 2) = 20 arm. **사전 지정 주 arm: `vote_x_0.1_1hop`.**

### 1-4. 자체 점검 (`nd/nd_vote.py --selfcheck`, 결과 `results/nd/nd01/selfcheck.json`)

- `W` 세 종류의 행 합이 1인지(최대 오차), nnz, isolated_count.
- 무작위 anchor 5개로 Q·D를 만들어 D 행 합 1 확인, D의 argmax가 Q의 argmax와 일치하는 노드 비율(1hop·soft 0.1)을 기록 — 이 비율이 0.99 이상이면 투표가 자기 배정과 사실상 같다는 뜻이므로 ④에 적는다(정지 아님, 관문 B의 `own_soft` 대조군이 그것을 측정한다).
- `H.verify_against_exp049`를 한 번 호출해 z0·z1·z2가 exp049와 같은지 확인(assert).

## 2. 관문 A — meta-test 분류 `nd/nd001_meta_test_gate.py`

### 2-1. 입력
- `pickle.load` 로 `save/downstream_tasks/Amazon_clothing/5way_{1,5}shot.pkl` → `(valid_pool, test_pool)`. 각 task는 `(supports, y_support, queries, y_query)`(`exp027.episode_accuracy`가 읽는 형식 그대로). shot ∈ {1, 5}.
- z0, z2 (`H.structural_scores`), W 세 종류.

### 2-2. 에피소드당 계산 (shot, f, tau, nbr 주어짐)
1. support를 클래스별로 묶는다(`exp027.episode_accuracy`의 검증 로직을 그대로 복사해 형식을 assert).
2. **anchor P** = 클래스별 support 특징의 평균(1-shot이면 support 그 자체), 행 정규화. N=5.
3. **특징 프로토타입 분류** `feat_<f>`: `exp027.episode_accuracy(zf, task, 5, shot)`의 식 — 프로토타입 = support 평균(정규화하지 않음, exp027과 동일), 제곱 유클리드 최근접. query별 logit_feat = −dist (shape [40, 5]).
4. **투표 분류** `vote_<f>_<tau>_<nbr>`: Q = assign(zf, P), D 행을 support·query 노드에 대해 vote_rows. π_k = 클래스 k support의 D 행 평균. logit_d = −‖D_q − π_k‖². 예측 = argmax.
5. **결합** `comb_<f>_<tau>_<nbr>_l<λ>`: log p_feat = log_softmax(logit_feat), log p_d = log_softmax(logit_d); logit = λ·log p_feat + (1−λ)·log p_d, λ ∈ {0.25, 0.5, 0.75}. 예측 = argmax.
6. 에피소드 정확도 = query 40개 중 맞춘 비율. **에피소드별 정확도 배열을 저장**(`results/nd/nd01/gateA/<shot>shot/<arm>.npy`, valid·test 각각).

### 2-3. 실행 범위
- shot ∈ {1, 5} × pool ∈ {valid, test} × arm: `feat_x`, `feat_a2x`, 20개 `vote_*`, 그리고 `comb_*`는 f별 주 arm(`vote_x_0.1_1hop`, `vote_a2x_0.1_1hop`)과 **valid에서 고른 최고 vote arm**(f별)에 대해서만 λ 3개 — 즉 comb는 f당 최대 2 × 3 = 6개.
- 선택 규칙(레이블은 valid 에피소드의 평가에만 쓰임, 사전 고정): f별로 `vote_*` 중 valid 1-shot 정확도 최고인 arm을 `vote_<f>_best`라 부르고, `comb_<f>_best_l<λ>` 중 valid 1-shot 최고 λ를 `comb_<f>_sel`이라 부른다. **관문 판정은 `comb_<f>_sel`의 test 1-shot으로 한다.** 5-shot은 같은 arm·λ를 그대로 쓴다(5-shot에서 다시 고르지 않는다).
- 계산량: 에피소드당 Q·D는 arm 20개가 공유하지 않으므로 (f, tau, nbr)마다 S 한 번·sparse 곱 한 번. 550 에피소드 × 2 shot × 10 (f,tau,nbr 조합 중 f별 5) — GPU에서 수 분. 전경 실행.

### 2-4. 표 `reports/ND01_tables/`
- **TA1** `TA1_gateA.csv`: arm × {valid_1shot, test_1shot, test_1shot_ci95, valid_5shot, test_5shot, test_5shot_ci95, delta_1shot_vs_feat_same_f, paired_delta_1shot_mean, paired_delta_1shot_ci95, paired_win_frac_1shot}. paired 열은 같은 f의 `feat_<f>`와 에피소드별 차이(500개)의 평균·95% CI·comb가 더 높은 에피소드 비율. 5-shot도 같은 세 열.
- **TA2** `TA2_grid_1shot.csv`: 20개 `vote_*`의 valid·test 1-shot을 f × tau × nbr 격자로.
- **TA3** `TA3_selection.csv`: f별 `vote_<f>_best`가 무엇인지, `comb_<f>_sel`의 λ, valid 값.

커밋 `ND01: gate A meta-test classification`, 태그 `nd01-gateA`, push.

## 3. 관문 B — 후보 same-class AUROC `nd/nd002_vote_auroc.py`

### 3-1. anchor와 후보
- `rng = np.random.default_rng(1212)`. **에피소드 400개**, 각 에피소드는 전체 노드에서 비복원 균등 5개 anchor(B_Choi의 uniform anchor와 같은 방식). 총 2,000 anchor(에피소드 간 중복 허용, 에피소드 안 중복 없음). `results/nd/nd01/gateB/anchors.npy`에 [400, 5]로 저장.
- 각 anchor a의 후보 = X 코사인 순위 **201–1000**(B4, 자기 자신 제외) 800개 — `hl008_distant_query_train.b4_scores`와 같은 방식(`data.Diffusion_matrix` 행에서 `row[a] = -inf` 후 stable 내림차순 정렬). 가능하면 `b4_scores`를 그대로 import해 `ranked`와 `ax`·`a2x`·`enc_n` 점수를 받는다(`emb`에는 enc_b 임베딩을 넣어 `enc_n` 자리에 enc_b가 오게 하되 열 이름은 `enc_b`로 쓴다; ppr은 쓰지 않으므로 `ppr_columns` 호출이 부담되면 b4_scores를 복사·축소해 `nd/`에 두고 출처를 적는다).

### 3-2. 점수 (anchor a, 후보 c, 에피소드의 anchor 집합 P = 5개 anchor의 f-특징)

| 열 | 정의 | 무엇을 재는가 |
|---|---|---|
| `ax` | cos(z1_a, z1_c) | 기존 판정기(평균 후 비교) |
| `a2x` | cos(z2_a, z2_c) | 기존 판정기 |
| `enc_b` | cos(emb_a, emb_c) (체크포인트 있을 때) | 학습된 판정기 참고 |
| `own_soft_<f>_<tau>` | assign(zf, P, tau)[c, slot(a)] | anchor 경쟁만(이웃 없음) |
| `own_hard_<f>` | one_hot(argmax)[c, slot(a)] | 〃, 이진 |
| `vote_<f>_<tau|hard>_<nbr>` | D[c, slot(a)] (§1-3 격자 20 arm) | 경쟁 + 이웃 투표 |

f=`x`에 대해 `own_soft` tau 3개 + `own_hard` 1개, f=`a2x`도 같이 — own 열 8개. 모든 점수는 anchor 행마다 800개 후보에 대해 계산한다.

### 3-3. 채점 (레이블은 여기서만)
- `same[a, c] = (label[c] == label[a])` (`H.labels_np`).
- `H`가 아니라 `hl005_diag_distance_purity.auroc_rows`로 anchor 행별 AUROC, 양·음 모두 있는 행만 평균(`auroc_all`, `n_rows`). `auroc_pooled`도. 추가로 B4 안 Â²X 순위 401–800 후보만 쓴 `auroc_far50`(HL06 P2의 열 정의와 같게: 후보를 a2x 점수로 내림차순 정렬해 뒤쪽 400개).
- 쌍 비교: `vote_x_0.1_1hop` − `a2x`, `vote_x_0.1_1hop` − `own_soft_x_0.1`의 행별 AUROC 차이의 평균과 95% CI.

### 3-4. 표
- **TB1** `TB1_auroc.csv`: 점수 열 × {auroc_all, n_rows, auroc_pooled, auroc_far50, n_rows_far50}.
- **TB2** `TB2_paired.csv`: 위 두 쌍 비교(평균, CI, 양수 비율).
- **TB3** `TB3_grid.csv`: 20개 vote arm의 auroc_all을 f × tau × nbr 격자로.

커밋 `ND01: gate B candidate AUROC`, 태그 `nd01-gateB`, push.

## 4. 사전 등록 (Agent는 계산값만 채운다; 판정은 이희태)

| 번호 | 예측 | 계산값 |
|---|---|---|
| NDP001 | **관문 A 핵심**: `comb_x_sel` test 1-shot − `feat_x` test 1-shot ≥ **+1.0** | |
| NDP002 | `comb_x_sel`의 paired_win_frac_1shot ≥ 0.55 | |
| NDP003 | 주 arm `vote_x_0.1_1hop` 단독 test 1-shot ≥ `feat_x` − 3.0 (투표만으로도 분류가 되지만 특징보다 약하다) | |
| NDP004 | `vote_x_0.1_self` ≥ `vote_x_0.1_1hop` (test 1-shot; 자기 표가 도움) | |
| NDP005 | `vote_x_0.1_1hop` ≥ `vote_x_hard_1hop` (soft가 hard보다 낫다) | |
| NDP006 | `comb_x_sel`의 5-shot Δ(vs `feat_x`) ≤ 1-shot Δ (5-shot은 여지가 작다) | |
| NDP007 | **관문 B 핵심**: `vote_x_0.1_1hop` auroc_all ≥ `a2x` auroc_all + **0.02** (같은 anchor) | |
| NDP008 | `vote_x_0.1_1hop` auroc_all ≥ `own_soft_x_0.1` auroc_all + 0.03 (이웃 집계가 실제로 기여) | |
| NDP009 | `vote_x_0.1_1hop`의 (auroc_all − auroc_far50) ≤ `a2x`의 (auroc_all − auroc_far50) (먼 후보에서 덜 무너진다) | |
| NDP010 | 이 run의 `a2x` auroc_all이 HL06 P2(0.7564)와 ±0.03 안 (anchor 집합이 달라도 일관) | |

## 5. 판정 기준(참고, 판정은 이희태)
- A 통과(NDP001) → 다음 지시서는 VISION식 네트워크 입력에 d_v 주입(Q2) 설계.
- A 미달·B 통과(NDP007·008) → d_v는 분류 기준이 아니라 **노이즈 점수**로만 유지; 다음 지시서는 pseudo-episode query 가중/배제 oracle.
- 둘 다 미달 → ND 트랙 종결 문서.
- NDP008 미달이면 NDP007 통과 여부와 무관하게 "투표 vs 평균" 차이는 없고 anchor 경쟁 효과뿐이라는 뜻이다 — 보고서 ②에 own_soft 행을 vote 행 바로 옆에 둔다.

## 6. 보고서 `reports/ND01_vote_gates.md`
CLAUDE.md §5 형식. 포함: 환경·md5·nvidia-smi 기록, selfcheck 요약(isolated_count, argmax 일치 비율), TA1 전체·TA2·TA3, TB1 전체·TB2·TB3, 사전 등록 표(계산값만), ④(지시서에 없어 정한 것·실패·enc_b 유무), ⑤ 산출물 경로. 커밋 `ND01: report`, push. 끝나면 채팅에 TA1(feat 2행 + comb_<f>_sel 2행 + 주 arm 2행), TB1(ax, a2x, enc_b, own_soft_x_0.1, vote_x_0.1_1hop 행), TB2, 사전 등록 표를 그대로 붙인다. 해석·제안은 쓰지 않는다.
