# HL02 — 학습 없는 진단, anchor oracle 사다리, B 학습 전 점검 (auto)

작성 2026-10-25 · 이희태 · 브랜치 `hl/anchor` · 태그 `hl02-diag` / `hl02-oracle` / `hl02-embed`
데이터셋: Amazon_clothing만. τ = 0.3(HL01 τ*). query 규칙은 **B_Choi 그대로**(`choi_exps/exp049.build_structural_queries`를 import해서 쓴다 — 복사하지 않는다).
목적: anchor 샘플링 자유도에 상한이 있는지(oracle), 레이블 없는 신호가 무엇을 재는지(진단), B의 전제가 임베딩에서 성립하는지(점검)를 한 큐로 잰다. **이 지시서의 결과로 스토리 A·B의 진행 여부가 결정된다.**

레이블 사용 범위: §2 진단과 §3 oracle arm, §4 점검의 GT 통계에서만 읽는다. 레이블 없는 샘플러는 이 지시서에 없다.

---

## 0. 시작

1. `git status` clean, HEAD가 `hl01-p0` 태그 커밋인지 확인. 아니면 **정지**.
2. 고정 에피소드 seed **1215, 1216**을 1-shot·5-shot 각각 생성(`choi_exps/exp026.py`, HL01 §4의 러너가 하는 방식 그대로). 생성 시간을 기록.
3. 결과 루트 `results/hl/hl02/`, 표 디렉터리 `reports/HL02_tables/` 생성.

## 1. 공통 유틸 `hl/hl_common.py`

HL02–HL04에서 반복될 것을 한 곳에 둔다. 모두 `choi_exps`의 함수·규칙을 그대로 따른다.

- `load_everything(args)`: `load_data` → `set_device` → `add_diffusion`(type=feature) → 반환. exp049 `main`의 앞부분과 동일.
- `structural_scores(data, device)`: exp049 `build_structural_queries` 앞부분과 **동일한 식**으로 `z0, z1, z2`(X, ÂX, Â²X의 행 정규화)를 계산해 반환. 정규화 인접행렬 구성도 그 코드와 같아야 한다(self-loop 추가, 대칭화, D^-1/2 A D^-1/2). 동일성 검증: 임의 노드 20개에 대해 exp049 결과(`diagnostics[s]["mean_ax_cosine"]`)와 우리 계산이 일치하는지 assert.
- `mine_queries_for(nodes, data, args, device)`: 노드 집합 `nodes`에 대해 B_Choi query 10개씩을 얻는다. 구현: `nodes`를 5개씩 묶어 더미 task `(supports, y_spt, dummy_queries(len=50), y_query)`를 만들고 `build_structural_queries(data, tasks, args, device)`를 호출해 `diagnostics[s]["selected_ids"]`, `raw_retained_count`, `mean_original_x_rank`, `fallback`을 꺼낸다. 마지막 묶음이 5개 미만이면 이미 포함된 노드로 채운다(`np.unique`가 중복을 지운다).
- `x_topL(data, nodes, L)`: `data.Diffusion_matrix`(X 코사인, 대각 0)에서 각 노드의 상위 L개 id와 코사인.
- `accepted_mask(z1, z2, src, dst, tau)`: 쌍 (src, dst)에 대해 `cos(z1)≥τ and cos(z2)≥τ`.
- `proto_softmax(Z, id_spt, id_query, y_spt, y_query)`: `meta.ProtoNet.forward`와 같은 식(제곱 유클리드, `log_softmax(-dists)`)으로 query별 확률 행렬과 정답 index를 반환. 학습 없음, `torch.no_grad()`.

## 2. 진단 `hl/hl002_diag_nodes.py` (학습 없음, GT 사용, 태그 `hl02-diag`)

seed 1212로 고정. 출력 `results/hl/hl02/diag/` + 표는 `reports/HL02_tables/`.

**2-1. 노드 표** `node_table_tau0.3.csv` — 전체 24,919 노드 각각에:
`node, label, degree, raw_retained, mean_x_rank, fallback, pass50, H50, pi_bchoi, pi_naq`
- `raw_retained, mean_x_rank, fallback`: `mine_queries_for(all nodes)`에서.
- `pass50`: X 상위 50개 중 필터 통과(accepted) 수. `H50 = 50 − pass50`(= 필터 탈락 후보 수, B의 재료).
- `pi_bchoi`: 그 노드의 B_Choi query 10개 중 노드와 GT가 같은 비율. `pi_naq`: X 상위 10개 중 같은 비율.
- 소요 시간을 기록(전체 노드 mining).

**2-2. 요약** `diag_summary.json` + `D1_node_stats.csv`:
- 분위수(10, 25, 50, 75, 90%)와 평균: `raw_retained`, `H50`, `pass50`, `pi_bchoi`, `pi_naq`, `degree`.
- Spearman ρ: (`raw_retained`, `degree`), (`raw_retained`, `pi_bchoi`), (`pass50`, `pi_bchoi`), (`degree`, `pi_bchoi`), (`H50`, `pi_bchoi`), (`raw_retained`, `H50`). degree를 통제한 편상관 (`raw_retained`, `pi_bchoi` | degree)도.
- `pi_bchoi` 5분위 경계값(§3 quintile 정책이 쓴다)과 각 분위의 노드 수·평균 degree·평균 `raw_retained`. `pi_bchoi_quintiles.csv`로 저장(node → quintile 1..5, 1이 최저).

**2-3. 쌍 통계** `D2_pair_stats.csv`:
- X 상위 50 쌍 중 **accepted**의 same-GT 비율, **rejected**의 same-GT 비율. 균등 무작위 쌍 100,000개의 same-GT 비율. (각각 쌍 수도.)
- rejected를 X 코사인 상위 10 / 11–30 / 31–50 구간으로 나눈 same-GT 비율.

**2-4. 에피소드 충돌 통계** (uniform anchor, B_Choi query, 2000 에피소드, seed 1212) `D3_episode_stats.csv`:
- anchor 5개의 GT가 모두 다른 비율, 정확히 한 쌍 같은 비율, 둘 이상 같은 비율.
- 서로 다른 anchor의 query 집합이 하나라도 겹치는 에피소드 비율; 한 anchor가 다른 anchor의 query에 들어 있는 에피소드 비율.
- 에피소드당 anchor 쌍(10쌍)의 평균 X 코사인, 평균 Â²X 코사인.

커밋 `HL02: node/pair/episode diagnostics`, 태그 `hl02-diag`.

## 3. anchor oracle 사다리 `hl/hl003_oracle_train.py` (학습, GT를 anchor 선택에 사용, 태그 `hl02-oracle`)

`choi_exps/exp049.py`의 `main`을 **복사**해 `hl/hl003_oracle_train.py`로 만들고(docstring 첫 줄에 출처), 바꾸는 곳은 **anchor 생성 한 군데**뿐이다: `original = [proto_task_generator(args, data) for _ in range(args.epoch)]` 대신 아래 정책으로 2000개의 `(id_spt, y_spt, dummy_query, y_query)`를 만들고, 그다음 `build_structural_queries(data, original, args, device)`로 query를 붙인다(method는 항상 ours, τ=0.3). 학습·평가·report.json 형식은 그대로. `--policy`, `--k`, `--L`(기본 50), `--node_table`(2-1 csv) 인자를 추가. 출력 `results/hl/hl02/oracle/<policy>/seed_<seed>/`.

anchor 정책(모두 비복원, N=5, `y_spt`는 exp049와 같은 방식의 무작위 pseudo-label):

| policy | 정의 |
|---|---|
| `uniform` | 전체 노드에서 5개 균등 = B_Choi (O0, 대조군. P0와 같은 설정이지만 **이 스크립트로 다시 돌린다**) |
| `quintile:q` (q=1..5) | `pi_bchoi` q분위 노드 풀에서 5개 균등 (O-A 곡선. q=5가 최고 순도) |
| `gt_distinct` | 균등으로 5개 뽑되 GT가 모두 다를 때까지 5개 전체를 다시 뽑음 (O-S, Choi CoraFull의 Clothing 판) |
| `hard:k` (k=2, 4) | a₁ 균등. a₁의 X 상위 L개 중 **필터 탈락(rejected)**이고 GT ≠ GT(a₁)인 후보를 X 코사인 순으로 보며, 이미 뽑힌 anchor들과 GT가 다른 것을 k개 선택. 부족분과 나머지(4−k)는 균등에서 뽑되 모든 기존 anchor와 GT가 다를 때까지 재추출 (O-B, oracle-safe hardness) |
| `hard_blind:k` (k=4) | `hard:k`와 같되 **GT를 전혀 보지 않음** — 탈락 후보 X 상위 k개 + 나머지 균등 (O1, 거짓 음성 비용 측정) |
| `ab` | a₁을 `pi_bchoi` 5분위 풀에서 균등, 나머지는 `hard:4` 규칙 (O-AB) |

재추출 상한 20회; 넘으면 균등으로 채우고 횟수를 기록. 정책별로 에피소드 진단을 report.json의 `anchor_diagnostics`에 넣는다: `k_eff` 평균, 에피소드 내 same-GT anchor 쌍 비율, anchor 쌍 평균 X 코사인·Â²X 코사인, 균등 fallback 비율, 뽑힌 anchor의 `pi_bchoi` 평균, 유일 anchor 수.

**실행**: seed **1212–1216**(5개) × 11 정책 = 55 run. 러너는 `hl/hl000_run.py`를 복사·수정한 `hl/hl000b_run_oracle.py`(policy·seed 격자, 재시작 가능, `summary.csv`를 `results/hl/hl02/oracle/`에). 예상 55 × ~55s ≈ 50분. 동시 2개까지 허용.

**표 T4** `T4_oracle.csv`: policy × {1-shot test 평균±std, 5-shot test 평균±std, 1-shot valid 평균, Δ_1shot vs uniform, Δ_5shot vs uniform, same-GT 쌍 비율, 평균 `pi_bchoi`, k_eff}. **표 T5** `T5_quintile_curve.csv`: q=1..5의 1-shot test 평균±std와 uniform.

일관성 점검: `uniform` 5-seed 평균이 P0 B_Choi 3-seed 평균(82.81)과 ±0.6 안인지 기록(같은 설정, 다른 샘플러 코드).

커밋 `HL02: anchor oracle ladder`, 태그 `hl02-oracle`.

## 4. B 학습 전 점검 `hl/hl004_embed_check.py` (학습 없음, 체크포인트 사용, 태그 `hl02-embed`)

체크포인트: `results/hl/p0/Amazon_clothing/ours/tau_0.3/seed_1212/best_1shot.pt`(주)와 `.../naq/baseline/seed_1212/best_1shot.pt`(보조). 각각 인코더를 eval 모드로 전체 임베딩 Z 계산.

**4-1. 쌍 거리** `E1_pair_dist.csv`: anchor 2,000개 균등 추출(seed 1212). 각 anchor에 대해 세 집합과의 임베딩 코사인 거리(1 − cos): (a) X 상위 50 중 rejected, (b) X 상위 50 중 accepted, (c) 균등 무작위 50개. 집합별 평균·중앙값·25/75%. 체크포인트 둘 다.

**4-2. negative 난이도** `E2_neg_softmax.csv`: 500 에피소드 × 정책 {`uniform`, `hard_blind:4`, `hard:4`}(§3의 함수 재사용, seed 1212), query는 B_Choi. `proto_softmax`로 query별: p(자기 anchor), Σp(negative), max p(negative), margin = d(가장 가까운 negative) − d(자기 anchor). 정책별 평균·중앙값, 그리고 "Σp(negative) < 0.05인 query 비율". 체크포인트 둘 다.

커밋 `HL02: embedding pre-checks`, 태그 `hl02-embed`.

## 5. 사전 등록 (계산값만 채운다; 임계값은 모두 편의 목표치)

| 번호 | 예측 | 계산값 |
|---|---|---|
| HLP007 | `pi_bchoi` 전체 평균은 0.42 ± 0.03, `pi_naq`는 0.60 ± 0.03 (HL01 T3와 일치) | |
| HLP008 | ρ(`raw_retained`, `degree`) ≥ 0.3 — 통과 수는 degree에 묶여 있다 | |
| HLP009 | ρ(`raw_retained`, `pi_bchoi`) < 0 — 생존자가 많을수록 middle이 멀어져 순도가 낮다 | |
| HLP010 | X 상위 50 중 rejected의 same-GT 비율 < accepted의 절반; 그러나 균등 쌍 비율보다는 높다 | |
| HLP011 | 2-4: anchor GT 모두 다름 ≥ 0.75, 한 쌍 겹침 ≤ 0.22 (Choi CoraFull과 같은 양상) | |
| HLP012 | 2-4: query 겹침 에피소드 비율 ≥ 0.1 (코드가 막지 않으므로 드물지 않다) | |
| HLP013 | T5: 5분위 곡선의 max − min ≥ 1.0%p (평평하지 않다) | |
| HLP014 | T5: q=5(최고 순도) − uniform ≤ +2.0 이고, 곡선은 단조 증가가 **아니다** (HL01에서 순도↓가 1-shot↑였으므로) | |
| HLP015 | T4: `gt_distinct` − `uniform` 은 ±0.4 안 (Choi CoraFull 재현) | |
| HLP016 | T4: `hard:4` − `uniform` ≥ +1.0 (B go 기준) | |
| HLP017 | T4: `hard:4` − `hard_blind:4` ≥ +0.5 (hard 샘플링 아래서는 거짓 음성 비용이 보인다) | |
| HLP018 | T4: `hard:2`는 `uniform`과 `hard:4` 사이 | |
| HLP019 | T4: `ab` ≥ max(`quintile:5`, `hard:4`) − 0.3 | |
| HLP020 | T4: 모든 oracle 정책에서 Δ_5shot < Δ_1shot | |
| HLP021 | 4-1 (ours ckpt): rejected 쌍 거리 중앙값 < 균등 쌍 거리 중앙값 (B 전제) | |
| HLP022 | 4-2 (ours ckpt, uniform): Σp(negative) 중앙값 < 0.10 | |
| HLP023 | 4-2: `hard_blind:4`의 Σp(negative) 평균 > `uniform`의 1.5배 (k가 hardness를 올린다) | |

**판정 규칙(이희태가 적용, Agent는 기록만)**: max(`quintile` 최고, `hard:4`, `ab`) − `uniform` < 1.0%p(1-shot, 5-seed 평균) → anchor 트랙(A·B 모두) 중단. HLP021 또는 HLP023 실패 → B의 hardness 경로 중단.

## 6. 보고서 `reports/HL02_diag_oracle.md`

CLAUDE.md §5 형식. 포함: 전체 노드 mining 소요 시간, D1–D3, T4–T5, E1–E2, 사전 등록 표, uniform vs P0 일관성 점검값, 실패·경고, 산출물 경로. 끝나면 채팅에 T4·T5·E1·E2·사전 등록 표를 그대로 붙인다.
