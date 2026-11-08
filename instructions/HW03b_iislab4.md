# HW03b [iislab4] — validation 500 + 대조군(`self`, `global_random`) + λ sweep + 차원별 Dirichlet 에너지 (auto, 220 run)

2026-11-08 · 작성: 이희태 (Claude 정리) · 서버 **iislab4** · 브랜치 `hl/hsan-weighting` · 선행 `fddf4b0`(HW04a-c) 이후 · 태그 `hw03b` · 사전 등록 HWP041–HWP048 · 선행 문서 `CLAUDE.md` 개정 6, `reports/HW04a_iislab4.md`

- auto 모드. 멈추는 조건: §0 허용 외 미커밋 변경, §1.1 G-valid 실패, §1.6 회귀 확인 실패, G6, 해결 못 한 오류.
- 결과 루트: **`HW_RESULTS_ROOT=results/hw_iislab4`**. `GPUS=0,1,2,3,4,5,6,7`. seed 1212–1221.
- 이번 지시서의 모든 학습 arm은 **`valid_tasks: 500`** 이다. 기존(validation 50) run과 짝짓지 않는다(CLAUDE.md §4.3).
- 시간 예산 2시간 (구현 40 / 확인 15 / 실행 ~50 / 측정 5 / 집계·보고 20).

## 0. 현황과 손실 형태 확인 (수정 없음, 보고서에 적고 계속)

1. `git fetch && git status -sb | head -1`, `git log --oneline -3`, `nvidia-smi --query-gpu=index,memory.used --format=csv`. **작업 시작 HEAD 해시를 기록한다.** 허용 미커밋 변경은 미추적 `instructions/HW03b_iislab4.md`와 수정된 `CLAUDE.md`(개정 6)뿐. 그 밖에 있으면 **멈추고 보고**. `git pull --rebase`. CLAUDE.md 개정 6을 다시 읽는다.
2. **손실 형태 확인 (기록만, 멈추지 않음)**:
   - `hw/losses.py`에서 L_original과 L_neighbor의 식을 그대로 옮겨 적는다(이웃 임베딩이 prototype을 대체하는지, 별도 항인지, 음성이 무엇인지).
   - optimizer 종류, lr, weight_decay, 기타 정규화 항을 적는다.
   - **동일성 수치 확인**: seed 1212, Cora_full, 학습 전 모델로 에피소드 1개에서 이웃 = support(n = s)로 넣었을 때 `|L_neighbor − L_original|`을 계산해 적는다. 스크래치 스크립트로 하고 커밋하지 않는다.
3. 기존 validation 캐시 구조를 확인해 적는다: `save/downstream_tasks/<ds>/5way_5shot.pkl`의 키, validation task 수와 형식(클래스 목록, support/query 수), 그 파일을 만든 코드 위치. 두 파일의 md5를 기록한다(작업 끝에 다시 확인).

## 1. HW03b-a: 구현

### 1.1 validation 500 캐시 (`save/valid500/<ds>.pkl`, git 밖)
- 새 파일 생성 코드는 `hw/valid500.py`(새 파일)에 둔다. 실행: `python -m hw.valid500 --dataset <ds>`.
- 구성: **앞 50개 = 기존 validation task 50개를 그대로 복사**, 뒤 450개 = 기존 validation 클래스 집합에서 기존과 같은 형식(5-way, 같은 support·query 수)으로 새로 뽑는다. 기존 생성 코드의 함수를 재사용하되, **전용 `numpy.random.Generator`/`torch.Generator`(seed 20261108)** 를 써서 전역 RNG를 소비하지 않는다.
- **G-valid (실패 시 멈추고 보고)**:
  - (a) 앞 50개가 기존 validation task와 원소 단위로 같다.
  - (b) 뒤 450개의 모든 클래스가 기존 validation 클래스 집합 안에 있다.
  - (c) 기존 `5way_5shot.pkl`의 md5가 작업 전과 같다.
- 기록: 새 파일 md5, 450개 중 앞 50개와 (클래스·노드 집합이) 완전히 같은 task 수.

### 1.2 `valid_tasks` (`hw/config.py`, `hw/run.py`, `hw/evaluate.py`)
- 설정 키 `valid_tasks ∈ {50, 500}`, 기본 50. 50이면 기존 경로를 한 줄도 바꾸지 않는다.
- 500이면 매 평가 시점(`eval_every`)마다 500개 전체 정확도 `val_acc_500`과 앞 50개 정확도 `val_acc_50`을 함께 기록한다(같은 임베딩으로 한 번에 계산해도 된다).
- best epoch을 두 기준으로 각각 고른다(`best_epoch_500`, `best_epoch_50`). 두 checkpoint의 state를 CPU에 복사해 두고, 학습 끝에 test 500을 **각 checkpoint에서 한 번씩** 잰다. 두 epoch이 같으면 한 번만 재고 값을 복사한다.
- `metrics.json`: `test_acc` = 500 기준(주 결과), `test_acc_at_best50` = 50 기준(측정용), `best_epoch_500`, `best_epoch_50`, `same_epoch`(bool), epoch별 `val_acc_500`·`val_acc_50`.
- 붕괴 표시(CLAUDE.md §4.3)는 500 기준 best-epoch validation으로 판정.
- validation 평가는 캐시된 task만 쓰므로 RNG를 소비하지 않아야 한다. 그렇지 않으면 그 사실을 적는다.

### 1.3 `pick: self`, `pick: global_random` (`hw/neighbors.py`)
- 기존 `max`/`contrast`/`random`/`sup_cos` 경로는 한 줄도 바꾸지 않는다.
- `self`: 모든 (query, support) 쌍에서 양성 = support 자신.
- `global_random`: (query, support) 쌍마다 전체 노드에서 균등 무작위 1개(support와 그 query 제외). **전용 `torch.Generator`(device 맞춤, `manual_seed(seed + 104729)`)**, 전역 RNG 소비 없음.
- 기존 진단(`nbr_eq_support`, `nbr_eq_query`, `nbr_in_own_queries` 등)은 그대로 기록된다. `self`에서는 `nbr_eq_support` = 1이 정상이다.

### 1.4 `hw/summarize.py`
- `--metric <key>` (기본 `test_acc`). 기본이 아니면 출력 파일에 `_<key>` 접미사(`summary_test_acc_at_best50.md`, `paired_<ref>_test_acc_at_best50.md`).

### 1.5 arm 추가 (`configs/hw.json`, 모두 `valid_tasks: 500`)

| arm | base | pick | λ (`lambda_neighbor`) | 단계 |
|---|---|---|---|---|
| `v500_naq_base` | `naq_base` | — | — | 1 |
| `v500_raw_l01` | `nbr_raw` | max | 0.1 | 1 |
| `v500_self_l01` | `nbr_raw` | self | 0.1 | 1 |
| `v500_grand_l01` | `nbr_raw` | global_random | 0.1 | 1 |
| `v500_raw_l03` / `_l07` / `_l10` | `nbr_raw` | max | 0.3 / 0.7 / 1.0 | 2 |
| `v500_grand_l03` / `_l07` / `_l10` | `nbr_raw` | global_random | 0.3 / 0.7 / 1.0 | 2 |
| `v500_self_l10` | `nbr_raw` | self | 1.0 | 2 |

(11 arm. `nbr_raw` 기반 arm은 selector raw, 손실·학습 길이 등 나머지는 `nbr_raw`와 같다.)

### 1.6 회귀 확인과 smoke
- 회귀(CLAUDE.md §6-7): `naq_base`, `nbr_raw`(둘 다 `valid_tasks` 기본 50) seed 1212 두 ds → `results/hw_iislab4/_regress/HW03b/`, 기존 값과 |Δ| ≤ 0.30. 어긋나면 **멈추고 보고**.
- 불변: `git diff <작업 시작 HEAD> -- hw/neighbors.py hw/evaluate.py hw/run.py`에서 기존 경로의 삭제 줄 수와 발췌를 적는다.
- smoke(epochs 4, `/tmp/hw_smoke`, 커밋 안 함): `v500_self_l01`, `v500_grand_l10` Cora_full 각 1회. `metrics.json`에 `test_acc`, `test_acc_at_best50`, `same_epoch` 기록 확인.

커밋 `HW03b-a [iislab4]: valid500 cache + valid_tasks, pick self/global_random, summarize --metric, arms` (CLAUDE.md 개정 6, 이 지시서, 사전 등록 포함. `save/`는 git 밖) → pull --rebase → push.

## 2. HW03b-b: 실행 (220 run, 한 번에)

- run 목록 `results/hw_iislab4/HW03b_runs.txt`: 11 arm × 2 ds × seed 1212–1221 = 220. 순서: seed 바깥, arm 안쪽. 1·2단계 구분은 집계용이며 한 큐로 돌린다.
- 집계 (모두 `HW_RESULTS_ROOT=results/hw_iislab4`):
  - `python -m hw.summarize --ref v500_naq_base --ref v500_raw_l01 --ref v500_grand_l01 --ref v500_grand_l03 --ref v500_grand_l07 --ref v500_grand_l10`
  - 같은 명령에 `--metric test_acc_at_best50`
- 커밋 `HW03b-b [iislab4]: 220 runs (valid500, controls, lambda sweep)` → pull --rebase → push.

## 3. HW03b-c: 차원별 Dirichlet 에너지 측정 (학습 없음)

- `hw/measure_positive.py`에 `--energy` 옵션 추가(또는 `hw/structure.py`에 함수 + 진입점). 실행: 두 ds.
- 입력: 학습 경로가 읽는 그대로의 특징 행렬 X(정규화 여부를 적는다: 이진인지, 행 정규화가 되어 있는지, 값 범위). 그래프 A = HW04m의 `normalize_graph`.
- 차원 m마다:
  - `E_raw` = ½ Σ_(i,j) A_ij (x_im − x_jm)² (SPGCL 식 13; 무향 edge 1회씩 합)
  - `E_rq` = E_raw / ‖x_:,m‖² (Rayleigh 몫; ‖x_:,m‖ = 0인 차원은 제외하고 수를 적는다)
  - `nnz` = x_:,m ≠ 0인 노드 수
- 출력 `results/measure/HW03b_energy_<ds>.{json,md}`:
  - 세 값의 요약 통계(최소·사분위·최대), 0인 차원 수
  - Spearman 순위 상관: (E_raw, nnz), (E_raw, E_rq), (E_rq, nnz)
  - 상위 K 겹침 비율(K = 16, 64, 512): E_raw 상위 K ∩ nnz 상위 K, E_raw 상위 K ∩ E_rq 상위 K
- selftest: 작은 그래프에서 E_raw = f^T L f (L = D − A)가 1e-9 안에서 같은지.
- 커밋 `HW03b-c [iislab4]: per-dimension Dirichlet energy` → pull --rebase → push.

## 4. 게이트

- **G-valid** (§1.1), **회귀 확인** (§1.6): 실패 시 멈춤.
- **G6**: run의 95% 이상 `completed`(1회 재시도 후). 못 미치면 멈추고 보고.
- **G7 (기록만)**: arm별 붕괴 수.
- 작업 끝에 기존 `5way_5shot.pkl` md5를 다시 확인해 적는다.
- **태그**: G-valid + 회귀 + G6 통과 → `git tag hw03b && git push origin hw03b`.

## 5. 사전 등록 (HW03b-a 커밋에 포함, 판정은 사람)

Δ = 같은 seed 짝지은 `test_acc`(500 기준) 차이의 10 seed 평균. "두 ds" = 두 데이터셋 모두.

| ID | 예측 | 기준 | 근거 | 확신도 |
|---|---|---|---|---|
| HWP041 | `self`는 `naq_base`와 같다 | `v500_self_l01 − v500_naq_base`, `v500_self_l10 − v500_naq_base` 모두 \|Δ\| ≤ 0.15, 두 ds | n = s이면 L_neighbor = L_original, Adam은 손실 상수배에 거의 불변 | 0.75 |
| HWP042 | λ 0.1에서 무작위 양성은 `nbr_raw`와 같다 | \|`v500_grand_l01 − v500_raw_l01`\| ≤ 0.15, 두 ds | HW03a·HW04a: 선택 방식 전체가 ±0.17 안 | 0.55 |
| HWP043 | λ 0.1에서 무작위 양성도 `naq_base`를 넘는다 | `v500_grand_l01 − v500_naq_base` > 0, 양수 ≥ 7/10, 두 ds | 같음 | 0.45 |
| HWP044 | 내용은 큰 λ에서 의미가 생긴다 | (`raw − grand` at λ 1.0) − (`raw − grand` at λ 0.1) ≥ +0.3, 두 ds | 무작위 양성(순도 ≈ 0.02)을 강하게 당기면 해롭다 | 0.55 |
| HWP045 | λ 1.0이 raw에서도 Cora를 올린다 | `v500_raw_l10 − v500_raw_l01` ≥ +0.4, 양수 ≥ 7/10, Cora_full | HW03a `nbr_bgrl`(λ 1.0) Cora +0.76, λ 맞춘 BGRL = raw | 0.55 |
| HWP046 | λ 1.0의 무작위 양성은 `naq_base`보다 낮다 | `v500_grand_l10 − v500_naq_base` < 0, 두 ds | — | 0.50 |
| HWP047 | validation 500이 짝지은 Δ의 흩어짐을 줄인다 | `paired_v500_naq_base`의 arm별 sd 평균: `test_acc` < `test_acc_at_best50`, 두 ds | 선택 잡음 감소 | 0.65 |
| HWP048 | 식 13 에너지는 단어 빈도와 거의 같은 순위다 | Spearman(E_raw, nnz) ≥ 0.8, 두 ds | 정규화 없는 이진 특징 | 0.70 |

## 6. 범위 밖

- `sup_cos`·`contrast`·BGRL arm, λ > 1.0, 학습 길이 변경, `valid_tasks: 50` 새 arm(회귀 확인 제외), 기존 `5way_5shot.pkl` 수정·재생성, EAP식 인코더 변경, seed 추가, 다른 데이터셋, `results/hw/` 수정.

## 7. 보고 (`reports/HW03b_iislab4.md`, CLAUDE.md §7)

핵심 표 (값의 종류를 표마다 명시, 주 결과 = `test_acc`(500 기준)):
1. 11 arm × ds 10 seed test 정확도 평균 ± 표준편차, 붕괴 수
2. **λ 표**: 행 λ ∈ {0.1, 0.3, 0.7, 1.0}, 열 {`raw − naq_base`, `grand − naq_base`, `raw − grand`} × 2 ds (짝지은 Δ ± sd, 양수 k/10). λ 0.1·1.0 행에 `self − naq_base` 추가
3. **선택 잡음 표**: arm × ds의 `same_epoch` 비율, 두 기준 test 차이의 평균 절댓값; `paired_v500_naq_base`의 arm별 sd를 두 지표로 나란히
4. 50 기준 값과 기존 run 비교: `v500_naq_base`·`v500_raw_l01`의 `test_acc_at_best50` vs 기존 `naq_base`·`nbr_raw`(같은 seed) 차이의 평균 절댓값과 최대(기록만)
5. 진단 표: arm × ds의 `nbr_eq_support`, `nbr_eq_query`, `query_eq_support`, `nbr_in_own_queries`, `pos_nbr_cos_raw`, `pos_nbr_cos_a2x`
6. best epoch 분포 (arm × ds, 500 기준과 50 기준 각각)
7. 에너지 표 (§3 출력 요약)
8. §0 손실 형태·optimizer·동일성 수치, G-valid, 회귀, 불변 diff 발췌, md5(전·후)
9. HWP041–048 대조
10. 소요 시간, HEAD (작업 시작 HEAD 포함)
