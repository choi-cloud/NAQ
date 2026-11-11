# HW08 [iislab4] — raw·l2 평가 병기 (`eval_l2`) + best checkpoint 저장 (`save_best`) + 핵심 5 arm 재실행 + 학습 없는 기준선 재측정 (auto, 4000 episode, 100 run)

2026-11-11 · 작성: 이희태 (Claude 정리) · 서버 **iislab4** · 브랜치 `hl/hsan-weighting` · 선행 태그 `hw07` 이후 · 태그 `hw08` · 사전 등록 HWP079–HWP087 · 선행 문서 `CLAUDE.md` 개정 10, `reports/HW06_iislab4.md`, `reports/HW07_iislab4.md`, `reports/HW03a_seio.md` §2.5

- auto 모드. 멈추는 조건: §0 허용 외 미커밋 변경, §1.6 G-cfg·G-l2·G-tf·회귀 확인 실패, G6, 해결 못 한 오류.
- 결과 루트 **`HW_RESULTS_ROOT=results/hw_iislab4`**, `GPUS=0,1,2,3,4,5,6,7`, seed 1212–1221. 모든 학습 arm은 `valid_tasks: 500`, `episodes: 4000`, `geom_diag: true`, `eval_l2: true`, `save_best: true`.
- 학습 경로(손실·샘플링·옵티마이저·RNG 소비)는 바꾸지 않는다. 바꾸는 것은 평가 시점의 기록과 저장뿐이다.
- 시간 예산 2시간 (구현 40 / 확인 15 / 실행 ~40 / 집계·보고 25).

## 0. 현황 (수정 없음, 보고서에 적고 계속)

`git fetch && git status -sb | head -1`, `git log --oneline -3`, `git tag --list 'hw0*'`, `nvidia-smi --query-gpu=index,memory.used --format=csv`, `df -h . | tail -1`. **작업 시작 HEAD 기록.** 허용 미커밋 변경은 미추적 `instructions/HW08_iislab4.md`와 수정된 `CLAUDE.md`(개정 10)뿐. 그 밖에 있으면 **멈추고 보고**. `git pull --rebase`. CLAUDE.md 개정 10을 다시 읽는다.

기록:
- `hw/run.py`에서 평가 임베딩을 만드는 줄과 `evaluate` 호출 줄, best 갱신 규칙(동률 처리 포함)의 줄 번호.
- `hw/trainfree.py`가 L2 행 정규화를 하는 줄과 식.
- 인코더 state_dict의 파라미터 수와 한 번 저장했을 때의 바이트 수(두 ds).
- 디스크 여유 용량. **20 GB 미만이면 모든 arm의 `save_best`를 false로 두고 보고서에 적는다**(멈추지 않는다).

## 1. HW08-a: 구현

### 1.1 `evaluate_l2` (`hw/evaluate.py`에 새 함수 추가)
- 기존 `evaluate`는 한 줄도 바꾸지 않는다.
- `evaluate_l2(embeddings, pool, ...)` = `evaluate(F.normalize(embeddings, p=2, dim=1, eps=1e-12), pool, ...)`. 인자와 반환 형식은 `evaluate`와 같다.
- 정규화는 `hw/trainfree.py`의 L2 행 정규화와 같은 식이어야 한다(§1.6 G-l2로 확인). 다르면 `trainfree.py`의 식을 따르고 차이를 보고서에 적는다.

### 1.2 `eval_l2` (`hw/run.py`, 기본 false)
false이면 기존 동작과 기록이 그대로다. true이면 기존 기록에 다음을 **더한다**(기존 키의 값과 뜻은 그대로).

- 매 평가 시점: 같은 `clean` 임베딩으로 `validation_l2 = evaluate_l2(clean, valid_pool)`, 앞 50개도 같은 방식. `epochs[i]`에 `val_acc_500_l2`, `val_acc_50_l2` 기록.
- best 추적을 하나 더 둔다: `validation_l2`의 500 기준 정확도로 고른 `best_state_l2`(갱신 규칙은 기존 500 기준 best와 같은 규칙).
- 학습 종료 뒤 test 측정 (선택 기준 × 평가 방식):

| 키 (최상위) | checkpoint 선택 기준 | test 평가 |
|---|---|---|
| `test_acc` (기존) | raw validation 500 | raw |
| `test_acc_l2_at_bestraw` | raw validation 500 | l2 |
| `test_acc_l2` | l2 validation 500 | l2 |
| `test_acc_raw_at_bestl2` | l2 validation 500 | raw |

- 그 밖의 최상위 키: `best_l2`(`epoch`, `validation`, `test` — `best`와 같은 형식, l2 값), `best_epoch_500_l2`, `same_epoch_l2`(`best_epoch_500 == best_epoch_500_l2`).
- 50 기준 기록(`best_50`, `test_acc_at_best50`)은 기존 그대로 두고 l2 판을 만들지 않는다.
- `geom_diag: true`이면 l2 기준 best checkpoint에서도 같은 진단을 계산해 `geom_l2`에 기록한다(기존 `geom`은 raw 기준 best 그대로).
- 노름 진단(레이블 불필요, 두 best checkpoint 각각, `norm_diag`·`norm_diag_l2`):
  - `zero_rows_all`: 전체 노드 중 ‖z‖ < 1e-12인 행 수
  - `zero_rows_test`: test 500 task에 등장하는 노드 중 같은 조건의 행 수
  - `test_norm_mean`, `test_norm_std`: test task에 등장하는 노드의 ‖z‖ 평균·표준편차
  - `proto_norm_cv`: test task마다 5개 프로토타입(support 평균, 정규화 전) 노름의 표준편차 / 평균, 그 500개 평균
- 새 계산은 모두 `torch.no_grad()`, 전역 RNG를 소비하지 않는다.

### 1.3 `save_best` (`hw/run.py`, 기본 false)
- true이면 학습 종료 뒤 `ckpt/hw_iislab4/<arm>/<ds>/seed<S>/best_raw.pt`(raw 기준 best의 `state_dict`, `epoch`)를 저장한다. `eval_l2`도 true이고 두 best의 epoch이 다르면 `best_l2.pt`도 저장한다.
- `metrics.json`에 `ckpt: {best_raw, best_l2(없으면 null), bytes}` 기록.
- `ckpt/`는 git 밖이다. `.gitignore`에 `ckpt/`를 추가한다.

### 1.4 arm 추가 (`configs/hw.json`; 모두 base `naq_base`, `valid_tasks 500`, `episodes 4000`, `geom_diag true`, `eval_l2 true`, `save_best true`)

| arm | 대응하는 기존 arm | `extra_neg` |
|---|---|---|
| `l2_naq` | `e4k_naq` | 없음 |
| `l2_rand_m20` | `e4k_rand_m20` | `{random, 20}` |
| `l2_rand_m50` | `e4k_rand_m50` | `{random, 50}` |
| `l2_rand_m200` | `e4k_rand_m200` | `{random, 200}` |
| `l2_clus_m20` | `e4k_clus_m20` | `{cluster, 20, k 100}` |

- `base`는 한 단계만 지원하므로 `e4k_*`를 base로 쓰지 않고, 대응 arm의 키를 그대로 옮겨 적는다.
- 5 arm × 2 ds × 10 seed = **100 run**.

### 1.5 `hw/summarize.py`
- `--metric`으로 `test_acc_l2`, `test_acc_l2_at_bestraw`, `test_acc_raw_at_bestl2`를 집계할 수 있어야 한다. 이미 최상위 키를 일반적으로 읽으면 변경 없음. 해당 키가 없는 run(기존 arm)은 그 지표 집계에서 뺀다.

### 1.6 확인 (실패 시 **멈추고 보고**; 결과는 `results/hw_iislab4/HW08_checks.json`)
- **G-cfg**: 다섯 `l2_*` arm의 병합된 설정에서 `eval_l2`, `save_best`를 뺀 것이 대응 `e4k_*` arm의 병합된 설정과 같다(dict 비교, arm 이름 키 제외).
- **G-l2**: 두 ds에서 정규화하지 않은 Â²X(`hw.structure.a2x_trainfree`의 정규화 전 값 또는 `hw/trainfree.py`가 쓰는 미정규화 Â²X)에 `evaluate_l2`를 적용한 test 정확도가 `hw/trainfree.py`의 "Â²X, L2" 값과 |Δ| ≤ 1e-4.
- **G-tf**: iislab4에서 `hw/trainfree.py`를 수정 없이 두 ds에 실행한다(§2). Â²X(L2) test가 Cora_full 80.12, Amazon_clothing 92.72와 |Δ| ≤ 0.01, Â²X(정규화 없음) test가 67.57, 83.03과 |Δ| ≤ 0.01.
- **회귀**: `naq_base`, `nbr_raw`(기본 설정) seed 1212 두 ds → `_regress/HW08/`, 기존 값과 |Δ| ≤ 0.30.
- **불변**: `git diff <작업 시작 HEAD> -- hw/losses.py hw/negatives.py hw/neighbors.py hw/episodes.py`가 비어 있다. `hw/run.py`, `hw/evaluate.py`는 삭제 줄 수와 기존 평가 경로 발췌를 적는다.
- **smoke** (episodes 4, `/tmp/hw_smoke`, 커밋 안 함): `l2_naq`, `l2_rand_m20`, `l2_clus_m20` Cora_full 각 1회. §1.2의 새 키 전부와 `ckpt` 기록, 저장 파일의 존재·크기를 확인한다.

커밋 `HW08-a [iislab4]: evaluate_l2, eval_l2/save_best, norm_diag, l2_* arms, checks` (CLAUDE.md 개정 10, 지시서, 사전 등록, `HW08_checks.json` 포함) → pull --rebase → push.

## 2. HW08-b: 학습 없는 기준선 (iislab4, 결정적)

- `python -m hw.trainfree --dataset <ds>` 두 ds. 출력 위치는 스크립트 기본값을 따르고(`HW_RESULTS_ROOT` 아래), 같은 표를 `results/measure/HW08_trainfree_<ds>.md`로도 복사한다.
- 커밋 `HW08-b [iislab4]: train-free baselines on iislab4` → pull --rebase → push.

## 3. HW08-c: 실행과 집계

- `results/hw_iislab4/HW08_runs.txt`: 100 run, 순서 seed 바깥·arm 안쪽.
- 집계 (지표 네 개 각각):
  - `python -m hw.summarize --ref l2_naq --ref l2_rand_m20`
  - 같은 명령에 `--metric test_acc_l2`, `--metric test_acc_l2_at_bestraw`, `--metric test_acc_raw_at_bestl2`
- 복제 비교(보고서에서 직접 계산): `l2_X`의 `test_acc` − 대응 `e4k_X`의 `test_acc`, seed별.
- 커밋 `HW08-c [iislab4]: 100 runs (raw + l2 evaluation), report` → pull --rebase → push.

## 4. 게이트

- **G-cfg**, **G-l2**, **G-tf**, **회귀**(§1.6): 실패 시 멈춤.
- **G6**: run의 95% 이상 `completed`(1회 재시도 후). 못 미치면 멈추고 보고.
- **G7 (기록)**: arm별 붕괴·NaN 수, `zero_rows_test` > 0인 run 수.
- **태그**: 위 게이트 통과 → `git tag hw08 && git push origin hw08`.

## 5. 사전 등록 (HW08-a 커밋에 포함, 판정은 사람)

Δ = 같은 seed 짝지은 차이의 10 seed 평균. "raw" = `test_acc`, "l2" = `test_acc_l2`. "두 ds" = 두 데이터셋 모두.

| ID | 예측 | 기준 | 근거 | 확신도 |
|---|---|---|---|---|
| HWP079 | raw 값은 HW06과 복제 잡음 안에서 같다 | 다섯 arm × 두 ds 모두 seed별 \|`l2_X` − `e4k_X`\|(raw)의 평균 ≤ 0.20 | HW06 복제 잡음 평균 0.12 / 0.09 | 0.70 |
| HWP080 | 같은 checkpoint에서 l2 평가가 raw 평가보다 높다 | `l2_naq`: `test_acc_l2_at_bestraw` − `test_acc` ≥ +0.50, 두 ds | HL11 Clothing 5-shot NaQ 90.5 → 92.6 | 0.65 |
| HWP081 | 무작위 음성의 이득은 l2 평가에서도 남는다 | l2: `l2_rand_m20 − l2_naq` > 0, 양수 ≥ 8/10, 두 ds | HW06 raw +1.15 / +0.61 (10/10) | 0.55 |
| HWP082 | l2에서의 이득은 raw 이득의 절반 이상이다 | l2 Δ(`rand_m20 − naq`) ≥ 0.5 × raw Δ(같은 run), 두 ds | HL11에서 B_Choi의 raw 이득이 l2에서 사라진 전례 | 0.40 |
| HWP083 | m을 늘리면 l2에서도 낮아지지 않는다 | l2: `l2_rand_m200 − l2_rand_m20` ≥ 0, Cora_full | HW06 raw +0.35 | 0.50 |
| HWP084 | Cora의 군집 제외 이득은 l2에서도 남는다 | l2: `l2_clus_m20 − l2_rand_m20` ≥ +0.20, Cora_full | HW06·HW07 raw +0.39 (10/10) | 0.45 |
| HWP085 | l2 평가에서 학습 arm 최고값이 Â²X(L2)를 넘는다 | 다섯 arm 중 l2 평균 최고값 ≥ 80.12 (Cora_full), ≥ 92.72 (Amazon_clothing) | HW06 raw 최고 80.20 / 92.08 + HWP080 | 0.55 |
| HWP086 | 선택 기준(raw / l2 validation)은 l2 test를 거의 바꾸지 않는다 | 다섯 arm × 두 ds 모두 \|`test_acc_l2` − `test_acc_l2_at_bestraw`\| 평균 ≤ 0.20 | best epoch이 후반에 몰려 있음(HW06) | 0.60 |
| HWP087 | 무작위 음성은 프로토타입 노름의 흩어짐을 줄인다 | `proto_norm_cv`(raw 기준 best): `l2_rand_m20` < `l2_naq`, 두 ds | HW06 uniformity·effective rank 변화 | 0.50 |

## 6. 범위 밖

- 다섯 arm 밖의 arm 재실행, 다른 데이터셋, 1-shot 평가, 새 손실·새 음성 모드, 전역 contrast 항, PCA 기준선, `hw/trainfree.py` 수정, 50 기준 l2 기록, 기존 `e4k_*` 결과 수정·삭제, seed 추가, `results/hw/` 수정, 저장한 checkpoint를 쓰는 추가 분석.

## 7. 보고 (`reports/HW08_iislab4.md`, CLAUDE.md §7)

값의 종류를 표마다 명시.
1. **정확도 표**: 5 arm × ds × 지표 4개(`test_acc`, `test_acc_l2_at_bestraw`, `test_acc_l2`, `test_acc_raw_at_bestl2`) 평균 ± sd, 붕괴·NaN 수. 같은 표 아래에 학습 없는 X / ÂX / Â²X의 raw·L2 값(iislab4, §2).
2. **짝지은 Δ 표**: 지표 4개 각각에서 {rand_m20, rand_m50, rand_m200, clus_m20} − naq, clus_m20 − rand_m20, rand_m200 − rand_m20 (Δ ± sd, 양수 k/10).
3. **평가 방식 차이 표**: arm × ds의 `test_acc_l2_at_bestraw` − `test_acc`(같은 checkpoint), `test_acc_l2` − `test_acc_l2_at_bestraw`(같은 평가, 선택 기준 차이) (평균 ± sd).
4. **복제 표**: `l2_X` − `e4k_X` (raw) seed별 Δ, 평균 |Δ|, 최대 |Δ|.
5. best epoch 분포(raw 기준 / l2 기준), `same_epoch_l2` 비율 (arm × ds).
6. 기하·노름 표: `geom`, `geom_l2`의 `norm_mean`, `norm_std`, `uniformity`, `eff_rank`, `corr_offdiag_mean`과 `norm_diag`, `norm_diag_l2` 전 항목 (arm × ds, 10 seed 평균).
7. G-cfg, G-l2, G-tf, 회귀, 불변 발췌, smoke.
8. HWP079–087 대조.
9. 소요 시간, HEAD(작업 시작 HEAD 포함), `ckpt/` 사용 용량.
