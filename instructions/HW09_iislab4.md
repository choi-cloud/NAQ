# HW09 [iislab4] — dblp · Amazon_electronics 확장 + 1-shot test 병기 (`test_shots`) + HW08 checkpoint의 1-shot 재평가 + 학습 없는 기준선 (auto, 4000 episode, 80 run)

2026-11-11 · 작성: 이희태 (Claude 정리) · 서버 **iislab4** · 브랜치 `hl/hsan-weighting` · 선행 태그 `hw08` 이후 · 태그 `hw09` · 사전 등록 HWP088–HWP095 · 선행 문서 `CLAUDE.md` 개정 11, `reports/HW08_iislab4.md`

- auto 모드. 멈추는 조건: §0 G-data 실패(**멈추고 사람**), §1.7 G-ckpt·G-tf·회귀 확인 실패, G6, 해결 못 한 오류.
- 결과 루트 **`HW_RESULTS_ROOT=results/hw_iislab4`**, `GPUS=0,1,2,3,4,5,6,7`, seed 1212–1221.
- **주 지표는 `test_acc_l2`(l2 validation 500 선택, l2 평가)다**(CLAUDE.md 개정 11 §5). raw 값은 계속 기록한다.
- 학습 경로(손실·샘플링·옵티마이저·RNG 소비)는 바꾸지 않는다. 바꾸는 것은 test 평가의 shot 수와 새 데이터셋의 캐시뿐이다.
- 시간 예산 3시간 (확인·캐시 40 / 구현 30 / 재평가·기준선 15 / 실행 ~70 / 집계·보고 25).

## 0. 현황과 G-data (수정 없음, 보고서에 적고 계속)

`git fetch && git status -sb | head -1`, `git log --oneline -3`, `git tag --list 'hw0*'`, `nvidia-smi --query-gpu=index,memory.used --format=csv`, `df -h . | tail -1`. **작업 시작 HEAD 기록.** 허용 미커밋 변경은 미추적 `instructions/HW09_iislab4.md`와 수정된 `CLAUDE.md`(개정 11)뿐. 그 밖에 있으면 **멈추고 보고**. `git pull --rebase`. CLAUDE.md 개정 11을 다시 읽는다.

**G-data** (새 데이터셋 두 개: dblp, Amazon 전자제품. 리포 안의 실제 이름을 쓴다 — NaQ 원본 표기는 `Amazon_eletronics`):
- `ls data/` 결과와 두 데이터셋의 파일 목록·크기.
- `hw/run.py`가 쓰는 데이터 로더로 두 데이터셋을 읽어 노드 수, 엣지 수(무방향), 특징 차원, 클래스 수, train / valid / test 클래스 수를 기록한다. 로더 인자 이름(데이터셋 문자열)도 기록한다.
- **데이터 파일이 없거나 로더가 읽지 못하면 멈추고 사람**: 아래를 그대로 출력하고 멈춘다.
  ```
  [사람이 할 일] 두 데이터셋의 data/ 파일을 iislab4 ~/wkspace/NAQ/data/에 넣어야 합니다.
  확인된 위치 후보: seio ~/wkspace/NAQ/data/ (HL 트랙에서 원저자 리포 data/ 복사), gpusystem ~/workspace/NAQ/data/
  없는 파일: <목록>
  ```
- 평가 캐시 존재 여부와 md5 (네 데이터셋 모두): `save/downstream_tasks/<ds>/5way_5shot.pkl`, `save/downstream_tasks/<ds>/5way_1shot.pkl`, `save/valid500/<ds>.pkl`. 없는 것은 §1.1에서 만든다.

기록: `hw/evaluate.py`가 에피소드 묶음을 받는 방식(`build_eval_pools`의 인자와 반환), 기존 캐시를 만든 코드 경로(NaQ 원본의 함수 이름과 seed 처리), HW03b가 valid500 캐시를 만든 함수 이름.

## 1. HW09-a: 캐시와 구현

### 1.1 평가 캐시 (없는 것만 새로 만든다)
- **기존 캐시는 수정·재생성하지 않는다.** 있는 파일은 md5만 기록한다.
- `5way_5shot.pkl`, `5way_1shot.pkl`이 없으면 기존 캐시를 만든 **NaQ 원본 코드 경로를 그대로** 써서 만든다(같은 함수, 같은 valid 50 / test 500 개수, 같은 seed 규칙). 만든 방법(함수, seed)과 md5를 보고서에 적는다.
- `save/valid500/<ds>.pkl`이 없으면 HW03b와 같은 함수로 만든다(앞 50개 = 그 데이터셋 5-shot 캐시의 validation 50개, 뒤 450개 = 같은 validation 클래스에서 새로 뽑은 task).
- 1-shot 캐시의 test task 수는 500, 5-shot 캐시와 같은 test 클래스 집합이어야 한다(확인해 기록).

### 1.2 `test_shots` (`hw/run.py`, 기본 `[5]`)
- `[5]`이면 기존 동작과 기록이 그대로다.
- `[5, 1]`이면 학습 종료 뒤 **같은 checkpoint**들(raw 기준 best, l2 기준 best)에서 1-shot test 500 task를 추가로 평가한다. checkpoint 선택(validation)은 기존대로 5-shot validation 500이다.
- 기록 키(최상위): 기존 test 키 네 개에 `_1shot`을 붙인 `test_acc_1shot`, `test_acc_l2_at_bestraw_1shot`, `test_acc_l2_1shot`, `test_acc_raw_at_bestl2_1shot`. 5-shot 키의 이름과 값은 그대로다.
- 평가는 `evaluate` / `evaluate_l2`를 그대로 쓴다(shot 수는 에피소드 묶음에서 온다).

### 1.3 checkpoint 재평가 (`hw/eval_ckpt.py`, 새 파일)
- 입력: `--arm`, `--dataset`, `--seed`, 결과 루트. `metrics.json`의 `ckpt`에 적힌 파일을 읽어(`best_l2`가 null이면 l2 기준 best = `best_raw.pt`), 그 run의 설정으로 인코더를 만들고 state_dict를 올린다.
- 평가: 5-shot test(raw·l2)와 1-shot test(raw·l2)를 두 checkpoint에서 각각 잰다.
- 출력: `results/hw_iislab4/ckpt_eval/<arm>/<ds>/seed<S>.json`. 기존 `metrics.json`은 수정하지 않는다.
- 대상: HW08의 `l2_naq`, `l2_rand_m20`, `l2_rand_m50`, `l2_clus_m20` × Cora_full, Amazon_clothing × 10 seed = 80개(학습 없음).

### 1.4 k-means 캐시 (새 데이터셋)
- `cluster` 모드의 기존 계산(`KMeans(n_clusters=100, n_init=4, random_state=20261108)`, Â²X L2 정규화)을 그대로 두 새 데이터셋에 적용해 `cache/kmeans/<ds>_k100.pt`에 저장한다. **k는 100으로 고정한다**(데이터셋별 조정 없음).

### 1.5 학습 없는 기준선 (`hw/trainfree.py`)
- `--shot ∈ {5, 1}` 인자를 추가한다(기본 5 = 기존 동작). 1이면 1-shot 캐시로 같은 6개 특징(X / ÂX / Â²X × 정규화 없음 / L2)을 평가한다.
- 기본값 실행의 출력은 HW08-b와 같아야 한다(G-tf).

### 1.6 arm 추가 (`configs/hw.json`; 모두 base `naq_base`, `valid_tasks 500`, `episodes 4000`, `geom_diag true`, `eval_l2 true`, `save_best true`, `test_shots [5, 1]`)

| arm | 대응하는 HW08 arm | `extra_neg` |
|---|---|---|
| `h9_naq` | `l2_naq` | 없음 |
| `h9_rand_m20` | `l2_rand_m20` | `{random, 20}` |
| `h9_rand_m50` | `l2_rand_m50` | `{random, 50}` |
| `h9_clus_m20` | `l2_clus_m20` | `{cluster, 20, k 100}` |

- 4 arm × 2 데이터셋(dblp, Amazon 전자제품) × 10 seed = **80 run**. Cora_full·Amazon_clothing은 학습하지 않는다(§1.3 재평가로 대신한다).

### 1.7 확인 (실패 시 **멈추고 보고**; 결과는 `results/hw_iislab4/HW09_checks.json`)
- **G-cfg**: 네 `h9_*` arm의 병합 설정에서 `test_shots`를 뺀 것이 대응 `l2_*` arm의 병합 설정과 같다(`arm`, `note` 키 제외).
- **G-ckpt**: §1.3 재평가의 5-shot 값이 HW08 `metrics.json`의 기록값과 같다. `test_acc`, `test_acc_l2_at_bestraw`, `test_acc_l2`, `test_acc_raw_at_bestl2` 네 개 모두, 80개 run 모두 |Δ| ≤ 1e-4.
- **G-tf**: `hw.trainfree` 기본값(5-shot)으로 Cora_full·Amazon_clothing을 실행해 Â²X L2 80.12 / 92.715, 정규화 없음 67.57 / 83.025와 |Δ| ≤ 0.01.
- **회귀**: `naq_base`, `nbr_raw`(기본 설정) seed 1212, Cora_full·Amazon_clothing → `_regress/HW09/`, 기존 값과 |Δ| ≤ 0.30.
- **불변**: `git diff <작업 시작 HEAD> -- hw/losses.py hw/negatives.py hw/neighbors.py hw/episodes.py hw/evaluate.py`가 비어 있다. `hw/run.py`, `hw/trainfree.py`는 삭제 줄 수와 기존 경로 발췌.
- **smoke** (episodes 8, `/tmp/hw_smoke`, 커밋 안 함): `h9_naq`, `h9_clus_m20` × 두 새 데이터셋. 1-shot 키 기록, checkpoint 저장, **run당 시간과 GPU 메모리 최대치**를 기록한다.
- **시간 규칙**: smoke의 run당 시간으로 4000 episode run 시간을 추정하고, 80 run ÷ 8 GPU의 예상이 **150분을 넘으면 두 새 데이터셋의 seed를 1212–1216(5개)으로 줄인다**. 멈추지 않고 보고서에 적는다.

커밋 `HW09-a [iislab4]: caches (dblp, electronics), test_shots, eval_ckpt, trainfree --shot, h9_* arms, checks` (CLAUDE.md 개정 11, 지시서, 사전 등록, 새 캐시의 md5 목록, `HW09_checks.json` 포함; `.pkl`·`.pt`는 git 밖) → pull --rebase → push.

## 2. HW09-b: 학습 없는 측정 (결정적)

1. §1.3 재평가 80개 실행. 출력 `results/hw_iislab4/ckpt_eval/`.
2. `hw.trainfree`를 네 데이터셋 × `--shot {5, 1}`로 실행. 같은 표를 `results/measure/HW09_trainfree_<ds>.md`로 복사.
3. 새 데이터셋 k-means 표(측정 전용 레이블): 군집 크기 최소·중앙값·최대, 같은 군집 내 같은 클래스 비율(노드 macro), q와 다른 군집 노드 중 같은 클래스 비율, 무작위 기대 FN rate Σ p_c², NaQ-Feat top-10 query 순도. Cora_full·Amazon_clothing 값(HW06 G-table)도 같은 표에 옮긴다. 출력 `results/measure/HW09_cluster_stats.md`.

커밋 `HW09-b [iislab4]: checkpoint 1-shot re-eval, train-free (4 ds × 2 shots), cluster stats` → pull --rebase → push.

## 3. HW09-c: 실행과 집계

- `results/hw_iislab4/HW09_runs.txt`: 80 run(또는 시간 규칙 적용 시 40 run), 순서 seed 바깥·arm 안쪽.
- 집계: `python -m hw.summarize --ref h9_naq --ref h9_rand_m20`를 지표 네 개(`test_acc_l2`, `test_acc_l2_1shot`, `test_acc`, `test_acc_1shot`)에 각각 실행.
- Cora_full·Amazon_clothing의 1-shot 짝지은 Δ는 `ckpt_eval/` 값으로 보고서에서 직접 계산한다(arm 이름은 `l2_*`).
- 커밋 `HW09-c [iislab4]: 80 runs (dblp, electronics; 5- and 1-shot), report` → pull --rebase → push.

## 4. 게이트

- **G-data**(§0): 실패 시 멈추고 사람.
- **G-cfg**, **G-ckpt**, **G-tf**, **회귀**(§1.7): 실패 시 멈춤.
- **G6**: run의 95% 이상 `completed`(1회 재시도 후). 못 미치면 멈추고 보고.
- **G7 (기록)**: arm별 붕괴·NaN 수, `zero_rows_test` > 0인 run 수.
- **태그**: 위 게이트 통과 → `git tag hw09 && git push origin hw09`.

## 5. 사전 등록 (HW09-a 커밋에 포함, 판정은 사람)

Δ = 같은 seed 짝지은 차이의 평균. "l2" = `test_acc_l2`(5-shot), "l2-1s" = `test_acc_l2_1shot`. "새 두 ds" = dblp, Amazon 전자제품. "네 ds" = 새 두 ds + Cora_full·Amazon_clothing(재평가 값).

| ID | 예측 | 기준 | 근거 | 확신도 |
|---|---|---|---|---|
| HWP088 | 무작위 음성은 새 데이터셋에서도 이득이다 | l2: `h9_rand_m20 − h9_naq` > 0, 양수 ≥ 7/10, 새 두 ds | HW08 +1.15 / +0.22 | 0.55 |
| HWP089 | 군집 제외는 새 데이터셋에서도 NaQ를 넘는다 | l2: `h9_clus_m20 − h9_naq` > 0, 양수 ≥ 8/10, 새 두 ds | HW08 +1.61 / +0.34 (10/10) | 0.50 |
| HWP090 | 군집 제외는 무작위보다 낮지 않다 | l2: `h9_clus_m20 − h9_rand_m20` ≥ 0, 새 두 ds | HW08 +0.46 / +0.12 | 0.45 |
| HWP091 | 새 데이터셋에서도 NaQ는 학습 없는 Â²X(L2)보다 낮다 | l2 `h9_naq` 평균 < Â²X L2 (5-shot), 새 두 ds | HW08 −0.98 / −0.13; TEG 트랙 D 5-shot dblp 86.1 / Electronics 89.0 | 0.60 |
| HWP092 | 최고 arm이 Â²X(L2)를 넘는 데이터셋이 새 두 ds 중 하나 이상이다 | 네 arm 중 l2 평균 최고 ≥ Â²X L2 (5-shot) | HW08 두 ds 모두 넘음 | 0.45 |
| HWP093 | 1-shot에서도 군집 제외가 NaQ를 넘는다 | l2-1s: `clus_m20 − naq` > 0, 네 ds | 5-shot 결과 | 0.50 |
| HWP094 | 1-shot 이득이 5-shot 이득보다 크다 | `clus_m20 − naq`의 l2-1s Δ > l2 Δ, 네 ds 중 3개 이상 | CaPE 트랙에서 이득이 1-shot에 집중 | 0.45 |
| HWP095 | 음성을 늘려도 낮아지지 않는다 | l2: `h9_rand_m50 − h9_rand_m20` ≥ 0, 새 두 ds | HW08 Cora +0.19 / Amazon +0.09 | 0.40 |

## 6. 범위 밖

- Cora_full·Amazon_clothing 재학습, 새 손실·새 음성 모드, k ≠ 100, 전역 contrast 항, VISION·TLP 기준선, NaQ-Diff, 1-shot validation 선택, PCA 기준선, 기존 캐시·`e4k_*`·`l2_*` 결과 수정, seed 추가, `results/hw/` 수정.

## 7. 보고 (`reports/HW09_iislab4.md`, CLAUDE.md §7)

값의 종류(정확도, shot, 평가 방식, checkpoint 선택 기준)를 표마다 명시. **주 결과 = `test_acc_l2`, 그다음 `test_acc_l2_1shot`.**
1. **데이터 표**: 네 ds의 노드·엣지·특징 차원·클래스 수(train/valid/test), 캐시 md5(새로 만든 것 표시), §2-3 군집 통계.
2. **네 데이터셋 주 결과 표** (5-shot과 1-shot 각각, l2): 행 = {Â²X L2(학습 없음), naq, rand_m20, rand_m50, clus_m20}, 열 = 네 ds. 셀은 평균 ± sd. Cora_full·Amazon_clothing은 HW08 `l2_*` 값(5-shot)과 재평가 값(1-shot)이다(출처 표시).
3. **짝지은 Δ 표** (l2 5-shot, l2 1-shot): {rand_m20, rand_m50, clus_m20} − naq, clus_m20 − rand_m20, rand_m50 − rand_m20 × 네 ds (Δ ± sd, 양수 k/n).
4. **학습 없는 기준선 표**: 네 ds × {X, ÂX, Â²X} × {raw, L2} × {5-shot, 1-shot}.
5. raw 지표 표: 새 두 ds의 `test_acc`, `test_acc_1shot`와 짝지은 Δ(같은 지표끼리).
6. best epoch 분포(raw 기준 / l2 기준), `same_epoch_l2`, 기하·노름 표(새 두 ds).
7. G-data, G-cfg, G-ckpt, G-tf, 회귀, 불변, smoke(run당 시간·메모리), 시간 규칙 적용 여부.
8. HWP088–095 대조.
9. 소요 시간, HEAD(작업 시작 HEAD 포함), `ckpt/` 사용 용량.
