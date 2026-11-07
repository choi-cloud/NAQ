# HW01 (개정 1) — 구조 개편 + exp012·exp015 재현 + 러너 이식 검증

2026-11-07 · 브랜치 `hl/hsan-weighting` · 지시서 `instructions/HW01_restructure_and_repro.md` (개정 1) · CLAUDE.md 개정 2

## 1. 결론

- 이 서버의 재현값(seed 1212)과 표 1의 차이(Δ = 이 서버 − 석휘)는 다음과 같다. exp015는 Cora_full 78.03(Δ −0.97), Amazon_clothing 90.91(Δ +0.19)이고, exp012는 Cora_full 78.10(Δ −0.45), Amazon_clothing 90.63(Δ −0.07)이다.
- 러너 `nbr_raw`를 exp015 재현과 비교하면 Cora_full Δ +0.07, Amazon_clothing Δ −0.04로, 두 값 모두 0.30 이하라서 G2를 통과한다. 다만 validation 곡선은 첫 평가 지점(epoch 160)부터 소수점 4자리에서 다르다(일치 지점: Cora_full 3/13, Amazon_clothing 1/13).
- 게이트 G1a·G1b·G2를 모두 통과했으므로 태그 `hw01`을 달았다. 사전 등록 6개 중 HWP002·003·005는 기준을 충족했고, HWP001·004·006은 충족하지 못했다.

## 2. 핵심 표

### 2.1 표 1 vs 이 서버 재현 vs 러너 (test accuracy %, 5-way 5-shot, 500 tasks, seed 1212)

| 실험 / arm | ds | 표 1 (석휘) | 이 서버 석휘 스크립트 | 러너 | Δ 재현 − 표 1 | Δ 러너 − 재현 |
|---|---|---|---|---|---|---|
| exp001 / `naq_base` | Cora_full | 77.72 | 미실행 (스크립트 없음) | 77.78 | — | — |
| exp001 / `naq_base` | Amazon_clothing | 90.48 | 미실행 (스크립트 없음) | 90.67 | — | — |
| exp012 | Cora_full | 78.55 | 78.10 | — | −0.45 | — |
| exp012 | Amazon_clothing | 90.70 | 90.63 | — | −0.07 | — |
| exp015 / `nbr_raw` | Cora_full | 79.00 | 78.03 | 78.10 | −0.97 | +0.07 |
| exp015 / `nbr_raw` | Amazon_clothing | 90.72 | 90.91 | 90.86 | +0.19 | −0.04 |
| exp016 / `nbr_bgrl` | Cora_full | 79.52 | 미실행 — BGRL 파일 대기 | 미실행 | — | — |
| exp016 / `nbr_bgrl` | Amazon_clothing | 91.21 | 미실행 — BGRL 파일 대기 | 미실행 | — | — |

- 러너 `naq_base` − 표 1 exp001: Cora_full Δ +0.06, Amazon_clothing Δ +0.19.
- Δ는 JSON의 반올림 전 값으로 계산했다. 예: exp015 Cora_full은 78.0250이다.

석휘 스크립트 재현 상세:

| run | best.epoch | test | ci95 | elapsed (s) |
|---|---|---|---|---|
| exp015 Cora_full | 2000 | 78.03 | 0.83 | 50.50 |
| exp015 Amazon_clothing | 1760 | 90.91 | 0.68 | 63.09 |
| exp012 Cora_full | 2000 | 78.10 | 0.83 | 49.07 |
| exp012 Amazon_clothing | 1600 | 90.63 | 0.71 | 62.86 |

`results/hw/summary.md` (러너):

| arm | Amazon_clothing | Cora_full |
|---|---|---|
| naq_base | 90.67 ± 0.00 (n=1) | 77.78 ± 0.00 (n=1) |
| nbr_raw | 90.86 ± 0.00 (n=1) | 78.10 ± 0.00 (n=1) |

| arm | ds | seed | best_epoch | val | test | ci95 | elapsed (s) |
|---|---|---|---|---|---|---|---|
| naq_base | Amazon_clothing | 1212 | 1600 | 87.70 | 90.67 | 0.69 | 60 |
| naq_base | Cora_full | 1212 | 2000 | 86.90 | 77.78 | 0.84 | 48 |
| nbr_raw | Amazon_clothing | 1212 | 1280 | 88.45 | 90.86 | 0.69 | 64 |
| nbr_raw | Cora_full | 1212 | 2000 | 86.65 | 78.10 | 0.83 | 48 |

### 2.2 validation 곡선 (exp015 재현 vs 러너 `nbr_raw`, 50 tasks, accuracy)

| epoch | Cora exp015 | Cora nbr_raw | 일치 | Amazon exp015 | Amazon nbr_raw | 일치 |
|---|---|---|---|---|---|---|
| 160 | 0.8010 | 0.8015 | N | 0.8070 | 0.8065 | N |
| 320 | 0.8350 | 0.8360 | N | 0.8400 | 0.8400 | Y |
| 480 | 0.8380 | 0.8390 | N | 0.8580 | 0.8585 | N |
| 640 | 0.8420 | 0.8435 | N | 0.8540 | 0.8545 | N |
| 800 | 0.8440 | 0.8455 | N | 0.8620 | 0.8625 | N |
| 960 | 0.8575 | 0.8580 | N | 0.8700 | 0.8690 | N |
| 1120 | 0.8525 | 0.8525 | Y | 0.8740 | 0.8750 | N |
| 1280 | 0.8630 | 0.8615 | N | 0.8835 | 0.8845 | N |
| 1440 | 0.8570 | 0.8570 | Y | 0.8820 | 0.8845 | N |
| 1600 | 0.8605 | 0.8580 | N | 0.8795 | 0.8805 | N |
| 1760 | 0.8635 | 0.8615 | N | 0.8840 | 0.8830 | N |
| 1920 | 0.8620 | 0.8600 | N | 0.8775 | 0.8800 | N |
| 2000 | 0.8665 | 0.8665 | Y | 0.8810 | 0.8815 | N |

- 첫 평가 구간(epoch 1–160)의 `train_loss_mean`: Cora_full은 1.436358(exp015)와 1.436318(러너), Amazon_clothing은 1.081748(exp015)와 1.081757(러너)이다.
- 두 run의 `settings`는 `exp_time`을 빼면 키와 값이 모두 같다. temperature 0.5, lambda_neighbor 0.1, max_neighbors 0도 같다.

## 3. 게이트·사전 등록 대조표

| ID | 기준 | 측정값 | 통과 여부 |
|---|---|---|---|
| G1a | Cora_full 내려받기 + 노드 19,793 | `Data(x=[19793, 8710], edge_index=[2, 126842], y=[19793])` | 통과 |
| G1b | smoke 4 run `status == "completed"` | 4/4 completed | 통과 |
| G2 | \|nbr_raw − exp015 재현\| ≤ 0.30, 두 데이터셋 | Cora_full +0.07, Amazon_clothing −0.04 | 통과 |
| HWP001 | exp015 Δ(표 1) ∈ [−0.5, +0.5], 두 데이터셋 | Cora_full −0.97, Amazon_clothing +0.19 | 기준 미충족 |
| HWP002 | exp012 Δ(표 1) ∈ [−0.5, +0.5], 두 데이터셋 | Cora_full −0.45, Amazon_clothing −0.07 | 기준 충족 |
| HWP003 | \|nbr_raw − exp015 재현\| ≤ 0.30, 두 데이터셋 | +0.07, −0.04 | 기준 충족 |
| HWP004 | validation 13개 지점 소수점 4자리 일치, 두 데이터셋 | Cora_full 3/13, Amazon_clothing 1/13 | 기준 미충족 |
| HWP005 | naq_base Δ(표 1 exp001) ∈ [−1.0, +1.0] | Cora_full +0.06, Amazon_clothing +0.19 | 기준 충족 |
| HWP006 | exp015 재현 − naq_base: Cora_full ≥ +0.5, Amazon_clothing ≥ 0 | Cora_full +0.24, Amazon_clothing +0.23 | 기준 미충족 (Cora_full) |

사전 등록의 최종 판정은 사람이 한다. 표의 "기준 충족 여부"는 측정값과 기준을 기계적으로 대조한 결과다.

## 4. 수행 내용

### HW01-a (`0835674`)
1. 드롭 폴더의 두 파일을 복사했다. `CLAUDE.md`(개정 2)는 루트로, 지시서는 `instructions/HW01_restructure_and_repro.md`로 옮겼다. 복사 후 CLAUDE.md를 다시 읽었고, 드롭 폴더 판본과 바이트가 같다.
2. 루트 파일 분류표. A = `git ls-tree --name-only 5cd9873` = `.gitignore arg.py data.py meta.py model.py naq_train.py sup_protonet.py sup_train.py task_generator.py unsup_protonet.py utils.py`.

| 파일 | A 포함 | 참조하는 파일 (A 또는 zip 6개) | 처리 |
|---|---|---|---|
| arg.py, data.py, meta.py, task_generator.py, utils.py | 예 | — | 루트 유지 |
| model.py, naq_train.py, sup_protonet.py, sup_train.py, unsup_protonet.py | 예 | — | 루트 유지 |
| exp001–exp049 `.py`/`.sh` (98개) | 아니오 | (규칙상 확인 불필요) | `git mv` → `choi_exps/legacy_naq_exp/` |
| run_exp049.py | 아니오 | (규칙상 확인 불필요) | `git mv` → `choi_exps/legacy_naq_exp/` |
| shell.sh | 아니오 | 없음 (`grep "shell"` 0건) | `git mv` → `choi_exps/legacy_naq_exp/` |

3. zip의 `exp012-016/` 여섯 파일을 `choi_exps/zip_20261107/`로 옮겼다. `__MACOSX/`는 빼고 옮겼다. `cmp`로 확인한 결과 zip 원본과 바이트가 같다.
4. 이 단계에서 작성한 파일은 `choi_exps/README.md`, `configs/hw.json`(CLAUDE.md §4.2 그대로), `hw/__init__.py`, `hw/repro_choi.sh`, `.gitkeep`이다.
5. `hw/repro_choi.sh`는 `.sh`의 `VAR="${VAR:-기본값}"` 줄과 `python -u <exp>.py` 인자 블록을 파일에서 직접 읽어 실행한다. DATASET과 (주어지면) EPOCHS만 치환한다. dry-run에서 만든 인자열은 exp012·exp015·exp016 `.sh`에 적힌 것과 같다.
6. 루트 `ls *.py *.sh` 결과: `arg.py data.py meta.py model.py naq_train.py sup_protonet.py sup_train.py task_generator.py unsup_protonet.py utils.py`. `*.sh`는 0개다.

### HW01-b (`7a4211c`)
- G1a: `CitationFull('./data/corafull','cora')`가 graph2gauss `cora.npz`를 내려받았다(약 5초).
- `data.py`의 `load_data_corafull`은 `random.shuffle(class_list)` 뒤에 클래스를 25 / 20 / 25(train / valid / test)로 나눈다. 분할은 `seed_everything` 이후의 python `random` 상태로 정해진다.
- G1b: smoke(EPOCHS=2) 결과는 다음과 같다. 모두 completed이고 run당 4–6 s 걸렸다.

| run | test |
|---|---|
| exp015 Cora_full | 0.5407 |
| exp015 Amazon_clothing | 0.7109 |
| exp012 Cora_full | 0.5415 |
| exp012 Amazon_clothing | 0.7098 |

### HW01-c (`3618a67`)
- 순서는 exp015 Cora_full → exp015 Amazon_clothing → exp012 Cora_full → exp012 Amazon_clothing이었다. run당 49–65 s, 4 run 합계 약 4분이 걸렸다. 지시서의 시간 규칙(총 4시간 초과 시 exp012 제외)은 적용되지 않았다.
- exp016은 실행하지 않았다(BGRL 파일 대기).

### HW01-d (이 커밋)
- 기준 코드 diff(`exp015.py` vs `exp016.py`)에서 나온 차이는 다음과 같다. 그 밖의 차이는 없다.
  - 이웃 선택 임베딩: raw feature 정규화 vs BGRL `.pt` 로드·검증·정규화
  - `--bgrl_dir` 인자(기본 `out/exp004`)
  - 변수명 `raw_features_normalized` / `selection_embeddings_normalized`
  - `EXP` 상수, docstring, `protocol` 설명 문자열
  - result의 `selection_embedding_path` 키
- `.py` 기본값과 `.sh` 값이 다른 항목은 exp015의 `lambda_neighbor`다(`.py` 1.0, `.sh` 0.1). `configs/hw.json`은 `.sh` 값을 쓴다.
- 구현 파일: `hw/config.py`, `episodes.py`, `neighbors.py`, `losses.py`, `evaluate.py`, `run.py`, `summarize.py`.
  - 함수는 `choi_exps/`를 import하지 않고 옮겨 적었다.
  - 호출 순서: `set_num_threads(4)` → `seed_everything` → `load_data` → `set_device` → `add_diffusion` → (선택 임베딩) → `ProtoNet` → `build_neighbor_lookup` → train_pool → `downstream_task_generator`.
  - `loss: "original"`이면 선택 임베딩, neighbor lookup, 이웃 선택을 모두 계산하지 않는다.
- `nbr_bgrl` 1회 실행(Cora_full): 0.79 s 만에 `FileNotFoundError: Missing BGRL embeddings: cache/bgrl/Cora_full.pt`로 끝났다. 생성된 실패 run 디렉토리(`results/hw/nbr_bgrl/`)는 지우고 커밋하지 않았다.
- `metrics.json`에는 석휘 result의 키 전부와 `runner`, `arm`, `config_path`, `config`, `first_episodes`(첫 5 에피소드의 loss·diagnostics)가 들어간다.
- 러너의 첫 5 에피소드 (original_loss, neighbor_loss). exp015.py는 에피소드별 값을 저장하지 않아 석휘 쪽 비교값은 없다.
  - Cora_full: (1.5540, 1.4647), (1.5198, 1.4946), (1.5606, 1.5571), (1.6732, 1.6599), (1.5517, 1.5723)
  - Amazon_clothing: (1.2584, 2.0645), (1.4532, 1.5550), (1.4431, 2.6477), (1.2963, 3.1789), (1.1176, 1.1187)

## 5. 판단한 사항

1. **`.gitignore`**: 기존 `.gitignore`는 `*` 전체를 무시하고 `.py`/`.sh`/`.gitignore`만 허용한다. 이대로는 지시서가 요구한 `.md`/`.json`/`.csv`/`train.log`를 추적할 수 없다.
   - 그래서 허용 규칙 `!*.md !*.json !*.csv !.gitkeep !results/hw/**/*.log`를 추가했다.
   - 이 허용 규칙 때문에 다른 트랙의 미추적 로컬 산출물(`results/hl/`, `results/nd/`)과 `save/`가 추적 대상으로 드러났다. 그래서 `results/hl/`, `results/nd/`, `save/`를 무시 목록에 추가했다. 그 파일들은 건드리지 않았다.
2. **`cache/bgrl/.gitkeep`**: `cache/` 무시 규칙과 겹쳐 추적되지 않는다. `cache/bgrl/` 디렉토리는 로컬에만 있다.
3. **평가 에피소드 캐시**: `task_generator.downstream_task_generator`는 `./save/downstream_tasks/<ds>/5way_5shot.pkl`이 있으면 그 파일을 읽고 RNG를 소비하지 않는다. 파일이 없으면 샘플링한 뒤 그 파일을 쓴다.
   - Amazon_clothing은 이 작업 전부터 캐시가 있었다(2026-10-24 06:03, md5 `ca3edaca02d541cbc0ee15a2122ef24a`, 생성 주체 미확인). 건드리지 않았다.
   - Cora_full은 캐시가 없었다. smoke(EPOCHS=2) run이 만든 캐시(md5 `567f67f0d114bed5e3a95802deb2a84a`)를 `out/hw_smoke_cache/`로 옮겼다. 그 결과 첫 전체 run(exp015 Cora_full, 2026-11-07 07:33:18)이 캐시를 새로 만들었다(md5 `cfef6a69020acde861b554cd2e8ac9c7`).
   - 이 판단의 근거는 다음과 같다. smoke는 train_pool을 2개만 만들어 RNG 상태가 전체 run과 다르다. 그래서 캐시가 없는 서버에서 석휘 스크립트를 전체로 돌린 경우와 같은 상태를 택했다(수정 범위가 작은 쪽).
   - 이후 모든 Cora_full run(exp015·exp012 재현, 러너)은 이 캐시를 읽는다.
   - 석휘 서버의 캐시 파일과 같은지는 확인되지 않았다.
4. **`hw/repro_choi.sh`**: 인자를 손으로 옮겨 적지 않고 `.sh` 파일에서 직접 추출한다. 값을 바꾸지 않는다는 지시서 조건을 기계적으로 보장하기 위해서다.
5. **`summary.md`의 열 순서**: 데이터셋 이름의 알파벳순이다(Amazon_clothing, Cora_full).

## 6. 제외·실패한 run과 사유

- exp016 재현, `nbr_bgrl` 학습: BGRL 임베딩 파일이 없다(지시서 §6 범위 밖).
- `nbr_bgrl` Cora_full 1회: 의도한 `FileNotFoundError` 확인용이다. 결과는 커밋하지 않았다.
- 새 계열 exp001: 스크립트가 없다. `naq_base`는 대응하는 석휘 스크립트가 없어 검증 대상이 아니다.

## 7. 소요 시간, 환경, HEAD

- 작업 시각: 2026-11-07 07:30–07:45 UTC, 약 15분. 시간 예산은 3시간이었다.
- run당 소요 시간: 석휘 스크립트 49–65 s, 러너 48–64 s(`elapsed_seconds`, 학습 + 평가).
- 환경
  - GPU: NVIDIA GeForce RTX 4080 SUPER(16 GB), device 0
  - conda `taskns`: torch 2.4.1+cu121, `yaml` 없음
  - PyG `CitationFull` 사용
- HEAD: 이 보고서를 포함한 HW01-d 커밋(해시는 `git log`). 태그 `hw01`.
