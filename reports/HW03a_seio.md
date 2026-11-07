# HW03a [seio] — BGRL 임베딩 생성 + `nbr_bgrl` 확인 run + 학습 없는 이웃 순도 측정 + 학습 없는 기준선

2026-11-07 · 서버 seio (RTX 4080 SUPER × 1) · 브랜치 `hl/hsan-weighting` · 지시서 `instructions/HW03a_seio.md` · CLAUDE.md 개정 4(드롭 폴더 판본) · 태그 없음

> s1–s4 완료.

## 1. 결론

- BGRL(exp004 + hw shim) 임베딩: Cora_full 19,793×256(md5 `c2ef0115…`), Amazon_clothing 24,919×256(md5 `bb1bd6d8…`), G-bgrl 통과.
  - `nbr_bgrl − nbr_raw`(seio 5 seed 짝지음): Amazon_clothing −0.09 (양수 3/5), Cora_full +0.62 (양수 5/5).
- 학습 없는 순도 `nbr_eq_support`(seed 1212, 2000 에피소드):
  - Cora_full: raw_max 0.63, raw_contrast 0.64, bgrl_max 0.59, random 0.59
  - Amazon_clothing: raw_max 0.66, bgrl_max 0.64, random 0.62
  - `query_eq_support`: Cora_full 0.35, Amazon_clothing 0.60
- 학습 없는 `Â²X`(L2 행 정규화) ProtoNet test는 Cora_full 80.12, Amazon_clothing 92.72다. 같은 서버의 학습 `nbr_raw` 5 seed 평균(77.94 / 91.00)보다 각각 +2.18 / +1.72 높다.
- 사전 등록 HWP024·HWP025는 둘 다 기준 미충족이다.

## 2. 핵심 표

### 2.1 BGRL(exp004 + hw shim) 생성

| 항목 | Cora_full | Amazon_clothing |
|---|---|---|
| 명령 | 아래 | 아래 |
| 소요 (wall / `elapsed_seconds`) | 18 s / 17.18 s | 25 s / 23.18 s |
| status | completed | completed |
| G-bgrl (a) `dataset` | ✓ | ✓ |
| G-bgrl (b) 2차원, shape[0] = N | ✓ (19,793 × 256) | ✓ (24,919 × 256) |
| G-bgrl (c) `node_ids` = arange(N) | ✓ | ✓ |
| md5 (`cache/bgrl/<ds>.pt`) | `c2ef0115b8f0e7732f206655f6eeee8b` | `bb1bd6d8fe8021dc4f2e93f269e53760` |
| 행 노름 평균 ± 표준편차 | 10.27 ± 3.64 | 6.43 ± 3.13 |
| 노름 0인 행 / 비유한 값 | 0 / 없음 | 0 / 없음 |
| 저장 키 | `dataset, embedding_type, embeddings, epoch, node_ids` | 같음 |

실행 명령(두 데이터셋 각각, `.sh` 값 그대로, 리포 루트에서):
```
PYTHONPATH=hw/compat:. /home/h2tae/.conda/envs/taskns/bin/python -u choi_exps/zip_20261107/exp004.py \
  --dataset <ds> --device 0 --epoch 400 --latent 256 --lr 0.001 --edge_drop 0.2 \
  --batch_size 512 --momentum 0.99 --eval_every 50 2>&1 | tee out/exp004/exp004_<ds>.log
```

### 2.2 exp004 출력 지표 (cosine Top-K 클래스 일치율, 고정 앵커 512개)

| ds | 대상 | top10 | top50 | top100 |
|---|---|---|---|---|
| Cora_full | raw_features | 0.38 | 0.25 | 0.20 |
| Cora_full | random_gcn (epoch 0) | 0.55 | 0.37 | 0.29 |
| Cora_full | BGRL epoch 50 | 0.58 | 0.46 | 0.40 |
| Cora_full | BGRL epoch 400 (최종) | 0.57 | 0.46 | 0.40 |
| Amazon_clothing | raw_features | 0.60 | 0.47 | 0.42 |
| Amazon_clothing | random_gcn (epoch 0) | 0.64 | 0.52 | 0.47 |
| Amazon_clothing | BGRL epoch 50 | 0.64 | 0.54 | 0.49 |
| Amazon_clothing | BGRL epoch 400 (최종) | 0.64 | 0.54 | 0.50 |

- `infonce_exp003`은 두 데이터셋 모두 `missing`이다(`out/exp003/exp003_<ds>_embeddings.pt` 없음).
- 학습 손실(epoch 1 → 400): Cora_full 1.9179 → 0.0035, Amazon_clothing 1.9072 → 0.0111.
- epoch별 전체 값은 `results/hw/repro/zip_20261107_exp004/exp004_<ds>.json`에 있다.

### 2.3 `nbr_bgrl` 5 seed (seio, `results/hw/summary.md`, `paired_nbr_raw.md`, `paired_naq_base.md`)

| arm | Amazon_clothing | Cora_full |
|---|---|---|
| naq_base | 90.80 ± 0.20 | 77.59 ± 0.70 |
| nbr_raw | 91.00 ± 0.20 | 77.94 ± 0.61 |
| nbr_bgrl | 90.91 ± 0.27 | 78.56 ± 0.39 |
| 표 1 exp016 (석휘 서버, seed 1212) | 91.21 | 79.52 |

| 짝지은 Δ (seio, 5 seed) | Amazon_clothing | Cora_full |
|---|---|---|
| `nbr_bgrl − nbr_raw` | −0.09 ± 0.35 (양수 3/5) | +0.62 ± 0.34 (양수 5/5) |
| `nbr_bgrl − naq_base` | +0.11 ± 0.33 (양수 3/5) | +0.97 ± 0.68 (양수 5/5) |

seed별 결과(`nbr_bgrl`, test % / best_epoch / Δ vs `nbr_raw`):

| seed | Amazon_clothing | Cora_full |
|---|---|---|
| 1212 | 91.02 / 1600 / +0.33 | 78.31 / 1920 / +0.15 |
| 1213 | 91.24 / 1760 / +0.05 | 78.38 / 2000 / +0.70 |
| 1214 | 90.75 / 1440 / −0.50 | 78.07 / 1920 / +1.18 |
| 1215 | 91.06 / 2000 / +0.18 | 79.13 / 2000 / +0.45 |
| 1216 | 90.48 / 960 / −0.52 | 78.90 / 1440 / +0.60 |

- 10 run 모두 `status == "completed"`이고 붕괴(collapsed)는 0이다.
- `selection_embedding_path`는 `cache/bgrl/<ds>.pt`다.
- 마지막 구간의 `fraction_valid_neighbor_selections`는 Cora_full 0.9990–0.9994, Amazon_clothing 0.9985–0.9988이다.
- `nbr_bgrl`(λ 1.0)과 `nbr_raw`(λ 0.1)는 λ_neighbor도 다르다(각각 exp016·exp015의 `.sh` 값).

### 2.4 학습 없는 이웃 순도 (seed 1212, 학습 에피소드 2000개 × 자기 클래스 쌍 50개, `results/hw/purity/`)

| 참고 행 | Cora_full | Amazon_clothing |
|---|---|---|
| query_eq_support | 0.3494 | 0.5977 |
| support_1hop_homophily | 0.5621 | 0.6686 |
| multi_nbr_frac | 0.8240 | 0.7288 |
| fallback_frac | 0.0046 | 0.0081 |

| 선택기 | Cora nbr_eq_support | Cora nbr_eq_query | Cora pos_nbr_cos_raw | Amazon nbr_eq_support | Amazon nbr_eq_query | Amazon pos_nbr_cos_raw |
|---|---|---|---|---|---|---|
| raw_max | 0.6275 | 0.3342 | 0.1730 | 0.6610 | 0.5563 | 0.4434 |
| raw_contrast | 0.6425 | 0.2928 | 0.1057 | 0.6357 | 0.4990 | 0.2964 |
| raw_random | 0.5871 | 0.2896 | 0.1090 | 0.6209 | 0.5076 | 0.3071 |
| bgrl_max | 0.5937 | 0.3472 | 0.1258 | 0.6395 | 0.5451 | 0.3576 |
| bgrl_contrast | 0.5478 | 0.2361 | 0.0976 | 0.5980 | 0.4732 | 0.2721 |
| bgrl_random | 0.5871 | 0.2896 | 0.1090 | 0.6209 | 0.5076 | 0.3071 |

- `raw_random`과 `bgrl_random`은 같은 값이다. `random`은 선택 공간을 쓰지 않고, 두 경우 모두 같은 Generator(seed + 7919)를 에피소드 순서대로 쓴다.
- `support_1hop_homophily`는 유효 이웃 단위 평균이고, 선택기 행은 (query, 자기 클래스 support) 쌍 단위 평균이다. 이웃이 없는 쌍(fallback)은 선택 결과가 support 자신이다.
- 교차 확인: 160 에피소드로 돌린 `bgrl_max` 값(Amazon_clothing)이 s2의 `nbr_bgrl` seed 1212 run의 첫 평가 구간 `nbr_diag`와 소수점 4자리까지 같다(0.6286 / 0.5156 / 0.5584 / 0.3567).

### 2.5 학습 없는 기준선 (ProtoNet을 특징 위에서 바로 평가, `results/hw/trainfree/`)

캐시 에피소드는 valid 50 / test 500이고, 정확도 단위는 %다. Â = `data.adj` = D^-1/2 (A+I) D^-1/2이며, 독립적으로 만든 행렬과의 최대 |차이|는 두 ds 모두 8.94e-08이다.

| 특징 | 행 정규화 | Cora val | Cora test (ci95) | Amazon val | Amazon test (ci95) |
|---|---|---|---|---|---|
| X (raw) | 없음 | 50.05 | 48.07 (0.92) | 66.25 | 66.58 (1.07) |
| X (raw) | L2 | 67.80 | 60.50 (0.83) | 84.20 | 87.61 (0.77) |
| ÂX | 없음 | 72.95 | 64.51 (1.02) | 70.10 | 73.64 (1.01) |
| ÂX | L2 | 85.25 | 77.41 (0.84) | 89.55 | 91.79 (0.69) |
| Â²X | 없음 | 77.00 | 67.57 (1.01) | 78.45 | 83.03 (0.86) |
| Â²X | L2 | 87.70 | 80.12 (0.81) | 90.30 | 92.72 (0.65) |
| naq_base (학습, seio 5 seed 평균 ± 표준편차) | — | 86.28 | 77.59 ± 0.70 | 87.66 | 90.80 ± 0.20 |
| nbr_raw (학습, seio 5 seed 평균 ± 표준편차) | — | 86.63 | 77.94 ± 0.61 | 88.11 | 91.00 ± 0.20 |

- 학습 없는 값은 결정적(seed 무관)이라 짝지은 비교가 아니다. 학습 arm의 seed 평균과 나란히 적었다.
- `Â²X`(L2) − `nbr_raw` 평균: Cora_full +2.18, Amazon_clothing +1.72.
- `ÂX`(L2) − `nbr_raw` 평균: Cora_full −0.53, Amazon_clothing +0.79.
- 학습 run의 best epoch 선택에는 validation 50 tasks를 썼다. 학습 없는 값에는 선택 단계가 없다.

## 3. 게이트·사전 등록 대조표

| ID | 기준 | 측정값 | 통과 여부 |
|---|---|---|---|
| G-bgrl | (a)(b)(c), 두 ds | 모두 ✓ | 통과 |
| HWP024 | `nbr_bgrl − nbr_raw` ≥ 0, 두 ds (seio 5 seed) | Amazon_clothing −0.09 (3/5), Cora_full +0.62 (5/5) | 기준 미충족 (Amazon_clothing) |
| HWP025 | `nbr_eq_support`: max(bgrl) ≥ max(raw) > contrast > random, 같은 공간 안 순서 포함, 두 ds | max(bgrl) − max(raw): Cora −0.0338, Amazon −0.0215. raw 공간: Cora contrast 0.6425 > max 0.6275 > random 0.5871; Amazon max 0.6610 > contrast 0.6357 > random 0.6209. bgrl 공간: Cora max 0.5937 > random 0.5871 > contrast 0.5478; Amazon max 0.6395 > random 0.6209 > contrast 0.5980 | 기준 미충족 |

## 4. 수행 내용

### §0 현황 (2026-11-07 11:33 UTC, 수정 없음)
- `git status -sb`: `## hl/hsan-weighting...origin/hl/hsan-weighting`. HEAD는 `d277ea2`이고 pull할 것은 없었다.
- GPU: 2 MiB 사용, 0%로 다른 프로세스가 없었다.
- `choi_exps/zip_20261107/`: `exp004.{py,sh} exp012.{py,sh} exp015.{py,sh} exp016.{py,sh}`.
- exp004 docstring: "exp004: BGRL pretraining and pool-purity comparison against baselines. … No labels are used for learning, model selection, or candidate retrieval: labels are diagnostics only. The CL encoder is NOT used to initialize the downstream few-shot encoder."
- exp004 `.sh`: `EPOCHS=400 LR=0.001 EDGE_DROP=0.2 BATCH_SIZE=512 MOMENTUM=0.99 EVAL_EVERY=50 DEVICE=0`, `--latent 256`. `for DATASET in Cora_full Amazon_clothing` 루프로 돌린다.
- exp004 `.py`에만 있는 기본값: `--seed 1212 --predictor_hidden 512 --diagnostic_anchors 512 --infonce_dir out/exp003`.
- 출력은 모두 `out/exp004/` 아래에 생긴다.
  - `exp004_<ds>.json`(결과)
  - `exp004_<ds>_encoder.pt`(키 `state_dict, config, dataset, epoch, seed, settings`)
  - `exp004_<ds>_embeddings.pt`(키 `embeddings, node_ids, dataset, epoch, embedding_type`)
- 의존성: `from exp003 import dropped_normalized_adj`, `from model import GNNEncoder`. 새 계열 `exp003.py`가 리포에 없어 그대로 실행하면 `ModuleNotFoundError: No module named 'exp003'`이 난다. `legacy_naq_exp/exp003.py`에는 이 함수가 없다.
- 이 오류로 한 번 멈추고 보고했다. 사람이 "선택지 2(shim)"를 정했다.

### HW03a-s1 (이 커밋)
- **작업 트리 정리(사람 지시)**: 리포 작업 트리에 있던 미커밋 CLAUDE.md 개정 4를 `~/wkspace/HW_drop/CLAUDE.md`로 복사했다(드롭 폴더에는 그때까지 개정 3이 있었다). 이어 `git checkout -- CLAUDE.md`로 리포 변경을 버렸다. `instructions/HW03a_seio.md`는 이 커밋에 넣었다.
- **shim `hw/compat/exp003.py`**: `dropped_normalized_adj(edge_index, num_nodes, drop_rate)` 하나만 구현했다.
  - 시그니처의 근거:
    - exp004.py 186–187행의 호출이 `dropped_normalized_adj(data.edge_index, n, args.edge_drop)`(위치 인자 3개)다.
    - 반환값은 `online(x, adj=view)`로 `model.GNNEncoder.forward`에 들어가 `torch.sparse.mm(adj, x)`를 거친다. 따라서 `edge_index.device`(set_device 이후 cuda) 위의 float32 sparse 텐서 [N, N]을 반환한다.
  - 동작은 다음과 같다. 난수는 전역 torch RNG(`torch.rand`, 장치는 `edge_index.device`)를 쓴다.
    1. self-loop을 제외한다.
    2. (u<v) 무방향 엣지로 정규화하고 unique를 취한다.
    3. 엣지마다 `rand ≥ drop_rate`인 것만 남긴다.
    4. 양방향으로 복원하고 self-loop을 추가한다.
    5. 행 차수 d로 D^-1/2 (A+I) D^-1/2를 계산한다(coalesce).
  - 확인:
    - `drop_rate=0`의 결과와 `data.adj`의 최대 |차이|는 8.94e-08이다(두 ds).
    - `drop_rate=0.2`의 결과는 대칭이고, 남은 비대각 비율은 Cora_full 0.7965, Amazon_clothing 0.7995다.
  - 참고: 두 데이터셋의 `edge_index`는 이미 대칭이고 중복과 self-loop이 없다(Cora_full E = 126,842, Amazon_clothing E = 183,360, 역방향 존재 비율 1.0). `data.adj`의 nnz는 E + N이다.
- `hw/repro_choi.sh`는 `PYTHONPATH=.`를 고정한다. 그래서 `.sh`의 인자를 그대로 옮겨 직접 실행했다(§2.1 명령).
- 복사: `out/exp004/exp004_<ds>_embeddings.pt` → `cache/bgrl/<ds>.pt`(md5 동일). `exp004_<ds>.{json,log}` → `results/hw/repro/zip_20261107_exp004/`.

### HW03a-s2
- run 목록 `results/hw/HW03a_s2.txt`: `nbr_bgrl` × 2 ds × seed 1212–1216. `GPUS=0 WORKERS=2 hw/run_queue.sh`로 12:20:17–12:27:11 UTC(6분 54초)에 돌렸다. run당 `elapsed_seconds`는 73.3–83.4 s다.
- `python -m hw.summarize --ref nbr_raw --ref naq_base`를 실행했다. `paired_nbr_raw_cos.md`는 이번에 다시 만들지 않았다(HW02 판본 그대로).
- 실행 전에 iislab4 커밋 `f8a6604`(HW03a-a: `pick`, 진단, collapse 표시)와 `0d9fbd7`을 pull했다. 비교 대상인 seio `naq_base`·`nbr_raw` 5 seed는 HW02(그 이전 코드)에서 돌린 run이다. 이 서버에서는 회귀 확인을 다시 하지 않았다(지시서에 해당 단계 없음).

### HW03a-s3
- 선행 조건 `HW03a-a [iislab4]` 커밋(`f8a6604`)은 s1 커밋 직전 pull로 이미 받았다. 대기 시간은 0이었다.
- `hw/purity.py`(새 파일)
  - 에피소드 풀은 `hw/run.py`와 같은 경로로 만든다. 순서는 seed → load_data → set_device → add_diffusion → 선택 임베딩 → ProtoNet 생성(학습 없음) → neighbor lookup → train_pool이다.
  - 선택은 `hw.neighbors.select_neighbors_pick`을 쓰고, 측정은 `hw.run.neighbor_diagnostics`를 쓴다. homophily에는 `hw.neighbors._eligible`을 쓴다. 셋 다 import만 했고 다시 구현하지 않았다.
  - 레이블 사용은 측정 전용이다(`# measurement-only label use`).
  - 설정은 arm `nbr_bgrl`의 프로토콜과 `bgrl_dir`(`cache/bgrl`)을 쓴다.
- 실행: `python -m hw.purity --dataset <ds>`(seed 1212, 2000 에피소드). 소요는 Cora_full 8.06 s, Amazon_clothing 9.91 s다.

### HW03a-s4 (이 커밋, 사람 지시로 추가된 단계)
- `hw/trainfree.py`(새 파일)
  - 특징 X, ÂX, Â²X 각각에 대해 미정규화와 L2 행 정규화를 평가한다(6가지).
  - 평가는 `hw.evaluate.evaluate`를 그대로 쓰고, 에피소드는 `hw.episodes.build_eval_pools`로 캐시에서 읽는다.
  - 프로토콜 인자는 arm `naq_base` 설정을 쓴다. `seed_everything(1212)`은 `load_data`의 클래스 셔플에만 영향이 있고, 평가 에피소드는 캐시에서 온다.
  - 레이블은 `evaluate` 안에서만 쓴다.
- 확인
  - Â가 `data.adj`와 같은지: `edge_index`로 (A+I)를 따로 만들어 D^-1/2 (A+I) D^-1/2를 계산했다. `data.adj`와의 최대 |차이|는 8.94e-08이다.
  - 두 ds의 `edge_index`는 대칭이고 중복과 self-loop이 없다(s1에서 확인).
  - 실행 전후 캐시 md5에 변화가 없다.
- 소요: Cora_full 1.86 s, Amazon_clothing 3.47 s.

## 5. 판단한 사항

1. exp004 출력 이름은 지시서가 예상한 것과 같다(`out/exp004/exp004_<ds>_embeddings.pt`).
2. 이 임베딩은 석휘 exp003의 `dropped_normalized_adj` 판본이 아니라 hw shim으로 만든 것이다. 이후 모든 문서에서 "BGRL(exp004 + hw shim)"으로 적는다.
3. `infonce_exp003` baseline의 `missing`은 그대로 두었다(사람 지시).

4. 커밋 직전 `git pull --rebase --autostash`를 쓰는 과정에서 s2 커밋 `9b1c214`에서 수정 파일(보고서·summary·paired)이 빠졌다. 이미 push한 뒤라 이력은 고치지 않고, 후속 커밋 `c34c8db`로 넣었다. s3부터는 `git stash -u` → `pull --rebase` → `stash pop` → `add` → `commit` 순서로 했다.
5. `paired_nbr_raw_cos.md`는 s2 집계에서 `--ref`로 다시 만들지 않아 HW02 판본 그대로다.

## 6. 제외·실패·붕괴 run

- 없다. s2의 10 run 모두 completed이고 붕괴는 0이다.

## 7. 소요 시간, 환경, HEAD

- 단계별 소요 (UTC, 멈춤·지시 대기 구간 제외)

| 단계 | 시간 | 비고 |
|---|---|---|
| s1 | 11:33–11:46 | exp004 실행 43 s |
| s2 | 12:20–12:27 | 10 run, 작업자 2개 |
| s3 | 12:27–12:29 | |
| s4 | 12:29–12:31 | |

- 환경: RTX 4080 SUPER(다른 사용자 프로세스 없음), conda `taskns`(torch 2.4.1+cu121).
- 커밋: s1 `92ac844`, s2 `9b1c214`·`c34c8db`, s3 `43a0d6e`, s4 `9c28bae`(시각 정정은 후속 커밋). 태그 없음(지시서).
