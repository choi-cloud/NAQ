# HW03a [seio] — BGRL 임베딩 생성 + `nbr_bgrl` 확인 run + 학습 없는 이웃 순도 측정 + 학습 없는 기준선

2026-11-07 · 서버 seio (RTX 4080 SUPER × 1) · 브랜치 `hl/hsan-weighting` · 지시서 `instructions/HW03a_seio.md` · CLAUDE.md 개정 4(드롭 폴더 판본) · 태그 없음

> 이 보고서는 단계마다 갱신한다. 현재: **s1 완료**, s2–s4 미실행.

## 1. 결론

- BGRL(exp004 + hw shim) 임베딩 두 개를 만들었다. Cora_full은 19,793 × 256(md5 `c2ef0115…`), Amazon_clothing은 24,919 × 256(md5 `bb1bd6d8…`)이고, G-bgrl을 통과했다.
- 최종 epoch(400)의 cosine Top-10 클래스 일치율은 Cora_full 0.57, Amazon_clothing 0.64다. 비교 대상은 다음과 같다.

| 데이터셋 | BGRL (epoch 400) | raw feature | 무작위 GCN |
|---|---|---|---|
| Cora_full | 0.57 | 0.38 | 0.55 |
| Amazon_clothing | 0.64 | 0.60 | 0.64 |

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

## 3. 게이트·사전 등록 대조표

| ID | 기준 | 측정값 | 통과 여부 |
|---|---|---|---|
| G-bgrl | (a)(b)(c), 두 ds | 모두 ✓ | 통과 |
| HWP024 | `nbr_bgrl − nbr_raw` ≥ 0, 두 ds (seio 5 seed) | s2 미실행 | — |
| HWP025 | `nbr_eq_support`: max(bgrl) ≥ max(raw) > contrast > random | s3 미실행 | — |

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

## 5. 판단한 사항

1. exp004 출력 이름은 지시서가 예상한 것과 같다(`out/exp004/exp004_<ds>_embeddings.pt`).
2. 이 임베딩은 석휘 exp003의 `dropped_normalized_adj` 판본이 아니라 hw shim으로 만든 것이다. 이후 모든 문서에서 "BGRL(exp004 + hw shim)"으로 적는다.
3. `infonce_exp003` baseline의 `missing`은 그대로 두었다(사람 지시).

## 6. 제외·실패·붕괴 run

- 없음(s1).

## 7. 소요 시간, 환경, HEAD

- s1: 11:33–11:46 UTC(멈춤·지시 대기 구간 제외).
- 환경: RTX 4080 SUPER, conda `taskns`(torch 2.4.1+cu121).
- HEAD: s1 커밋.
