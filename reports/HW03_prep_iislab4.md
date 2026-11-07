# HW03-prep — iislab4 작업 환경 구축 + 기준선 브릿지

2026-11-07 · 서버 iislab4 (RTX 3090 24GB × 8) · 브랜치 `hl/hsan-weighting` · 태그 없음 · 사전 등록 없음
지시서: `~/wkspace/HW_drop/HW03_prep_iislab4.md` + 사용자 정정 1건(모든 run을 `HW_RESULTS_ROOT=results/hw_iislab4`로 실행, G-smoke 경로도 `results/hw_iislab4/...`).

## 1. 결론

1. 환경: python 3.9.25, torch 2.4.1+cu121 (CUDA 12.1), torch_geometric 2.6.1, numpy 1.26.4, sklearn 1.5.2, tqdm 4.70.0. seio와 일치. `pip list --format=freeze` 58줄 전체가 seio와 동일. `device_count() == 8`. G-env 통과.
2. G-smoke: `results/hw_iislab4/nbr_raw/Cora_full/seed1212/metrics.json` status = `completed`, test 78.15 (seio 78.16). 통과.
3. G-bridge: 20개 (arm, ds, seed) 전부 |Δ| ≤ 1.0. 최대 |Δ| = 0.92 (`naq_base` Cora_full seed 1213). 통과(판정은 사람).

## 2. 환경 표

| 항목 | seio (`taskns_versions.txt`) | iislab4 |
|---|---|---|
| GPU | RTX 4080 SUPER 16GB × 1 | RTX 3090 24GB × 8 |
| 드라이버 / 드라이버 CUDA | 미기록 | 580.178.04 / 13.0 |
| nvcc | 미기록 | 12.1 (cuda_12.1.r12.1/compiler.32688072_0) |
| python | 3.9.25 | 3.9.25 |
| torch | 2.4.1+cu121 | 2.4.1+cu121 |
| torch.version.cuda | 12.1 | 12.1 |
| torch_geometric | 2.6.1 | 2.6.1 |
| PyG 확장 (scatter/sparse/cluster/pyg-lib) | 없음 | 없음 |
| numpy | 1.26.4 | 1.26.4 |
| scikit-learn | 1.5.2 | 1.5.2 |
| tqdm | 4.70.0 | 4.70.0 |
| conda | — | 26.9.4 (miniforge, `/opt/miniforge3`) |
| env 경로 | `/home/h2tae/.conda/envs/taskns` | `/home/h2tae/.conda/envs/taskns` |

`taskns_versions.txt` 원문:
```
python 3.9.25
torch 2.4.1+cu121 cuda 12.1
torch_geometric 2.6.1
numpy 1.26.4 sklearn 1.5.2

torch==2.4.1+cu121
torch-geometric==2.6.1
```

`env/taskns_iislab4.yml`과 `taskns_seio.yml`의 차이(`prefix` 제외): `channels` 목록만 다르다 (seio: defaults, pkgs/main, pkgs/r / iislab4: defaults, conda-forge). 패키지·판본 목록은 동일.

## 3. 브릿지 표

test accuracy (%), 5-way 5-shot, 500 tasks. seio 값은 `results/hw/<arm>/<ds>/seed<S>/metrics.json`(HW02 실행분). Δ = iislab4 − seio. 값은 metrics.json에서 직접 계산(`summary.md`와 반올림 방식 차이로 소수점 둘째 자리 ±0.01).

| arm | ds | seed | best epoch (iislab4 / seio) | iislab4 | seio | Δ |
|---|---|---|---|---|---|---|
| naq_base | Cora_full | 1212 | 1760 / 2000 | 77.96 | 77.84 | +0.12 |
| naq_base | Cora_full | 1213 | 1920 / 1120 | 77.11 | 76.20 | +0.92 |
| naq_base | Cora_full | 1214 | 1600 / 1760 | 77.24 | 77.85 | −0.61 |
| naq_base | Cora_full | 1215 | 2000 / 2000 | 77.90 | 77.96 | −0.06 |
| naq_base | Cora_full | 1216 | 2000 / 2000 | 78.21 | 78.09 | +0.12 |
| **naq_base** | **Cora_full** | **평균** | | **77.68** | **77.59** | **+0.10** (max \|Δ\| 0.92) |
| naq_base | Amazon_clothing | 1212 | 1600 / 1600 | 90.60 | 90.60 | +0.00 |
| naq_base | Amazon_clothing | 1213 | 2000 / 1920 | 90.80 | 90.92 | −0.11 |
| naq_base | Amazon_clothing | 1214 | 2000 / 2000 | 91.11 | 91.13 | −0.02 |
| naq_base | Amazon_clothing | 1215 | 1920 / 2000 | 90.66 | 90.67 | −0.01 |
| naq_base | Amazon_clothing | 1216 | 1600 / 1440 | 90.67 | 90.68 | −0.00 |
| **naq_base** | **Amazon_clothing** | **평균** | | **90.77** | **90.80** | **−0.03** (max \|Δ\| 0.11) |
| nbr_raw | Cora_full | 1212 | 2000 / 2000 | 78.15 | 78.16 | −0.02 |
| nbr_raw | Cora_full | 1213 | 1920 / 1920 | 77.68 | 77.68 | +0.00 |
| nbr_raw | Cora_full | 1214 | 1440 / 2000 | 77.56 | 76.89 | +0.67 |
| nbr_raw | Cora_full | 1215 | 1920 / 1920 | 78.67 | 78.67 | −0.00 |
| nbr_raw | Cora_full | 1216 | 2000 / 2000 | 78.36 | 78.30 | +0.06 |
| **nbr_raw** | **Cora_full** | **평균** | | **78.08** | **77.94** | **+0.14** (max \|Δ\| 0.67) |
| nbr_raw | Amazon_clothing | 1212 | 1280 / 1440 | 90.89 | 90.69 | +0.20 |
| nbr_raw | Amazon_clothing | 1213 | 1760 / 1760 | 91.15 | 91.19 | −0.03 |
| nbr_raw | Amazon_clothing | 1214 | 2000 / 2000 | 91.26 | 91.25 | +0.01 |
| nbr_raw | Amazon_clothing | 1215 | 2000 / 1920 | 90.95 | 90.88 | +0.07 |
| nbr_raw | Amazon_clothing | 1216 | 960 / 1600 | 90.64 | 91.00 | −0.36 |
| **nbr_raw** | **Amazon_clothing** | **평균** | | **90.98** | **91.00** | **−0.02** (max \|Δ\| 0.36) |

전체 20개 중 최대 |Δ| = 0.92. |Δ| ≥ 0.30인 4개(naq_base Cora 1213·1214, nbr_raw Cora 1214, nbr_raw Amazon 1216)는 모두 best epoch가 두 서버에서 다르다. best epoch가 같은 10개의 |Δ| 최대는 0.12 (naq_base Cora 1216).

### 짝지은 Δ `nbr_raw − naq_base` (%p, 평균 ± 표준편차(ddof=0), 양수 seed k/5)

| 서버 | Amazon_clothing | Cora_full |
|---|---|---|
| iislab4 | +0.21 ± 0.14 (4/5) | +0.40 ± 0.24 (5/5) |
| seio (`results/hw/paired_naq_base.md`) | +0.20 ± 0.09 (5/5) | +0.35 ± 0.79 (4/5) |

seed별: iislab4 Cora +0.19 +0.57 +0.32 +0.77 +0.15 / seio Cora +0.32 +1.48 −0.96 +0.71 +0.21; iislab4 Amazon +0.29 +0.35 +0.15 +0.28 −0.03 / seio Amazon +0.09 +0.27 +0.13 +0.21 +0.33 (seed 1212→1216).

## 4. 수행 내용과 커밋

- §0 확인: `~/wkspace/NAQ` HEAD = `origin/hl/hsan-weighting` = `1b59332` (HW02-c), 태그 `hw02` 있음, 작업 트리 깨끗함. `data/`, `save/downstream_tasks/`, `HW_drop/env/` 3개 파일 존재.
- §1 현황: GPU 8장 모두 memory.used 0 MiB. `conda env list`: base만. md5 일치 — Cora_full `cfef6a69020acde861b554cd2e8ac9c7`, Amazon_clothing `ca3edaca02d541cbc0ee15a2122ef24a`. `data/`: Amazon_clothing_{network,test.mat,train.mat}, Amazon_eletronics_{network,test.mat,train.mat}, corafull/, dblp_{network,test.mat,train.mat}; 734M.
- §2 환경: 1차 `conda env create -f taskns_seio.yml` → conda 단계 완료, pip 단계 `CondaEnvException: Pip failed` (torch `+cu121` 로컬 판본). 2차: 1차에서 생긴 env(python 3.9.25)에 `pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121` → torch 2.4.1+cu121 및 nvidia-* 의존성 설치; 나머지 57개(`torch`, `pip`, `setuptools`, `wheel` 제외) `pip install -r` → 실패 패키지 없음. G-env 출력: `2.4.1+cu121 12.1 True 8 2.6.1 1.26.4 1.5.2`. `env/taskns_iislab4.yml` export.
- §3 코드 (커밋 1 `e8e928c` `HW03-prep: iislab4 env export, run_queue GPU assignment, HW_RESULTS_ROOT`):
  - `hw/run.py`, `hw/summarize.py`: `RESULTS_ROOT = Path(os.environ.get("HW_RESULTS_ROOT", "results/hw"))`. 미설정 시 기존과 같은 경로.
  - `hw/run_queue.sh`: `GPUS`(쉼표 목록, 기본 `0`), `WORKERS`(우선순위: 위치 인자 > 환경변수 > GPUS 개수). 작업자 i(1부터)가 `--device GPUS[(i−1) mod len]`. `status_of`가 `HW_RESULTS_ROOT`를 따른다. 큐 로그에 device 기록.
  - `env/taskns_iislab4.yml`.
- §4 smoke: `HW_RESULTS_ROOT=results/hw_iislab4 conda run -n taskns python -m hw.run --arm nbr_raw --dataset Cora_full --seed 1212 --device 0`, 2000 에피소드, rc=0, `completed`, best epoch 2000, val 86.70, test 78.15 ± 0.83.
- §5 브릿지: `HW_RESULTS_ROOT=results/hw_iislab4 GPUS=0,1,2,3,4 WORKERS=5 hw/run_queue.sh <목록 20줄>` → 19 run 실행 + 1 skip(smoke), 전부 attempt 1에서 `completed`, 285 s. `HW_RESULTS_ROOT=results/hw_iislab4 python -m hw.summarize --ref naq_base` → `results/hw_iislab4/{summary.csv, summary.md, paired_naq_base.md}`.
- 커밋 2 `1319e7d` `HW03-prep: iislab4 baseline bridge (naq_base, nbr_raw × 5 seeds)`: `results/hw_iislab4/**` (metrics.json 20, train.log 20, summary.csv, summary.md, paired_naq_base.md).
- 커밋 3: 이 보고서.
- push: 3회 모두 `git pull --rebase` 후 push, 원격 신규 커밋 없음, 충돌 없음.
- `results/hw/` 아래 파일: 쓰기·수정 없음 (`git status` 확인).

## 5. 판단한 사항

1. 2차 시도에서 `conda create -n taskns python=3.9.25`로 env를 다시 만들지 않고 1차에서 conda 단계까지 완료된 env를 그대로 썼다. 이 env의 conda 패키지는 `taskns_seio.yml`의 conda 목록과 판본이 같다(export 비교).
2. 설치 실패 패키지: 없음. `pip`, `setuptools`, `wheel`은 conda 단계에서 seio와 같은 판본(26.0.1, 80.9.0, 0.45.1)으로 설치되어 pip 목록에서 뺐다.
3. PyG 확장(`torch-scatter`, `torch-sparse`, `torch-cluster`, `pyg-lib`)은 `taskns_pip_seio.txt`에 없어 설치하지 않았다.
4. `run_queue.sh`의 `WORKERS` 기본값: 기존은 위치 인자 없으면 2, 변경 후는 `GPUS` 개수(기본 1). 지시서 §3 규정을 따랐다. seio 기존 호출 `hw/run_queue.sh <list> 2`는 위치 인자가 우선하므로 그대로 2개 작업자·device 0. `GPUS=0 WORKERS=2`도 같은 동작. 변경 후 작업자는 항상 `--device`를 넘기며, `GPUS=0`이면 `--device 0` = 설정 기본값 `device: 0`이라 `metrics.json`의 `config`는 바뀌지 않는다.
5. `run_queue.sh`의 `PYTHON` 기본 경로 `/home/h2tae/.conda/envs/taskns/bin/python`이 iislab4에서도 같은 위치라 고치지 않았다.
6. `.gitignore`가 `*.yml`과 `results/hw_iislab4/**/*.log`를 허용하지 않아 `env/taskns_iislab4.yml`과 `train.log` 20개를 `git add -f`로 추가했다. `.gitignore`는 고치지 않았다.
7. 지시서 제목·§5의 "10 run"과 §5 목록({naq_base, nbr_raw} × {Cora_full, Amazon_clothing} × 5 seed = 20)이 다르다. 목록대로 20 run(smoke 포함)을 돌렸다.
8. iislab4에 git 사용자 정보가 없어 리포 로컬 설정(`git config user.name/user.email`)에 기존 커밋과 같은 작성자(dragor0123)를 넣었다.
9. 사용자 정정에 따라 smoke도 `HW_RESULTS_ROOT=results/hw_iislab4`로 실행했고, 이를 위해 §3 코드 변경을 smoke 전에 끝냈다(커밋 1은 smoke 후).
10. 브릿지 run 목록 파일은 리포 밖(세션 scratchpad)에 두었다. 큐 로그는 `out/hw_queue/hw03prep_bridge.log`(추적 밖).

## 6. 제외·실패한 run

없음. 20/20 `completed`, 재시도 0.

## 7. 소요 시간, 환경, HEAD

- run당 시간(`elapsed_seconds`, 2000 에피소드 + 평가, RTX 3090):
  - 단독(GPU 1장에 run 1개): `nbr_raw` Cora_full 58.6 s (smoke). 명령 전체(`conda run` 포함) 81 s.
  - 큐(GPU 5장, GPU당 run 1개 동시): `naq_base` Cora_full 51–53 s, Amazon_clothing 66–68 s; `nbr_raw` Cora_full 57–59 s, Amazon_clothing 72–73 s.
  - 참고 seio(RTX 4080 SUPER, 작업자 2개가 GPU 1장 공유): 74–115 s.
- 단계별: 환경 생성 834 s(1차 637 s 포함), smoke 81 s, 브릿지 큐 285 s, 전체 약 25분(10:10–10:35 UTC).
- HEAD: 이 보고서 커밋. 직전 `1319e7d`.
