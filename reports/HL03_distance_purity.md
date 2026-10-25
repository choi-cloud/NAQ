# HL03 — 새 서버 기준선 + 먼 후보 순도 판정(G0) + 거리 고정 순도 oracle(G1)

작성 2026-10-25 · Agent · 지시서 `instructions/HL03_distance_purity.md` (auto) · 브랜치 `hl/anchor`
서버 `gpusystem`, GPU 3번만 사용(`CUDA_VISIBLE_DEVICES=3`, `--device 0`). 데이터셋 Amazon_clothing, τ = 0.3.

레이블 사용 범위: `hl005_diag_distance_purity.py`는 모든 점수 계산이 끝난 뒤 GT를 한 번 읽어 same-GT 채점에만 사용. `hl006_purity_oracle_train.py`는 GT를 oracle query 선택과 진단 통계에만 사용. `hl003`(기준선)은 anchor 진단 통계에만 사용.

## ① 실행 명령 · 환경 · commit

### 커밋·태그
| 태그 | commit | 메시지 |
|---|---|---|
| (사전) | `66e238e` | HL03: closure doc, CLAUDE.md rev4, HL03 instruction |
| `hl03-setup` | `4e5ea45` | HL03: new-server setup and baseline |
| `hl03-g0` | `63af38b` | HL03: G0 distance-purity diagnostics |
| `hl03-g1` | `faba27c` | HL03: G1 purity-at-distance oracle |

### 환경 `NAQ` (새로 생성, `/home/h2tae/anaconda3/envs/NAQ`)
```bash
conda env create -f env/NAQ_conda.yml
$NAQ/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
$NAQ/bin/python -m pip install torch-scatter==2.1.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
$NAQ/bin/python -m pip install -r env/requirements_pip.txt
```
- 세 단계 모두 exit 0. requirements에서 뺀 줄 없음. torch-scatter/sparse wheel 존재(`2.1.2+pt25cu121`, `0.6.18+pt25cu121`).
- conda 24.11.3, driver 580.95.05, GPU 3 = RTX 6000 Ada (49,140 MiB).
- `env/gpusystem_env_list.txt`(생성 전 `conda env list`: `base`만 있음), `env/envcheck_gpusystem_NAQ.txt`, `env/gpusystem_NAQ_inventory.txt`(`conda list -n NAQ --export`, 152줄).

환경 점검 출력(`CUDA_VISIBLE_DEVICES=3 python hl/hl_envcheck.py`):
```
python 3.10.21
torch 2.5.1+cu121 cuda_build 12.1 cuda_available True
pyg 2.8.0.post1 numpy 1.26.3 scipy 1.15.3 sklearn 1.7.2
OK
```

짧은 동작 테스트 2–4(`hl/hl_setup_check.py all`) 출력:
```
[0-2] fixed episodes sha256 self-check passed: 20 files
[test2] device=cuda:0 features=(24919, 9034) Diffusion_matrix=(24919, 24919)
[test3] seed1212 1-shot valid=50 test=500
[test4] results/hl/p0/Amazon_clothing/ours/tau_0.3/seed_1212/best_1shot.pt epoch=1920 model_keys=4
[test4] results/hl/p0/Amazon_clothing/naq/baseline/seed_1212/best_1shot.pt epoch=1600 model_keys=4
```

### §0-1, §0-2 점검
- 브랜치 `hl/anchor`, HEAD `66e238e`(부모 `a5e6712`). 추적 파일 변경 없음(④ 참고).
- `data/` Amazon_clothing 3개 파일 존재. md5 `5way_1shot.pkl` = `5f901727f81903cc12159d280be6915d`, `5way_5shot.pkl` = `ca3edaca02d541cbc0ee15a2122ef24a`(§9와 일치). 고정 에피소드 20개 파일 sha256 자체 검사 통과. 체크포인트 2개·노드표 존재.
- GPU 3: 0 MiB 사용, 프로세스 없음. 디스크 여유 420G.

### 실행 명령 (모두 리포 루트, `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD`, python = 환경 `NAQ`)
```bash
python hl/hl000_run.py --arms naq --seeds 1212 --output_root results/hl/hl03/base              # §0-4(a)
python hl/hl000b_run_oracle.py --policies uniform --jobs 2 --output_root results/hl/hl03/base  # §0-4(b)
python hl/hl005_diag_distance_purity.py --device 0                                             # §1 G0
python hl/hl000c_run_purity.py --jobs 3 --output_root results/hl/hl03/g1                       # §2 G1
python hl/hl_hl03_tables.py t0 | t6 | prereg                                                   # 표
```
새 스크립트: `hl/hl_setup_check.py`, `hl/hl005_diag_distance_purity.py`, `hl/hl006_purity_oracle_train.py`(hl003 복사), `hl/hl000c_run_purity.py`(hl000b 복사), `hl/hl_hl03_tables.py`.

## ② 표

### 새 서버 run 시간 · 메모리
| 항목 | 값 |
|---|---|
| NaQ smoke (단독) | wall 46.3 s, elapsed 33.66 s |
| B_Choi uniform, 2개 동시 (seed 1212–1215) | wall 94.1–102.1 s, elapsed 83.99–85.82 s |
| B_Choi uniform, 단독 (seed 1216) | wall 62.1 s, elapsed 47.31 s |
| G1 oracle, 3개 동시 (18 run) | wall 116.1–132.2 s |
| G1 oracle, 2개 동시 (마지막 2 run) | wall 92.1 s |
| run당 `torch.cuda.max_memory_reserved` (hl003·hl006 전 run) | 8.22 GB |
| GPU 3 전체 사용량 최대(nvidia-smi 2 s 간격, 3개 동시) | 26,779 MiB |
| run당 프로세스 사용량(nvidia-smi, 2개 동시 구간) | 8,932 MiB |
| G0 (`hl005`) `max_memory_reserved` | 10.74 GB |

### T0 `T0_new_server_baseline.csv`
| arm | seed | test 1-shot | test 5-shot | valid 1-shot | wall s | 구 서버 1-shot | 구 서버 5-shot |
|---|---|---|---|---|---|---|---|
| NaQ smoke | 1212 | 80.65 | 90.52 | 73.55 | 46.3 | 80.64 | |
| B_Choi uniform | 1212 | 83.2 | 90.44 | 76.55 | 98.1 | | |
| B_Choi uniform | 1213 | 82.71 | 89.84 | 76.1 | 96.1 | | |
| B_Choi uniform | 1214 | 82.47 | 90.17 | 77.9 | 102.1 | | |
| B_Choi uniform | 1215 | 82.76 | 90.14 | 78.3 | 94.1 | | |
| B_Choi uniform | 1216 | 82.4 | 88.96 | 76.85 | 62.1 | | |
| B_Choi uniform | mean ± std | 82.71 ± 0.31 | 89.91 ± 0.57 | 77.14 | 90.5 | 82.70 ± 0.29 | 89.91 |
| G-S | Δ = new − 82.70 | 0.01 | | | | | |

**게이트 G-S**: |Δ| = 0.01 ≤ 1.0 → 통과. B_Choi uniform 5 seed 평균 query GT 순도 0.4200, 평균 X 순위(exp049, z0 기준) 750.39, fallback 0.0339–0.0372.

### G0 계산 시간 (`G0_summary.json`, 누적 초)
load 4.76 → 점수 준비(verify·PPR·임베딩) 5.24 → 쌍 점수 5.63 → 채점 13.56. `verify_against_exp049` max_abs_err 1.19e-07.

### G0 `G0_purity.csv` — B4 행 전체
| band | score | raw | AUROC_mean | AUROC_pooled | R2_purity | R1_purity | R1_purity_anchor_mean | R1_selected_pairs | k_a_zero_frac |
|---|---|---|---|---|---|---|---|---|---|
| B4 | x | 0.1783 | 0.5943 | 0.6258 | 0.2143 | 0.2789 | 0.2589 | 378574 | 0.0375 |
| B4 | ax | 0.1783 | 0.7301 | 0.7453 | 0.2609 | 0.3480 | 0.4134 | 378574 | 0.0375 |
| B4 | a2x | 0.1783 | 0.7564 | 0.7684 | 0.2707 | 0.3589 | 0.4293 | 378574 | 0.0375 |
| B4 | ppr | 0.1783 | 0.6994 | 0.6980 | 0.2497 | 0.3438 | 0.4026 | 378574 | 0.0375 |
| B4 | choi | 0.1783 | | | | 0.3487 | 0.4140 | 378574 | 0.0375 |
| B4 | mv | 0.1783 | 0.7334 | 0.7476 | 0.2584 | 0.3527 | 0.4229 | 378574 | 0.0375 |
| B4 | enc_b | 0.1783 | 0.7786 | 0.8141 | 0.2779 | 0.3657 | 0.4411 | 378574 | 0.0375 |
| B4 | enc_n | 0.1783 | 0.7878 | 0.8217 | 0.2794 | 0.3706 | 0.4419 | 378574 | 0.0375 |
| B4 | comb | 0.1783 | 0.7709 | 0.8011 | 0.2740 | 0.3615 | 0.4352 | 378574 | 0.0375 |

### G0 `G0_purity.csv` — B1–B5의 raw · `choi` · `comb` (`x` 행 병기)
| band | score | raw | AUROC_mean | AUROC_pooled | R1_purity | R1_purity_anchor_mean | R2_purity |
|---|---|---|---|---|---|---|---|
| B1 | x | 0.6010 | 0.6633 | 0.7097 | 0.6523 | 0.6290 | 0.6557 |
| B1 | choi | 0.6010 | | | 0.6583 | 0.6413 | |
| B1 | comb | 0.6010 | 0.7097 | 0.7306 | 0.6598 | 0.6437 | 0.6707 |
| B2 | x | 0.4529 | 0.5602 | 0.6252 | 0.5289 | 0.4837 | 0.4749 |
| B2 | choi | 0.4529 | | | 0.5542 | 0.5253 | |
| B2 | comb | 0.4529 | 0.6948 | 0.7424 | 0.5558 | 0.5314 | 0.5345 |
| B3 | x | 0.3280 | 0.5581 | 0.5897 | 0.4151 | 0.3709 | 0.3609 |
| B3 | choi | 0.3280 | | | 0.4603 | 0.4600 | |
| B3 | comb | 0.3280 | 0.7283 | 0.7584 | 0.4624 | 0.4714 | 0.4366 |
| B4 | x | 0.1783 | 0.5943 | 0.6258 | 0.2789 | 0.2589 | 0.2143 |
| B4 | choi | 0.1783 | | | 0.3487 | 0.4140 | |
| B4 | comb | 0.1783 | 0.7709 | 0.8011 | 0.3615 | 0.4352 | 0.2740 |
| B5 | x | 0.0623 | 0.6091 | 0.6471 | 0.1539 | 0.1176 | 0.0816 |
| B5 | choi | 0.0623 | | | 0.2663 | 0.3751 | |
| B5 | comb | 0.0623 | 0.8398 | 0.8629 | 0.2913 | 0.4067 | 0.1127 |

열 정의: `R1_purity` = 선택된 쌍 전체를 모은 순도(Σsame/Σk_a, k_a=0 anchor는 자연히 제외), `R1_purity_anchor_mean` = k_a>0 anchor별 순도의 평균. `AUROC_mean`은 구간 안에 양·음성이 모두 있는 anchor만(개수는 아래 표 마지막 열).

### G0 `G0_coverage.csv`
| band | size | k_a q0 | q10 | q25 | q50 | q75 | q90 | q100 | mean | k_a=0 비율 | query 충족률 | 양·음성 모두 있는 anchor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 10 | 0 | 5 | 8 | 10 | 10 | 10 | 10 | 8.6475 | 0.0105 | 0.645 | 1294 |
| B2 | 40 | 0 | 4 | 18 | 35 | 40 | 40 | 40 | 28.1 | 0.034 | 0.8405 | 1765 |
| B3 | 150 | 0 | 4 | 23 | 76 | 131 | 145 | 150 | 76.3075 | 0.0415 | 0.846 | 1953 |
| B4 | 800 | 0 | 3 | 19 | 99.5 | 322.25 | 529.1 | 782 | 189.287 | 0.0375 | 0.8215 | 1992 |
| B5 | 2000 | 0 | 1 | 5 | 29 | 135.5 | 453.4 | 1283 | 131.0495 | 0.088 | 0.675 | 1980 |

### T6 `T6_purity_curve.csv` (기준 = 새 서버 B_Choi uniform 82.71 / 89.91; 구 서버 82.70 / 89.91은 참고)
| p | n | 1-shot test | 5-shot test | Δ1 vs B_Choi(새) | Δ5 vs B_Choi(새) | 실제 순도 | 평균 X 순위 | 보충 query 비율 | 보충 anchor 비율 |
|---|---|---|---|---|---|---|---|---|---|
| 0.4 | 5 | 83.92 ± 0.40 | 91.12 ± 0.24 | +1.21 | +1.21 | 0.3981 | 581.5 | 0.0038 | 0.0149 |
| 0.6 | 5 | 86.73 ± 0.60 | 92.87 ± 0.19 | +4.02 | +2.96 | 0.5964 | 563.5 | 0.0082 | 0.0244 |
| 0.8 | 5 | 88.83 ± 0.33 | 93.92 ± 0.18 | +6.12 | +4.01 | 0.7942 | 547.2 | 0.0143 | 0.0332 |
| 1.0 | 5 | 89.42 ± 0.26 | 94.17 ± 0.20 | +6.71 | +4.26 | 0.9916 | 532.8 | 0.0227 | 0.0444 |

seed별 값은 `T6b_purity_per_seed.csv`. 보충 query 비율 = (B5 same + diff 대체) / 전체 query; 보충 anchor 비율 = 보충이 한 번이라도 일어난 anchor 출현 비율. 평균 X 순위는 `data.Diffusion_matrix` 기준(④ 참고).

## ③ 사전 등록 대조표 (`prereg_HL03.csv`, 계산값만)
| 번호 | 예측 | 계산값 |
|---|---|---|
| HLP024 | raw 순도는 B1 → B5로 단조 감소 | raw B1–B5 = 0.601 / 0.4529 / 0.328 / 0.1783 / 0.0623; 단조 감소 True |
| HLP025 | B4에서 `choi` R1 순도 > B4 raw 순도 | choi R1 0.3487 vs raw 0.1783 |
| HLP026 | B4에서 `enc_b` AUROC_mean ≥ 0.70 | enc_b AUROC_mean 0.7786 |
| HLP027 | B4에서 `mv` AUROC_mean − `x` AUROC_mean ≥ 0.05 | mv 0.7334 − x 0.5943 = 0.1391 |
| HLP028 | B4에서 `enc_n` AUROC_mean < `enc_b` AUROC_mean | enc_n 0.7878 vs enc_b 0.7786 |
| HLP029 | G0 게이트: B4 `comb` R1 ≥ max(0.55, `choi` R1 + 0.10) | comb R1 0.3615 vs 기준 0.5500 |
| HLP030 | B4에서 `choi` query 충족률 ≥ 0.5 | query_fill_rate 0.8215 |
| HLP031 | T6: 1-shot test는 p에 대해 단조 증가 | 83.92 / 86.73 / 88.83 / 89.42; 단조 증가 True |
| HLP032 | T6: p=1.0 − p=0.4 ≥ +4.0 (1-shot) | 89.42 − 83.92 = +5.50 |
| HLP033 | T6: p=0.6 − B_Choi(새 서버) ≥ +1.0 (1-shot) | 86.73 − 82.71 = +4.02 |
| HLP034 | T6: p=0.4의 1-shot은 B_Choi(새 서버)와 ±0.8 안 | 83.92 − 82.71 = +1.21 |
| HLP035 | T0: 새 서버 uniform 5-seed 1-shot 평균은 82.70과 ±0.5 안 | 82.71 − 82.70 = +0.01 |

## ④ 실패 · 경고 · 이상
- **git identity 없음**: 새 서버에 git user 설정이 없어 첫 커밋이 실패했다. 이전 커밋과 같은 이름·메일(`dragor0123`, `blader137@gmail.com`)을 **리포 로컬** config로 설정했다(전역 설정은 건드리지 않음).
- **§0-1 git status**: 추적 파일 변경은 없었고, 비추적 항목은 rsync된 `results/hl/hl02/`(노드표 csv, `!*.csv` 규칙 때문에 무시되지 않음)뿐이었다. 이를 지시서가 둔 자산으로 보고 진행했다.
- **환경**: `conda activate` 대신 환경의 python을 전체 경로로 호출했다. torch-scatter/sparse 설치 시 numpy 2.2.6이 함께 설치됐고, 이어진 requirements 설치가 numpy 1.26.3으로 되돌렸다(최종 1.26.3).
- **동시 실행 수**: 기준선(§0-4b)은 기존 `hl000b`(최대 2개로 제한된 코드)를 그대로 써서 2개 동시로 돌렸다. smoke와 기준선은 `summary.csv` 덮어쓰기를 피하려고 순차 실행했고 smoke 결과는 `results/hl/hl03/base/summary_smoke.csv`로 복사해 두었다. G1은 새 러너 `hl000c`(최대 3개)로 3개 동시.
- 기준선 러너 첫 실행이 셸 변수 오류(Exit 126, run 시작 전)로 실패해 같은 명령으로 다시 띄웠다. 결과에 영향 없음.
- NaQ smoke의 GPU 메모리: exp049는 `max_memory_reserved`를 기록하지 않고, smoke 중에는 nvidia-smi 샘플러가 돌지 않아 값이 없다.
- G1 시작 전 `hl006`을 `--epoch 20`으로 scratchpad에 한 번 돌려 동작을 확인했다(결과 디렉터리 밖, 표에 미포함).
- **G0 구현상 선택**(지시서에 정의가 없던 부분):
  - anchor는 전체 24,919 노드(train·test 클래스 모두)에서 `np.random.default_rng(1212).choice(n, 2000, replace=False)`.
  - X 순위: `data.Diffusion_matrix` 행에서 자기 자신을 −inf로 두고 내림차순 stable 정렬(동률은 노드 id 순).
  - 백분위: 자기 자신을 뺀 24,918 후보 안의 평균 순위(동률 평균) / 24,918. `mv`는 ax·a2x·ppr 백분위 평균, `comb`는 (후보 전체에서 `mv`의 백분위, `enc_b` 백분위)의 평균.
  - R1/R2의 동률은 X 순위 순으로 끊음. `x` 점수의 R1 = 구간 안 X 상위 k_a개. `choi`는 R1만 계산(R2·AUROC 빈칸).
  - R1 순도는 쌍을 모은 값(주 열)과 anchor별 평균(보조 열) 두 가지를 기록. HLP025·HLP029 계산값은 주 열(`R1_purity`) 기준.
  - `G0_summary.json`의 `seconds`는 누적 시간.
- **G1 구현상 선택**:
  - X 순위는 G0와 같은 `data.Diffusion_matrix` 기준. B_Choi의 750.39는 exp049의 z0(행 정규화 X) 순위 기준이라 두 값의 정의가 같지 않을 수 있다(둘 다 X 코사인).
  - query는 exp049처럼 anchor 노드별로 한 번 정하고 캐시한다(같은 anchor = 같은 query).
  - anchor는 hl003 `uniform`과 같은 코드 경로·난수 상태(seed 1212 `original_gt_match` 0.59771이 기준선과 같음).
  - p=1.0의 실제 순도 0.9916: B4+B5 same이 10개가 안 되는 anchor에서 diff로 채운 결과(diff 대체 query 비율은 report.json `diff_fill_query_frac`).
- 표 스크립트 이름은 레지스트리 번호를 쓰지 않도록 `hl/hl_hl03_tables.py`로 했다.
- 한 run 실패 없음. push 실패 없음.

## ⑤ 산출물 경로
- 표(커밋): `reports/HL03_tables/{T0_new_server_baseline, G0_purity, G0_coverage, T6_purity_curve, T6b_purity_per_seed, prereg_HL03}.csv`
- 환경(커밋): `env/envcheck_gpusystem_NAQ.txt`, `env/gpusystem_NAQ_inventory.txt`, `env/gpusystem_env_list.txt`
- 결과(미추적): `results/hl/hl03/base/`(smoke `Amazon_clothing/naq/baseline/seed_1212/`, `uniform/seed_<seed>/`, `summary.csv`, `summary_smoke.csv`), `results/hl/hl03/g0/`(`anchors.npy`, `G0_purity.csv`, `G0_coverage.csv`, `G0_summary.json`), `results/hl/hl03/g1/p_<p>/seed_<seed>/`(report.json, best_{1,5}shot.pt), `results/hl/hl03/g1/summary.csv`
- 로그(미추적): `out/hl/hl03_*.log`, `out/hl/hl03_gpu3_{mem,apps}.csv`, `results/hl/hl03/{base,g1}/logs/`
