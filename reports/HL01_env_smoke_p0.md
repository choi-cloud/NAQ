# HL01 보고서 — 환경 구축, Choi 스크립트 정리, smoke 재현, P0 기준선

지시서: `instructions/HL01_env_smoke_p0.md` · 브랜치 `hl/anchor` · 태그 `hl01-env` / `hl01-smoke` / `hl01-p0` · 2026-10-24

## ① 실행 명령·환경·commit

**채택 환경: `taskns`** (`/home/h2tae/.conda/envs/taskns`). 설치·업그레이드·삭제는 하지 않았다.

`hl/hl_envcheck.py` 출력 (`env/envcheck_taskns.txt`, pandas import를 뺀 수정판, ④ 참조):
```
python 3.9.25
torch 2.4.1+cu121 cuda_build 12.1 cuda_available True
pyg 2.6.1 numpy 1.26.4 scipy 1.13.1 sklearn 1.5.2
OK
```
- 패키지 목록: `env/taskns_inventory.txt` (`conda list --export`).
- `ctx-teg` 점검 출력 (`env/envcheck_ctx-teg.txt`): `FAIL ModuleNotFoundError: No module named 'torch'`
- `taskns` 첫 점검 출력 (pandas 포함 원본 스크립트): `FAIL ModuleNotFoundError: No module named 'pandas'` (파일은 재점검 출력으로 덮어씀)
- GPU: NVIDIA GeForce RTX 4080 SUPER, driver 550.163.01, `--device 0`. 디스크 여유 337G.

commit (브랜치 `hl/anchor`, 분기점 `1a20f23`):
| commit | 내용 |
|---|---|
| `66d276d` | HL01: track md/yml/txt/csv |
| `840d2e7` | HL01: env check (taskns) — 태그 `hl01-env` |
| `6a06ce1` | HL01: move Choi experiments into choi_exps/ (100개 파일 `renamed`) |
| `b0ee4e1` | HL01: add repo-root runner hl/hl000_run.py |
| `9f23c85` | HL01: smoke NaQ seed1212 — 태그 `hl01-smoke` |
| (태그 `hl01-p0`) | HL01: P0 baselines (이 보고서 포함) |

모든 학습 run은 `b0ee4e1`/`9f23c85` 시점의 동일한 코드(`hl/hl000_run.py`, `choi_exps/`)로 실행했다. 실행 명령(리포 루트, `conda activate taskns`):
```bash
# smoke
PYTHONPATH=$PWD nohup python -u hl/hl000_run.py --datasets Amazon_clothing --arms naq --seeds 1212 \
  --output_root results/hl/p0 > out/hl/HL01_smoke.log 2>&1 & echo $! > out/hl/HL01_smoke.pid
# 5-1 τ 탐색
PYTHONPATH=$PWD nohup python -u hl/hl000_run.py --datasets Amazon_clothing \
  --arms ours:0.2,ours:0.3,ours:0.4 --seeds 1212 --output_root results/hl/p0 \
  > out/hl/HL01_tau.log 2>&1 & echo $! > out/hl/HL01_tau.pid
# 5-2 3 seed (τ* = 0.3)
PYTHONPATH=$PWD nohup python -u hl/hl000_run.py --datasets Amazon_clothing \
  --arms naq,ours:0.3 --seeds 1212,1213,1214 --output_root results/hl/p0 \
  > out/hl/HL01_p0.log 2>&1 & echo $! > out/hl/HL01_p0.pid
# 표 생성
python hl/hl001_hl01_tables.py --tau_star 0.3
```

## ② 표

모든 수치는 스크립트가 쓴 csv를 그대로 옮긴 것이다(정확도 단위 %, `hl/hl001_hl01_tables.py`가 report.json에서 생성).

### T1 — smoke NaQ seed 1212 (`reports/HL01_tables/T1_smoke.csv`)
```
run,valid_1shot,test_1shot,valid_5shot,test_5shot,best_epoch_1shot,best_epoch_5shot,elapsed_seconds,choi_ref_1shot_test,delta_1shot_test
naq/baseline/seed_1212,73.4,80.64,87.65,90.51,1600,1600,40.07,80.66,-0.02
```
- report.json 원값: 1-shot test `0.8063500106930733` (Δ = 80.635 − 80.66 = −0.025).
- 고정 에피소드 생성 시간 (`out/hl/HL01_smoke.log`): seed 1212 1-shot 4.7s, 5-shot 5.0s. (`out/hl/HL01_p0.log`: seed 1213 1-shot 4.1s·5-shot 4.1s, seed 1214 1-shot 4.0s·5-shot 4.0s.)
- **G1: |Δ| = 0.025 ≤ 0.5 → 통과.**

### τ 탐색 (seed 1212, `reports/HL01_tables/tau_search.csv`)
```
tau,valid_1shot,valid_5shot,best_epoch_1shot
0.2,76.25,88.25,1440
0.3,76.6,87.95,1920
0.4,74.5,87.5,1920
```
τ* = 0.3 (1-shot valid 최대, test는 보지 않음).

### T2 — P0 3 seed (`reports/HL01_tables/T2_p0.csv`)
```
arm,seed,status,test_1shot,test_5shot,valid_1shot,best_epoch_1shot,best_epoch_5shot
naq/baseline,1212,complete,80.64,90.51,73.4,1600,1600
naq/baseline,1213,complete,80.88,90.68,73.25,2000,1920
naq/baseline,1214,complete,81.44,91.16,74.85,2000,2000
naq/baseline,mean,,80.99,90.78,73.83,1866.67,1840
naq/baseline,std,,0.41,0.34,0.88,230.94,211.66
ours/tau_0.3,1212,complete,83.2,90.46,76.6,1920,1920
ours/tau_0.3,1213,complete,82.77,89.9,76.1,2000,1920
ours/tau_0.3,1214,complete,82.45,90.17,77.85,1920,2000
ours/tau_0.3,mean,,82.81,90.18,76.85,1946.67,1946.67
ours/tau_0.3,std,,0.38,0.28,0.9,46.19,46.19
```
std는 표본 표준편차(n−1).

### T3 — ours τ*=0.3 진단 (`reports/HL01_tables/T3_diagnostics.csv`)
```
seed,query_gt_match,original_gt_match,original_x_rank,coverage_without_fallback,fallback_occurrence_rate
1212,0.41924,0.59771,735.65395,0.9649251004749726,0.0345
1213,0.42127,0.60123,757.5946999999999,0.9656428311278378,0.0339
1214,0.41482,0.60101,750.4647700000002,0.964571705896627,0.0354
```

### run당 wall time (runner 측정, `results/hl/p0/summary.csv`·`summary_tau.csv`의 `wall_seconds`)
```
dataset,shot,method,tau,seed,wall_seconds
Amazon_clothing,1,naq,,1212,41.7
Amazon_clothing,1,naq,,1213,42.0
Amazon_clothing,1,naq,,1214,42.0
Amazon_clothing,1,ours,0.2,1212,50.6
Amazon_clothing,1,ours,0.3,1212,50.9
Amazon_clothing,1,ours,0.3,1213,50.7
Amazon_clothing,1,ours,0.3,1214,51.2
Amazon_clothing,1,ours,0.4,1212,51.7
```
(같은 run의 shot=5 행은 같은 값. smoke 41.7s × 남은 7 run은 8시간보다 훨씬 짧아 5-2는 3 seed 그대로 실행.)

## ③ 사전 등록 대조표 (계산값만, `reports/HL01_tables/prereg.csv`)

| 번호 | 예측 | 계산값 |
|---|---|---|
| HLP001 | smoke NaQ seed 1212의 1-shot test는 80.66 ± 0.5 안에 든다 | Δ = −0.02 (원값 80.635 − 80.66 = −0.025) |
| HLP002 | τ 탐색에서 1-shot valid는 τ에 대해 단조가 아니다(최적이 0.2·0.4가 아닌 0.3) | τ별 valid = 0.2: 76.25, 0.3: 76.6, 0.4: 74.5 |
| HLP003 | 3-seed 평균에서 ours(τ*) 1-shot test > NaQ 1-shot test, 차이 ≥ +0.5%p | Δ_mean = +1.82, seed별 부호 = 1212: + (2.56), 1213: + (1.89), 1214: + (1.01) |
| HLP004 | 같은 비교에서 5-shot 차이는 1-shot 차이보다 작다 | Δ_1shot = +1.82, Δ_5shot = −0.61 (seed별 5-shot: 1212 −0.05, 1213 −0.78, 1214 −0.99) |
| HLP005 | ours τ*의 `query_gt_match`는 `original_gt_match`보다 높다 | 값 = query 0.41924 / 0.42127 / 0.41482 (평균 0.41844), original 0.59771 / 0.60123 / 0.60101 (평균 0.59998) — seed 1212/1213/1214 |
| HLP006 | ours τ*의 `coverage_without_fallback` ≥ 0.8 | 값 = 0.9649 / 0.9656 / 0.9646 (평균 0.9650) — seed 1212/1213/1214 |

레이블 사용 범위: GT 레이블은 Choi `exp049` 내부 진단(`query_gt_match`, `original_gt_match`)에서만 읽혔다. `hl/` 코드는 레이블을 읽지 않는다(`hl001`은 report.json의 진단값만 읽음).

## ④ 실패·경고·이상

- **envcheck 수정(이희태 채팅 지시)**: `hl/hl_envcheck.py` 첫 import 줄에서 `pandas`를 뺐다(리포 어디에서도 import되지 않음). 수정 후 taskns 재점검 결과 OK, taskns를 채택했다.
- **새 환경 NAQ 일부 생성**: taskns(pandas 없음)·ctx-teg(torch 없음)가 원본 점검에서 실패해, §1-2에 따라 `conda env create -f env/NAQ_conda.yml`를 실행했다(exit 0, 로그 `out/hl/HL01_env_create.log`). 다음 단계(torch pip 설치)는 사용자가 중단시켜 실행하지 않았다. 보고서 작성 시점에 `conda env list`에는 `NAQ`가 없다(Agent는 삭제하지 않았다). NAQ 환경은 쓰지 않았다.
- `env/envcheck_taskns.txt`는 재점검 출력으로 덮어써져 첫 점검의 FAIL 출력은 ①에만 남아 있다.
- 경고: 모든 exp026/exp049 로그에 `utils.py:83: UserWarning: torch.sparse.SparseTensor(...) is deprecated` (14회, `results/hl/p0/logs/*.log`). 오류·Traceback은 없다. 실패한 run은 없다.
- 부작용: `choi_exps/exp026.py`가 고정 경로 `results/Amazon_clothing/exp026/result.txt`에 에피소드 sha256을 쓴다(호출마다 덮어씀, 현재 마지막 생성분인 seed 1214 5-shot). 커밋하지 않았다.
- `results/hl/p0/summary.csv`는 runner를 호출할 때마다 그 호출의 arm·seed로 덮어써진다(run_exp049와 같은 동작). 단계별 사본을 남겼다: `summary_smoke.csv`(smoke), `summary_tau.csv`(τ 탐색), `summary.csv`(5-2 최종: naq·ours 0.3 × 3 seed).
- runner가 run 디렉터리마다 `wall_seconds.txt`를 쓴다(`.txt` 추적 규칙에 걸리지만 커밋하지 않았다).
- 지시서 §4 정지 시 첨부 대상인 `env/NAQ_installed.txt`는 G1이 통과했고 NAQ 환경도 쓰지 않아 만들지 않았다.
- G1: |Δ| = 0.025, 통과(≤ 0.5). 기록할 범위 이탈 없음.

## ⑤ 산출물 경로

- 러너: `hl/hl000_run.py`; 표 생성: `hl/hl001_hl01_tables.py`; 환경 점검: `hl/hl_envcheck.py`
- 표 csv: `reports/HL01_tables/{T1_smoke,tau_search,T2_p0,T3_diagnostics,prereg}.csv`
- 요약 csv: `results/hl/p0/summary.csv`, `results/hl/p0/summary_smoke.csv`, `results/hl/p0/summary_tau.csv`
- report.json 및 체크포인트 (각 디렉터리에 `report.json`, `best_1shot.pt`, `best_5shot.pt`):
  - `results/hl/p0/Amazon_clothing/naq/baseline/seed_{1212,1213,1214}/`
  - `results/hl/p0/Amazon_clothing/ours/tau_0.3/seed_{1212,1213,1214}/`
  - `results/hl/p0/Amazon_clothing/ours/tau_0.2/seed_1212/`, `results/hl/p0/Amazon_clothing/ours/tau_0.4/seed_1212/`
- 고정 에피소드: `save/fixed_episodes/Amazon_clothing/5way_{1,5}shot_8query_seed{1212,1213,1214}/{valid,test}_episodes.json`
- 에피소드 메타 json: `results/hl/p0/episodes_Amazon_clothing_{1,5}shot_seed{1212,1213,1214}.json`
- 로그: `out/hl/HL01_{smoke,tau,p0}.log`, `out/hl/HL01_env_create.log`, run별 `results/hl/p0/logs/*.log`; PID: `out/hl/HL01_{smoke,tau,p0}.pid`
- 환경 기록: `env/envcheck_taskns.txt`, `env/envcheck_ctx-teg.txt`, `env/taskns_inventory.txt`
