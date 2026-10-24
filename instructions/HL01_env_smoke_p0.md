# HL01 — 환경 구축, Choi 스크립트 정리, smoke 재현, P0 기준선 (auto)

작성 2026-10-24 · 이희태 · 브랜치 `hl/anchor` · 태그 `hl01-env` / `hl01-smoke` / `hl01-p0`
목적: (1) 실행 환경을 만들고 (2) Choi 스크립트를 한 디렉터리로 옮기되 그대로 돌아가게 하고 (3) NaQ 기준선이 Choi 수치(1-shot test 80.66, seed 1212)와 비슷하게 재현되는지 확인하고 (4) P0 기준선 — NaQ와 B_Choi를 3 seed로 — 를 확보한다.
GT 레이블은 이 지시서에서 Choi 스크립트 내부 진단(`query_gt_match` 등)에만 쓰인다. 우리 코드는 레이블을 읽지 않는다.

---

## 0. 시작 상태 확인과 브랜치

1. `git status`가 clean이고 HEAD가 `1a20f23`(naq_exp)인지 확인. 아니면 **정지**.
2. `git checkout -b hl/anchor`.
3. `.gitignore` 맨 끝에 다음을 추가하고 커밋(`HL01: track md/yml/txt/csv`):
   ```
   # HL track: track docs, env specs, summaries
   !*.md
   !*.yml
   !*.txt
   !*.csv
   ```
4. `data/` 확인: `data/Amazon_clothing_network`, `Amazon_clothing_{train,test}.mat`, `dblp_*`, `Amazon_eletronics_*` 여섯 조각이 있는지. 없으면 **정지**.
5. `nvidia-smi`로 GPU 1개 이상, `df -h .`로 여유 20GB 이상 확인. 아니면 **정지**.

## 1. 실행 환경 점검 (기존 환경 우선, 설치 없음)

후보 순서: **`taskns`** → `ctx-teg` → (둘 다 실패 시) 새 환경 `NAQ`. 기존 환경에는 **아무것도 설치하지 않는다.**

**1-1. 점검 스크립트** — `hl/hl_envcheck.py`로 저장하고 각 후보 환경에서 실행한다.
```python
"""hl_envcheck: import/version check for running choi_exps/exp049.py. No installs."""
import sys
out = []
try:
    import torch, torch_geometric, numpy, scipy, sklearn, tqdm, pandas
    import torch_geometric.transforms as T
    import inspect
    out.append(f"python {sys.version.split()[0]}")
    out.append(f"torch {torch.__version__} cuda_build {torch.version.cuda} cuda_available {torch.cuda.is_available()}")
    out.append(f"pyg {torch_geometric.__version__} numpy {numpy.__version__} scipy {scipy.__version__} sklearn {sklearn.__version__}")
    assert "weights_only" in inspect.signature(torch.load).parameters, "torch>=1.13 required (weights_only)"
    assert torch.cuda.is_available(), "CUDA not available"
    T.GDC  # used by utils.graph_diffusion
    sparse = torch.sparse_coo_tensor(torch.tensor([[0, 1], [1, 0]]), torch.tensor([1.0, 1.0]), (2, 2)).coalesce()
    torch.sparse.mm(sparse, torch.ones(2, 3))  # used by exp048/049 mining
    out.append("OK")
except Exception as e:
    out.append(f"FAIL {type(e).__name__}: {e}")
print("\n".join(out))
```
```bash
conda activate taskns && python hl/hl_envcheck.py | tee env/envcheck_taskns.txt
```
- `OK`가 나오면 그 환경을 채택한다. `conda list --export > env/<ENV>_inventory.txt`. 이후 모든 명령은 이 환경에서. 커밋 `HL01: env check (<ENV>)`, 태그 `hl01-env`.
- `FAIL`이면 `ctx-teg`로 같은 점검. 그것도 `FAIL`이면 1-2로.

**1-2. 새 환경 `NAQ` (1-1이 모두 실패했을 때만)**
허용 범위: `env/NAQ_conda.yml`, `env/requirements_pip.txt`, 아래 torch 계열 3개. 그 외는 CLAUDE.md §7 멈춤 사유.
```bash
conda env create -f env/NAQ_conda.yml
conda activate NAQ
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install -r env/requirements_pip.txt
python hl/hl_envcheck.py | tee env/envcheck_NAQ.txt
```
- `requirements_pip.txt`에서 실패하는 줄은 빼고 재시도하되 뺀 목록을 보고서 ④에 적는다. `lightning*`, `wandb`, `fastapi`, `pygcl`, `ogb`는 빼도 된다. `numpy`, `scipy`, `scikit-learn`, `torch-geometric`, `networkx`, `pandas`, `tqdm`은 필요하다.
- 점검이 또 `FAIL`이면 **정지**.

**보고서 ①에 채택한 환경 이름과 점검 출력을 그대로 적는다.** §4 G1이 실패하면 환경 차이가 원인 후보이므로, 이 기록이 필요하다.

## 2. Choi 스크립트 정리 (내용 수정 없음)

```bash
mkdir -p choi_exps hl instructions reports results/hl out/hl
git mv exp0*.py exp0*.sh run_exp049.py shell.sh choi_exps/
```

- 루트의 NaQ 원본 10개(`data.py` 등)는 옮기지 않는다.
- 옮긴 뒤 import가 풀리는지 확인: `PYTHONPATH=$PWD python -c "import sys; sys.path.insert(0,'choi_exps'); import exp026, exp027, exp049; print('ok')"`.
- 커밋 `HL01: move Choi experiments into choi_exps/` (git이 rename으로 인식해야 한다 — `git status`에 `renamed:`가 보여야 함).

## 3. 러너 `hl/hl000_run.py`

`choi_exps/run_exp049.py`의 기능을 루트에서 돌 수 있게 다시 쓴 것. **`choi_exps/run_exp049.py`를 읽고 그 논리를 따르되** 다음을 바꾼다.

- subprocess 명령을 `[sys.executable, "-u", "choi_exps/exp026.py", ...]`, `[sys.executable, "-u", "choi_exps/exp049.py", ...]`로. 환경변수 `PYTHONPATH`에 리포 루트를 넣어 전달.
- 인자: `--datasets`(기본 `Amazon_clothing`), `--arms`(쉼표 구분, 각 항목은 `naq` 또는 `ours:<tau>`), `--seeds`(쉼표 구분), `--output_root`(기본 `results/hl/p0`), `--device`, `--epoch`(기본 2000).
- 고정 에피소드 생성은 run_exp049와 같은 규칙(1-shot·5-shot 둘 다, seed별 디렉터리). **이미 있으면 건너뛴다.**
- 각 run의 출력 디렉터리는 exp049 규칙 그대로(`<root>/<dataset>/<method>/<tau_name>/seed_<seed>/report.json`). report.json이 있으면 건너뛴다(재시작 가능).
- 끝나면 `summary.csv`를 run_exp049와 같은 열로 쓰되 `wall_seconds` 열을 추가.
- 한 run이 실패하면 `failure.txt`를 남기고 다음 run으로.

docstring 첫 줄: `"""hl000_run: runner derived from choi_exps/run_exp049.py; paths fixed for repo-root execution."""`

## 4. smoke — NaQ seed 1212 재현 (태그 `hl01-smoke`)

```bash
PYTHONPATH=$PWD nohup python -u hl/hl000_run.py --datasets Amazon_clothing --arms naq --seeds 1212 \
  --output_root results/hl/p0 > out/hl/HL01_smoke.log 2>&1 & echo $! > out/hl/HL01_smoke.pid
```

끝나면 `results/hl/p0/Amazon_clothing/naq/baseline/seed_1212/report.json`에서 다음을 읽어 보고서 표 T1에 적는다: 1-shot valid/test, 5-shot valid/test, best epoch(둘), `elapsed_seconds`, 고정 에피소드 생성 시간(로그).

**게이트 G1 (정지 게이트)**: 1-shot test와 80.66의 차 |Δ|.
- |Δ| ≤ 0.5 → 통과.
- 0.5 < |Δ| ≤ 1.5 → 통과하되 보고서 ④에 기록.
- |Δ| > 1.5 → **정지**. 로그 전체, `report.json`, `env/NAQ_installed.txt`를 채팅에 붙인다.

커밋 `HL01: smoke NaQ seed1212` (summary.csv 포함), 태그 `hl01-smoke`.

## 5. P0 — 기준선 3 seed (태그 `hl01-p0`)

Choi의 best τ는 모른다(summary.csv 미수령). 두 단계로 간다.

**5-1. τ 탐색 (seed 1212만)**
```bash
PYTHONPATH=$PWD nohup python -u hl/hl000_run.py --datasets Amazon_clothing \
  --arms ours:0.2,ours:0.3,ours:0.4 --seeds 1212 --output_root results/hl/p0 \
  > out/hl/HL01_tau.log 2>&1 & echo $! > out/hl/HL01_tau.pid
```
τ* = 세 run 중 **1-shot valid**가 가장 높은 τ(test는 보지 않는다). 동률이면 작은 τ.

**5-2. 3 seed**
```bash
PYTHONPATH=$PWD nohup python -u hl/hl000_run.py --datasets Amazon_clothing \
  --arms naq,ours:<τ*> --seeds 1212,1213,1214 --output_root results/hl/p0 \
  > out/hl/HL01_p0.log 2>&1 & echo $! > out/hl/HL01_p0.pid
```
(seed 1212의 두 run은 이미 있으므로 건너뛴다.)

smoke run 시간 × 남은 run 수(3 + 4 = 7)가 **8시간을 넘으면** 5-2를 seed 1212·1213의 2 seed로 줄이고 보고서 ④에 적는다.

**표 T2** (seed별 + 평균±표준편차): arm × {1-shot test, 5-shot test, 1-shot valid, best epoch}. **표 T3** (ours τ*의 진단, report.json `diagnostics`): `query_gt_match`, `original_gt_match`, `original_x_rank`, `coverage_without_fallback`, `fallback_occurrence_rate` — seed별.

커밋 `HL01: P0 baselines`, 태그 `hl01-p0`, push.

## 6. 사전 등록 (계산값만 채운다)

| 번호 | 예측 | 계산값 |
|---|---|---|
| HLP001 | smoke NaQ seed 1212의 1-shot test는 80.66 ± 0.5 안에 든다 | Δ = |
| HLP002 | τ 탐색에서 1-shot valid는 τ에 대해 단조가 아니다(최적이 0.2·0.4가 아닌 0.3) | τ별 valid = |
| HLP003 | 3-seed 평균에서 ours(τ*) 1-shot test > NaQ 1-shot test, 차이 ≥ +0.5%p | Δ_mean = , seed별 부호 = |
| HLP004 | 같은 비교에서 5-shot 차이는 1-shot 차이보다 작다 | Δ_1shot = , Δ_5shot = |
| HLP005 | ours τ*의 `query_gt_match`는 `original_gt_match`보다 높다 | 값 = |
| HLP006 | ours τ*의 `coverage_without_fallback` ≥ 0.8 | 값 = |

## 7. 보고서 `reports/HL01_env_smoke_p0.md`

CLAUDE.md §5 형식. 반드시 포함: 채택한 환경 이름과 `hl_envcheck.py` 출력(새 환경을 만들었다면 뺀 패키지 목록), run당 wall time, 표 T1–T3, 사전 등록 표, 실패·경고, 산출물 경로(report.json·체크포인트 `best_{1,5}shot.pt` 경로 — HL02의 임베딩 진단이 이 체크포인트를 쓴다).

끝나면 채팅에 T1·T2·T3와 사전 등록 표를 그대로 붙인다.