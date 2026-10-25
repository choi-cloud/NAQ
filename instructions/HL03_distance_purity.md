# HL03 — 먼 후보의 순도 판정 (G0, 학습 없음) + 거리 고정 순도 oracle 곡선 (G1, 학습) (auto)

작성 2026-10-25 · 이희태 · 브랜치 `hl/anchor` · 태그 `hl03-setup` / `hl03-g0` / `hl03-g1` · **서버 gpusystem, GPU 3번**
데이터셋: Amazon_clothing만. τ = 0.3.
배경: anchor 트랙은 HL02로 종결했다(`reports/HL_anchor_track_closure.md` 참고). 이번 스토리 C는 **query 쪽**이다. B_Choi는 먼 query(평균 X 순위 ≈ 750)를 써서 1-shot을 올렸지만 query 순도는 0.60 → 0.42로 떨어졌다. Choi의 oracle middle-K(먼 + 순도 100%)는 89.9였다. 질문: **먼 후보 중 같은 클래스를 레이블 없이 가려낼 판정기가 있는가(G0)**, 그리고 **거리를 고정했을 때 순도가 얼마나 올라야 성능이 따라오는가(G1)**.

레이블 사용 범위: G0에서는 판정기 채점에만, G1에서는 oracle query 선택에만 읽는다. 판정기 점수 계산에는 레이블을 넘기지 않는다.

---

## 0. 시작 — 새 서버 이전 점검과 기준선 (태그 `hl03-setup`)

HL03부터 새 서버(`gpusystem`)에서 실행한다. CLAUDE.md §2를 먼저 읽는다. **GPU는 별도 지시가 없는 한 3번(마지막)**: 모든 명령 앞에 `CUDA_VISIBLE_DEVICES=3`, 스크립트 인자는 `--device 0`.

**0-1. 리포 상태.** 브랜치 `hl/anchor`, `git status` clean, HEAD가 `a5e6712` 위의 "closure doc, CLAUDE.md rev4, HL03 instruction" 커밋인지 확인. 아니면 **정지**.

**0-2. 비추적 자산 확인.** 다음이 모두 있는지 확인하고, 없으면 **정지**:
- `data/` 의 Amazon_clothing 3개 파일
- `save/downstream_tasks/Amazon_clothing/5way_{1,5}shot.pkl` — `md5sum` 이 CLAUDE.md §9의 값과 같아야 함
- `save/fixed_episodes/Amazon_clothing/5way_{1,5}shot_8query_seed{1212..1216}/{valid,test}_episodes.json` (20개 파일; `choi_exps/exp026.load_fixed_episodes`로 각각 로드해 sha256 자체 검사 통과 확인)
- `results/hl/p0/Amazon_clothing/ours/tau_0.3/seed_1212/best_1shot.pt`, `results/hl/p0/Amazon_clothing/naq/baseline/seed_1212/best_1shot.pt`
- `results/hl/hl02/diag/node_table_tau0.3.csv`
`nvidia-smi -i 3`로 GPU 3의 메모리 여유, `df -h .`로 디스크 여유 20GB 이상 확인.

**0-3. conda 환경 `NAQ` 생성 (필수 — 새 서버에는 쓸 만한 환경이 없다).** 기존 환경을 찾거나 쓰지 않는다. `conda env list`만 기록해 둔다(`env/gpusystem_env_list.txt`).

허용 범위: `env/NAQ_conda.yml`, `env/requirements_pip.txt`, 아래 torch 계열 3개. 그 밖의 설치는 CLAUDE.md §7 멈춤 사유.
```bash
conda env create -f env/NAQ_conda.yml            # python 3.10 + 기본 라이브러리
conda activate NAQ
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install -r env/requirements_pip.txt
```
- `requirements_pip.txt` 설치가 실패하면 실패한 줄을 빼고 재시도하고, 뺀 목록을 보고서 ④에 적는다. `lightning*`, `wandb`, `fastapi`, `pygcl`, `ogb`, `dgl`, `sentry-sdk`, `uvicorn`, `starlette`, `starsessions`는 이 리포 실행에 필요 없으므로 빼도 된다. `numpy==1.26.3`, `scipy==1.15.3`, `scikit-learn==1.7.2`, `torch-geometric==2.8.0.post1`, `networkx`, `tqdm`은 필요하다.
- torch-scatter/sparse wheel이 그 인덱스에 없으면 그 두 줄은 건너뛴다(이 리포는 `torch_geometric.transforms.GDC`와 `torch.sparse.mm`만 쓰며 scatter/sparse 확장은 필요 없다). 건너뛴 사실을 ④에 적는다.

**짧은 동작 테스트(순서대로, 하나라도 실패하면 정지):**
1. `CUDA_VISIBLE_DEVICES=3 python hl/hl_envcheck.py` → `OK`. 출력을 `env/envcheck_gpusystem_NAQ.txt`에 저장.
2. 데이터 로드 테스트: `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD python -c "import sys; sys.path.insert(0,'choi_exps'); from hl.hl_common import load_everything; ..."` 형태로 Amazon_clothing을 `load_data` → `set_device` → `add_diffusion`까지 돌려 `Data.features.shape`와 `Diffusion_matrix.shape`를 출력(둘 다 24919가 들어가야 함). `hl_common`의 함수 시그니처는 파일을 읽고 맞춘다.
3. 고정 에피소드 로드: `choi_exps/exp026.load_fixed_episodes`로 seed 1212 1-shot valid/test를 로드해 개수(50/500) 출력.
4. 체크포인트 로드: 0-2의 `best_1shot.pt` 둘을 `torch.load(..., map_location="cpu", weights_only=True)`로 열어 `state["model"]` 키 수를 출력.
5. `conda list --export > env/gpusystem_NAQ_inventory.txt`.

0-4의 smoke run(NaQ seed 1212)이 이 환경의 최종 테스트다.

**0-4. 새 서버 기준선.** 구 서버 수치와 비교하지 않고, 이 서버의 기준선을 새로 잡는다.
- (a) smoke: NaQ seed 1212 1 run (`hl/hl000_run.py --arms naq --seeds 1212 --output_root results/hl/hl03/base`).
- (b) B_Choi 기준선: `hl/hl003_oracle_train.py --policy uniform` × seed 1212–1216 = 5 run (HL02의 `uniform`과 같은 스크립트·설정). 결과 `results/hl/hl03/base/uniform/seed_<seed>/`.
- 동시 최대 3개. run당 wall time과 GPU 최대 예약 메모리(`torch.cuda.max_memory_reserved`, 없으면 `nvidia-smi` 관찰값)를 기록.
- **표 T0** `T0_new_server_baseline.csv`: run × {1-shot test, 5-shot test, 1-shot valid, wall_seconds} + uniform 5-seed 평균±std. 구 서버 값(NaQ 1212: 80.64 / B_Choi uniform: 82.70 ± 0.29, 89.91)을 옆 열에 참고로.
- **게이트 G-S (정지 게이트)**: 새 서버 uniform 5-seed 1-shot 평균과 구 서버 82.70의 차 |Δ|. |Δ| ≤ 1.0 → 통과(기록만). |Δ| > 1.0 → **정지**하고 T0와 환경 점검 출력을 채팅에 붙인다.
- 이후 §2 G1의 비교 기준은 **이 새 서버 uniform 평균**이다.

커밋 `HL03: new-server setup and baseline`, 태그 `hl03-setup`.

**0-5.** 결과 루트 `results/hl/hl03/`, 표 디렉터리 `reports/HL03_tables/`. 재사용 자산: `hl/hl_common.py`, 0-2의 체크포인트 둘(인코더 판정기용, 구 서버에서 학습된 것 — 추론만 하므로 그대로 쓴다), 노드표.

## 1. G0 — 판정기 채점 `hl/hl005_diag_distance_purity.py` (학습 없음, 태그 `hl03-g0`)

### 1-1. 표본
- anchor 2,000개를 전체 노드에서 균등 추출(seed 1212, 비복원). anchor id 목록을 `results/hl/hl03/g0/anchors.npy`로 저장.
- 각 anchor의 후보 = 자기 자신을 뺀 전체 노드. X 코사인(`data.Diffusion_matrix`, 대각 0)으로 순위를 매기고 **거리 구간**으로 나눈다: `B1=[1,10]`, `B2=[11,50]`, `B3=[51,200]`, `B4=[201,1000]`, `B5=[1001,3000]`. **주 구간은 B4**(B_Choi query의 평균 순위가 여기 있다).

### 1-2. 점수 (레이블 없음)
각 (anchor a, 후보 q) 쌍에 대해 아래 점수를 계산한다. 모두 "클수록 같은 클래스일 것 같다".

| 이름 | 정의 |
|---|---|
| `x` | X 코사인 (구간 안 순위만 다시 매기는 기준선) |
| `ax` | ÂX 코사인 (`hl_common.structural_scores`의 z1) |
| `a2x` | Â²X 코사인 (z2) |
| `ppr` | a를 seed로 한 personalized PageRank 점수(α=0.15). 2,000개 anchor를 열로 둔 dense 행렬 E에 대해 `P ← (1−α)·Â_sym·P + α·E`를 30회 반복(Â_sym은 exp049와 같은 정규화 인접행렬). |
| `choi` | 석휘님 필터의 이진값: `ax ≥ 0.3 and a2x ≥ 0.3` |
| `mv` | 다중 뷰 순위 평균: anchor별로 후보 전체에서 `ax`, `a2x`, `ppr` 각각의 백분위 순위를 구해 셋을 평균 (0–1). X는 구간 정의에 쓰였으므로 넣지 않는다. |
| `enc_b` | B_Choi 체크포인트 인코더 임베딩의 코사인 |
| `enc_n` | NaQ 체크포인트 인코더 임베딩의 코사인 |
| `comb` | `mv`와 `enc_b`의 anchor별 백분위 순위 평균 |

인코더 임베딩: `meta.ProtoNet`과 같은 config(`GCN(F→512)`, `GCN(512→256)`, `linear(256→5)`)로 `GNNEncoder`를 만들고 체크포인트의 `model` state_dict를 로드, eval 모드로 `network(features, adj=adj)` 전체 출력.

메모리: 2,000 × 24,918 행렬을 점수별로 만들면 float32 약 200MB씩이다. 필요하면 anchor를 500개씩 나눠 계산한다.

### 1-3. 채점 (GT 사용)
구간 B1–B5 각각, 점수별로:
- **raw 순도**: 구간 안 후보 중 GT가 a와 같은 비율(점수와 무관, 구간의 기본값).
- **AUROC**: 구간 안 후보를 점수로 정렬했을 때 same-GT를 가르는 AUROC. anchor별로 계산해(구간 안에 양·음성이 모두 있는 anchor만) 평균, 그리고 쌍을 모두 모은 pooled AUROC. `choi`는 이진이므로 AUROC 대신 아래 R1만.
- **R1 (Choi와 같은 개수)**: anchor별로 구간 안에서 `choi`가 통과시킨 개수 k_a를 세고, 각 점수로 상위 k_a개를 골랐을 때의 순도. `choi` 자체의 순도와 나란히 놓는다. k_a = 0인 anchor는 제외하고 그 비율을 기록.
- **R2 (상위 절반)**: anchor별 구간 후보의 상위 50%를 골랐을 때의 순도.
- **query 충족률**: R1 규칙에서 k_a ≥ 10인 anchor 비율(= 그 구간만으로 query 10개를 채울 수 있는 anchor 비율).

출력: `G0_purity.csv`(구간 × 점수 × {raw, AUROC_mean, AUROC_pooled, R1_purity, R2_purity}), `G0_coverage.csv`(구간별 k_a 분포 분위수, k_a=0 비율, query 충족률), `G0_summary.json`.

### 1-4. 게이트 (판정은 이희태, Agent는 계산값만)
**G0 통과 기준(B4)**: `comb`의 R1 순도 ≥ max(0.55, `choi` R1 순도 + 0.10).

커밋 `HL03: G0 distance-purity diagnostics`, 태그 `hl03-g0`.

## 2. G1 — 거리 고정·순도 가변 oracle `hl/hl006_purity_oracle_train.py` (학습, 태그 `hl03-g1`)

**G0 결과와 무관하게 실행한다**(통과 기준을 해석하는 데 필요하다).

`hl/hl003_oracle_train.py`를 복사해 만든다(docstring에 출처). anchor는 **uniform**(HL02의 `uniform` 정책 그대로). 바꾸는 곳은 **query 선택 한 군데**: `build_structural_queries` 대신 아래 함수로 anchor마다 query 10개를 고른다.

`purity_oracle_queries(anchor, p)`:
- 후보 = anchor의 X 순위 B4 구간(201–1000위).
- 구간을 X 순위로 정렬한 뒤 GT same / GT diff 두 목록으로 나누고, 각 목록에서 **목록 중앙부터** 필요한 개수를 뽑는다: same에서 round(10·p)개, diff에서 나머지. (중앙에서 뽑아 두 목록의 평균 X 순위를 비슷하게 유지한다.)
- same이 부족하면 B5(1001–3000위) same에서 X 순위 앞쪽부터 채우고, 그래도 부족하면 diff로 채운다. 채운 개수를 기록.
- 진단: 선택 query의 평균 X 순위, 실제 순도, 보충 비율을 report.json `query_diagnostics`에.

arm: `p ∈ {0.4, 0.6, 0.8, 1.0}`. seed 1212–1216. 4 × 5 = 20 run, 동시 최대 3개. 러너는 `hl/hl000b_run_oracle.py`를 재사용하거나 같은 방식으로 `hl/hl000c_run_purity.py`를 만든다. 결과 `results/hl/hl03/g1/p_<p>/seed_<seed>/`.

비교 기준: **§0-4(b)의 새 서버 uniform(B_Choi) 5-seed 평균.** 구 서버 82.70 / 89.91은 참고 열로만.

**표 T6** `T6_purity_curve.csv`: p × {1-shot test 평균±std, 5-shot test 평균±std, Δ vs 새 서버 B_Choi, 실제 순도 평균, 평균 X 순위, 보충 비율}.

커밋 `HL03: G1 purity-at-distance oracle`, 태그 `hl03-g1`, push.

## 3. 사전 등록 (계산값만; 임계값은 모두 편의 목표치)

| 번호 | 예측 | 계산값 |
|---|---|---|
| HLP024 | raw 순도는 B1 → B5로 단조 감소 | |
| HLP025 | B4에서 `choi` R1 순도 > B4 raw 순도 (필터는 먼 영역에서도 순도를 올린다) | |
| HLP026 | B4에서 `enc_b` AUROC_mean ≥ 0.70 | |
| HLP027 | B4에서 `mv` AUROC_mean − `x` AUROC_mean ≥ 0.05 | |
| HLP028 | B4에서 `enc_n` AUROC_mean < `enc_b` AUROC_mean | |
| HLP029 | **G0 게이트**: B4에서 `comb` R1 순도 ≥ max(0.55, `choi` R1 + 0.10) | |
| HLP030 | B4에서 `choi`의 query 충족률 ≥ 0.5 | |
| HLP031 | T6: 1-shot test는 p에 대해 단조 증가 | |
| HLP032 | T6: p=1.0 − p=0.4 ≥ +4.0 (1-shot) | |
| HLP033 | T6: p=0.6 − B_Choi(새 서버) ≥ +1.0 (1-shot) — 먼 영역 순도를 0.6으로만 올려도 의미 있다 | |
| HLP034 | T6: p=0.4의 1-shot은 B_Choi(새 서버)와 ±0.8 안 (순도 0.4 근처의 B_Choi와 같은 지점) | |
| HLP035 | T0: 새 서버 uniform 5-seed 1-shot 평균은 구 서버 82.70과 ±0.5 안 | |

**판정 규칙(이희태 적용)**: HLP029 실패 → 스토리 C 중단. HLP029 통과이고 T6에서 G0가 달성한 순도 수준의 Δ가 +1.0 미만 → 스토리 C 중단(판정기가 올린 순도로는 성능이 안 따라옴).

## 4. 보고서 `reports/HL03_distance_purity.md`

CLAUDE.md §5 형식. 포함: 채택 환경과 점검 출력, 새 서버 run 시간·메모리, T0, G0 계산 시간, `G0_purity.csv`의 B4 행 전체와 B1–B5의 raw·`choi`·`comb` 행, `G0_coverage.csv`, T6, 사전 등록 표, 실패·경고, 산출물 경로. 끝나면 채팅에 채택 환경, T0, G0의 B4 표, G0_coverage의 B4 행, T6, 사전 등록 표를 그대로 붙인다.
