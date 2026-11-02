# CLAUDE.md — NAQ 리포 위 이웃 분포 패턴 연구 (ND 트랙)

개정 2 · 2026-11-02 · 작성 이희태 (개정 1: 11-02)
이 파일은 Agent가 매 세션 처음에 읽는 상태 파일이다. 수정은 이희태만 한다. Agent는 이 파일을 고치지 않는다.
이 파일은 HL 트랙의 CLAUDE.md(개정 8, `hl/anchor` 브랜치)를 ND 트랙용으로 다시 쓴 것이다. HL 트랙의 야간 모드·스토리 D 상태는 이 브랜치에 적용되지 않는다.
개정 2 변경: §0 ND01 결과·ND02 질문, §2 GPU 메모리 실측, §5 레지스트리 ND02·스크립트 번호, §8 ND01 확정 사실, §9 ND01 코드 재사용.

## 0. 이 리포가 무엇인가

- 원격: `choi-cloud/NAQ`. 원본은 NaQ(ICML'24, `JhngJng/NaQ-PyTorch`) 공식 코드이고, 그 위에 공동연구자 Choi가 `exp001`–`exp049`를 쌓았으며(`choi_exps/`), 그 위에 HL 트랙(HL01–HL10, `hl/`)이 쌓였다.
- **작업 브랜치: `nd/anchor-vote`** (`hl/anchor` HEAD `f86f817`에서 분기). `hl/anchor`·`naq_exp`에는 아무것도 쓰지 않는다. `hl/`의 파일은 **import만 하고 수정하지 않는다**(HL 트랙과의 충돌 방지). 우리 코드는 전부 `nd/`에 둔다.
- **HL 트랙 상태(참고, 이 브랜치의 작업 대상 아님)**: 스토리 A·B(anchor, HL01–02) 종결, 스토리 C(먼 query 순도 회복, HL03–05) 종결, 스토리 D(순도·거리 분리, HL06–10) 야간 시퀀스 완료·판정 대기(이희태). 문서 `reports/HL_anchor_track_closure.md`, `reports/HL_storyC_closure.md`, `reports/OVERNIGHT_2026-10-26.md`.
- **ND 트랙이 묻는 것**: pseudo-class 그룹(anchor + 유사도 상위 노드)의 노드들이 **다른 그룹으로 가는 실제 그래프 연결의 분포(inter-class 패턴)** 를 공유하는가, 그 패턴만으로 그룹을 식별하거나 잘못 들어온 노드를 가려낼 수 있는가. 배경(이희태 판단, 2026-11-02): VISION·NaQ식 pseudo-task(anchor top-(K+M) 유사도 샘플링)는 같은 pseudo-label로 묶기엔 신뢰도 낮은 샘플이 섞이는 noisy-label 문제로 볼 수 있다(GAPNet의 IE noise에 대응). DCGC(2025)는 pseudo-label보다 이웃 분포가 더 안정적인 감독 신호임을 클러스터링에서 보였고, heterophily 문헌은 이웃 클래스 분포 패턴이 중심 노드 클래스 예측에 유용하다고 본다. ND는 그 관찰을 episodic·비지도 형태로 옮긴다.
  - **ND01(완료, 2026-11-02)**: 자기 그룹 성분을 *포함한* 이웃 투표 분포 d_v = normalize(Σ_{u∈N(v)} softmax(cos(f_u, anchors)/τ))를 시험했다. 결과: Â²X 프로토타입 위에 분류(meta-test)·노이즈 점수(B4 AUROC) 어느 쪽으로도 이득 없음(§8). 자기 그룹 성분이 inter 성분을 묻는 설계였으므로 원래 질문의 답은 아니다.
  - **ND02(현재)**: 자기 그룹 슬롯을 **뺀** inter-profile로 다시 묻는다. 세 질문 — Q1 inter-profile만으로 query의 pseudo-label 식별(우연 20% 대비, 엣지 섞기 대조), Q2 패턴 불일치가 GT 노이즈 query를 가려내는가(`a2x`·호모필 비율 대조), Q3 같은 GT 쌍이 inter-profile이 더 비슷한가(특징 유사 쌍 층화). GT 그룹 oracle 에피소드로 "Clothing에 패턴이 있는가"를 먼저 가른다.
  - 종결된 스토리 C와의 차이: C는 구조적 *거리*(Â²X 순위·PPR)로 먼 query의 순도를 올리려 했고, ND는 그룹 간 *연결 분포 패턴*을 쓴다. 재료가 다르다.
- **ND 트랙이 하지 않는 것**: 새 점수를 pseudo-task *query 선택* 판정기로 쓰는 학습 실험. HL06–HL10에서 같은 계열 판정기 12개가 세 데이터셋 모두에서 B_Choi를 넘지 못했다(§8). 지시서에 명시될 때만 한다.

## 1. 디렉터리 규약

```
<repo root>
├── data.py, meta.py, model.py, utils.py, task_generator.py, arg.py,
│   unsup_protonet.py, sup_protonet.py, naq_train.py, sup_train.py   ← NaQ 원본 (불변)
├── choi_exps/        ← Choi의 exp001–exp049 (불변; 함수 import는 허용)
├── hl/               ← HL 트랙 스크립트 (이 브랜치에서는 불변; 함수 import는 허용)
├── instructions/HL*, reports/HL*, reports/OVERNIGHT*  ← HL 트랙 문서 (불변)
├── data/             ← 원저자 공개 데이터 (불변, git 미추적)
├── save/downstream_tasks/<dataset>/5way_{1,5}shot.pkl  ← 고정 valid 50 / test 500 에피소드 (불변, git 미추적)
├── save/fixed_episodes/   ← HL 고정 에피소드 JSON (불변, git 미추적)
├── env/              ← 환경 점검 기록
├── nd/               ← 우리 스크립트. 번호 접두사 nd###_*.py, 공용 모듈 nd_*.py
├── instructions/     ← 지시서 ND##_*.md (이희태 작성)
├── reports/          ← Agent 보고서 ND##_*.md + reports/ND##_tables/*.csv (Agent 작성)
├── results/nd/<ND##>/  ← 지시서별 결과 json·npy·csv (git 미추적)
└── out/nd/           ← 실행 로그·PID 파일 (git 미추적)
```

- **불변 파일**: 루트의 NaQ 원본 10개, `choi_exps/` 전체, `hl/` 전체, HL 문서, `data/`, `save/` 아래 전부. 수정·삭제·덮어쓰기 금지. 필요한 변형은 `nd/`에 복사해 만들고 복사 출처를 docstring 첫 줄에 적는다.
- `.gitignore`는 HL 트랙 것을 그대로 쓴다. `git add`는 `nd/`, `instructions/ND*`, `reports/ND*`, `env/`, `CLAUDE.md`로 한정한다. `results/`, `out/`은 커밋하지 않는다.

## 2. 실행 환경 (구 서버)

- **서버**: `seio`, 리포 경로 `~/wkspace/NAQ`. GPU **RTX 4080 SUPER 1장**(16GB), `--device 0`. 다른 사람과 공유하므로 시작 전 `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv`를 기록하고, 여유가 **10GB 미만**이면 멈춘다(§7(d)). ND01 실측 최대 할당: selfcheck 9.9GB, 관문 A·B 7.4GB(X 코사인 dense 2.5GB + z0·z1·z2 2.7GB + 작업 메모리).
- **conda 환경 `taskns`**(`/home/h2tae/.conda/envs/taskns`): python 3.9.25, torch 2.4.1+cu121, PyG 2.6.1(`env/envcheck_taskns.txt`). HL01·HL02가 이 환경에서 돌았다. **아무것도 설치하지 않는다.** 다른 GFSL 트랙과 공유하는 환경이다. `taskns`가 깨져 있으면 2순위 `ctx-teg`, 둘 다 안 되면 멈춘다(§7(c)). 새 환경을 만들지 않는다.
- 이 서버에 있는 비추적 자산: `data/`(Amazon_clothing·dblp·Amazon_eletronics), `save/downstream_tasks/Amazon_clothing/5way_{1,5}shot.pkl`(md5 1-shot `5f901727f81903cc12159d280be6915d`, 5-shot `ca3edaca02d541cbc0ee15a2122ef24a` — 시작 시 확인), `save/fixed_episodes/`(seed 1212–1216), `results/hl/p0/Amazon_clothing/{ours/tau_0.3,naq/baseline}/seed_1212/best_1shot.pt`, `results/hl/hl02/`. **HL03–HL10의 결과(`results/hl/hl03`–`hl10`)는 이 서버에 없다**(gpusystem에만 있음). 필요한 수치는 커밋된 `reports/HL*_tables/*.csv`를 인용한다.
- 모든 스크립트는 **리포 루트에서** 실행한다. `nd/` 스크립트는 `hl/hl_common.py` 방식으로 리포 루트와 `choi_exps/`, `hl/`를 sys.path에 넣고 함수를 import한다(`hl_common`, `hl005_diag_distance_purity`, `exp027` 등). `PYTHONPATH=$PWD`.
- 백그라운드 실행은 `nohup ... > out/nd/<이름>.log 2>&1 & echo $! > out/nd/<이름>.pid`로 하고 감시는 PID 파일로만 한다. 로그 tail을 반복 polling하지 않는다. ND01은 학습이 없어 대부분 전경 실행으로 충분하다.
- 메모리 메모: Clothing X 코사인 dense 행렬(`data.Diffusion_matrix`) 24,919² × 4B ≈ 2.5GB. `structural_scores`의 z0·z1·z2(각 24,919 × 9,034 float32 ≈ 0.9GB). dblp·electronics는 dense 행렬이 6.6–7.2GB라 이 서버에서는 **Clothing만** 한다.

## 3. 고정 프로토콜

- **데이터셋**: Amazon_clothing(24,919 노드, 9,034 특징, 77 클래스, train 57 / valid·test 20). ND01·ND02는 Clothing만.
- **meta-test 평가(ND01 관문 A)**: `save/downstream_tasks/Amazon_clothing/5way_{shot}shot.pkl`의 `(valid_pool, test_pool)` — valid 50, test 500 에피소드, 각 `(supports, y_support, queries, y_query)`, query 8/클래스. 정확도 식은 `choi_exps/exp027.py: episode_accuracy`와 동일(프로토타입 = support 평균, 제곱 유클리드 최근접). **파라미터·임계값 선택은 valid 50에서만, 보고는 test 500에서.** 고정 에피소드라 seed 분산은 없다 — 대신 500 에피소드의 쌍 비교(paired Δ, 95% CI)를 보고한다.
- **특징 공간**: `hl_common.structural_scores` → z0(X), z1(ÂX), z2(Â²X), 행 정규화. Â = D^-1/2(A+I)D^-1/2(exp049와 동일). 인접행렬 A는 `data.edge_index`로부터 대칭화한 것.
- **B4 구간(ND01 관문 B)**: anchor의 X 코사인 순위 201–1000(자기 자신 제외), `hl008_distant_query_train.b4_scores`·`hl005` 정의 그대로. AUROC는 `hl005.auroc_rows`(anchor 행별 Mann-Whitney, 양·음 모두 있는 행만 평균)와 `auroc_pooled`.
- **기준선 정의(참고)**: **NaQ** = `exp049 --method naq`. **B_Choi** = `exp049 --method ours --tau 0.3`. 학습 기준선 수치는 §8.
- **pseudo 에피소드(ND02)**: anchor 균등 5개 → anchor별 X 코사인 상위(자기 자신 제외, 순차 배제) K−1개 support + M개 query. 주 설정 K=3, M=10, E=300. 정의는 ND02 §1.
- seed 표준: **1212** 하나(학습이 없으므로). 무작위 추출은 모두 `np.random.default_rng(1212)`.
- AUROC·CI 식은 ND01과 같다: `hl005.auroc_rows`(행별 Mann-Whitney, 양·음 모두 있는 행만 평균)·`auroc_pooled`, CI = 1.96·std(ddof=0)/√n.

## 4. 레이블 사용 규칙

- Ground-truth 레이블은 **채점(AUROC·정확도·순도)과 지시서가 명시한 oracle 에피소드 생성**에서만 읽는다. 배정·profile·패턴 계산, anchor 추출, 후보 선택, 파라미터 선택 과정에는 넣지 않는다. 레이블을 읽는 함수는 이름에 `grade`·`diag`·`eval`·`gt`가 들어가고, oracle 에피소드 파일명에는 `gt`가 들어간다. 보고서에 "레이블 사용 범위"를 한 줄로 적는다.
- `hl_common.labels_np`만으로 레이블을 읽는다.

## 5. 지시서·사전 등록·보고서

- 지시서 `instructions/ND##_<이름>.md`는 이희태가 쓴다. Agent는 지시서에 적힌 것만 한다.
- 사전 등록 `NDP###`: Agent는 **계산값만** 채운다. 판정은 이희태. 결과를 본 뒤 예측 문구를 바꾸지 않는다.
- 보고서 `reports/ND##_<이름>.md`: 사실만(해석·추측·제안 금지). ① 실행 명령·환경·commit ② 표 ③ 사전 등록 대조표 ④ 실패·경고·이상·지시서에 없어 정한 것 ⑤ 산출물 경로. 수치는 스크립트가 쓴 csv에서 옮긴다. 표 csv는 `reports/ND##_tables/`에 두고 커밋한다.
- 번호 레지스트리: ND01 투표 분포 두 관문(완료, NDP001–010) / **ND02 inter-profile — Q1 식별·Q2 노이즈·Q3 쌍 비교 + 엣지 섞기·GT 그룹 oracle 대조(NDP011–022)** / 이후 이희태가 추가.
- 스크립트 번호: `nd_vote.py` 공용 모듈(ND01; 인접행렬·Â² 가중 `build_weights`·`_sym_a` 재사용), `nd001` 관문 A, `nd002` 관문 B, `nd_nd01_tables.py`(ND01 완료) / **`nd003_episodes.py` 에피소드 생성, `nd004_profiles.py` 배정·profile·패턴·분류·쌍 비교·커버리지(`--stage`, `--rewire`), `nd_nd02_tables.py` 표**. 이후 지시서는 `nd005`부터. 표 스크립트는 `nd/nd_nd##_tables.py`.

## 6. git 규약

- 지시서 하나 = `nd/anchor-vote` 위의 커밋 묶음 = 태그 묶음(`nd01-setup`, `nd01-gateA`, `nd01-gateB`, …). 독립 실험마다 커밋을 나눈다.
- 커밋 메시지 첫 줄: `ND01: <무엇을>`. 결과 csv·보고서 `.md`는 포함, npy·json·로그는 제외.
- push: `git push origin nd/anchor-vote --tags`. 실패하면 보고서 ④에 적고 계속. `hl/anchor`·`naq_exp` push 금지, force push 금지.
- git identity는 리포 로컬 config(`dragor0123`, 이희태 메일). 전역 설정은 건드리지 않는다.

## 7. 멈춤·auto 규칙

- **멈추는 경우(채팅 보고 후 대기)**: (a) 불변 파일을 고쳐야만 진행되는 상황, (b) 지시서가 "정지"로 표시한 게이트 실패, (c) 환경에 지시서에 없는 설치가 필요해 보이는 상황(설치하지 말고 멈춘다), (d) GPU 여유 6GB 미만·디스크 부족, (e) pkl md5 불일치.
- **멈추지 않는 경우**: 수치가 예측과 다름(기록만), 한 arm 실패(오류 로그 남기고 다음 arm), push 실패.
- auto 모드는 지시서 또는 채팅에 "auto"라고 명시된 경우에만. auto에서는 질문하지 않고 §7의 멈춤 사유가 아니면 끝까지 간다.
- 애매하면 **덜 하는 쪽**을 택하고 보고서 ④에 적는다. 이 트랙에는 야간 모드가 없다.

## 8. 확정 사실 (Agent가 전제로 삼아도 되는 것)

- **학습 기준선(Clothing, 1-shot / 5-shot test, 고정 test 500 에피소드)**: NaQ 80.65 / 90.52(seed 1212), **B_Choi(τ=0.3) 82.71 ± 0.31 / 89.91 ± 0.57**(5 seed). dblp: NaQ 68.51 / 81.96, B_Choi 71.01 / 80.94. electronics: NaQ 68.93 / 83.68, B_Choi 70.26 / 82.93.
- **ND01 관문 A(학습 없음, 같은 test 500 에피소드, 같은 분류기)**: `feat_x`(X 행 정규화) 72.35 / 87.61, **`feat_a2x`(Â²X 행 정규화) 85.10 / 92.72** — 학습 없는 Â²X가 NaQ·B_Choi를 넘는다(1-shot −2.4, 5-shot −2.8). Â²X 위에 d_v를 결합한 어떤 변형도 ±0.5 안(`comb_a2x_sel` −0.16, CI ±0.12). X 특징에서는 τ=0.05·hard가 τ≥0.1보다 훨씬 낫고(코사인 분산이 작아 softmax가 균등해짐), `self`·`a2` 이웃이 `1hop`보다 일관되게 낫다.
- **ND01 관문 B(B4, 400 에피소드 × 5 anchor = 2,000 anchor, `results/nd/nd01/gateB/anchors.npy`)**: AUROC_all `ax` 0.728, `a2x` 0.753, `enc_b` 0.775, `own_soft_x_0.1` 0.595, `vote_x_0.1_1hop` 0.675, `vote_x_0.2_a2` 0.703(X 계열 최고). f=`a2x`에서는 `own_soft_a2x_0.2` 0.753 ≈ `vote_a2x_0.2_self` 0.754 ≈ `a2x` 0.753 — 구조가 든 특징 위에서는 anchor 경쟁도 이웃 투표도 0.002 이내. `a2x` 0.7534는 HL06 P2(다른 anchor 집합) 0.7564와 일치.
- **HL06 P2(Clothing, B4, anchor 2,000개 — 결과 파일은 이 서버에 없음)**: AUROC_all `x` 0.594, `ax` 0.730, `a2x` 0.756, `ppr` 0.699, `mv` 0.733, `enc_b` 0.779, `enc_n` 0.788, `nbr_b` 0.781, `clu_b` 0.780.
- **HL06–HL10 요지(스토리 D)**: 레이블 없는 판정기 12종은 AUROC 0.74–0.80인데도 B4 구간에서 10개를 고르면 순도 0.32 수준(Choi 필터 0.42)이고, 레이블 없는 학습 arm은 세 데이터셋 모두 B_Choi 미달(Clothing 최고 `knnppr_ex20` −0.78, dblp −2.71, electronics −0.45). oracle 쪽은 일관되게 강함: 순도 1.0에서 `strat5` 90.12 > `far50` 89.67 > `t1` 87.20; dblp far − near +3.38, electronics +1.53. 핵심 가설 HLP078(`encb_strat5` − `encb_top10` ≥ +2.0)은 +0.51로 기각.
- **HL02 진단**: X 상위 50 중 Choi 필터 탈락 쌍의 same-GT 0.218, 통과 쌍 0.574, 무작위 쌍 0.020. B_Choi query 순도 0.417, NaQ 0.597.
- **HL05**: 순도 0.99 고정에서 구조적으로 먼 same-class positive(89.46)가 가까운 것(85.49)보다 1-shot을 더 올린다.
- 평가 에피소드는 seed와 무관하게 같다(pkl 캐시). 같은 설정 재학습 시 1-shot test는 0.1%p 안에서 흔들리고, 0.3%p 미만 차이는 판단 근거로 쓰지 않는다 — ND01은 학습이 없어 결정론적이지만, 500 에피소드 평균의 표집 오차(CI 반폭 약 ±0.8%p)는 남는다. 그래서 쌍 비교를 쓴다.

## 9. 운영 메모

- `data.Diffusion_matrix`는 X 코사인 dense 행렬(대각 포함). anchor 행에서 자기 자신을 빼려면 `row[a] = -inf`(hl008 방식).
- `exp026.py`는 `results/Amazon_clothing/exp026/result.txt`를 덮어쓴다(부작용, 무해). ND01은 호출할 일이 없다.
- `utils.py:83`의 `torch.sparse.SparseTensor` deprecation 경고는 무해. 보고서에 반복해서 적지 않는다.
- `hl005.encoder_embeddings(data, path, 256, device)`로 B_Choi 체크포인트의 행 정규화 임베딩(enc_b)을 얻는다. 체크포인트가 없으면 enc_b 열을 비우고 ④에 적는다(정지 사유 아님).
- Clothing 대칭 인접행렬(self-loop 제외) nnz 183,360, 고립 노드 0(ND01 selfcheck). Â² 가중 행렬 nnz 2,150,025(41MB).
- ND01 코드 재사용: `nd_vote._sym_a(data)`(대칭 이진 A, self-loop 없음, scipy csr), `nd_vote.build_weights(data, device)`(`1hop`·`self`·`a2` 행 정규화 가중, torch sparse), `nd_vote.vote_rows`(CSR 행 선택 곱). `hl005.auroc_rows`·`auroc_pooled`는 ND01 관문 B가 쓴 그대로.
- NaQ 원본의 pseudo-task는 anchor 1개 + X 코사인 top-q query(`task_generator.query_generation_NaQ`, `Diffusion_matrix` topk — 자기 자신 포함)이고, HL·ND는 자기 자신을 `-inf`로 빼고 정렬한다. VISION은 anchor당 top-(K+M)을 순차 배제로 묶는다. ND02 §1이 쓰는 방식은 후자.
- networkx는 `taskns`에 있다고 가정하지 않는다. 그래프 조작은 numpy·scipy로 한다.
