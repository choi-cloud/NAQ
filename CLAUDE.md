# CLAUDE.md — NAQ 리포 위 anchor 연구 (HL 트랙)

개정 4 · 2026-10-25 · 작성 이희태 (개정 1: 10-24, 개정 2·3: 10-25)
이 파일은 Agent가 매 세션 처음에 읽는 상태 파일이다. 수정은 이희태만 한다. Agent는 이 파일을 고치지 않는다.
개정 4 변경: **서버 이전**(1×4080 SUPER → gpusystem 4×RTX 6000 Ada, GPU 3번만 사용). §2 환경·GPU·동시 실행, §3 run 시간·기준선은 새 서버에서 재측정, §8 수치에 측정 서버 표기, §9 이전된 비추적 자산 목록.
개정 3 변경: §0 트랙 상태(anchor 트랙 종결, 스토리 C 시작), §5 레지스트리 HL03, §8 HL02 사실 추가, §9 평가 에피소드 캐시·mining 배치 의존성.

## 0. 이 리포가 무엇인가

- 원격: `choi-cloud/NAQ`(브랜치 `naq_exp`, HEAD `1a20f23`)를 clone한 것. 원본은 NaQ(ICML'24, `JhngJng/NaQ-PyTorch`) 공식 코드이고, 그 위에 공동연구자 Choi가 `exp001`–`exp049` 실험 스크립트를 쌓았다.
- **트랙 상태**:
  - anchor 트랙(스토리 A·B, HL01–HL02): **종결**(2026-10-25). oracle 상한 +0.19%p < 1.0%p. 문서 `reports/HL_anchor_track_closure.md`. anchor 샘플링을 바꾸는 새 실험은 지시서에 명시될 때만 한다.
  - **스토리 C(HL03–)**: query 쪽. B_Choi가 얻은 "먼 query"의 순도를 레이블 없이 되찾을 수 있는가. Choi의 query 작업(문서 17–19, exp044–049)의 연장이다.
- 작업 브랜치: `hl/anchor` (`naq_exp` HEAD에서 분기, 이름은 유지). `naq_exp`에는 아무것도 쓰지 않는다.

## 1. 디렉터리 규약

```
<repo root>
├── data.py, meta.py, model.py, utils.py, task_generator.py, arg.py,
│   unsup_protonet.py, sup_protonet.py, naq_train.py, sup_train.py   ← NaQ 원본 (불변)
├── choi_exps/        ← Choi의 exp001–exp049 .py/.sh, run_exp049.py, shell.sh (불변)
├── data/             ← 원저자 공개 데이터 (불변, git 미추적)
├── save/fixed_episodes/   ← 고정 valid/test 에피소드 JSON (생성 후 불변, git 미추적)
├── env/              ← 환경 점검 기록, NAQ_conda.yml(fallback), requirements_pip.txt(fallback)
├── hl/               ← 우리 스크립트. 번호 접두사 hl###_*.py  (hl000 러너, hl001 HL01 표, hl_envcheck)
├── instructions/     ← 지시서 HL##_*.md (이희태 작성)
├── reports/          ← Agent 보고서 HL##_*.md + reports/HL##_tables/*.csv (Agent 작성)
├── results/hl/<HL##>/  ← 지시서별 결과 json·체크포인트·csv (git 미추적)
└── out/hl/           ← 실행 로그·PID 파일 (git 미추적)
```

- **불변 파일**: 루트의 NaQ 원본 10개, `choi_exps/` 전체, `data/`, 생성된 `save/fixed_episodes/*.json`. 수정·삭제·덮어쓰기 금지. 필요한 변형은 `hl/`에 복사해 만든다(복사 출처를 docstring 첫 줄에 적는다). `choi_exps/`의 **함수를 import해서 쓰는 것은 허용**된다(예: `from exp049 import build_structural_queries`).
- `.gitignore`: Choi의 규칙 + HL01에서 추가한 `!*.md !*.yml !*.txt !*.csv`. 단 run 디렉터리 안의 `wall_seconds.txt` 같은 실행 부산물 `.txt`는 커밋하지 않는다(`git add`를 `reports/`, `env/`, `hl/`, `instructions/`, `CLAUDE.md`로 한정).

## 2. 실행 환경 (HL03부터 새 서버)

- **서버**: `gpusystem`, 리포 경로 `~/workspace/NAQ`. GPU 4×RTX 6000 Ada(48GB), driver 580.95.05.
- **GPU 기본값은 3번(마지막 GPU)이다.** 지시서나 채팅에 다른 번호가 명시되지 않는 한, 모든 명령 앞에 `CUDA_VISIBLE_DEVICES=3`을 붙이고 스크립트 인자는 `--device 0` 그대로 둔다(보이는 GPU가 하나뿐이므로 0 = 물리 3번). 0–2번에 다른 사람의 프로세스가 있어도 건드리지 않는다. 이 규칙은 HL03 이후 모든 지시서에 적용된다.
- **동시 실행**: run당 예약 메모리 약 8.2GB(구 서버 측정)이므로 학습 run은 **동시 최대 3개**. 학습 없는 진단 스크립트(dense 행렬 사용)는 학습 run과 겹치지 않게 돌린다. 새 서버에서 측정한 run당 메모리를 HL03 보고서에 적는다.
- **conda 환경 `NAQ`** — 우리 전용, `env/NAQ_conda.yml` + `env/requirements_pip.txt`로 HL03 §0-3에서 만든다. 새 서버에는 쓸 만한 기존 환경이 없다(이희태 확인). 서버에 원래 있던 다른 환경은 쓰지도, 건드리지도 않는다. `NAQ`에 추가 설치가 필요하면 HL03 §0-3의 허용 범위 안에서만 하고, 범위 밖이면 멈춘다(§7). 확정 후 `hl/hl_envcheck.py` 출력과 `conda list --export`를 `env/`에 남긴다.
- 구 서버 기록(참고): `taskns` — python 3.9.25, torch 2.4.1+cu121, PyG 2.6.1(`env/envcheck_taskns.txt`). HL01–HL02 수치는 이 환경·RTX 4080 SUPER에서 측정됐다. 새 환경은 torch 2.5.1+cu121, PyG 2.8.0.post1(Choi의 원래 환경과 같음)이다.
- 모든 스크립트는 **리포 루트에서** 실행한다. Choi 스크립트·함수는 `PYTHONPATH=$PWD`를 주고 `choi_exps/`를 sys.path에 넣어 쓴다(`hl/hl000_run.py`가 하는 방식). `choi_exps/run_exp049.py`는 쓰지 않는다.
- 백그라운드 실행은 `nohup ... > out/hl/<이름>.log 2>&1 & echo $! > out/hl/<이름>.pid`로 하고, **감시는 PID 파일로만** 한다(`kill -0 $(cat pid)`). 로그 tail을 반복 polling하지 않는다.

## 3. 고정 프로토콜 (Choi exp027/exp049와 동일, 변경 금지)

- 학습: 5-way 1-shot pseudo-episode, q_query 10, 2000 에피소드(학습 전 일괄 생성), latent 256, lr 1e-4, Adam, `meta.ProtoNet.forward` 그대로(제곱 유클리드 → log_softmax → NLL).
- 평가: 160 에피소드마다 고정 valid 50개로 제곱 유클리드 최근접 프로토타입 정확도. 1-shot·5-shot 각각 best valid 체크포인트 → 고정 test 500개 1회. `exp027.evaluate` 사용.
- 고정 에피소드: `save/fixed_episodes/{dataset}/5way_{shot}shot_8query_seed{seed}/{valid,test}_episodes.json`. seed 1212·1213·1214는 생성됨. 새 seed는 `choi_exps/exp026.py`로 생성(있으면 거부함 — 정상).
- 기준선 정의: **NaQ** = `exp049 --method naq`. **B_Choi** = `exp049 --method ours --tau 0.3`(τ*=0.3은 HL01 §5-1에서 1-shot valid로 선택). 필터 = ÂX·Â²X 코사인이 **둘 다** τ 이상인 노드만 남김(전체 노드 대상) → X 코사인 정렬의 정중앙 10개 → 10개 미만이면 min(s1,s2) 상위 10개로 relax.
- 데이터셋 순서: Amazon_clothing → dblp → Amazon_eletronics(철자는 원본 그대로). Cora_full은 쓰지 않는다.
- **run 시간**: 구 서버(RTX 4080 SUPER) NaQ 약 42초, B_Choi 약 51초. 새 서버 값은 HL03 §0에서 재측정해 보고서에 적는다.
- **기준선은 서버별로 따로 잡는다.** GPU·환경이 바뀌면 같은 설정도 수치가 달라질 수 있다. 새 서버의 비교 기준은 HL03 §0-4에서 다시 돌린 B_Choi(uniform anchor, 5 seed)이고, HL02의 82.70은 참고값으로만 쓴다.
- seed 표준: **5개 — 1212, 1213, 1214, 1215, 1216**. 탐색은 1212 하나, 판정은 5개.

## 4. 레이블 사용 규칙

- Ground-truth 레이블은 **oracle arm과 진단 통계에서만** 읽는다. 실제(레이블 없는) 샘플러에는 절대 쓰지 않는다. 레이블 없는 샘플러 함수에는 `labels`를 인자로 넘기지 않는다.
- oracle·진단 스크립트는 파일명에 `oracle` 또는 `diag`를 넣고, 보고서에 "레이블 사용 범위"를 한 줄로 적는다.

## 5. 지시서·사전 등록·보고서

- 지시서 `instructions/HL##_<이름>.md`는 이희태가 쓴다. Agent는 지시서에 적힌 것만 한다. 지시서에 없는 판단이 필요하면 §7 규칙을 따른다.
- 사전 등록 `HLP###`: Agent는 **계산값만** 표에 채운다. 판정은 이희태. 결과를 본 뒤 예측 문구를 바꾸지 않는다.
- 보고서 `reports/HL##_<이름>.md`: 사실만(해석·추측·제안 금지). ① 실행 명령·환경·commit ② 표 ③ 사전 등록 대조표 ④ 실패·경고·이상 ⑤ 산출물 경로. 수치는 스크립트가 쓴 csv에서 옮긴다. 표 csv는 `reports/HL##_tables/`에 두고 커밋한다.
- 번호 레지스트리: HL01 환경·smoke·P0(완료, HLP001–006) / HL02 진단 + anchor oracle 사다리 + B 점검(완료, HLP007–023) / **HL03 새 서버 기준선 + 먼 후보 순도 판정 G0 + 거리 고정 순도 oracle G1(HLP024–035)** / 이후 이희태가 추가.
- 스크립트 번호: hl000(·b·c) 러너, hl001 HL01 표, hl002 노드 진단, hl003 anchor oracle 학습, hl004 임베딩 점검, **hl005 G0 순도 판정, hl006 G1 순도 oracle 학습**.

## 6. git 규약

- 지시서 하나 = `hl/anchor` 위의 커밋 묶음 = 태그 묶음(`hl02-diag`, `hl02-oracle`, …). 독립 실험마다 커밋을 나눈다.
- 커밋 메시지 첫 줄: `HL02: <무엇을>`. 결과 csv·보고서 `.md`는 포함, 체크포인트·json·로그·`wall_seconds.txt`는 제외.
- push: `git push origin hl/anchor --tags`. 실패하면 보고서 ④에 적고 계속. `naq_exp` push 금지, force push 금지.

## 7. 멈춤·auto 규칙

- **멈추는 경우(채팅 보고 후 대기)**: (a) 불변 파일을 고쳐야만 진행되는 상황, (b) 지시서가 "정지"로 표시한 게이트 실패, (c) `taskns`에 설치가 필요해 보이는 상황(설치하지 말고 멈춘다), (d) GPU 메모리 부족·디스크 부족.
- **멈추지 않는 경우**: 수치가 예측과 다름(기록만), 한 run 실패(오류 로그 남기고 다음 run), push 실패.
- auto 모드는 지시서 또는 채팅에 "auto"라고 명시된 경우에만. auto에서는 질문하지 않고 §7의 멈춤 사유가 아니면 끝까지 간다.
- 애매하면 **덜 하는 쪽**을 택하고 보고서 ④에 적는다.

## 8. 확정 사실 (Agent가 전제로 삼아도 되는 것)

아래 HL01·HL02 수치는 **구 서버(RTX 4080 SUPER, taskns)** 측정값이다. 정확도의 절대값은 새 서버에서 소수점 단위로 다를 수 있고, 같은 서버 안의 차이(Δ)가 판단 근거다.

- HL01 (Amazon_clothing, 3 seed, 1-shot test): NaQ 80.99±0.41, B_Choi(τ=0.3) 82.81±0.38 → **+1.82**. 5-shot: 90.78 vs 90.18 → **−0.61**(B_Choi가 5-shot에서는 낮다).
- B_Choi의 query GT 순도는 **0.42**(NaQ 0.60). 즉 B_Choi는 순도를 낮추면서 1-shot을 올렸다. 선택된 query의 평균 X 순위 ≈ 750(전체 24,919 노드 중). fallback 발생률 3.4%.
- Amazon_clothing: 24,919 노드, 77 클래스(train 57 / test 20). 2000 에피소드 × 5 anchor = 10,000회 추출 → 학습 전체에서 anchor로 한 번이라도 뽑히는 노드는 약 1/3.
- Choi 보고(CoraFull, 구두): uniform 5 anchor에서 GT가 모두 다른 에피소드 80%, 한 쌍 겹침 18%. 겹치는 쌍을 레이블로 제거해도 test 개선 없음 → "uniform에서 같은 클래스 anchor 회피"는 레버가 아니다.
- smoke 재현: NaQ seed 1212 1-shot test 80.64 (Choi 80.66).
- HL02 (5 seed, uniform anchor + B_Choi query = 82.70 ± 0.29 / 5-shot 89.91): anchor 순도 5분위 q1–q5 = 77.35 / 77.41 / 79.10 / 82.56 / 81.84. oracle-safe hard co-anchor `hard:4` −0.58, `hard_blind:4` −2.40, `gt_distinct` +0.25, `ab` +0.19.
- HL02 진단: 노드별 B_Choi query 순도 평균 0.417, NaQ 0.597. X 상위 50 중 필터 탈락 쌍의 same-GT 0.218, 통과 쌍 0.574, 무작위 쌍 0.020. 필터 통과 수와 degree ρ = 0.35.
- HL02 임베딩(B_Choi ckpt): 거리 중앙값 통과 0.04 / 탈락 0.35 / 무작위 0.78. uniform 에피소드에서 negative 확률 합 < 0.05인 query 50%.
- Choi oracle(문서 19, 1 seed, 다른 평가 코드로 같은 프로토콜): query 순도 100% top-K 85.02, middle-K 89.90, reverse 88.68.

## 9. 운영 메모

- `hl/hl000_run.py`는 호출마다 `summary.csv`를 덮어쓴다. 단계별 사본을 남긴다(HL01 방식). 새 지시서부터는 결과 루트를 `results/hl/<HL##>/`로 분리한다.
- `choi_exps/exp026.py`는 `results/Amazon_clothing/exp026/result.txt`를 덮어쓴다(부작용, 무해). 커밋하지 않는다.
- `utils.py:83`의 `torch.sparse.SparseTensor` deprecation 경고는 무해. 보고서에 반복해서 적지 않는다.
- **평가 에피소드는 seed와 무관하게 같다.** NaQ 원본 `downstream_task_generator`가 `save/downstream_tasks/{dataset}/5way_{shot}shot.pkl` 캐시를 재사용하기 때문이다. seed 간 분산은 학습 쪽만 반영한다. 캐시 pkl은 건드리지 않는다.
- exp049 query mining은 노드 배치 구성에 따라 동률 노드에서 결과가 미세하게 달라질 수 있다(부동소수 1e-7). `hl_common`의 함수는 exp049와 같은 배치 방식으로 계산한다.
- 같은 설정 재학습 시 1-shot test가 0.1%p 안에서 흔들린다. 0.3%p 미만의 차이는 판단 근거로 쓰지 않는다.
- **새 서버로 rsync한 비추적 자산**(구 서버와 동일, 2026-10-25): `data/`, `save/downstream_tasks/`(md5 1-shot `5f901727f81903cc12159d280be6915d`, 5-shot `ca3edaca02d541cbc0ee15a2122ef24a`), `save/fixed_episodes/`(seed 1212–1216), `results/hl/p0/Amazon_clothing/{ours/tau_0.3,naq/baseline}/seed_1212/best_1shot.pt`, `results/hl/hl02/diag/node_table_tau0.3.csv`. 그 밖의 `results/hl/` 내용(HL01·HL02의 다른 run 결과)은 새 서버에 없다 — 필요하면 보고서·csv(커밋됨)를 인용한다.
