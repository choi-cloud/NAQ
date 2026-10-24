# CLAUDE.md — NAQ 리포 위 anchor 연구 (HL 트랙)

개정 2 · 2026-10-25 · 작성 이희태 (개정 1: 10-24)
이 파일은 Agent가 매 세션 처음에 읽는 상태 파일이다. 수정은 이희태만 한다. Agent는 이 파일을 고치지 않는다.
개정 2 변경: §2 환경 확정, §3 τ*·run 시간, §5 레지스트리 HL02, §8 확정 사실 신설, §9 운영 메모 신설.

## 0. 이 리포가 무엇인가

- 원격: `choi-cloud/NAQ`(브랜치 `naq_exp`, HEAD `1a20f23`)를 clone한 것. 원본은 NaQ(ICML'24, `JhngJng/NaQ-PyTorch`) 공식 코드이고, 그 위에 공동연구자 Choi가 `exp001`–`exp049` 실험 스크립트를 쌓았다.
- 우리(HL 트랙)는 **anchor(support) 샘플링** 자유도를 다룬다. Choi의 작업은 query 쪽이다. 두 작업은 같은 원시 신호(ÂX·Â²X 코사인 필터)를 쓰되 자유도가 다르다.
- 작업 브랜치: `hl/anchor` (`naq_exp` HEAD에서 분기). `naq_exp`에는 아무것도 쓰지 않는다.

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

## 2. 실행 환경 (HL01에서 확정)

- conda 환경 **`taskns`**(`/home/h2tae/.conda/envs/taskns`): python 3.9.25, torch 2.4.1+cu121, PyG 2.6.1, numpy 1.26.4, scipy 1.13.1, sklearn 1.5.2. 점검 기록 `env/envcheck_taskns.txt`, 인벤토리 `env/taskns_inventory.txt`.
- **`taskns`에는 패키지를 설치·업그레이드·삭제하지 않는다**(다른 트랙과 공유). 설치가 필요해 보이면 멈추고 보고한다(§7).
- GPU: RTX 4080 SUPER 1장, `--device 0`. 동시에 돌리는 run은 최대 2개.
- 모든 스크립트는 **리포 루트에서** 실행한다. Choi 스크립트·함수는 `PYTHONPATH=$PWD`를 주고 `choi_exps/`를 sys.path에 넣어 쓴다(`hl/hl000_run.py`가 하는 방식). `choi_exps/run_exp049.py`는 쓰지 않는다.
- 백그라운드 실행은 `nohup ... > out/hl/<이름>.log 2>&1 & echo $! > out/hl/<이름>.pid`로 하고, **감시는 PID 파일로만** 한다(`kill -0 $(cat pid)`). 로그 tail을 반복 polling하지 않는다.

## 3. 고정 프로토콜 (Choi exp027/exp049와 동일, 변경 금지)

- 학습: 5-way 1-shot pseudo-episode, q_query 10, 2000 에피소드(학습 전 일괄 생성), latent 256, lr 1e-4, Adam, `meta.ProtoNet.forward` 그대로(제곱 유클리드 → log_softmax → NLL).
- 평가: 160 에피소드마다 고정 valid 50개로 제곱 유클리드 최근접 프로토타입 정확도. 1-shot·5-shot 각각 best valid 체크포인트 → 고정 test 500개 1회. `exp027.evaluate` 사용.
- 고정 에피소드: `save/fixed_episodes/{dataset}/5way_{shot}shot_8query_seed{seed}/{valid,test}_episodes.json`. seed 1212·1213·1214는 생성됨. 새 seed는 `choi_exps/exp026.py`로 생성(있으면 거부함 — 정상).
- 기준선 정의: **NaQ** = `exp049 --method naq`. **B_Choi** = `exp049 --method ours --tau 0.3`(τ*=0.3은 HL01 §5-1에서 1-shot valid로 선택). 필터 = ÂX·Â²X 코사인이 **둘 다** τ 이상인 노드만 남김(전체 노드 대상) → X 코사인 정렬의 정중앙 10개 → 10개 미만이면 min(s1,s2) 상위 10개로 relax.
- 데이터셋 순서: Amazon_clothing → dblp → Amazon_eletronics(철자는 원본 그대로). Cora_full은 쓰지 않는다.
- **run 시간**: NaQ 약 42초, B_Choi 약 51초 (RTX 4080 SUPER, 2000 에피소드, 두 shot 평가 포함). 큐 설계는 이 값으로 한다 — 60 run ≈ 1시간.
- seed 표준: **5개 — 1212, 1213, 1214, 1215, 1216**. 탐색은 1212 하나, 판정은 5개.

## 4. 레이블 사용 규칙

- Ground-truth 레이블은 **oracle arm과 진단 통계에서만** 읽는다. 실제(레이블 없는) 샘플러에는 절대 쓰지 않는다. 레이블 없는 샘플러 함수에는 `labels`를 인자로 넘기지 않는다.
- oracle·진단 스크립트는 파일명에 `oracle` 또는 `diag`를 넣고, 보고서에 "레이블 사용 범위"를 한 줄로 적는다.

## 5. 지시서·사전 등록·보고서

- 지시서 `instructions/HL##_<이름>.md`는 이희태가 쓴다. Agent는 지시서에 적힌 것만 한다. 지시서에 없는 판단이 필요하면 §7 규칙을 따른다.
- 사전 등록 `HLP###`: Agent는 **계산값만** 표에 채운다. 판정은 이희태. 결과를 본 뒤 예측 문구를 바꾸지 않는다.
- 보고서 `reports/HL##_<이름>.md`: 사실만(해석·추측·제안 금지). ① 실행 명령·환경·commit ② 표 ③ 사전 등록 대조표 ④ 실패·경고·이상 ⑤ 산출물 경로. 수치는 스크립트가 쓴 csv에서 옮긴다. 표 csv는 `reports/HL##_tables/`에 두고 커밋한다.
- 번호 레지스트리: HL01 환경·smoke·P0(완료, HLP001–006) / **HL02 학습 없는 진단 + anchor oracle 사다리 + B 학습 전 점검(HLP007–)** / 이후 이희태가 추가.

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

- HL01 (Amazon_clothing, 3 seed, 1-shot test): NaQ 80.99±0.41, B_Choi(τ=0.3) 82.81±0.38 → **+1.82**. 5-shot: 90.78 vs 90.18 → **−0.61**(B_Choi가 5-shot에서는 낮다).
- B_Choi의 query GT 순도는 **0.42**(NaQ 0.60). 즉 B_Choi는 순도를 낮추면서 1-shot을 올렸다. 선택된 query의 평균 X 순위 ≈ 750(전체 24,919 노드 중). fallback 발생률 3.4%.
- Amazon_clothing: 24,919 노드, 77 클래스(train 57 / test 20). 2000 에피소드 × 5 anchor = 10,000회 추출 → 학습 전체에서 anchor로 한 번이라도 뽑히는 노드는 약 1/3.
- Choi 보고(CoraFull, 구두): uniform 5 anchor에서 GT가 모두 다른 에피소드 80%, 한 쌍 겹침 18%. 겹치는 쌍을 레이블로 제거해도 test 개선 없음 → "uniform에서 같은 클래스 anchor 회피"는 레버가 아니다.
- smoke 재현: NaQ seed 1212 1-shot test 80.64 (Choi 80.66).

## 9. 운영 메모

- `hl/hl000_run.py`는 호출마다 `summary.csv`를 덮어쓴다. 단계별 사본을 남긴다(HL01 방식). 새 지시서부터는 결과 루트를 `results/hl/<HL##>/`로 분리한다.
- `choi_exps/exp026.py`는 `results/Amazon_clothing/exp026/result.txt`를 덮어쓴다(부작용, 무해). 커밋하지 않는다.
- `utils.py:83`의 `torch.sparse.SparseTensor` deprecation 경고는 무해. 보고서에 반복해서 적지 않는다.