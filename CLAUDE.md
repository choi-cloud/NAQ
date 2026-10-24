# CLAUDE.md — NAQ 리포 위 anchor 연구 (HL 트랙)

개정 1 · 2026-10-24 · 작성 이희태
이 파일은 Agent가 매 세션 처음에 읽는 상태 파일이다. 수정은 이희태만 한다. Agent는 이 파일을 고치지 않는다.

## 0. 이 리포가 무엇인가

- 원격: `choi-cloud/NAQ`(브랜치 `naq_exp`, HEAD `1a20f23`)를 clone한 것. 원본은 NaQ(ICML'24, `JhngJng/NaQ-PyTorch`) 공식 코드이고, 그 위에 공동연구자 Choi가 `exp001`–`exp049` 실험 스크립트를 쌓았다.
- 우리(HL 트랙)는 **anchor(support) 샘플링** 자유도를 다룬다. Choi의 작업은 query 쪽이다. 두 작업은 같은 원시 신호(ÂX·Â²X 코사인 필터)를 쓰되 자유도가 다르다.
- 작업 브랜치: `hl/anchor` (`naq_exp` HEAD에서 분기). `naq_exp`에는 아무것도 쓰지 않는다.

## 1. 디렉터리 규약

```
<repo root>
├── data.py, meta.py, model.py, utils.py, task_generator.py, arg.py,
│   unsup_protonet.py, sup_protonet.py, naq_train.py, sup_train.py   ← NaQ 원본 (불변)
├── choi_exps/        ← Choi의 exp001–exp049 .py/.sh, run_exp049.py, shell.sh (HL01에서 git mv; 이후 불변)
├── data/             ← 원저자 공개 데이터 (불변, git 미추적)
├── save/fixed_episodes/   ← 고정 valid/test 에피소드 JSON (생성 후 불변, git 미추적)
├── env/              ← NAQ_conda.yml, requirements_pip.txt
├── hl/               ← 우리 스크립트. 번호 접두사 hl###_*.py
├── instructions/     ← 지시서 HL##_*.md (이희태 작성)
├── reports/          ← Agent 보고서 HL##_*.md (Agent 작성)
├── results/hl/       ← 결과 json·체크포인트·csv (git 미추적)
└── out/hl/           ← 실행 로그·PID 파일 (git 미추적)
```

- **불변 파일**: 루트의 NaQ 원본 10개, `choi_exps/` 전체, `data/`, 생성된 `save/fixed_episodes/*.json`. 수정·삭제·덮어쓰기 금지. 필요한 변형은 `hl/`에 복사해 만든다(복사 출처를 docstring 첫 줄에 적는다).
- `.gitignore`는 Choi의 것을 그대로 둔다(`.py`/`.sh` 외 미추적). `.md`·`.yml`·`.txt`·`.csv`를 추적하려면 HL01의 `.gitignore` 추가 규칙을 따른다.

## 2. 실행 환경

- conda 환경: 서버의 기존 환경 **`taskns`**(`/home/h2tae/.conda/envs/taskns`)를 쓴다. HL01 §1의 점검을 통과하지 못하면 2순위 `ctx-teg`, 둘 다 실패하면 `env/NAQ_conda.yml`로 새 환경 `NAQ`를 만든다. **기존 환경(`taskns`, `ctx-teg`)에는 패키지를 설치·업그레이드·삭제하지 않는다** — 다른 트랙과 공유한다. 실제로 쓴 환경 이름은 모든 보고서 ①에 적는다.
- 모든 스크립트는 **리포 루트에서** 실행한다. Choi 스크립트는 `PYTHONPATH=$PWD python choi_exps/expNNN.py ...` 형태로 호출한다(스크립트 디렉터리가 sys.path[0]이 되어 `from exp026 import`가 풀리고, PYTHONPATH로 루트의 `data.py` 등이 풀린다). `choi_exps/run_exp049.py`는 상대 경로 `exp026.py`를 subprocess로 부르므로 **그대로는 쓰지 않는다**. 우리 러너 `hl/hl000_run.py`를 쓴다.
- GPU: `--device 0` 고정. 동시에 돌리는 run은 최대 2개.
- 백그라운드 실행은 `nohup ... > out/hl/<이름>.log 2>&1 & echo $! > out/hl/<이름>.pid`로 하고, **감시는 PID 파일로만** 한다(`kill -0 $(cat pid)`). 로그 tail을 반복 polling하지 않는다.

## 3. 고정 프로토콜 (Choi exp027/exp049와 동일, 변경 금지)

- 학습: 5-way 1-shot pseudo-episode, q_query 10, 2000 에피소드(학습 전 일괄 생성), latent 256, lr 1e-4, Adam, `meta.ProtoNet.forward` 그대로.
- 평가: 160 에피소드마다 고정 valid 50개로 제곱 유클리드 최근접 프로토타입 정확도. 1-shot·5-shot 각각 best valid 체크포인트 선택 → 고정 test 500개 1회. `exp027.evaluate` 사용.
- 고정 에피소드: `save/fixed_episodes/{dataset}/5way_{shot}shot_8query_seed{seed}/{valid,test}_episodes.json`. 없으면 `choi_exps/exp026.py`로 생성(파일이 있으면 exp026이 거부한다 — 정상).
- 기준선 정의: **NaQ** = `exp049 --method naq`. **B_Choi** = `exp049 --method ours --tau τ*`(ÂX·Â²X 코사인이 **둘 다** τ 이상인 노드만 남기고, X 코사인 순 정렬의 정중앙 10개를 query로; 10개 미만이면 min(s1,s2) 상위 10개로 relax).
- 데이터셋 순서: Amazon_clothing → dblp → Amazon_eletronics(철자는 원본 그대로 `eletronics`). Cora_full은 쓰지 않는다.

## 4. 레이블 사용 규칙

- Ground-truth 레이블은 **oracle arm과 진단 통계에서만** 읽는다. 실제 샘플러(anchor·query 선택)에는 절대 쓰지 않는다. 샘플러 함수에는 `labels`를 인자로 넘기지 않는다.
- oracle 스크립트는 파일명에 `oracle`을 넣고, 보고서에 "레이블 사용 범위"를 한 줄로 적는다.

## 5. 지시서·사전 등록·보고서

- 지시서 `instructions/HL##_<이름>.md`는 이희태가 쓴다. Agent는 지시서에 적힌 것만 한다. 지시서에 없는 판단이 필요하면 §7 규칙을 따른다.
- 사전 등록 `HLP###`: 지시서에 적힌 예측. Agent는 **계산값만** 표에 채운다. 맞음/틀림 판정은 이희태가 한다. 결과를 본 뒤 예측 문구를 바꾸지 않는다.
- 보고서 `reports/HL##_<이름>.md`: 사실만 쓴다(해석·추측·제안 금지). 구성: ① 실행 명령·환경·commit ② 표(요청된 모든 수치) ③ 사전 등록 대조표(계산값만) ④ 실패·경고·이상 ⑤ 산출물 경로. 수치는 보고서에 손으로 옮기지 말고 스크립트가 쓴 csv/json을 인용한다.
- 번호 레지스트리: HL01 환경·smoke·P0 / HL02 학습 없는 진단 + anchor oracle 사다리(예정) / 이후 이희태가 추가.
- seed: 1212(Choi 재현용), 1213, 1214. 다른 값은 지시서에 명시될 때만.

## 6. git 규약

- 지시서 하나 = 브랜치 `hl/anchor` 위의 커밋 묶음 하나 = 태그 하나(`hl01`, `hl02`, …). 지시서 안에 독립 실험이 여럿이면 실험별로 커밋을 나눈다(`hl01-env`, `hl01-smoke`, `hl01-p0`).
- 커밋 메시지 첫 줄: `HL01: <무엇을>`. 결과 csv·보고서 `.md`는 커밋에 포함한다(`.gitignore` 예외 규칙은 HL01 §0). 체크포인트·json·로그는 포함하지 않는다.
- push: `git push origin hl/anchor` 와 태그. 실패하면 오류를 보고서 ④에 적고 계속한다. `naq_exp`에 push 금지, force push 금지.

## 7. 멈춤·auto 규칙

- **멈추는 경우(채팅으로 보고하고 대기)**: (a) 불변 파일을 고쳐야만 진행되는 상황, (b) 지시서가 "정지"로 표시한 게이트 실패, (c) 기존 환경에 설치가 필요해 보이는 상황(설치하지 말고 멈춘다) 또는 새 환경 생성이 지시서의 허용 범위 밖 패키지를 요구, (d) GPU 메모리 부족·디스크 부족.
- **멈추지 않는 경우**: 수치가 예측과 다름(기록만), 한 run 실패(오류 로그 남기고 다음 run), push 실패.
- auto 모드는 지시서 또는 채팅 지시에 "auto"라고 명시된 경우에만. auto에서는 질문하지 않고 §7의 멈춤 사유가 아니면 끝까지 간다.
- 애매하면 **덜 하는 쪽**을 택하고 보고서 ④에 적는다.