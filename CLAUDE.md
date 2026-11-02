# CLAUDE.md — NAQ 리포 위 에피소드 연구 (HL 트랙)

개정 11 · 2026-11-02 · 작성 이희태 (개정 1: 10-24, 개정 2–8: 10-25, 개정 9·10: 11-02)
이 파일은 Agent가 매 세션 처음에 읽는 상태 파일이다. 수정은 이희태만 한다. Agent는 이 파일을 고치지 않는다.
개정 11 변경: §0 HL12 결과·현재 질문, **§0-1 야간 시퀀스(11-02 21:30 → 11-03 08:30 KST, HL13)**, **§4 base 클래스 = 40(train.mat 57 − 고정 valid 17)**, §5 레지스트리 HL13·스크립트 hl015·hl016, **§7-1 야간 모드(seio 한 장용) 복원**, §8 HL12 확정 사실·Clothing 클래스 구성 정정.
개정 10 변경: §0 현재 질문(HL11 결과 반영 — 정규화·PCA), **§3 평가 규약(raw·l2 둘 다, 주 지표 l2)**, §4 base 레이블 사용 허용 명시, §5 레지스트리 HL12·스크립트, §8 HL11 확정 사실, §9 ReLU 출력·노름 메모.
개정 9 변경: 서버 복귀(gpusystem → seio, 환경 `taskns`, GPU 1장), §0 스토리 D·ND 종결, 야간 모드 삭제, §5 HL11, §8 HL06–10·ND01, §9 seio 자산.
개정 8 변경: §0-1 야간 시퀀스(HL06–HL10), §7-1 야간 모드. 개정 7: 스토리 D 시작, GPU 2·3번. 개정 4–6: 서버 이전(gpusystem), 스토리 C. 개정 3: anchor 트랙 종결.

## 0. 이 리포가 무엇인가

- 원격: `choi-cloud/NAQ`(브랜치 `naq_exp`, HEAD `1a20f23`)를 clone한 것. 원본은 NaQ(ICML'24, `JhngJng/NaQ-PyTorch`) 공식 코드이고, 그 위에 공동연구자 Choi가 `exp001`–`exp049` 실험 스크립트를 쌓았다.
- 작업 브랜치: `hl/anchor` (`naq_exp` HEAD에서 분기, 이름은 유지). `naq_exp`에는 아무것도 쓰지 않는다. 별도 브랜치 `nd/anchor-vote`(ND 트랙, 종결)는 건드리지 않는다.
- **트랙 상태(2026-11-02)**:
  - 스토리 A·B(anchor, HL01–02): 종결. `reports/HL_anchor_track_closure.md`.
  - 스토리 C(먼 query 순도 회복, HL03–05): 종결. `reports/HL_storyC_closure.md`.
  - 스토리 D(순도·거리 분리 query 선택, HL06–10): **종결**(2026-11-02). HLP078 +0.51 < +2.0; 레이블 없는 arm이 세 데이터셋 모두 B_Choi 미달. `reports/HL_storyD_closure.md`(사실 D1–D9, C2 정정 포함).
  - ND 트랙(이웃 분포 패턴, 브랜치 `nd/anchor-vote`, ND01–02): 종결. Â²X 위 이득 없음. 그 브랜치의 `reports/ND_closure.md`.
  - HL11(완료, 2026-11-02): 학습 없는 Â²X 기준선의 통제. 결과가 그림을 바꿨다(§8) — **L2 정규화 평가에서 학습 모델(NaQ 87.84, B_Choi 87.75)은 Â²X_l2 85.10을 넘고, PCA-64(Â²X)_l2 89.05가 그 위에 있다.** 비정규화 평가(exp027)는 ReLU 출력의 노름에 지배된다. B_Choi의 NaQ 대비 +2(raw)는 l2에서 사라진다.
  - HL12(완료, 2026-11-02): 에피소드 출처 × 인코더 × 손실(§8). **base 레이블 pure 에피소드로 GCN을 학습하면 레이블 없는 pseudo-episode보다 novel 전이가 훨씬 나쁘다**(81.2 vs 87.7, l2). PCA-64 위 항등 초기화 선형층은 어느 에피소드로도 +0.1뿐. cosine 손실은 GCN에 해롭다(84.2). 무작위 초기화 MLP는 PCA 정보를 망가뜨린다(83.8).
  - **현재 질문(HL13–)**: 학습이 PCA-64(89.05)를 **넘을 수 있는가** — (A) 사영을 처음부터 에피소드로 배워 PCA가 하는 일(저분산 방향 제거)에 닿는가, (B) PCA-64 위 선형층이 안 움직인 것이 lr 때문인가, (C) GCN(raw)에 구조 soft target·소량 base 혼합이 더하는가, (D) oracle 상한은 l2에서 몇 점인가. 그리고 RQ1의 새 형태: **왜 base 클래스 순도 1.0 에피소드가 novel 전이를 깎는가**(supervision collapse).
- 레이블 없는 판정기로 query를 고르는 실험(스토리 C·D 계열)은 지시서에 명시될 때만 한다. 같은 계열 12종이 세 데이터셋 모두에서 실패했다(§8).

### 0-1. 야간 시퀀스 (2026-11-02 21:30 → 11-03 08:30 KST) — 지시서 `instructions/HL13_overnight_mix_soft_oracle.md`

HL12가 끝난 직후 시작한다. arm은 지시서에 고정되어 있다(HL12 결과 반영). 단계는 서로 독립이고 한 단계가 실패해도 다음으로 간다(§7-1). 마감 08:15 이후 새 run 금지, 08:30까지 보고서 push.

## 1. 디렉터리 규약

```
<repo root>
├── data.py, meta.py, model.py, utils.py, task_generator.py, arg.py,
│   unsup_protonet.py, sup_protonet.py, naq_train.py, sup_train.py   ← NaQ 원본 (불변)
├── choi_exps/        ← Choi의 exp001–exp049 .py/.sh, run_exp049.py, shell.sh (불변)
├── data/             ← 원저자 공개 데이터 (불변, git 미추적)
├── save/downstream_tasks/<dataset>/5way_{1,5}shot.pkl  ← 고정 valid 50 / test 500 (불변, git 미추적)
├── save/fixed_episodes/   ← 고정 valid/test 에피소드 JSON (생성 후 불변, git 미추적)
├── env/              ← 환경 점검 기록
├── hl/               ← 우리 스크립트. 번호 접두사 hl###_*.py, 공용 hl_*.py
├── instructions/     ← 지시서 HL##_*.md (이희태 작성)
├── reports/          ← Agent 보고서 HL##_*.md + reports/HL##_tables/*.csv (Agent 작성), 종결 문서
├── results/hl/<HL##>/  ← 지시서별 결과 json·체크포인트·csv (git 미추적)
└── out/hl/           ← 실행 로그·PID 파일 (git 미추적)
```

- **불변 파일**: 루트의 NaQ 원본 10개, `choi_exps/` 전체, `data/`, `save/` 아래 전부. 수정·삭제·덮어쓰기 금지. 필요한 변형은 `hl/`에 복사해 만든다(복사 출처를 docstring 첫 줄에 적는다). `choi_exps/`의 **함수를 import해서 쓰는 것은 허용**된다.
- 기존 `hl/` 스크립트(hl000–hl011, hl_common, hl_pdd 등)는 HL06–10 보고서가 참조하므로 **수정하지 않는다**. 새 지시서는 새 번호 스크립트를 만든다(함수 import는 허용).
- `.gitignore`: Choi의 규칙 + `!*.md !*.yml !*.txt !*.csv`. `git add`는 `reports/`, `env/`, `hl/`, `instructions/`, `CLAUDE.md`로 한정. `results/`·`out/`은 커밋하지 않는다.

## 2. 실행 환경 (seio, 구 서버 복귀)

- **서버**: `seio`, 리포 경로 `~/wkspace/NAQ`. GPU **RTX 4080 SUPER 1장(16GB)**, `--device 0`. 다른 사람과 공유하므로 시작 전 `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv`를 기록하고, 여유가 **10GB 미만**이면 멈춘다(§7(d)).
- **conda 환경 `taskns`**(`/home/h2tae/.conda/envs/taskns`): python 3.9.25, torch 2.4.1+cu121, PyG 2.6.1, scipy 1.13.1, sklearn 1.5.2(`env/envcheck_taskns.txt`). **아무것도 설치하지 않는다**(다른 트랙과 공유). 새 환경을 만들지 않는다. `taskns`가 깨져 있으면 2순위 `ctx-teg`, 둘 다 안 되면 멈춘다(§7(c)).
- **동시 실행**: 학습 run은 **동시 1개**(run당 예약 8.2GB, 작업 메모리 포함 약 10GB). 학습 없는 진단(dense 행렬)도 학습 run과 겹치지 않게. run 시간: Clothing B_Choi 약 50초(HL02 측정), NaQ 약 45초.
- **데이터셋: 이 서버에서는 Amazon_clothing만.** dblp·electronics는 X 코사인 dense 행렬(6.6–7.2GB)과 학습 run 메모리(13.9–16.3GB)가 16GB를 넘거나 근접해 돌리지 않는다. gpusystem 복구 뒤 그쪽에서 한다.
- 모든 스크립트는 **리포 루트에서** 실행한다. `PYTHONPATH=$PWD`, `choi_exps/`를 sys.path에 넣어 함수를 import한다(`hl/hl_common.py` 방식).
- 백그라운드 실행은 `nohup ... > out/hl/<이름>.log 2>&1 & echo $! > out/hl/<이름>.pid`, **감시는 PID 파일로만**. 로그 tail을 반복 polling하지 않는다.
- gpusystem 기록(참고): 4×RTX 6000 Ada, 환경 `NAQ`(torch 2.5.1, PyG 2.8.0). HL03–10 수치는 거기서 측정됐다. 절대값은 서버·환경 간에 0.1–0.3 다를 수 있고, **같은 서버 안의 Δ**만 판단 근거다.

## 3. 고정 프로토콜 (Choi exp027/exp049와 동일, 변경 금지)

- 학습: 5-way 1-shot pseudo-episode, q_query 10, 2000 에피소드(학습 전 일괄 생성), latent 256, lr 1e-4, Adam, `meta.ProtoNet.forward` 그대로(제곱 유클리드 → log_softmax → NLL).
- 평가: 160 에피소드마다 고정 valid 50개로 제곱 유클리드 최근접 프로토타입 정확도. 1-shot·5-shot 각각 best valid 체크포인트 → 고정 test 500개 1회. `exp027.evaluate`·`episode_accuracy` 사용. **학습 없는 특징의 평가도 같은 함수·같은 pkl**.
- **평가 규약(HL12부터)**: 모든 arm을 **raw 임베딩과 L2 정규화 임베딩 둘 다**로 평가하고, 체크포인트 선택도 raw valid(규칙 R)·l2 valid(규칙 L) 둘 다 둔다. **주 지표는 l2 valid로 고른 체크포인트의 l2 test(`bestL_test_l2`)**. raw 값은 exp027·HL01–11 수치와의 연결용으로 항상 같이 보고한다. 이유: GNNEncoder 출력이 ReLU 뒤라 비음수이고 노름 편차가 커서 raw 제곱 유클리드가 노름에 지배된다(HL11, `encb_raw` 83.2 → `encb_l2` 87.7).
- 고정 에피소드: `save/downstream_tasks/Amazon_clothing/5way_{1,5}shot.pkl`(md5 1-shot `5f901727f81903cc12159d280be6915d`, 5-shot `ca3edaca02d541cbc0ee15a2122ef24a` — **시작 시 확인**). `save/fixed_episodes/Amazon_clothing/.../seed{1212..1216}` 존재.
- 기준선 정의: **NaQ** = `exp049 --method naq`. **B_Choi** = `exp049 --method ours --tau 0.3`. 필터 = ÂX·Â²X 코사인이 **둘 다** τ 이상인 노드만 남김 → X 코사인 정렬의 정중앙 10개 → 10개 미만이면 relax.
- Â = D^-1/2 (A + I) D^-1/2 (대칭화한 A + self-loop). z0·z1·z2 = 행 정규화 X·ÂX·Â²X — `hl_common.structural_scores`(exp049와의 동일성 assert 있음).
- seed 표준: **5개 — 1212, 1213, 1214, 1215, 1216**. 탐색은 1212 하나, 판정은 5개. 학습 없는 결정론적 평가는 seed 없음(500 에피소드 표집 CI만).
- CI: `1.96 × std(ddof=0) / √n`(exp027과 같은 식). 고정 에피소드는 모든 arm이 같으므로 arm 간 비교는 **에피소드별 쌍 차이**(paired Δ, CI, 승률)로 보고한다.

## 4. 레이블 사용 규칙

- 기준은 하나다: **novel(valid·test) 클래스의 레이블이 학습에 들어가지 않는다.** **base 클래스 = train.mat의 57개에서 고정 valid pkl의 17개를 뺀 40개**(seed와 무관하게 고정; seed 1212의 `data.class_list_train`과 같은 집합. `load_data`는 seed마다 57개를 40/17로 다르게 나누므로 `class_list_train`을 그대로 쓰면 seed 1213–1216에서 고정 valid 클래스가 섞인다 — HL12 ④-1). 이 40개의 레이블은 에피소드 생성·학습에 써도 된다(arm 이름에 `sup`). 레이블 없는 노드는 규칙 (A)대로 전체 노드를 수동적으로 쓰고, "base가 아닌 노드"만 골라 쓰는 규칙 (B)는 금지. valid·test 클래스 레이블은 oracle arm·진단·meta-test 평가에서만. 특징 변환(PCA 등)에는 어떤 레이블도 쓰지 않는다. `hl_common.labels_np`로만 읽는다.
- oracle·진단 스크립트는 파일명에 `oracle` 또는 `diag`를 넣고, 보고서에 "레이블 사용 범위"를 한 줄로 적는다.

## 5. 지시서·사전 등록·보고서

- 지시서 `instructions/HL##_<이름>.md`는 이희태가 쓴다. Agent는 지시서에 적힌 것만 한다.
- 사전 등록 `HLP###`: Agent는 **계산값만** 표에 채운다. 판정은 이희태. 결과를 본 뒤 예측 문구를 바꾸지 않는다.
- 보고서 `reports/HL##_<이름>.md`: 사실만(해석·추측·제안 금지). ① 실행 명령·환경·commit ② 표 ③ 사전 등록 대조표 ④ 실패·경고·이상·지시서에 없어 정한 것 ⑤ 산출물 경로. 표 csv는 `reports/HL##_tables/`에 두고 커밋한다.
- 번호 레지스트리: HL01(HLP001–006) / HL02(007–023) / HL03(024–035) / HL04(036–049) / HL05(050–056) / HL06(057–079) / HL07(080–092) / HL08(093–102) / HL09(103–112) / HL10(113–118) — 모두 완료 / HL11 학습 없는 Â²X 기준선 + 학습 곡선(완료, HLP119–130) / HL12 에피소드 출처 × 인코더 × 손실(완료, HLP131–142) / **HL13 야간: 사영 학습(SGC)·lr 스윕·soft target·GCN 혼합·oracle l2·inductive PCA·10-seed·노름 진단(HLP153–168)** / 이후 이희태가 추가.
- 스크립트 번호: hl000(·b–f) 러너, hl001–hl011·hl_common·hl_pdd·hl_pdd_diag·hl_hl##_tables(완료, 불변) / hl012 학습 없는 특징 평가, hl013 학습 곡선, hl000g 러너, hl_hl11_tables(HL11 완료, 불변) / hl014 학습 변형(에피소드 출처·인코더·손실·선택 규칙 R/L), hl000h 러너, hl_hl12_tables(HL12) / **hl015 HL13 학습(hl014 복사 + sgc·pcainit 인코더, lr, soft 손실, mix·oracle 에피소드, pca_fit), hl016 노름 진단(학습 없음), hl000j 러너(deadline, gcn/lin 2슬롯), hl_hl13_tables** / 이후 hl017부터.

## 6. git 규약

- 지시서 하나 = `hl/anchor` 위의 커밋 묶음 = 태그 묶음(`hl11-feat`, `hl11-curve`, …). 독립 실험마다 커밋을 나눈다.
- 커밋 메시지 첫 줄: `HL11: <무엇을>`. 결과 csv·보고서 `.md`는 포함, 체크포인트·json·로그·`wall_seconds.txt`는 제외.
- push: `git push origin hl/anchor --tags`. 실패하면 보고서 ④에 적고 계속. `naq_exp`·`nd/anchor-vote` push 금지, force push 금지.
- git identity는 리포 로컬 config(`dragor0123`, 이희태 메일). 전역 설정은 건드리지 않는다.

## 7. 멈춤·auto 규칙

- **멈추는 경우(채팅 보고 후 대기)**: (a) 불변 파일을 고쳐야만 진행되는 상황, (b) 지시서가 "정지"로 표시한 게이트 실패, (c) `taskns`에 지시서에 없는 설치가 필요해 보이는 상황(설치하지 말고 멈춘다), (d) GPU 여유 10GB 미만·디스크 부족, (e) pkl md5 불일치.
- **멈추지 않는 경우**: 수치가 예측과 다름(기록만), 한 run 실패(오류 로그 남기고 다음 run), push 실패.
- auto 모드는 지시서 또는 채팅에 "auto"라고 명시된 경우에만. auto에서는 질문하지 않고 §7의 멈춤 사유가 아니면 끝까지 간다.
- 애매하면 **덜 하는 쪽**을 택하고 보고서 ④에 적는다.

### 7-1. 야간 모드 (night-auto; 채팅에 "night-auto"라고 명시된 경우, seio 한 장용)

목적: 사람이 없는 동안 **GPU가 비는 상태를 만들지 않는다.** §7의 (a)·(c)·(d)와 지시서의 "정지"는 야간에는 **"그 단계(또는 arm)를 건너뛰고 기록한 뒤 다음으로"**가 된다. 멈추는 경우는 둘뿐: ① 시작 시점에 HEAD·브랜치가 지시서와 다를 때, ② GPU 여유가 10GB 미만일 때(30분 간격 세 번 재확인 뒤). pkl md5 불일치(e)는 야간에도 멈춘다.
- 단계 독립: 한 단계의 코드·smoke·실행이 실패하면 `results/hl/hl13/progress.md`와 보고서 ④에 적고 다음 단계로. 되돌아와 재시도하지 않는다.
- arm 독립: smoke(epoch 20)에서 실패한 arm은 한 번만 고쳐 보고 재실패 시 그 arm만 뺀다.
- 시간 상한: 단계별 목표 시각과 큐 deadline이 있다. deadline이 지나면 새 run을 시작하지 않고(실행 중인 것은 끝까지) 완료된 run으로 표를 만들고(seed 수 표기) 다음 단계로. run 하나가 wall 10분을 넘기면 러너가 kill.
- 동시 실행: GCN run 1개 + 선형(`pca*`·`sgc*`) run 1개까지 동시(러너 슬롯 2개). `sgc*`는 Â²X 9,034차원 dense(0.9GB)를 GPU에 올리므로 GCN `ours`(7.6GB)와 겹쳐도 16GB 안이다.
- 진행 기록: 단계 시작·끝·건너뛰기마다 `results/hl/hl13/progress.md`에 한 줄(`TZ=Asia/Seoul date`, 단계, 상태, 완료/전체 run, 커밋). 단계 커밋 때 `reports/hl13_progress.md`로 복사해 커밋.
- 세션 복구: 끊기면 CLAUDE.md → progress.md → 미완료 단계 지시서 순으로 읽고 완료된 run은 건너뛰며 이어간다.
- 채팅 출력 최소화: 단계 전환 때 한 줄. 표는 아침 보고에 한 번. 질문하지 않는다.
- 야간에도 변하지 않는 것: 불변 파일 금지, 레이블 규칙(§4), 사전 등록 문구 변경 금지, `naq_exp`·`nd/anchor-vote` push 금지, 추가 설치 금지.

## 8. 확정 사실 (Agent가 전제로 삼아도 되는 것)

HL01·HL02는 구 서버(seio, `taskns`), HL03–10은 gpusystem(`NAQ`), ND01–02는 seio(`taskns`) 측정값. 절대값은 서버 간 소수점 단위로 다를 수 있다.

- **기준선(Clothing, 1-shot / 5-shot, 고정 test 500)**: NaQ 80.65 / 90.52(seed 1212; 3 seed 80.99 ± 0.41 / 90.78), **B_Choi(τ=0.3) 82.71 ± 0.31 / 89.91 ± 0.57**(5 seed; 구 서버 82.70 ± 0.29 / 89.91). dblp: NaQ 68.51 / 81.96, B_Choi 71.01 / 80.94. electronics: NaQ 68.93 / 83.68, B_Choi 70.26 / 82.93. B_Choi의 5-shot은 세 데이터셋 모두 NaQ보다 낮다.
- **학습 없는 특징(HL11, Clothing, test 500, 1-shot / 5-shot)**: X_l2 72.35 / 87.61, ÂX_l2 82.47 / 91.79, **Â²X_l2 85.10 / 92.72**, Â³X_l2 85.47 / 92.40, [ÂX‖Â²X]_l2 85.04. 비정규화는 전부 붕괴(X_raw 36.4, Â²X_raw 60.6). **PCA(Â²X_l2)_l2: 64차원 89.05 / 93.72, 256차원 88.55 / 93.14, 1024차원 87.92** — 차원이 낮을수록 좋다(설명 분산 0.53 / 0.75 / 0.93). 무작위 사영 256은 81.8 ± 0.7 → PCA의 이득은 저분산 방향 제거. PCA는 전체 노드에 레이블 없이 fit.
- **학습 모델의 l2 평가(HL11)**: B_Choi `encb_raw` 83.20 → `encb_l2` **87.75**, NaQ `encn_raw` 80.64 → `encn_l2` **87.84**(seed 1212 체크포인트). 학습 곡선(5 seed, l2 test 1-shot): epoch 0(무작위 GCN) 72.55 = X_l2 수준 → NaQ 최대 87.72(epoch 1120), B_Choi 최대 87.19(epoch 1600). raw 곡선은 54.3 → 80.7 / 82.8. **학습은 l2 기준 +15를 더하지만 PCA-64 Â²X(89.05)에 −1.3 못 미친다.** B_Choi − NaQ: raw +1.8, l2 −0.1 ~ −0.5.
- **oracle 레버(HL03–07)**: B4 구간 순도 p 0.4 → 1.0: 83.92 → 89.42. 순도 1.0에서 strat5(Â²X 층에 고루) **90.12 / 94.74** > far50 89.67 > t1 87.20; 순도 0.3에서 strat5 84.45 > t1 82.68 > far50 79.49. **레버는 순도 × 구조적 다양성**(스토리 D 문서 D3, C2 정정). dblp·electronics에서 재현.
- **레이블 없는 query 선택은 전부 실패(HL04·06–10)**: 판정기 12종 × 규칙 7종 × 변형, 최고 Clothing `knnppr_ex20` −0.78, dblp −2.71, electronics −0.45. 판정기 순도 벽 0.32–0.35(Choi 필터 0.42). Choi 필터 + 거리 통제 ±0.2. FP를 정확히 빼도(−1.59) teacher를 갱신해도(−2.25) 나빠진다.
- **ND(별도 브랜치)**: 이웃 투표 분포는 Â²X 위에 −0.16(승률 0.08). B4 AUROC: `ax` 0.728, `a2x` 0.753, `enc_b` 0.775; 구조가 든 특징 위에서는 anchor 경쟁·이웃 투표·`a2x`가 0.002 이내. Clothing 호모필 비율 0.86, inter 엣지 없는 노드 2/3.
- **HL02 진단**: X 상위 50 중 Choi 필터 탈락 쌍의 same-GT 0.218, 통과 쌍 0.574, 무작위 쌍 0.020. B_Choi query 순도 0.417, NaQ 0.597. 2000 에피소드 × 5 anchor로 anchor에 뽑히는 노드는 전체의 약 1/3.
- Amazon_clothing: 24,919 노드, 9,034 특징, 77 클래스 — **train.mat 57(= base 40 + 고정 valid 17) / test 20**(HL12 ④-1·15에서 정정; 이전 판의 "train 57 / valid·test 20"은 틀림). base 40클래스 크기 최소 101 / 중앙값 201 / 최대 923. 대칭 A(self-loop 제외) nnz 183,360, 고립 노드 0.
- **HL12(Clothing, 1-shot / 5-shot, `bestL_test_l2`)**: `gcn_raw_naq` **87.72** / 92.13, `gcn_raw_ours` 87.09 / 91.32, `gcn_raw_sup`(base 40 pure) **81.19** / 90.28(epoch 160 정점 82.6 뒤 단조 하락 → 79.5), `gcn_raw_sup_strat` 81.69, `gcn_cos_naq` 84.17, `gcn_cos_ours` 86.54, `gcn_cos_sup` 78.75. `pca64_lin_cos_{naq,ours,sup,sup_strat}` 89.14–89.17 / 93.76–93.97(시작 89.05 / 93.72; 쌍 Δ +0.10~+0.13, CI ±0.1, 승률 0.21–0.24; lr 1e-4). `pca256_lin_cos_sup` 88.97, `pca256_mlp_cos_sup` 83.75(epoch 0 81.1). scale s=5/10/20: 89.33/89.12/89.20(seed 1212). 사실: (i) base pure 에피소드 < 레이블 없는 에피소드(−6.5); (ii) PCA-64 위 선형층은 어느 에피소드로도 +0.1; (iii) cosine 손실은 GCN에 −3.5, 단 그 아래서는 B_Choi > NaQ +2.4; (iv) `sup_strat` − `sup` = +0.5(GCN), −0.01(PCA).

## 9. 운영 메모

- **seio에 없는 자산(gpusystem에만)**: `results/hl/hl03/`–`hl10/`(run 결과·체크포인트·`hl03/g0/anchors.npy`), dblp·electronics 고정 에피소드와 pkl. 필요한 수치는 커밋된 `reports/HL*_tables/*.csv`를 인용한다. anchor 집합이 필요하면 새로 뽑고(`default_rng(1212)`) 파일을 새 경로에 둔다 — HL06 P2와 ND01 TB1의 `a2x` AUROC 차이(0.7564 vs 0.7534)가 anchor 집합 차이의 크기다.
- **seio에 있는 자산**: `data/`(세 데이터셋), Clothing pkl·고정 에피소드(seed 1212–1216), `results/hl/p0/Amazon_clothing/{ours/tau_0.3,naq/baseline}/seed_1212/best_1shot.pt`(B_Choi·NaQ 체크포인트, enc_b·enc_n), `results/hl/hl02/diag/node_table_tau0.3.csv`, `results/nd/`(ND01–02, 참고용).
- `hl/hl000_run.py`는 호출마다 `summary.csv`를 덮어쓴다. 단계별 사본을 남기고 결과 루트를 `results/hl/<HL##>/`로 분리한다.
- `choi_exps/exp026.py`는 `results/Amazon_clothing/exp026/result.txt`를 덮어쓴다(부작용, 무해). 커밋하지 않는다.
- `utils.py:83`의 `torch.sparse.SparseTensor` deprecation 경고는 무해. 보고서에 반복해서 적지 않는다.
- **평가 에피소드는 seed와 무관하게 같다**(pkl 캐시). seed 간 분산은 학습 쪽만 반영한다. 캐시 pkl은 건드리지 않는다.
- `data.add_diffusion(args)`가 X 코사인 dense 행렬(`Diffusion_matrix`, Clothing 2.5GB)을 만든다. 학습 없는 특징 평가에는 필요 없으므로 **호출하지 않는 경로**를 둔다(`hl_common.load_everything`은 호출함 — 복사해 분기).
- `hl005.encoder_embeddings(data, path, 256, device)`로 체크포인트의 행 정규화 임베딩을 얻는다. 정규화 **전** 출력이 필요하면 같은 함수를 복사해 `F.normalize`를 뺀 판을 만든다(exp027은 정규화하지 않은 출력으로 평가한다).
- 같은 설정 재학습 시 1-shot test가 0.1%p 안에서 흔들린다(HL11에서 seed 1212 B_Choi가 valid 동률 때문에 −0.44, 나머지 ±0.05). 0.3%p 미만의 차이는 판단 근거로 쓰지 않는다. `H.structural_scores`의 z1·z2는 프로세스마다 소수 5자리에서 다를 수 있다(원인 미확인, 판정에 영향 없음).
- `GNNEncoder`는 config의 `linear` 항목을 건너뛰고 GCN 층만 적용하며, **마지막 GCN 층 뒤에도 ReLU**가 있다(`model.py:55`). 따라서 256차원 출력은 비음수이고 노름 편차가 크다 — raw 제곱 유클리드 평가가 노름에 지배되는 원인. `meta.ProtoNet`도 수정하지 않고, 변형은 `hl/`에 복사한 클래스로 만든다.
- PCA 특징은 run마다 다시 계산하지 않고 `results/hl/hl12/feat/pca{64,256}.npy`(md5 `3d432fd1…`, `eb97f813…`; `pca_meta.json`에 components·mean)에 저장해 공유한다. HL13의 `pca1024.npy`·`pca64_base.npy`는 `results/hl/hl13/feat/`.
- `sup_strat`·`ours` 에피소드 풀은 Â²X 코사인의 프로세스 간 미세 차이 때문에 같은 seed라도 run마다 query 슬롯이 일부(sup_strat 3.8%) 다르다(HL12 ④-3). 풀 해시는 기록만 하고 assert하지 않는다.
