# CLAUDE.md — NAQ 리포, 브랜치 `hl/hsan-weighting` (HW 트랙)

개정 4 · 2026-11-07 · 작성: 이희태 (Claude 정리)
이 파일은 이 브랜치에서 Claude Code Agent가 따르는 규약이다. 지시서(`instructions/HW##_*.md`)가 이 파일과 어긋나면 **지시서가 우선**하되, 어긋난 항목을 보고서에 적는다.

> 개정 4 (HW02·HW03-prep 반영): 두 서버 운영 규칙(§2, §6), 서버별 결과 루트와 짝지은 비교 규칙(§4.3), iislab4 10 seed(§4.4), `euclid_gradscale` 종결과 cos τ·이웃 선택 방식(`pick`) 추가(§4.2), 측정 전용 레이블 사용 허용(§5), git 규칙(§6-9), 사전 등록 번호(§8), 현재 상태(§9).

---

## 0. 이 트랙이 하는 일

- 석휘(Choi)의 새 계열 NaQ 실험(표 1: exp001, exp012, exp015, exp016) 위에서 석휘 아이디어 1(NCLA식 이웃 양성 확장)과 2(HSAN식 hard-sample 가중치)를 결합해 성능을 끌어올린다. 아이디어 3(hard negative)은 그 뒤.
- 석휘의 입장(2026-11-07): BGRL 사전학습 자체가 중요한 것이 아니다. 어떤 사전학습 임베딩 위에서도 1+2가 이득을 주는지가 핵심이다. 따라서 선택 임베딩(raw/BGRL/…)은 실험 축이다.
- 모든 실험은 **단일 실행 파일(`hw/run.py`) + 단일 설정 파일(`configs/hw.json`) + results/ + reports/** 위에서 돌린다.

## 1. 트랙·브랜치·번호

| 항목 | 값 |
|---|---|
| 리포 | `choi-cloud/NAQ` |
| 브랜치 | `hl/hsan-weighting` (`origin/naq_exp` `1a20f23`에서 분기). **upstream = `origin/hl/hsan-weighting`** |
| 다른 브랜치 | `naq_exp`, `hl/anchor`, `nd/anchor-vote`, `naq_query_sampling_pre_train` — 손대지 않는다 |
| 지시서 | `instructions/HW##[a-z]_<server>.md` |
| 사전 등록 | `HWP###` |
| 보고서 | `reports/HW##[a-z]_<server>.md` |
| 태그 | `hw01`(b7124c3), `hw02`(1b59332), `hw03a`, … (지시서가 정한 서버만 태그를 단다) |
| 커밋 메시지 | `HW03a-a: <내용>` (지시서 번호 + 단계 문자). 서버를 앞에 붙인다: `HW03a-a [iislab4]: …` |

## 2. 서버와 환경

| 서버 | GPU | 리포 | 결과 루트 | 주 용도 |
|---|---|---|---|---|
| iislab4 | RTX 3090 24GB × 8 | `~/wkspace/NAQ` | `results/hw_iislab4/` | 본실험 격자, 코드·설정 수정 |
| seio | RTX 4080 SUPER 16GB × 1 | `~/wkspace/NAQ` | `results/hw/` | 산출물 생성(BGRL), 학습 없는 측정, 소규모 확인 |

- 두 서버의 conda 환경 `taskns`는 판본이 같다(python 3.9.25, torch 2.4.1+cu121, torch_geometric 2.6.1, numpy 1.26.4, sklearn 1.5.2; HW03-prep). **패키지 설치 금지** — seio는 다른 트랙과 공유하고, iislab4는 seio와 판본을 맞춰 두었다.
- 평가 에피소드 캐시 `save/downstream_tasks/<ds>/5way_5shot.pkl`: 두 서버 md5 동일(Cora_full `cfef6a69…`, Amazon_clothing `ca3edaca…`). 지우거나 다시 만들지 않는다.
- `data/`, `save/`, `cache/`, `out/`은 git 밖. 서버 간 이동은 사람이 rsync로 한다.
- BGRL 임베딩 `cache/bgrl/<ds>.pt`: seio에서 한 번만 만들고 iislab4로 복사한다. 두 서버가 각자 BGRL을 학습하지 않는다.
- run당 시간: iislab4 GPU당 1 run 51–73 s, seio 작업자 2개 74–115 s.

## 3. 디렉토리 스키마

```
NAQ/
├── CLAUDE.md
├── arg.py data.py meta.py model.py naq_train.py sup_protonet.py sup_train.py
│   task_generator.py unsup_protonet.py utils.py        # NaQ 원본(5cd9873) — 수정 금지
├── choi_exps/                     # 석휘 스크립트 — 동결, 실행은 hw/repro_choi.sh로만
│   ├── legacy_naq_exp/            # 구 계열 exp001–049
│   └── zip_20261107/              # 새 계열 exp004(BGRL, 메일)·012·015·016 — 표 1의 번호
├── hw/                            # 트랙 코드. 실행 진입점은 hw/run.py
│   ├── run.py config.py episodes.py neighbors.py losses.py weights.py evaluate.py summarize.py
│   ├── run_queue.sh repro_choi.sh
│   └── (HW03a+) purity.py
├── configs/hw.json
├── env/taskns_iislab4.yml
├── results/hw/                    # seio
├── results/hw_iislab4/            # iislab4
├── reports/  instructions/
```

- 루트에 새 `.py`/`.sh`를 만들지 않는다.

## 4. 코드 규칙

### 4.1 동결 영역
- NaQ 원본 모듈, `choi_exps/`: 수정 금지. 러너는 `choi_exps/`를 import하지 않는다.
- 새 기능은 설정 키로 켠다. **기본값이면 기존 동작이 바뀌지 않아야 한다**(회귀 확인, §6-7).

### 4.2 설정 `configs/hw.json`
- `defaults` + `arms[arm]` 병합. arm에 `base` 키가 있으면 그 arm을 먼저 깐다(한 단계).
- CLI는 `--arm`, `--dataset`, `--seed`, `--device`만. 하이퍼파라미터는 설정에서만.
- 기존 arm: `naq_base`(L_original만), `nbr_raw`(+0.1·L_neighbor, raw 선택, = exp015), `nbr_bgrl`(+1.0·L_neighbor, BGRL 선택, = exp016), HW02의 `hsan_*`, `nbr_raw_cos`, `hsan_pn_orig_b2_cos`.
- 이웃 선택 키
  - `selector ∈ {raw, bgrl}`: 선택에 쓰는 임베딩 공간.
  - `pick ∈ {max, contrast, random}` (기본 `max`, HW03a에서 추가): `max` = argmax cos(이웃, query) (기존); `contrast` = argmax [cos(이웃, support) − cos(이웃, query)]; `random` = support의 유효 이웃 중 균등 무작위(전용 `torch.Generator` 사용, 전역 RNG를 소비하지 않음).
- `hsan` 블록: `mode ∈ {none, p, pn}` (`none`은 `logit: cos_hsan`과만), `apply ∈ {orig, both}`, `beta`, `space ∈ {gcn, raw}`, `logit ∈ {euclid_gradscale, cos_hsan}`, `tau_cos`.
  - **`euclid_gradscale`은 HW02로 종결**: 손실 폭주(최대 4.6e7)와 붕괴를 보였다. 새 arm을 만들지 않는다(코드는 남겨 둔다).
  - `cos_hsan`이면 두 항(orig·nbr)의 logit이 `cos(z_q, z_c)/tau_cos`가 되고, M은 `apply`가 가리키는 항에만 곱한다. 평가는 기존 유클리드 ProtoNet 그대로.

### 4.3 결과 저장과 집계
- `<결과 루트>/<arm>/<ds>/seed<S>/{metrics.json, train.log}`. 결과 루트는 환경변수 `HW_RESULTS_ROOT`(기본 `results/hw`). **iislab4는 항상 `HW_RESULTS_ROOT=results/hw_iislab4`.**
- `hw/summarize.py`: `summary.{csv,md}`, `--ref <arm>`마다 `paired_<ref>.md`(같은 seed끼리 짝지은 Δ의 평균 ± 표준편차, 양수 seed k/n).
- **짝지은 비교는 한 결과 루트 안에서만 한다.** 서버 간 seed 하나의 차이가 최대 0.92(HW03-prep)로, 재려는 효과(+0.2~0.4)보다 크다. 서로 다른 서버의 run을 짝짓지 않는다.
- **붕괴 표시**: best epoch의 validation 정확도가 25% 이하인 run은 `summary.csv`에 `collapsed=1`로 표시하고 `summary.md` 셀에 붕괴 수를 `[c=k]`로 적는다. 집계에는 포함한다.

### 4.4 재현성과 seed
- RNG 소비 순서: seed → load_data → set_device → add_diffusion → (선택 임베딩) → ProtoNet → neighbor lookup → train_pool → downstream pool. 새 기능은 전역 RNG를 소비하지 않는다.
- seed: **iislab4는 1212–1221 (10개)**, seio는 1212–1216 (5개, HW02와 같은 묶음).

## 5. 세팅 규칙

- 학습 손실과 선택에 어떤 레이블도 쓰지 않는다. valid/test 레이블은 평가에만.
- **측정 전용 레이블 사용 허용** (HW03a부터): 학습에 쓰이는 노드의 GT 레이블을 `torch.no_grad()` 진단(순도 등)에만 쓸 수 있다. 진단값은 손실·선택·체크포인트 선택에 들어가지 않는다. 코드에 `# measurement-only label use` 주석을 단다.
- 프로토콜 고정: 학습 5-way 1-shot NaQ-Feat Top-10 pseudo-query, 평가 5-way 5-shot ProtoNet(유클리드), validation 50 tasks로 best 선택, test 500 tasks 1회. 인코더 2-layer GCN, 그래프 증강 없음.

## 6. 작업 흐름

1. 0단계(현황) → 구현 → 회귀 확인 → 실행 → 집계 → 보고서 → 커밋 → (게이트 통과 시) 태그 → push.
2. 단계마다 커밋을 나눈다.
3. 게이트 실패: 보고서를 쓰고 커밋하되 태그 없음. "멈추고 보고" 게이트면 멈춘다.
4. auto 모드 판단은 지시서 규칙으로, 없으면 수정 범위가 작은 쪽으로. "판단한 사항"에 적는다.
5. 범위 밖 작업은 하지 않는다.
6. 사람에게 넘길 일(rsync 등)이 있으면 지시서의 "멈추고 사람" 지점에서 명령을 그대로 적어 보이고 멈춘다.
7. **회귀 확인**: `hw/` 코드를 바꾼 지시서는 실행 전에 `naq_base`·`nbr_raw` seed 1212(두 데이터셋)를 같은 서버의 기존 값과 비교해 |Δ| ≤ 0.30. 어긋나면 멈추고 보고.
8. 병렬: `hw/run_queue.sh`. `GPUS`(쉼표 목록), `WORKERS`(위치 인자 > 환경변수 > GPUS 개수). iislab4는 `GPUS=0,1,2,3,4,5,6,7`.
9. **git 규칙 (두 서버 공용)**
   - 커밋 직전에 `git pull --rebase`.
   - push는 항상 `git push origin hl/hsan-weighting` (태그는 `git push origin <tag>`).
   - **`configs/hw.json`과 `hw/*.py`의 기존 파일은 iislab4에서만 고친다.** seio는 새 파일 추가와 결과·보고서 커밋만 한다(지시서가 달리 정하면 그쪽을 따른다).
   - 각 서버는 자기 결과 루트에만 쓴다. 다른 서버의 결과 루트를 수정하지 않는다.
   - `.gitignore`가 막는 파일은 `git add -f` 대신 `.gitignore` 허용 규칙을 추가한다(iislab4에서).

## 7. 보고서 규칙

### 7.1 형식
1. 결론 (3줄 이내, 숫자로) 2. 핵심 표 3. 게이트·사전 등록 대조표 4. 수행 내용 (커밋 해시) 5. 판단한 사항 6. 제외·실패·붕괴 run 7. 소요 시간, 환경, HEAD

### 7.2 금지 (사실만)
- 해석·평가 어휘 금지: "효과적", "개선", "유의미", "시사한다", "보인다", "성공", "좋다/나쁘다", "기대", "아마".
- 원인 추정 금지. 차이의 크기와 조건만. 제안 금지.
- 숫자는 소수점 둘째 자리, 차이는 `Δ`(%p). 짝지은 Δ는 "평균 ± 표준편차 (양수 k/n)".

## 8. 사전 등록 등록부

| 범위 | 지시서 | 상태 |
|---|---|---|
| HWP001–006 | HW01 | 완료 |
| HWP007–013 | HW02 | 완료 (충족 007·008, 미충족 009–013) |
| HWP014–023 | HW03a_iislab4 | 배정 |
| HWP024–025 | HW03a_seio | 배정 |
| HWP026– | 다음 | 미배정 |

## 9. 현재 상태와 계획

- **HW01** (hw01): 구조 개편, 러너 이식(스크립트 대비 |Δ| ≤ 0.07).
- **HW02** (hw02, seio 5 seed): 아이디어 1 유지 — `nbr_raw − naq_base` +0.20 (5/5) / +0.35 (4/5) (Amazon / Cora). `euclid_gradscale` HSAN 9 arm 전부 `nbr_raw` 아래, 양성 가중치를 많이 줄일수록 낮음, 일부 붕괴. HSAN 원식형 cos 쌍은 `hsan_pn_orig_b2_cos − nbr_raw_cos` +1.13 (4/5) / +1.46 (5/5), 다만 `nbr_raw_cos`가 `nbr_raw`보다 −5.74 / −4.32 (τ_cos 0.5). 진단: NaQ 양성은 query와 매우 가깝다(gcn cos 0.87–0.94) → hard positive가 구성상 거의 없다.
- **HW03-prep** (iislab4): 환경 일치, 브릿지 평균 |Δ| ≤ 0.14, seed 단위 최대 0.92. iislab4에서도 `nbr_raw − naq_base` +0.21 (4/5) / +0.40 (5/5).
- `choi_exps/zip_20261107/exp004.*`: 석휘 메일로 받은 새 계열 BGRL 사전학습 스크립트(d277ea2).
- **HW03a**: iislab4 = cos τ 격자 × HSAN 모드, hard positive 선택(`contrast`, `random`), 자동 2단계, 조건부 BGRL 단계. seio = BGRL `.pt` 생성(→ 사람이 iislab4로 rsync), `nbr_bgrl` 확인 run, 학습 없는 순도 측정.
- HW03a 해석 규칙(사람이 판정): ① 최고 HSAN-cos가 `nbr_raw`를 +0.3 이상(양수 7/10 이상) 넘으면 그것이 아이디어 2의 구현. ② 아니고 hard positive arm이 넘으면 아이디어 2를 "hard positive 구성"으로 재정의해 석휘와 상의. ③ 둘 다 아니면 아이디어 1만 남기고 아이디어 3(HW04)으로.
- 석휘 표 1 (석휘 서버, seed 1212): exp001 77.72/90.48; exp012 78.55/90.70; exp015 79.00/90.72; exp016 79.52/91.21 (Cora_full/Amazon_clothing).
