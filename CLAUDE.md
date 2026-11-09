# CLAUDE.md — NAQ 리포, 브랜치 `hl/hsan-weighting` (HW 트랙)

개정 9 · 2026-11-09 · 작성: 이희태 (Claude 정리)
이 파일은 이 브랜치에서 Claude Code Agent가 따르는 규약이다. 지시서(`instructions/HW##_*.md`)가 이 파일과 어긋나면 **지시서가 우선**하되, 어긋난 항목을 보고서에 적는다.

> 개정 9 (HW06 반영, HW07 준비): `extra_neg`의 `cluster_ref`·`kmeans_seed`·`far_top` 키와 진단 `xneg_in_qcluster`·`xneg_in_farset`(§4.2), HW06 결과·band 보류(§9), 사전 등록부(§8).
> 개정 8 (HW05 반영, HW06 준비): 큰 그림과 방향(§0), `extra_neg` 확장(far·band·cluster)·에피소드 안 query 양성/음성·`geom_diag` 상관 지표(§4.2), HW05 결과와 종결 항목(§9), 사전 등록부(§8).
> 개정 7 (HW03b 반영, HW05 준비): 트랙 방향 전환(§0), 추가 음성 블록 `extra_neg`·이웃 항 분해 `nbr_term`·기하 진단 `geom_diag`·`episodes` 재정의(§4.2), 복제 run 명명(§4.3), HW03b 결과와 종결 항목(§9), 사전 등록부(§8).
> 개정 6 (HW04a 반영, HW03b 준비): `pick` 확장 `self`·`global_random`(§4.2), `valid_tasks`와 두 checkpoint test 측정(§4.2, §5), `summarize --metric`(§4.3), validation 500 캐시(§2), HW04a 결과(§9), 사전 등록부(§8).
> 개정 5 (HW03a·HW04m 반영): 파일 전달 방식(§2.1), 측정 루트 `results/measure/`, `pick: sup_cos`, 회귀·불변 확인의 기준점 = 작업 시작 HEAD(§6), 종결 항목과 1차 목표(§9).

---

## 0. 이 트랙이 하는 일

- 석휘(Choi)의 새 계열 NaQ 실험(표 1) 위에서 출발했다. 아이디어 2(HSAN식 가중치, W)는 HW02·HW03a로, 아이디어 1의 "이웃 양성의 내용" 서사는 HW03b로 종결(§9).
- **큰 그림 (희태·석휘 합의, 2026-11-08)**: ① 더 나은 양성/음성 샘플링 → ② 갖춰진 양성/음성 위에서 HSAN식 당김/밀어냄 강도 조절 → ③ 임베딩 차원 간 과상관 해결. 별도로 에피소드를 넘는 전역 손실을 검토한다(형태 미정).
- **현재 단계 (HW06 이후)**: ①의 음성 쪽. 4000 episode에서 무작위 음성 m 20/50/100/200은 `naq` 대비 +1.15~+1.50 / +0.61~+0.70(HW06). 같은 m 20에서 무작위를 넘은 전략은 Cora의 cluster(+0.39, 10/10) 하나다. 지금은 **cluster 효과가 실재하는지와 그 기제**를 검증하고, far를 큰 m에서 재확인한다.
- **1차 목표 (희태 결정 1, 2026-11-07)**: 같은 조건에서 NaQ·VISION을 넘는 것. train-free Â²X 기준선은 측정값으로만 기록하고 지금은 목표로 삼지 않는다.
- 모든 실험은 **단일 실행 파일(`hw/run.py`) + 단일 설정 파일(`configs/hw.json`) + results/ + reports/** 위에서 돌린다.

## 1. 트랙·브랜치·번호

| 항목 | 값 |
|---|---|
| 리포 | `choi-cloud/NAQ` |
| 브랜치 | `hl/hsan-weighting`. **upstream = `origin/hl/hsan-weighting`** |
| 다른 브랜치 | `naq_exp`, `hl/anchor`, `nd/anchor-vote`, `naq_query_sampling_pre_train` — 손대지 않는다 |
| 지시서 | `instructions/HW##[a-z]_<server>.md` |
| 사전 등록 | `HWP###` |
| 보고서 | `reports/HW##[a-z]_<server>.md` |
| 태그 | `hw01`, `hw02`, `hw03a`, `hw04m`, `hw04a`, `hw03b`, `hw05`, `hw06`, … (지시서가 정한 서버만) |
| 커밋 메시지 | `HW07-a [iislab4]: <내용>` (지시서 번호 + 단계 문자 + 서버) |

## 2. 서버와 환경

| 서버 | GPU | 리포 | 결과 루트 | 주 용도 |
|---|---|---|---|---|
| iislab4 | RTX 3090 24GB × 8 | `~/wkspace/NAQ` | `results/hw_iislab4/` | 본실험, 코드·설정 수정, 측정 |
| seio | RTX 4080 SUPER 16GB × 1 | `~/wkspace/NAQ` | `results/hw/` | 소규모 확인 (현재 대기) |

- conda 환경 `taskns` 두 서버 동일(python 3.9.25, torch 2.4.1+cu121, torch_geometric 2.6.1, numpy 1.26.4, scipy 1.13.1, sklearn 1.5.2). **패키지 설치 금지.**
- 평가 에피소드 캐시 `save/downstream_tasks/<ds>/5way_5shot.pkl` (md5 Cora_full `cfef6a69…`, Amazon_clothing `ca3edaca…`). 지우거나 다시 만들지 않는다.
- validation 500 캐시 `save/valid500/<ds>.pkl` (HW03b): 앞 50개 = 기존 validation 50개, 뒤 450개 = 같은 validation 클래스에서 새로 뽑은 task. test task는 기존 캐시 그대로.
- `data/`, `save/`, `cache/`, `out/`은 git 밖. 서버 간 이동은 사람이 rsync로 한다.
- git 밖 캐시: `cache/bgrl/<ds>.pt`(HW03a), `cache/ppr/<ds>_a{015,050}_top100.pt`, `cache/curv/<ds>.pt`(HW04m), `cache/rawcos/<ds>_top500.pt`, `cache/kmeans/<ds>_k100.pt`(HW06).
- run당 시간: iislab4 GPU당 1 run 51–73 s.

### 2.1 파일 전달 (개정 5)
- 사람이 지시서를 **`instructions/<지시서>.md`에 직접** 두고, 새 판 CLAUDE.md는 **리포 루트에 직접 덮어쓴다.** 드롭 폴더(`~/wkspace/HW_drop/`)는 쓰지 않는다.
- §0 현황에서 미추적 지시서 파일과 수정된 `CLAUDE.md`는 "미커밋 변경"으로 치지 않는다. 그 밖의 미커밋 변경이 있으면 멈추고 보고한다. 두 파일은 지시서의 첫 커밋에 포함한다.

## 3. 디렉토리 스키마

```
NAQ/
├── CLAUDE.md
├── arg.py data.py meta.py model.py naq_train.py sup_protonet.py sup_train.py
│   task_generator.py unsup_protonet.py utils.py        # NaQ 원본(5cd9873) — 수정 금지
├── choi_exps/                     # 석휘 스크립트 — 동결
│   ├── legacy_naq_exp/            # 구 계열 exp001–049
│   └── zip_20261107/              # 새 계열 exp004(BGRL)·012·015·016
├── hw/                            # 트랙 코드. 실행 진입점은 hw/run.py
│   ├── run.py config.py episodes.py neighbors.py losses.py weights.py evaluate.py summarize.py
│   ├── run_queue.sh repro_choi.sh trainfree.py
│   ├── purity.py                  # HW03a, build_pool() (HW04m)
│   ├── structure.py               # HW04m: 그래프 구조량 (hop, Â²X, PPR, 곡률)
│   ├── measure_positive.py        # HW04m: 학습 없는 측정 진입점
│   └── compat/exp003.py           # exp004 shim
├── configs/hw.json
├── env/taskns_iislab4.yml
├── results/hw/                    # seio
├── results/hw_iislab4/            # iislab4
├── results/measure/               # 학습 없는 결정적 측정 (서버 무관)
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
- 주요 arm: `naq_base`(L_original만), `nbr_raw`(+0.1·L_neighbor, raw 선택, = exp015), `nbr_bgrl`(λ 1.0, BGRL 선택, = exp016), HW02·HW03a의 `hsan_*`·`cos_t*`·`nbr_raw_{contrast,random}`·`nbr_bgrl_l01*`.
- 이웃 선택 키
  - `selector ∈ {raw, bgrl}`: `max`/`contrast`에서 쓰는 임베딩 공간.
  - `pick ∈ {max, contrast, random, sup_cos}` (기본 `max`):
    - `max` = support의 유효 1-hop 이웃 중 argmax cos(이웃, query) (기존)
    - `contrast` = argmax [cos(이웃, support) − cos(이웃, query)]
    - `random` = 유효 이웃 중 균등 무작위 (전용 Generator, seed + 7919)
    - `sup_cos` (HW04a) = 후보 집합 안에서 argmax cos(후보, **support**). 키 `pick_hop ∈ {1, 2}`(후보 = 유효 1-hop 이웃 / 최단거리 정확히 2인 노드), `pick_space ∈ {a2x, raw}`(Â²X 또는 X를 L2 행 정규화). query 자신은 제외. 후보가 없으면 support 자신.
    - `self` (HW03b) = 양성 = support 자신 (대조군)
    - `global_random` (HW03b) = 그래프 전체에서 균등 무작위 1개 (support·query 제외, 전용 Generator seed + 104729) (대조군)
- **손실 구조 (HW03b §0 확인)**: L_original = InfoNCE(query q, 로짓 −‖z_q − z_s‖²/T, s = support 5개, T 0.5). L_neighbor = 같은 InfoNCE에서 z_s를 (q, s)마다 고른 노드 z_{n(q,s)}로 바꾼 별도 항. L = L_original + λ·L_neighbor. Adam(lr 1e-4, weight_decay 0).
- `nbr_term ∈ {infonce, pull, push}` (기본 `infonce`, HW05): L_neighbor를 `pull` = mean_q ‖z_q − z_{n(q,s⁺)}‖²/T (당김만), `push` = mean_q logsumexp_s(−‖z_q − z_{n(q,s)}‖²/T) (밀어냄만)으로 바꾼다. `pull + push = infonce`(상수 log|P| 제외).
- `extra_neg` 블록 (HW05): `mode ∈ {none, random, hard_raw, hard_a2x}` (기본 `none`), `m`(개수). L_original의 분모에 query마다 노드 m개를 음성으로 추가한다(양성·λ 없음). 후보에서 그 에피소드의 support 5개와 pseudo-query 50개, query 자신을 제외한다. `random` = 균등(전용 Generator seed + 23757), `hard_*` = query와 cos 상위(raw 또는 Â²X L2 정규화, 사전 표 top-100에서 제외 후 m개; 부족분은 random으로 채우고 비율 기록).
- `extra_neg` 확장 (HW06): `mode`에 `far`(q의 2-hop 이내 ∪ raw cos 상위 100 제외 후 균등), `band`(q의 raw cos 순위 `band_lo`–`band_hi`, 기본 101–500 구간에서 균등), `cluster`(Â²X L2 정규화 k-means, `k` 기본 100, q와 다른 군집에서 균등)를 추가. 후보 제외 규칙은 `random`과 같다. 각 모드는 전용 Generator를 쓴다.
- `extra_neg` 키 추가 (HW07): `cluster_ref ∈ {query, anchor, shuffled}`(기본 `query` = HW06 `cluster`와 같음; `anchor` = q의 양성 support s⁺의 군집 제외; `shuffled` = q의 군집 밖에서 균등하게 뽑은 노드 u의 군집 제외 — 제외 군집 크기 분포를 q 기준과 같게 맞춘 대조군), `kmeans_seed`(기본 20261108; 다르면 `cache/kmeans/<ds>_k<k>_s<seed>.pt`), `far_top`(기본 100; far의 raw cos 제외 상위 개수).
- 진단 (HW07): 모든 `extra_neg` arm에 `xneg_in_qcluster`(추가 음성 중 q의 k100·seed 20261108 군집에 속한 비율), `xneg_in_farset`(추가 음성 중 q의 2-hop ∪ raw cos 상위 100에 속한 비율). 레이블 불필요.
- `ep_query_neg` (기본 false, HW06): 같은 에피소드에서 q와 다른 pseudo-class의 pseudo-query(40개)를 L_original 분모에 음성으로 추가. `ep_query_pos` (기본 false, HW06): 같은 pseudo-class의 다른 pseudo-query(9개)를 양성 집합 P(q)에 추가(기존 다중 양성 식 logsumexp_all − logsumexp_P + log|P| 그대로).
- `geom_diag` (기본 false, HW05): 500 기준 best checkpoint에서 고정 표본 4096 노드(전용 Generator seed 2026)의 z로 평균 노름, 노름 표준편차, 평균 쌍 거리, uniformity(log mean exp(−2‖û−v̂‖²), L2 정규화), effective rank(중심화 표본 특이값 분포의 엔트로피 지수)를 `metrics.json`의 `geom`에 기록한다. HW06부터 `corr_offdiag_mean`(표본 z의 차원 간 Pearson 상관 행렬에서 비대각 |상관|의 평균)과 `dim`을 추가.
- `episodes` (기본 2000): arm에서 재정의할 수 있다. `eval_every`는 그대로.
- `valid_tasks ∈ {50, 500}` (기본 50, HW03b): 500이면 validation 500 캐시로 매 평가 시점에 500개 정확도와 앞 50개 정확도를 함께 기록하고, best epoch을 두 기준으로 각각 골라 test를 각 checkpoint에서 잰다. 주 결과 `test_acc` = 500 기준, 측정용 `test_acc_at_best50` = 50 기준.
- `hsan` 블록은 HW02·HW03a 기록용으로 남긴다. **새 HSAN arm을 만들지 않는다**(§9 종결).

### 4.3 결과 저장과 집계
- 학습 run: `<결과 루트>/<arm>/<ds>/seed<S>/{metrics.json, train.log}`. 결과 루트는 `HW_RESULTS_ROOT`(기본 `results/hw`). **iislab4는 항상 `HW_RESULTS_ROOT=results/hw_iislab4`.**
- 학습 없는 결정적 측정: `results/measure/`. 서버 간 비교 규칙의 대상이 아니다.
- `hw/summarize.py`: `summary.{csv,md}`, `--ref <arm>`마다 `paired_<ref>.md`(같은 seed 짝지은 Δ 평균 ± 표준편차, 양수 seed k/n). `--metric <key>`(기본 `test_acc`)로 다른 지표를 집계하면 파일 이름에 `_<key>` 접미사를 붙인다.
- **validation 기준이 다른 run끼리는 짝짓지 않는다** (`valid_tasks` 50 run과 500 run).
- **복제 run**: 같은 설정을 같은 서버에서 다시 돌릴 때는 arm 이름에 `_r2`를 붙인다(예: `v500_naq_base_r2`). 원본과의 seed별 |Δ|는 같은 서버 안 복제 잡음의 추정치로 보고서에 적는다.
- **짝지은 비교는 한 결과 루트 안에서만 한다.** 서버 간 seed 하나 차이가 최대 0.92(HW03-prep).
- **붕괴 표시**: best epoch의 validation 정확도 ≤ 25% → `collapsed=1`, `summary.md` 셀에 `[c=k]`. 집계에는 포함.

### 4.4 재현성과 seed
- RNG 소비 순서: seed → load_data → set_device → add_diffusion → (선택 임베딩·선택 표) → ProtoNet → neighbor lookup → train_pool → downstream pool. 새 기능은 전역 RNG를 소비하지 않는다.
- seed: **iislab4는 1212–1221 (10개)**, seio는 1212–1216.

## 5. 세팅 규칙

- 학습 손실과 선택에 어떤 레이블도 쓰지 않는다. valid/test 레이블은 평가에만.
- **측정 전용 레이블 사용 허용**: GT 레이블을 `torch.no_grad()` 진단(순도 등)에만 쓸 수 있다. 진단값은 손실·선택·체크포인트 선택에 들어가지 않는다. 코드에 `# measurement-only label use` 주석.
- 프로토콜 고정: 학습 5-way 1-shot NaQ-Feat Top-10 pseudo-query, 평가 5-way 5-shot ProtoNet(유클리드), test 500 tasks. 인코더 2-layer GCN, 그래프 증강 없음.
- validation: HW04a까지는 50 tasks. HW03b부터 `valid_tasks: 500` arm은 500 기준 best epoch의 test가 주 결과다. 50 기준 checkpoint의 test는 **측정용 예외**로 함께 잰다(선택 잡음 측정). 보고서에는 주 결과를 먼저, 측정용 값은 따로 적는다.

## 6. 작업 흐름

1. 0단계(현황) → 구현 → 회귀 확인 → 실행 → 집계 → 보고서 → 커밋 → (게이트 통과 시) 태그 → push.
2. 단계마다 커밋을 나눈다.
3. 게이트 실패: 보고서를 쓰고 커밋하되 태그 없음. "멈추고 보고" 게이트면 멈춘다.
4. auto 모드 판단은 지시서 규칙으로, 없으면 수정 범위가 작은 쪽으로. "판단한 사항"에 적는다.
5. 범위 밖 작업은 하지 않는다.
6. 사람에게 넘길 일이 있으면 지시서의 "멈추고 사람" 지점에서 명령을 그대로 적어 보이고 멈춘다.
7. **회귀 확인**: `hw/` 학습 경로 코드를 바꾼 지시서는 실행 전에 `naq_base`·`nbr_raw` seed 1212(두 데이터셋)를 `HW_RESULTS_ROOT=results/hw_iislab4/_regress`에 다시 돌려 기존 값과 비교, |Δ| ≤ 0.30. 어긋나면 멈추고 보고. `_regress/`는 커밋하되 집계에서 뺀다.
8. **불변 확인의 기준점은 작업 시작 시 HEAD**(§0에서 기록한 해시)다. 태그를 기준점으로 쓰지 않는다.
9. 병렬: `hw/run_queue.sh`. iislab4는 `GPUS=0,1,2,3,4,5,6,7`.
10. **git 규칙**
    - 커밋 직전에 `git pull --rebase`.
    - push는 `git push origin hl/hsan-weighting` (태그는 `git push origin <tag>`).
    - `configs/hw.json`과 `hw/*.py`의 기존 파일은 iislab4에서만 고친다.
    - 각 서버는 자기 결과 루트(와 `results/measure/`)에만 쓴다.
    - `.gitignore`가 막는 파일은 `git add -f` 대신 허용 규칙을 추가한다(iislab4에서).

## 7. 보고서 규칙

### 7.1 형식
1. 결론 (3줄 이내, 숫자로) 2. 핵심 표 3. 게이트·사전 등록 대조표 4. 수행 내용 (커밋 해시) 5. 판단한 사항 6. 제외·실패·붕괴 run 7. 소요 시간, 환경, HEAD
- 표마다 값의 종류(정확도, 노드 macro 순도, edge 비율, 에피소드 쌍 비율 등)와 정의를 적는다(HW04m-c2 방식).

### 7.2 금지 (사실만)
- 해석·평가 어휘 금지: "효과적", "개선", "유의미", "시사한다", "보인다", "성공", "좋다/나쁘다", "기대", "아마".
- 원인 추정 금지. 차이의 크기와 조건만. 제안 금지.
- 숫자는 소수점 둘째 자리(순도는 넷째 자리), 차이는 `Δ`(%p). 짝지은 Δ는 "평균 ± 표준편차 (양수 k/n)".

## 8. 사전 등록 등록부

| 범위 | 지시서 | 상태 |
|---|---|---|
| HWP001–006 | HW01 | 완료 |
| HWP007–013 | HW02 | 완료 (충족 007·008) |
| HWP014–023 | HW03a_iislab4 | 완료 |
| HWP024–025 | HW03a_seio | 완료 |
| HWP026–032 | HW04m_iislab4 | 완료 (충족 027·031·032) |
| HWP033–040 | HW04a_iislab4 | 완료 (충족 033·039·040) |
| HWP041–048 | HW03b_iislab4 | 완료 (충족 041·043·045·048) |
| HWP049–058 | HW05_iislab4 | 완료 (충족 049–051·053–055·057) |
| HWP059–069 | HW06_iislab4 | 완료 (충족 059–061·063·064·066·068·069) |
| HWP070–078 | HW07_iislab4 | 배정 |
| HWP079– | 다음 | 미배정 |

## 9. 현재 상태와 계획

- **HW01** (hw01): 구조 개편, 러너 이식(스크립트 대비 |Δ| ≤ 0.07).
- **HW02** (hw02): 아이디어 1 유지 — `nbr_raw − naq_base` +0.20/+0.35 (Amazon/Cora, seio 5 seed). iislab4 10 seed +0.33/+0.27 계열.
- **HW03a** (hw03a): cos-HSAN 전 τ에서 `nbr_raw` 아래(−3 ~ −20). `pick` contrast/random은 `nbr_raw`와 차이 ±0.15 이내. BGRL λ 1.0은 Cora +0.76이나 λ 맞춘 BGRL은 raw와 같음. 이웃 순도 0.63–0.66, NaQ query 순도 0.35/0.60. Â²X(L2) train-free 80.12/92.72.
- **HW04m** (hw04m): 학습 없는 측정(`results/measure/`). hop1 0.586/0.621 → hop2 0.422/0.529, 그러나 hop2 안 support와 Â²X cos 1위 0.615/0.709. PPR 구간은 hop1보다 낮음. `jl > 0` edge 0.738/0.753(coverage 20–25%). query 순도는 support 차수 하위 0.29/0.55 vs 상위 0.42/0.68.
- **HW04a** (hw04a): `sup_cos` 3 arm(h2 Â²X, h2 raw, h1 Â²X) 모두 `nbr_raw` 대비 −0.17 ~ +0.08, `naq_base` 대비 +0.15 ~ +0.35. 학습 중 순도는 측정값을 따름(h2 0.61–0.74). raw 2-hop 양성의 58% / 75%가 이미 그 support의 pseudo-query. λ 0.1에서 이웃 선택 방식 전체(max, contrast, random, BGRL, sup_cos)가 `nbr_raw` ±0.17 안.
- **HW03b** (hw03b, validation 500): `self`(양성 = support) = `naq_base`(|Δ| ≤ 0.04). **무작위 양성(순도 0.02)이 모든 λ·두 ds에서 이웃 양성보다 높음**: λ 0.3 `grand − naq` +1.80 / +0.95 (10/10), `raw − grand` −0.94 / −0.48. `raw`는 λ 단조 증가(λ 1.0 Cora +1.46). validation 500은 `self`–`naq` 쌍 sd 0.04 vs 0.35. best epoch 다수 2000. Dirichlet 에너지(식 13) ≈ 단어 빈도(ρ 0.95 / 0.81).
- **종결 (재개하지 않는다)**: `euclid_gradscale`; HSAN 가중(W) 전 형태; cos τ 실험(결정 2); BGRL 실험(결정 3); **양성 선택 정교화**(1-hop/2-hop·기준·공간·순도 — HW03a·HW04a·HW03b: 모두 ±0.17 안이거나 무작위보다 낮음); SPGCL EAP 이식(에너지 = 빈도 문제 미해결); 아이디어 A(이웃 겹침 anchor 병합, HW05 측정); 특징 근접 top-k hard negative(HW05).
- **HW05** (hw05): 무작위 양성 이득의 기제 = **추가 음성**. L_original 분모에 무작위 음성 m 20: +1.62 / +0.80, m 5: +1.16 / +0.54 (10/10). pull만 −6.69 / −0.46, push만 −3.79 / −11.22. raw cos 상위 20 음성은 FN rate 0.30 / 0.54, 무작위 음성 대비 −7.63 / −7.85(Â²X 상위는 −43.71 / −12.65). 4000 episode: naq +0.92 / +0.55, grand +0.63 / +0.39. 이웃 항 λ1.0 + 무작위 20: +2.10 / +0.87. 아이디어 A: Jaccard θ0.3 병합 정밀도 0.55 / 0.65(1-hop 순도 수준), 같은 클래스 쌍의 90–93%는 공통 이웃 0.
- **HW06** (hw06, 4000 episode): `naq` 78.69 / 91.38. 무작위 m 20 / 50 / 100 / 200: +1.15 / +1.31 / +1.39 / +1.50 (Cora), +0.61 / +0.70 / +0.66 / +0.66 (Amazon). m 20 전략 `− rand_m20`: cluster +0.39 (10/10) / +0.00, far +0.07 / −0.05, band −1.04 / −8.44(FN 0.099 / 0.262). query 음성만 +0.33 / +0.19, query 음성 + 무작위 20은 무작위 20보다 −0.19 / −0.14, query 양성 추가 −2.94 / −2.77. 차원 |상관|은 음성 m이 클수록 감소, band가 가장 낮음(0.114 / 0.083).
- **진행**: HW07 — cluster 효과 검증(무작위 군집 대조, k-means seed, anchor 군집, m 100/200), far m 200·far_strong, 제외 집합 측정.
- **보류**: band(semi-hard) — 고정 raw 순위 기준은 FN에 따라 손해. 섞어 쓰기·학습 임베딩 기준 동적 어려움은 큰 그림 ②에서 다시 검토.
- **대기**: 이긴 전략의 최적 m 재확인, 큰 그림 ②(HSAN식 강도 조절 — 가까운 음성 = FN 위험이라는 HW05 결과를 반영해 설계), 전역 손실, VISION 동일 프로토콜 기준선.
- 석휘 표 1 (석휘 서버, seed 1212): exp001 77.72/90.48; exp012 78.55/90.70; exp015 79.00/90.72; exp016 79.52/91.21 (Cora_full/Amazon_clothing).
