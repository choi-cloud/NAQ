# CLAUDE.md — NAQ 리포, 브랜치 `hl/hsan-weighting` (HW 트랙)

개정 3 · 2026-11-07 · 작성: 이희태 (Claude 정리)
이 파일은 이 브랜치에서 Claude Code Agent가 따르는 규약이다. 지시서(`instructions/HW##_*.md`)가 이 파일과 어긋나면 **지시서가 우선**하되, 어긋난 항목을 보고서에 적는다.

> 개정 3 (HW01 완료 반영): §4.4 seed를 5개로, §4.2 arm 목록에 HSAN 가중치 arm 규약 추가, §4.3에 짝지은 차이 표 추가, §8 번호 배정, §9 현재 상태 갱신. 개정 2의 구조(§3)는 HW01-a로 실제 적용되었다.

---

## 0. 이 트랙이 하는 일

- 석휘(Choi)의 새 계열 NaQ 실험(표 1: exp001 baseline, exp012, exp015, exp016)을 이 서버에서 재현하고, 그 위에 **HSAN식 차등 가중치(아이디어 2)**를 얹어 1+2 결합 성능을 끌어올린다.
- 석휘의 `exp0NN.py + exp0NN.sh` 방식은 쓰지 않는다. **단일 실행 파일 + 단일 설정 파일 + results/ + reports/** 구조 위에서 모든 실험을 돌린다 (§3, §4).
- 석휘의 입장(2026-11-07 카톡): BGRL 사전학습 자체가 중요한 것이 아니라, 어떤 사전학습 임베딩 위에서도 "NCLA식 양성 확장 + HSAN식 가중치"가 이득을 주는지가 핵심. 따라서 `nbr_bgrl`은 재현 의무가 있는 기준선이 아니라 선택 임베딩의 한 옵션이다.
- 순서: HW01(완료) → HW02(HSAN 가중치, `nbr_raw` 위) → HW03+(클러스터 게이팅·pushing·선택 임베딩 축 확장). 지시서 없이 다음 단계로 넘어가지 않는다.

## 1. 트랙·브랜치·번호

| 항목 | 값 |
|---|---|
| 리포 | `choi-cloud/NAQ`, 서버 `seio` `~/wkspace/NAQ` |
| 브랜치 | `hl/hsan-weighting` (`origin/naq_exp` `1a20f23`에서 분기) |
| 다른 브랜치 | `naq_exp`, `hl/anchor`, `nd/anchor-vote`, `naq_query_sampling_pre_train` — **손대지 않는다** |
| 지시서 | `instructions/HW##_<slug>.md` |
| 사전 등록 | `HWP###` (지시서 안 표) |
| 보고서 | `reports/HW##_<slug>.md` |
| 태그 | `hw01`(b7124c3), `hw02`, … (게이트 통과 시에만) |
| 커밋 메시지 | `HW02-a: <내용>` 처럼 지시서 번호 + 단계 문자 |
| 실험 arm 이름 | `configs/hw.json`의 `arms` 키 (§4.2) |

## 2. 환경

- GPU: RTX 4080 SUPER 1장(16 GB). `device 0`. run 하나(2000 에피소드 + 평가)는 약 1분.
- conda 환경 `taskns` (torch 2.4.1+cu121). **패키지 설치 금지**. `yaml` 없음 → 설정은 JSON.
- 데이터: `data/` (추적 밖). `Cora_full`(PyG CitationFull), `Amazon_clothing`.
- 평가 에피소드 캐시 `save/downstream_tasks/<ds>/5way_5shot.pkl`: 있으면 그대로 읽는다. 지우거나 다시 만들지 않는다. (Cora_full은 2026-11-07 07:33 exp015 전체 run이 생성, md5 `cfef6a69…`; Amazon_clothing은 2026-10-24 생성, 출처 미확인.)
- push는 Agent가 직접 한다 (`git push origin hl/hsan-weighting`).

## 3. 디렉토리 스키마 (HW01-a 적용됨)

```
NAQ/
├── CLAUDE.md
├── arg.py data.py meta.py model.py naq_train.py sup_protonet.py sup_train.py
│   task_generator.py unsup_protonet.py utils.py        # NaQ 원본(커밋 5cd9873) — 수정 금지
├── data/  save/                   # 추적 밖
├── choi_exps/                     # 석휘 스크립트 — 동결, 실행은 hw/repro_choi.sh로만
│   ├── legacy_naq_exp/            # origin/naq_exp 구 계열 exp001–049
│   └── zip_20261107/              # 새 계열 exp012/015/016 (표 1의 번호)
├── hw/                            # 트랙 코드. 실행 진입점은 hw/run.py 하나
│   ├── run.py config.py episodes.py neighbors.py losses.py evaluate.py summarize.py
│   ├── repro_choi.sh
│   └── (HW02+) weights.py  run_queue.sh
├── configs/hw.json                # 단일 설정 파일
├── cache/bgrl/<ds>.pt             # 추적 밖, 미수령
├── results/hw/
│   ├── repro/<계열>_<exp>/<ds>.json
│   ├── <arm>/<ds>/seed<S>/{metrics.json, train.log}
│   └── summary.csv  summary.md  (HW02+) paired_<ref>.md
├── reports/HW##_*.md
└── instructions/HW##_*.md
```

- `.gitignore`: 원본은 `*` 무시 + `.py/.sh` 허용. HW01-a에서 `!*.md !*.json !*.csv !.gitkeep !results/hw/**/*.log`를 허용하고 `results/hl/ results/nd/ save/ out/ cache/ data/ *.pt`를 무시한다.
- 루트에 새 `.py`/`.sh`를 만들지 않는다.

## 4. 코드 규칙

### 4.1 동결 영역
- NaQ 원본 모듈: 수정 금지. `hw/`에서 import해서 감싼다.
- `choi_exps/`: 수정 금지. 러너는 `choi_exps/`를 import하지 않는다.
- `hw/`의 기존 파일: 새 arm이 기존 arm의 동작을 바꾸면 안 된다. `naq_base`·`nbr_raw`는 seed 1212에서 HW01 값(77.78/90.67, 78.10/90.86)을 다시 내야 한다(회귀 확인, §6-7).

### 4.2 단일 설정 파일 `configs/hw.json`
```json
{
  "defaults": {
    "n_way": 5, "k_shot": 1, "k_shot_test": 5,
    "q_query": 10, "q_query_test": 8,
    "query_generation": "NaQ", "type": "feature",
    "lr": 1e-4, "epochs": 2000, "eval_every": 160,
    "temperature": 0.5,
    "valid_tasks": 50, "test_tasks": 500,
    "seed": 1212, "device": 0,
    "hsan": null
  },
  "arms": {
    "naq_base": { "loss": "original" },
    "nbr_raw":  { "loss": "original+neighbor", "selector": "raw",  "lambda_neighbor": 0.1, "max_neighbors": 0 },
    "nbr_bgrl": { "loss": "original+neighbor", "selector": "bgrl", "lambda_neighbor": 1.0, "max_neighbors": 0, "bgrl_dir": "cache/bgrl" },

    "hsan_p_orig_b2":  { "base": "nbr_raw", "hsan": { "mode": "p",  "apply": "orig", "beta": 2, "space": "gcn", "logit": "euclid_gradscale" } }
  }
}
```
- `base` 키가 있으면 그 arm의 설정을 먼저 깔고 덮어쓴다(한 단계만).
- `hsan` 블록 규약 (HW02 지시서 §2에 정의): `mode ∈ {p, pn}`, `apply ∈ {orig, both}`, `beta ∈ {1,2,3}`, `space ∈ {gcn, raw}`, `logit ∈ {euclid_gradscale, cos_hsan}`. arm 이름은 `hsan_<mode>_<apply>_b<beta>[_raw][_cos]`.
- CLI는 `--arm`, `--dataset`, `--seed`, `--device`만. 하이퍼파라미터를 CLI로 넘기지 않는다.
- 석휘 `.sh`와 `.py` 기본값이 다르면 `.sh` 값.

### 4.3 결과 저장과 집계
- `metrics.json`: 석휘 result 키 + `runner`, `arm`, `config_path`, `config`, `first_episodes`. HSAN arm은 `epochs[]`에 가중치 진단(§HW02)을 추가.
- `hw/summarize.py`
  - `summary.csv`(arm, ds, seed, best_epoch, val, test, ci95, elapsed), `summary.md`(arm × ds, seed 평균 ± 표준편차(ddof=0), n).
  - `--ref <arm>` 옵션 → `paired_<ref>.md`: 각 arm에 대해 데이터셋별 **같은 seed끼리 짝지은 Δ = arm − ref**의 평균, 표준편차, 양수 seed 수 / n. 같은 seed는 같은 에피소드를 쓰므로 짝지은 비교가 맞다.
- 체크포인트는 저장하지 않는다.

### 4.4 재현성
- RNG 소비 순서: seed → load_data → set_device → add_diffusion → (선택 임베딩) → ProtoNet → neighbor lookup → train_pool → downstream pool. HSAN 가중치 계산은 RNG를 소비하지 않아야 한다.
- **seed: 1212·1213·1214·1215·1216 (5개)** — HW02부터. seed 1212 단독 비교는 HW01 회귀 확인에만 쓴다.
- 같은 seed라도 GPU 비결정성으로 validation 곡선이 소수점 4자리에서 흔들린다(HW01 §2.2). test ±0.1 수준의 차이는 의미를 두지 않는다.

## 5. 세팅 규칙

- 학습에 어떤 레이블도 쓰지 않는다. valid/test 레이블은 평가에만.
- 프로토콜 고정: 학습 5-way 1-shot NaQ-Feat Top-10 pseudo-query, 평가 5-way 5-shot ProtoNet(유클리드), validation 50 tasks로 best 선택, test 500 tasks 1회.
- 인코더 2-layer GCN. 그래프 증강 없음. 선택 임베딩은 이웃 선택에만.
- HSAN 가중치의 유사도 공간(`space`)은 가중치 계산에만 쓰고 gradient를 흘리지 않는다(detach).

## 6. 작업 흐름

1. 0단계(현황 파악)가 있으면 수정·커밋 없이 보고. "멈춤"이면 멈추고 "auto"면 계속.
2. 구현 → 회귀 확인 → 실행 → 집계 → 보고서 → 커밋 → (게이트 통과 시) 태그 → push.
3. 단계(-a, -b, …)마다 커밋을 나눈다.
4. 게이트 실패: 보고서를 쓰고 커밋하되 태그 없음. "멈추고 보고" 게이트가 실패하면 멈춘다.
5. auto 모드의 판단은 지시서 규칙으로, 없으면 수정 범위가 작은 쪽으로. "판단한 사항"에 적는다.
6. 범위 밖 작업(새 arm, 하이퍼파라미터 탐색, 다른 데이터셋, 원본 모듈 수정)은 하지 않는다.
7. **회귀 확인**: `hw/` 코드를 바꾼 지시서는 실행 전에 `naq_base`·`nbr_raw`를 seed 1212로 한 번씩 돌려 HW01 값과 |Δ| ≤ 0.30인지 확인한다. 어긋나면 **멈추고 보고**.
8. 병렬 실행: `hw/run_queue.sh`로 작업자 2개(flock 큐). OOM이면 1개로 내리고 적는다.

## 7. 보고서 규칙

### 7.1 형식 (`reports/HW##_<slug>.md`)
1. 결론 (3줄 이내, 숫자로) 2. 핵심 표 (summary·paired에서 복사) 3. 게이트·사전 등록 대조표 4. 수행 내용 (단계별 커밋 해시) 5. 판단한 사항 6. 제외·실패한 run 7. 소요 시간, 환경, HEAD

### 7.2 금지 (사실만 쓴다)
- 해석·평가 어휘 금지: "효과적", "개선", "유의미", "시사한다", "보인다", "성공", "좋다/나쁘다", "기대", "아마".
- 원인 추정 금지. 차이의 크기와 조건만 적는다. 제안 금지.
- 숫자는 소수점 둘째 자리, 차이는 `Δ`(%p). 짝지은 Δ는 "평균 (양수 seed k/5)" 형식.

## 8. 사전 등록 등록부

| 범위 | 지시서 | 상태 |
|---|---|---|
| HWP001–HWP006 | HW01 | 완료 (충족 002·003·005, 미충족 001·004·006) |
| HWP007–HWP013 | HW02 | 배정 |
| HWP014– | HW03 | 미배정 |

## 9. 현재 상태와 계획

- **HW01 완료** (2026-11-07, 태그 `hw01` = b7124c3). 구조 개편 적용. seed 1212 결과: `naq_base` 77.78/90.67, `nbr_raw` 78.10/90.86 (Cora_full/Amazon_clothing). 석휘 스크립트 재현 exp015 78.03/90.91, exp012 78.10/90.63. 러너–스크립트 |Δ| ≤ 0.07 (G2 통과).
- 희태 판정(HW01): HWP001 미충족(exp015 Cora −0.97)은 단일 seed 변동과 평가 캐시 차이 범위로 보고 추가 조치 없음(추정). HWP004 미충족(곡선 소수점 4자리 불일치)은 GPU 비결정성으로 보고 수용(추정, 손실 차이 소수점 5자리). HWP006 미충족(Cora +0.24)은 seed 1개로는 판정 불가 → HW02부터 5 seed.
- **석휘에게 받을 것(선택)**: 새 계열 exp001·exp004 스크립트, BGRL `.pt`, `save/downstream_tasks/`. 없어도 HW02는 진행한다.
- HW02: `nbr_raw` 위 HSAN 가중치(episode pseudo-label을 Q로, 클러스터링 없음). 5 seed, 2단계(β=2 격자 → 상위 2 arm β 탐색 + cos 원식 쌍).
- HW03 후보(사람이 정함): 클러스터 기반 high-confidence 게이팅·pushing 완화, 선택 임베딩 축(raw/BGRL/기타), 학습 중 재클러스터링.
- 석휘 표 1 (석휘 서버, seed 1212, 5-way 5-shot): exp001 77.72/90.48; exp012 78.55/90.70; exp015 79.00/90.72; exp016 79.52/91.21.
