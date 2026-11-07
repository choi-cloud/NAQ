# CLAUDE.md — NAQ 리포, 브랜치 `hl/hsan-weighting` (HW 트랙)

개정 2 · 2026-11-07 · 작성: 이희태 (Claude 정리)
이 파일은 이 브랜치에서 Claude Code Agent가 따르는 규약이다. 지시서(`instructions/HW##_*.md`)가 이 파일과 어긋나면 **지시서가 우선**하되, 어긋난 항목을 보고서에 적는다.

> 개정 2 (HW01 §0 보고 반영): 브랜치의 exp 스크립트(구 계열)와 zip의 exp012/015/016(새 계열)은 이름만 같은 다른 실험이다 → 덮어쓰지 않고 분리 보관(§3, §4.1). 새 계열의 exp001·exp004(BGRL)는 아직 없다 → 재현·이식 기준을 exp015로 바꾼다(§4.2, §9). Cora_full은 PyG로 내려받는다(§2). yaml 없음 → 설정 파일은 JSON(§4.2).

---

## 0. 이 트랙이 하는 일

- 석휘(Choi)의 새 계열 NaQ 실험(표 1: exp001 baseline, exp012, exp015, exp016)을 이 서버에서 재현하고, 그 위에 **HSAN식 차등 가중치(아이디어 2)**를 얹어 1+2 결합 성능을 끌어올린다.
- 석휘의 `exp0NN.py + exp0NN.sh` 방식은 쓰지 않는다. **단일 실행 파일 + 단일 설정 파일 + results/ + reports/** 구조로 바꾼 뒤 모든 실험을 그 위에서 돌린다 (§3, §4).
- 순서: HW01(구조 개편 + 재현 + 러너 이식 검증) → [석휘 BGRL 파일 수령 후] HW01b(exp016 재현) → HW02(HSAN pulling 가중치) → HW03+(결합 변형). 지시서 없이 다음 단계로 넘어가지 않는다.

## 1. 트랙·브랜치·번호

| 항목 | 값 |
|---|---|
| 리포 | `choi-cloud/NAQ`, 서버 `seio` `~/wkspace/NAQ` |
| 브랜치 | `hl/hsan-weighting` (`origin/naq_exp` `1a20f23`에서 분기, 2026-11-07) |
| 다른 브랜치 | `naq_exp`(석휘 원본), `hl/anchor`, `nd/anchor-vote`, `naq_query_sampling_pre_train` — **손대지 않는다** |
| 지시서 | `instructions/HW##_<slug>.md` |
| 사전 등록 | `HWP###` (지시서 안 표) |
| 보고서 | `reports/HW##_<slug>.md` |
| 태그 | `hw01`, `hw02`, … (게이트 통과 시에만) |
| 커밋 메시지 | `HW01-a: <내용>` 처럼 지시서 번호 + 단계 문자 |
| 실험 arm 이름 | `configs/hw.json`의 `arms` 키 (§4.2). 석휘의 exp 번호는 arm 설명에만 적는다 |

## 2. 환경

- GPU: RTX 4080 SUPER 1장. `device 0`.
- conda 환경 `taskns` (torch 2.4.1+cu121). **패키지 설치 금지** — 다른 트랙과 공유한다. `yaml` 없음.
- 데이터: `data/` (git 추적 밖). 이 트랙은 `Cora_full`, `Amazon_clothing` 두 개만 쓴다. `Cora_full`은 `data.py`의 경로(`CitationFull("./data/corafull", "cora")`)로 PyG가 내려받는다.
- push는 Agent가 직접 한다 (`git push origin hl/hsan-weighting`).

## 3. 디렉토리 스키마

```
NAQ/
├── CLAUDE.md
├── arg.py data.py meta.py task_generator.py utils.py ...   # NaQ 원본(커밋 5cd9873에 있던 파일) — 수정 금지
├── data/                         # 데이터셋 (추적 밖)
├── choi_exps/                    # 석휘 스크립트 — 내용 동결, 실행은 hw/repro_choi.sh로만
│   ├── README.md
│   ├── legacy_naq_exp/           # origin/naq_exp의 구 계열 exp001–exp049 등 (git mv)
│   └── zip_20261107/             # 석휘가 준 zip의 새 계열 exp012/015/016 .py/.sh
├── hw/                           # 이 트랙의 코드. 실행 진입점은 hw/run.py 하나
│   ├── run.py                    # python -m hw.run --arm <arm> --dataset <ds> [--seed S]
│   ├── config.py                 # configs/hw.json 로더: defaults + arms[arm] 병합
│   ├── episodes.py               # 에피소드 풀 생성 (RNG 순서 보존)
│   ├── neighbors.py              # 1-hop lookup, 선택 임베딩(raw/BGRL) Top-1
│   ├── losses.py                 # original / neighbor / (HW02+) hsan 가중 손실
│   ├── evaluate.py               # 5-way 5-shot ProtoNet 평가
│   ├── summarize.py              # results/hw/** → results/hw/summary.{csv,md}
│   └── repro_choi.sh             # choi_exps/ 스크립트 실행 래퍼 (재현 전용)
├── configs/hw.json               # 단일 설정 파일 (§4.2)
├── cache/bgrl/<ds>.pt            # BGRL 임베딩 (추적 밖, 석휘에게서 받음 — §9)
├── out/                          # 석휘 스크립트 원래 출력 위치 (추적 밖)
├── results/hw/                   # 추적 (JSON·CSV·MD·log만)
│   ├── repro/<계열>_<exp>/<ds>.json        # 석휘 스크립트 결과 사본
│   ├── <arm>/<ds>/seed<S>/metrics.json
│   ├── <arm>/<ds>/seed<S>/train.log
│   └── summary.csv, summary.md
├── reports/HW##_*.md
└── instructions/HW##_*.md
```

- `.gitignore`: `out/`, `cache/`, `data/`, `*.pt`, `__pycache__/`.
- 루트에 새 `.py`/`.sh`를 만들지 않는다.

## 4. 코드 규칙

### 4.1 동결 영역
- NaQ 원본 모듈(커밋 `5cd9873`에 있던 파일): 수정 금지. 필요한 동작은 `hw/`에서 import해서 감싼다.
- `choi_exps/`: 내용 수정 금지. 두 계열은 **이름이 겹쳐도 서로 다른 실험**이므로 서로 덮어쓰지 않는다. 표 1의 번호는 `zip_20261107/` 계열을 가리킨다.
- 이후 실험은 전부 `hw/run.py`. 러너는 `choi_exps/`를 import하지 않고 필요한 함수를 옮겨 적는다.

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
    "seed": 1212, "device": 0
  },
  "arms": {
    "naq_base": { "loss": "original", "note": "L_original만 (석휘 새 계열 exp001 대응 가정, 미확인)" },
    "nbr_raw":  { "loss": "original+neighbor", "selector": "raw",  "lambda_neighbor": 0.1, "max_neighbors": 0, "note": "= zip exp015 (.sh 값)" },
    "nbr_bgrl": { "loss": "original+neighbor", "selector": "bgrl", "lambda_neighbor": 1.0, "max_neighbors": 0, "bgrl_dir": "cache/bgrl", "note": "= zip exp016 (.sh 값)" }
  }
}
```
- arm 값은 `defaults`를 덮어쓴다. CLI는 `--arm`, `--dataset`, `--seed`, `--device`만 받는다. 하이퍼파라미터를 CLI로 넘기지 않는다.
- 새 실험 = 새 arm 블록 하나. 새 `.py`가 아니다.
- 석휘 `.sh`와 `.py`의 기본값이 다르면 **`.sh` 값**을 쓴다(표 1은 `.sh`로 돌린 값으로 본다).

### 4.3 결과 저장
- `metrics.json`: 석휘 result dict 키 유지(`settings`, `protocol`, `epochs[]`, `best{epoch, validation, test}`, `status`, `elapsed_seconds`) + `runner`, `arm`, `config_path`.
- `train.log`: stdout 복사. 체크포인트는 저장하지 않는다.
- `hw/summarize.py` → `summary.csv`(arm, ds, seed, best_epoch, val, test, ci95, elapsed), `summary.md`(arm × ds 표, seed 평균 ± 표준편차). 보고서 표는 여기서 복사.

### 4.4 재현성
- `seed_everything(seed)` 위치와 **RNG 소비 순서**를 석휘 코드와 같게: seed → load_data → add_diffusion → (선택 임베딩 로드) → 모델 생성 → train_pool(`epochs`개) → valid/test pool.
- 기본 seed 1212. 다중 seed는 1212·1213·1214.

## 5. 세팅 규칙

- 학습에 **어떤 레이블도 쓰지 않는다**. valid/test 레이블은 평가에만.
- 프로토콜 고정: 학습 5-way 1-shot NaQ-Feat Top-10 pseudo-query, 평가 5-way 5-shot ProtoNet(유클리드), validation 50 tasks로 best 선택, test 500 tasks 1회.
- 인코더: 2-layer GCN (`meta.ProtoNet`). 그래프 증강 없음.
- 선택 임베딩(raw/BGRL)은 **이웃 선택에만** 쓴다. 손실의 임베딩은 학습 중 GCN.

## 6. 작업 흐름

1. 지시서의 0단계(현황 파악)가 있으면 수정·커밋 없이 보고한다. "멈춤"이면 멈추고, "auto"면 계속한다.
2. 구현 → 실행 → 집계 → 보고서 → 커밋 → (게이트 통과 시) 태그 → push.
3. 지시서의 단계(-a, -b, -c …)마다 커밋을 나눈다.
4. 게이트 실패: 보고서를 쓰고 커밋하되 태그를 달지 않는다. "멈추고 보고" 게이트가 실패하면 멈춘다.
5. auto 모드에서 질문이 생기면 지시서 규칙으로 판단하고 "판단한 사항"에 적는다. 규칙이 없으면 **수정 범위가 작은 쪽**을 택한다.
6. 범위 밖 작업(새 arm, 하이퍼파라미터 탐색, 다른 데이터셋, 원본 모듈 수정)은 하지 않는다.

## 7. 보고서 규칙

### 7.1 형식 (`reports/HW##_<slug>.md`)
1. 결론 (3줄 이내, 숫자로)
2. 핵심 표 (summary.md에서 복사)
3. 게이트·사전 등록 대조표 (ID / 기준 / 측정값 / 통과 여부)
4. 수행 내용 (단계별, 커밋 해시)
5. 판단한 사항
6. 제외·실패한 run과 사유
7. 소요 시간, 환경, HEAD

### 7.2 금지 (사실만 쓴다)
- 해석·평가 어휘 금지: "효과적", "개선", "유의미", "시사한다", "보인다", "성공", "좋다/나쁘다", "기대", "아마".
- 원인 추정 금지. 차이의 크기와 조건만 적는다.
- 제안 금지. "다음에 할 일"을 쓰지 않는다.
- 숫자는 소수점 둘째 자리, 차이는 `Δ`(%p).

## 8. 사전 등록 등록부

| 범위 | 지시서 | 상태 |
|---|---|---|
| HWP001–HWP006 | HW01 (개정 1) | 배정 |
| HWP007– | 다음 지시서 | 미배정 |

## 9. 현재 상태와 계획

- 2026-11-07 HW01 §0 완료(수정 없음). 사실: (1) 브랜치 exp012/015/016과 zip의 같은 이름 파일은 docstring상 다른 실험, (2) 브랜치 exp004는 BGRL 생성기가 아님("bgrl" 0건), 브랜치 exp001은 cross-attention residual + LR 평가, (3) `data/`에 Cora_full 없음, (4) `yaml` 없음, (5) 루트에 exp001–049와 model.py·naq_train.py·run_exp049.py·shell.sh·sup_protonet.py·sup_train.py·unsup_protonet.py.
- **석휘에게서 받을 것**(희태가 요청): 새 계열 `exp001.py/.sh`, `exp004.py/.sh`(BGRL), 가능하면 `out/exp004/exp004_{Cora_full,Amazon_clothing}_embeddings.pt` 두 파일, 그리고 로컬의 `arg/data/meta/task_generator/utils`가 `origin/naq_exp`와 같은지 여부.
- HW01(개정 1): 구조 개편 → Cora_full → zip smoke → exp012·exp015 재현 → 러너 이식(G2는 exp015 기준).
- HW01b(파일 수령 후): BGRL `.pt`를 `cache/bgrl/`에 두고 `nbr_bgrl` 실행, exp016 재현과 비교.
- HW02: HSAN pulling 가중치(아이디어 2).
- 석휘 표 1 (석휘 서버, seed 1212, 5-way 5-shot test): exp001 Cora_full 77.72 / Amazon_clothing 90.48; exp012 78.55 / 90.70; exp015 79.00 / 90.72; exp016 79.52 / 91.21.
