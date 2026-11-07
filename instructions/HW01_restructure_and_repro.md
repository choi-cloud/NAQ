# HW01 (개정 1) — 구조 개편 + exp012·exp015 재현 + 러너 이식 검증 (auto)

2026-11-07 · 작성: 이희태 (Claude 정리) · 브랜치 `hl/hsan-weighting` · 태그 `hw01` · 사전 등록 HWP001–HWP006 · 선행 문서 `CLAUDE.md` 개정 2

> **개정 1 (§0 보고 반영).** §0은 완료된 것으로 본다. 원래 지시서의 §1–§3은 이 문서로 대체한다. 바뀐 점:
> - zip과 브랜치의 exp012/015/016은 다른 실험이므로 **덮어쓰지 않고 분리 보관**한다.
> - 새 계열의 exp001·exp004(BGRL)가 없으므로 **exp016 재현과 `nbr_bgrl` 실행은 이번 범위 밖**이다. 재현 대상은 exp012·exp015, 러너 검증 기준은 exp015.
> - Cora_full은 PyG로 내려받는다. 설정 파일은 JSON.
> - zip 스크립트가 브랜치 모듈과 맞는지 smoke test 게이트를 넣는다.

- auto 모드. "멈추고 보고" 게이트만 지킨다.
- 시간 예산 **3시간**. run 하나의 소요 시간을 첫 run에서 재고 보고서에 적는다. 총 예상이 4시간을 넘으면 §3의 exp012를 빼고 진행하고 그 사실을 적는다.
- 새 방법은 없다.

## 1. HW01-a: 구조 개편

1. 사용자가 `~/wkspace/HW_drop/`에 새 `CLAUDE.md`(개정 2)와 이 지시서를 두었다. `CLAUDE.md` → 루트(덮어쓰기), 이 지시서 → `instructions/HW01_restructure_and_repro.md`(덮어쓰기). 복사 후 **CLAUDE.md를 다시 읽는다.**
2. 루트 파일 분류. `git ls-tree --name-only 5cd9873`(커밋 `original`)의 파일 목록을 A라 한다.
   - A에 있는 루트 파일 → **루트에 남긴다.**
   - A에 없는 루트 파일 중 `exp\d{3}\.(py|sh)`, `run_exp049.py` → `git mv`로 `choi_exps/legacy_naq_exp/`.
   - A에 없는 나머지 파일(예: `model.py`, `naq_train.py`, `shell.sh`, `sup_protonet.py`, `sup_train.py`, `unsup_protonet.py` 중 A에 없는 것): A에 있는 파일 또는 zip 여섯 파일 중 하나라도 `import`하거나 이름을 참조하면(`grep -l`) **루트에 남긴다**. 아니면 `choi_exps/legacy_naq_exp/`로 옮긴다.
   - 분류 결과를 표(파일 / A 포함 여부 / 참조하는 파일 / 처리)로 보고서에 적는다.
3. zip(`~/wkspace/HW_drop/exp012-016.zip`)의 `exp012-016/` 여섯 파일 → `choi_exps/zip_20261107/`. `__MACOSX/` 제외. 내용 수정 없음.
4. `choi_exps/README.md`:
   ```
   석휘(Choi) 실험 스크립트. 내용 동결.
   - legacy_naq_exp/: origin/naq_exp(1a20f23)의 구 계열. 2026-11-07 루트에서 이동.
   - zip_20261107/: 석휘가 2026-11-07 직접 준 새 계열 exp012/015/016. 표 1의 번호는 이 계열.
   두 계열은 번호가 겹쳐도 다른 실험이다. 실행은 hw/repro_choi.sh로만.
   ```
5. 디렉토리: `hw/`(빈 `__init__.py`), `configs/`, `cache/bgrl/`, `results/hw/repro/`, `reports/`, `instructions/`. 빈 디렉토리는 `.gitkeep`.
6. `.gitignore` (없으면 생성, 있으면 추가): `out/`, `cache/`, `data/`, `*.pt`, `__pycache__/`.
7. `configs/hw.json`: CLAUDE.md §4.2 내용 그대로.
8. `hw/repro_choi.sh <계열> <exp> <dataset> [EPOCHS]`:
   - 리포 루트에서 `choi_exps/<계열>/<exp>.sh`에 적힌 `python -u <exp>.py ...` 인자를 **그대로** 써서 `PYTHONPATH=. python -u choi_exps/<계열>/<exp>.py <인자>`를 실행한다(`DATASET`만 바꾼다). `EPOCHS`가 주어지면 그 값을, 아니면 `.sh`의 기본값을 쓴다. 다른 인자 값을 바꾸지 않는다.
   - 석휘 `.py`는 `Path("out")/EXP`에 쓰므로 루트에서 실행한다. 끝나면 `out/<exp>/<exp>_<dataset>.json`을 `results/hw/repro/<계열>_<exp>/<dataset>.json`으로 복사한다(EPOCHS를 줬으면 파일명에 `_smoke` 접미사).
   - 이번 지시서에서 `<계열>`은 `zip_20261107`만 쓴다.
9. 루트 `ls *.py *.sh` 결과를 보고서에 적는다.
10. 커밋 `HW01-a: restructure (choi_exps/{legacy_naq_exp,zip_20261107}, hw/ skeleton, configs/hw.json)` → push.

## 2. HW01-b: 데이터와 smoke test

1. **Cora_full**: `conda run -n taskns python -c "from torch_geometric.datasets import CitationFull; d=CitationFull('./data/corafull','cora'); print(d[0])"`. 노드 수 19,793을 확인한다.
   - **G1a**: 내려받기 성공 + 노드 수 일치. 실패(네트워크 차단 등)하면 **멈추고 보고**(에러 메시지 전문).
   - `load_data`가 `Cora_full` 이름으로 이 경로를 읽는지 `data.py`에서 확인한다. 클래스 분할(25/20/25)이 data.py 안에서 정해지면 그 값을 적는다.
2. **smoke**: `hw/repro_choi.sh zip_20261107 exp015 Cora_full 2`, `… Amazon_clothing 2`, `hw/repro_choi.sh zip_20261107 exp012 Cora_full 2`, `… Amazon_clothing 2`.
   - **G1b**: 네 run 모두 결과 JSON의 `status == "completed"`. 하나라도 ImportError·AttributeError·인자 오류로 실패하면 **멈추고 보고** — traceback 전문과, 실패한 이름이 브랜치의 `arg.py`/`data.py`/`meta.py`/`task_generator.py`/`utils.py` 중 어디에 있어야 하는지(`grep -n`) 적는다. 모듈은 고치지 않는다.
3. 커밋 `HW01-b: Cora_full + smoke (zip exp012/exp015)` → push. (`results/hw/repro/*_smoke.json` 포함)

## 3. HW01-c: 석휘 스크립트 재현 (전체 2000 epoch)

1. 순서: `exp015 Cora_full` → `exp015 Amazon_clothing` → `exp012 Cora_full` → `exp012 Amazon_clothing`. 첫 run의 소요 시간으로 남은 예산을 계산한다(시간 규칙은 머리말).
2. 각 run의 `best.test.accuracy_mean`, `best.test.accuracy_ci95_half_width`, `best.epoch`, `elapsed_seconds`를 표로 만들고 표 1 값과의 Δ(이 서버 − 석휘)를 적는다.
3. exp016은 실행하지 않는다(BGRL 없음). 표에 "미실행 — BGRL 파일 대기"로 적는다.
4. 커밋 `HW01-c: Choi repro (zip exp015, exp012)` → push.

## 4. HW01-d: 러너 이식

### 4.1 구현 (`hw/` 아래)
- 기준 코드: `choi_exps/zip_20261107/exp015.py`와 `exp016.py`. **import하지 않고 옮겨 적는다.** 두 파일의 차이는 이웃 선택 임베딩(raw feature 정규화 vs BGRL 정규화)과 기본값뿐임을 `diff`로 확인하고 다른 차이가 있으면 보고서에 적는다.
- `hw/config.py`: `configs/hw.json` 로드, `defaults` + `arms[arm]` 병합. 석휘 `parse_args`처럼 `arg.parse_args_proto("unsup")`를 거쳐 `args`를 만들어, `vars(args)`가 석휘 결과 JSON의 `settings`와 같은 키·값을 갖게 한다.
- `hw/episodes.py`, `hw/neighbors.py`, `hw/losses.py`, `hw/evaluate.py`, `hw/run.py`, `hw/summarize.py`: CLAUDE.md §3·§4. 호출 순서는 CLAUDE.md §4.4.
- `loss: "original"`이면 neighbor 항과 이웃 선택을 **계산하지 않는다**(RNG를 소비하지 않게). 이 arm은 대응하는 석휘 스크립트가 없으므로 검증 대상이 아니다.
- `selector: "bgrl"` 경로는 구현만 하고 실행하지 않는다. `cache/bgrl/<ds>.pt`가 없으면 파일 경로를 담은 `FileNotFoundError`로 끝나는지만 확인한다(`--dataset Cora_full --arm nbr_bgrl` 1회, 즉시 실패 기대).

### 4.2 실행 (seed 1212)
- `python -m hw.run --arm nbr_raw --dataset Cora_full`
- `python -m hw.run --arm nbr_raw --dataset Amazon_clothing`
- `python -m hw.run --arm naq_base --dataset Cora_full`
- `python -m hw.run --arm naq_base --dataset Amazon_clothing`
- `python -m hw.summarize`

### 4.3 게이트
- **G2 (이식 일치)**: 두 데이터셋에서 `|러너 nbr_raw test − exp015 재현 test| ≤ 0.30`. 추가로 13개 평가 지점(160…2000)의 validation 정확도를 소수점 4자리로 나란히 놓은 표를 적는다.
- G2 실패: 태그 없음. (a) 갈라지는 첫 평가 지점, (b) 러너와 exp015.py의 RNG 소비 호출 목록 비교, (c) 첫 5 에피소드의 `original_loss`·`neighbor_loss`를 나란히 적고 **멈추고 보고**. 원인은 쓰지 않는다.
- 커밋 `HW01-d: hw runner port + verification`. G2 통과 시 `git tag hw01` → `git push origin hl/hsan-weighting --tags`.

## 5. 사전 등록 (HW01-a 커밋에 포함, 판정은 사람)

| ID | 예측 | 기준 | 근거 | 확신도 |
|---|---|---|---|---|
| HWP001 | zip exp015가 이 서버에서 표 1과 맞는다 | Cora_full·Amazon_clothing 모두 Δ ∈ [−0.5, +0.5] | 같은 seed·코드. 단 모듈 판본 미확인 | 0.55 |
| HWP002 | zip exp012가 이 서버에서 표 1과 맞는다 | 위와 같음 | 위와 같음 | 0.55 |
| HWP003 | 러너 `nbr_raw`가 exp015 재현값과 일치한다 | 두 데이터셋 \|Δ\| ≤ 0.30 | RNG 순서 보존 | 0.75 |
| HWP004 | 러너와 exp015의 validation 곡선이 같다 | 13개 지점 소수점 4자리 일치, 두 데이터셋 | 호출 순서 동일 | 0.50 |
| HWP005 | `naq_base`가 표 1 exp001에 가깝다 | 두 데이터셋 Δ ∈ [−1.0, +1.0] | exp001 스크립트 미확인, λ=0 가정 | 0.40 |
| HWP006 | 이웃 양성 이득이 이 서버에서도 있다 | exp015 재현 − `naq_base`: Cora_full ≥ +0.5, Amazon_clothing ≥ 0 | 표 1 +1.28 / +0.24 | 0.55 |

## 6. 범위 밖

- exp016 재현, `nbr_bgrl` 실행, BGRL 학습 코드 작성, 구 계열(`legacy_naq_exp/`) 실행.
- HSAN 가중치, 새 arm, 하이퍼파라미터 변경, seed 1213·1214, 다른 데이터셋, 원본 모듈 수정, `choi_exps/` 내용 수정.

## 7. 완료 기준과 보고

- [ ] HW01-a: 루트 분류표, `choi_exps/` 두 계열, `configs/hw.json`, `hw/repro_choi.sh`, 사전 등록 포함 커밋·push
- [ ] HW01-b: G1a·G1b, smoke JSON 네 개
- [ ] HW01-c: exp015·exp012 재현 네 run
- [ ] HW01-d: `hw/*.py`, `results/hw/{nbr_raw,naq_base}/**`, `summary.{csv,md}`, G2, 통과 시 태그 `hw01`
- [ ] `reports/HW01_restructure_and_repro.md` — CLAUDE.md §7. 핵심 표: (1) 표 1 vs 이 서버 재현 vs 러너(exp001/naq_base, exp012, exp015/nbr_raw, exp016 미실행) × 두 데이터셋 + Δ; (2) validation 곡선 일치 표; (3) G1a·G1b·G2·HWP001–006 대조; (4) 루트 분류표; (5) run당 소요 시간.
