# HW03a [seio] — BGRL 임베딩 생성 + `nbr_bgrl` 확인 run + 학습 없는 이웃 순도 측정

2026-11-07 · 작성: 이희태 (Claude 정리) · 서버 **seio** (RTX 4080 SUPER × 1) · 브랜치 `hl/hsan-weighting` · 태그 없음 · 사전 등록 HWP024–HWP025 · 선행 문서 `CLAUDE.md` 개정 4 (드롭 폴더 판본을 읽는다. 리포 반영은 iislab4가 한다)

- auto 모드. 멈추는 조건: §1 G-bgrl 실패, **§1 끝(사람이 rsync)**, 구현 중 해결 못 한 오류.
- 결과 루트: 기본 `results/hw/`. `results/hw_iislab4/`에는 쓰지 않는다.
- **이 서버는 `configs/hw.json`과 기존 `hw/*.py`를 고치지 않는다.** 새 파일 `hw/purity.py`만 추가한다. CLAUDE.md도 리포에 복사하지 않는다(iislab4가 커밋한다. pull로 받는다).
- 같은 시간에 iislab4가 HW03a_iislab4를 돌린다. 그쪽 HW03a-a 커밋(이웃 선택기 `pick`)이 §3에 필요하다.
- 시간 예산 2시간.

## 0. 현황 (수정 없음, 보고서에 적고 계속)

`git fetch && git status -sb | head -1`(upstream = `origin/hl/hsan-weighting`), `git log --oneline -3`, `nvidia-smi`(다른 사용자 프로세스 유무), `ls choi_exps/zip_20261107/`, `head -40 choi_exps/zip_20261107/exp004.py`, `cat choi_exps/zip_20261107/exp004.sh`. exp004의 docstring, 인자, 출력 경로·파일명·저장 키를 보고서에 옮긴다.

## 1. HW03a-s1: BGRL 임베딩 생성 → **멈추고 사람**

1. `hw/repro_choi.sh zip_20261107 exp004 <ds>`로 Cora_full, Amazon_clothing을 각각 실행한다(`.sh`가 두 데이터셋을 한 루프로 돌리면 그 방식대로 한 번). `.sh`의 값(EPOCHS=400 등)을 바꾸지 않는다. `repro_choi.sh`가 exp004의 `.sh` 형식을 처리하지 못하면, `.sh`에 적힌 인자를 그대로 옮겨 `PYTHONPATH=. python -u choi_exps/zip_20261107/exp004.py ...`로 직접 실행하고 그 명령을 보고서에 적는다.
2. 산출물: `out/exp004/exp004_<ds>_embeddings.pt` (exp016이 읽는 경로). 이름이 다르면 실제 이름을 적는다.
3. **G-bgrl** (exp016.py의 검증과 같다): 각 파일에서 (a) `dataset` 키 == 데이터셋 이름, (b) `embeddings`가 2차원이고 `shape[0]` == 노드 수(Cora_full 19,793, Amazon_clothing 24,919), (c) `node_ids` == `arange(N)`. 하나라도 어긋나면 **멈추고 보고**.
4. 복사: `cache/bgrl/Cora_full.pt`, `cache/bgrl/Amazon_clothing.pt`. md5, `embeddings.shape`, 행 노름 평균·표준편차를 보고서에 적는다. exp004가 출력한 지표(pool purity 비교 등)는 값만 옮긴다.
5. exp004 결과 JSON이 있으면 `results/hw/repro/zip_20261107_exp004/`에 복사.
6. 커밋 `HW03a-s1 [seio]: BGRL embeddings via Choi exp004 (md5 …)` (`.pt`는 git 밖이므로 결과 JSON·로그만) → `git pull --rebase` → `git push origin hl/hsan-weighting`.
7. **멈추고 사람**: 아래를 그대로 출력하고 멈춘다. 사람이 "계속"이라고 하면 §2로 간다.
   ```
   [iislab4에서 실행]
   mkdir -p ~/wkspace/NAQ/cache/bgrl
   rsync -avz h2tae@172.16.163.176:~/wkspace/NAQ/cache/bgrl/ ~/wkspace/NAQ/cache/bgrl/
   md5sum ~/wkspace/NAQ/cache/bgrl/*.pt
   # 기대 md5: Cora_full <값>, Amazon_clothing <값>
   ```

## 2. HW03a-s2: `nbr_bgrl` 확인 run (10 run)

- `hw/run_queue.sh`, 작업자 2개(`GPUS=0 WORKERS=2`), run 목록: `nbr_bgrl` × {Cora_full, Amazon_clothing} × seed 1212–1216.
- `python -m hw.summarize --ref nbr_raw --ref naq_base` (seio의 HW02 기준선과 짝짓기).
- 목적은 `.pt` 확인이다: 학습이 정상 완료되는가, 표 1 exp016(79.52/91.21) 근처인가, `nbr_raw`(같은 seio 5 seed) 대비 차이. 주 비교는 iislab4 10 seed에서 한다.
- 커밋 `HW03a-s2 [seio]: nbr_bgrl 5 seeds` → pull --rebase → push.

## 3. HW03a-s3: 학습 없는 이웃 순도 측정

### 3.1 선행 조건
`git fetch`로 `origin/hl/hsan-weighting`에 메시지가 `HW03a-a [iislab4]`로 시작하는 커밋이 있는지 확인한다. 없으면 5분마다 다시 확인, 최대 60분. 그래도 없으면 §3을 건너뛰고 보고서에 적는다. 있으면 `git pull --rebase`.

### 3.2 `hw/purity.py` (새 파일, 측정 전용 레이블 사용, CLAUDE.md §5)
- 입력: `--dataset`, `--seed`(기본 1212), `--episodes`(기본 2000).
- 학습 에피소드 풀은 `hw/run.py`의 설정 경로를 그대로 따라 만든다(seed → load → set_device → add_diffusion → ProtoNet 생성 → neighbor lookup → train_pool). 모델은 학습하지 않는다.
- 선택은 `hw/neighbors.py`의 함수를 import해서 쓴다(iislab4가 추가한 `pick`). 직접 다시 구현하지 않는다.
- 측정 (모든 에피소드의 query × 자기 클래스 선택에 대해 평균):

| 행 | 값 |
|---|---|
| `query_eq_support` | NaQ query의 GT == support의 GT 비율 |
| `support_1hop_homophily` | support의 유효 이웃 중 support와 같은 GT 비율 (선택 전) |
| `multi_nbr_frac`, `fallback_frac` | 유효 이웃 ≥2 비율, 이웃 없음 비율 |
| 선택기 6개: {raw, bgrl} × {max, contrast, random} | `nbr_eq_support`, `nbr_eq_query`, `pos_nbr_cos_raw`(query–선택 이웃 raw cos 평균) |

- `random`은 `pick` 구현의 전용 Generator(seed + 7919)를 쓴다.
- 출력: `results/hw/purity/HW03a_purity_<ds>.{json,md}`.

### 3.3 실행·커밋
두 데이터셋, seed 1212. 커밋 `HW03a-s3 [seio]: train-free neighbor purity` → pull --rebase → push.

## 4. 사전 등록 (s1 커밋에 포함, 판정은 사람)

| ID | 예측 | 기준 | 근거 | 확신도 |
|---|---|---|---|---|
| HWP024 | BGRL 선택(λ 1.0)이 seio에서 `nbr_raw` 이상이다 | `nbr_bgrl − nbr_raw` ≥ 0, 두 ds (seio 5 seed 짝지은 평균) | 표 1 exp016 > exp015 | 0.55 |
| HWP025 | 이웃 순도 순서는 max(bgrl) ≥ max(raw) > contrast > random | `nbr_eq_support` 기준, 두 ds, 같은 공간 안의 contrast·random 순서도 포함 | BGRL은 구조 정보를 담음; contrast는 query에서 먼 쪽을 고름 | 0.50 |

## 5. 범위 밖

- `configs/hw.json`·기존 `hw/*.py`·CLAUDE.md 수정, `results/hw_iislab4/` 수정, cos·HSAN arm 실행, 다른 데이터셋, BGRL 하이퍼파라미터 변경.

## 6. 보고 (`reports/HW03a_seio.md`, CLAUDE.md §7)

핵심 표: (1) exp004 실행 명령·소요 시간·G-bgrl·md5·shape·노름 통계; (2) `nbr_bgrl` 5 seed 값과 `paired_nbr_raw`, `paired_naq_base`(seio); (3) 순도 표(두 ds × 선택기 6개 + 참고 행); (4) HWP024–025 대조; (5) 판단한 사항; (6) HEAD.
