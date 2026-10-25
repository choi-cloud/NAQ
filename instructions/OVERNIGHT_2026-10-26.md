# OVERNIGHT 2026-10-25 23:30 → 10-26 08:00 KST — 야간 작업 시퀀스 (night-auto)

작성 2026-10-25 · 이희태 · 브랜치 `hl/anchor` · 서버 gpusystem, GPU **2·3번** · 환경 `NAQ`
이 문서가 야간의 마스터다. 각 단계의 상세는 해당 지시서에 있다. 규칙은 CLAUDE.md §7-1(야간 모드).

## 목적
1. 스토리 D(순도·거리 분리 query 선택)의 핵심 가설 H-D를 Clothing에서 판정할 수치(HL06 HLP078·073)를 아침까지 낸다.
2. 가설이 틀리더라도 아침의 판단 재료가 남게 한다: (순도, 거리) 2D oracle 격자(HL07), 두 번째·세 번째 데이터셋의 기준선·진단·oracle 끝점(HL08·HL09), 변형 통제(HL07·HL10).
3. **GPU가 비는 시간을 만들지 않는다.** 어떤 단계가 실패해도 다음 단계로 간다.

## 시퀀스와 시각
| # | 지시서 | 핵심 산출 | run | 시작 조건 | 목표 시각 | 큐 deadline |
|---|---|---|---|---|---|---|
| 0 | 이 문서 §시작 | 문서 커밋, GPU 기록, progress.md 생성 | — | — | 23:45 | — |
| 1 | HL06 | P1·P2·R1 진단, T10(oracle 4 + 레이블 없는 6) | 50 | 항상 | 01:45 | 01:30 |
| 2 | HL07 | T11 격자, T12 변형 | 85 | hl011 존재 | 03:15 | 03:00 |
| 3 | HL08 | T13 dblp 기준선, dblp P1·P2, T14 | 55 | 데이터·에피소드 생성 성공 | 05:15 | 05:00 |
| 4 | HL09 | electronics 같은 구성 | 55 | **05:30 이전 시작** + HL08 §0-3 성공 | 07:00 | 06:45 |
| 5 | HL10 | T17 채움 | 40 | hl011 존재; 다른 단계 건너뛰어졌을 때 즉시, 또는 남는 시간 | 07:15 | 07:15 |
| 마감 | 이 문서 §마감 | `reports/OVERNIGHT_2026-10-26.md`, push, 채팅 보고 | — | 07:15 이후 새 run 금지 | 07:45 | — |

예상 GPU 시간(6슬롯): HL06 25분, HL07 40분, HL08 60–75분, HL09 60–75분, HL10 20분 — 합 약 4시간. 나머지는 코드 작성·smoke·표·보고서 시간이다. 한 단계가 목표 시각을 넘기면 그 단계의 큐를 deadline에서 멈추고 다음으로 간다. 모든 단계가 일찍 끝나면 HL10 → 미완 run 재시작(HL06 → HL07 → HL08 순) → 그래도 남으면 §여분.

## 시작 (0단계)
1. `TZ=Asia/Seoul date` 기록. 브랜치 `hl/anchor`, HEAD `1d0a4ce`, 추적 파일 변경은 이희태가 넣은 문서들뿐인지 확인: `CLAUDE.md`, `instructions/{OVERNIGHT_2026-10-26,HL06_purity_distance_split,HL07_clothing_grid_variants,HL08_dblp_transfer,HL09_electronics_transfer,HL10_clothing_fillers}.md`, `reports/HL_storyC_closure.md`. 이들을 한 커밋으로: `OVERNIGHT: closure doc, CLAUDE.md rev8, HL06-HL10 instructions`. push.
2. GPU 2·3 상태 기록(`results/hl/overnight/gpu_start.txt`). 둘 다 쓸 수 없으면 30분 간격 세 번 재확인 뒤 멈춤(유일한 두 번째 멈춤 사유).
3. `results/hl/overnight/progress.md` 생성(첫 줄: 시작 시각, HEAD). 이후 모든 단계 전환마다 한 줄 추가하고, 단계 커밋 때 `reports/overnight_progress.md`로 복사해 함께 커밋.
4. 디스크 여유 확인(`df -h .`): 20GB 미만이면 `results/hl/hl0{3,4}/**/best_5shot.pt`를 지워 확보(1-shot 체크포인트와 report.json은 남긴다). 기록.

## 단계 공통
- 각 단계: 지시서 읽기 → 코드 추가 → smoke(epoch 20, 실패 arm 1회 수정 후 제외) → 큐(`hl000f`, `--deadline`) → 표 → 보고서 → 커밋·태그·push → progress.md → 채팅 한 줄.
- **큐가 도는 동안 다음 단계의 코드를 쓴다**(GPU는 큐가 쓰고 있으므로 다음 단계 smoke는 큐가 끝난 뒤). 러너 감시는 PID 파일로만.
- 큐 안의 arm 순서는 지시서가 정한다(레이블 없는 arm 먼저 — deadline에 걸려도 핵심 결과가 남게).
- 보고서는 사실만(CLAUDE.md §5). 사전 등록은 계산값만. 판정은 아침에 이희태.

## 여분(모든 단계가 07:00 전에 끝났을 때만)
HL06 레이블 없는 6 arm을 seed 1217–1221로 5 run씩 더(고정 에피소드 생성 필요: `hl000_run.py`가 만든다). 결과 `results/hl/hl06/train_extra/`. 표에 10-seed 평균을 추가 열로.

## 마감 (07:15 이후)
1. 실행 중인 run은 끝까지(20분 상한). 새 run 금지.
2. `reports/OVERNIGHT_2026-10-26.md` 작성(사실만):
   - 단계별 상태 표(progress.md 그대로), 실제로 쓴 GPU·시간, 건너뛴 단계와 이유, 실패 arm.
   - 단계별 핵심 표를 **그대로 붙인다**: HL06 P1(strat5·ex20·far50·top10·무작위·choi 행)·P2·R1·T10, HL07 T11·T12, HL08 T13·T14·dblp P2, HL09 T15·T16, HL10 T17.
   - 사전 등록 HLP057–118 통합 대조표(계산값만; 미실행은 "미실행").
   - 산출물 경로, 커밋·태그 목록.
3. 커밋 `OVERNIGHT: morning report`, push.
4. 채팅: progress.md 전체 + HL06 T10 + HL07 T12 + HL08 T14(있으면) + 통합 사전 등록 표를 붙이고 끝낸다. 질문·해석 없음.
