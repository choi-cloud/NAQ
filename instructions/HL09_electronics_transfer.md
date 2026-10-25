# HL09 — Amazon_eletronics 전이 (night-auto, 야간 4단계, 조건부)

작성 2026-10-25 · 이희태 · 브랜치 `hl/anchor` · 태그 `hl09-base` / `hl09` · GPU 2·3번 · 환경 `NAQ`
데이터셋 **Amazon_eletronics**(철자는 원본 그대로; 42,318 노드, 8,669 특징, 167 클래스 — 데이터에서 다시 읽어 기록), τ = 0.3.

**시작 조건**: `TZ=Asia/Seoul date`가 **05:30 이전**이고 HL08 §0-3(고정 에피소드 생성)이 성공했을 때만. 아니면 건너뛰고 HL10으로. **목표 시각 07:00 KST**, 큐 `--deadline 06:45`.

## 내용
HL08과 **완전히 같은 구성**을 데이터셋 이름만 바꿔 수행한다: §0 자산(pkl md5, 고정 에피소드 seed 1212–1216, 첫 run 단독), §1 기준선 10 run(`hl000_run.py --datasets Amazon_eletronics`, 태그 `hl09-base`), §2 진단(`hl010 --dataset Amazon_eletronics`, 파일명 앞 `elec_`), §4 레이블 없는 5 arm 25 run을 큐 앞에, §3 oracle 4 arm 20 run을 뒤에. 결과 `results/hl/hl09/`, 표 `reports/HL09_tables/`(T15 기준선, T16 학습).

시간이 모자라면 큐는 deadline에서 멈추고 완료된 seed만으로 표를 만든다(seed 수 표기). 기준선 10 run과 진단은 그 자체로 재료이므로 §1·§2는 반드시 끝낸다(시작했다면).

커밋 `HL09: electronics baselines` / `HL09: electronics diagnostics, oracle and label-free training`, 태그 `hl09-base` / `hl09`, push. 보고서 `reports/HL09_electronics_transfer.md`. progress.md 한 줄, 채팅 한 줄.

## 사전 등록 (계산값만; HL08과 같은 문항을 electronics에)
| 번호 | 예측 | 계산값 |
|---|---|---|
| HLP103 | B_Choi − NaQ (1-shot) > 0 | |
| HLP104 | B4 기본 순도 ≤ 0.18 | |
| HLP105 | `enc_b` `auroc_all` ≥ 0.70 | |
| HLP106 | `g1_p10` − `g1_p04` ≥ +3.0 | |
| HLP107 | `far` − `near` ≥ +2.0 | |
| HLP108 | `encb_strat5` − `encb_top10` ≥ +1.5 | |
| HLP109 | `encb_strat5` − B_Choi ≥ +0.5 | |
| HLP110 | `choi_far` − B_Choi ≥ 0 | |
| HLP111 | `rand_strat5` ≤ B_Choi | |
| HLP112 | `enc_b` `strat5` 순도 ∈ [0.25, 0.45] | |
