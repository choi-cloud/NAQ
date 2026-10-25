# HL08 — dblp 전이: 기준선, 진단, oracle 20 run, 레이블 없는 25 run (night-auto, 야간 3단계)

작성 2026-10-25 · 이희태 · 브랜치 `hl/anchor` · 태그 `hl08-base` / `hl08` · GPU 2·3번 · 환경 `NAQ`
데이터셋 **dblp**(40,672 노드, 7,202 특징, 137 클래스 — 수치는 데이터에서 다시 읽어 기록), τ = 0.3(Clothing 값 그대로; τ 탐색은 하지 않는다). **목표 시각 05:15 KST**, 큐 `--deadline 05:00`.

## 배경
Clothing에서 본 두 축(순도, 구조적 거리)의 그림이 다른 그래프에서도 성립하는지가 논문의 생사다. 이 단계는 dblp에서 (i) NaQ vs B_Choi 기준선, (ii) 판정기 × 거리 규칙 순도 진단, (iii) 두 oracle 곡선의 끝점(G1 p=0.4/1.0, near/far), (iv) 레이블 없는 핵심 arm을 한 번에 돈다. 결과가 어느 쪽이든 아침의 판단 재료다.

레이블 사용 범위: 진단 채점, oracle arm 선택, 사후 진단에만.

## 0. 시작·자산 (태그 `hl08-base`)
1. `data/dblp_network`, `data/dblp_{train,test}.mat` 존재 확인. 없으면 이 단계를 건너뛰고(progress.md 기록) HL10으로.
2. `save/downstream_tasks/dblp/5way_{1,5}shot.pkl` 존재·md5 기록. **없으면 §1의 첫 run을 단독으로 돌려 만든 뒤** md5를 기록한다(CLAUDE.md §9).
3. 고정 에피소드: `save/fixed_episodes/dblp/5way_{1,5}shot_8query_seed{1212..1216}/` — `hl/hl000_run.py`가 없으면 exp026으로 만든다. 만들어진 뒤 `exp026.load_fixed_episodes`로 20개 파일 로드 확인. 생성이 실패하면 이 단계를 건너뛰고 HL10으로.
4. GPU 메모리: dblp는 run당 메모리가 크다(CLAUDE.md §9). 첫 run의 `max_memory_reserved`를 보고 GPU당 동시 수를 정한다(여유 20GB 이상이면 3, 아니면 2).
5. 결과 `results/hl/hl08/`, 표 `reports/HL08_tables/`.

## 1. 기준선 (10 run)
- `hl/hl000_run.py --datasets dblp --arms naq,ours --seeds 1212,1213,1214,1215,1216 --output_root results/hl/hl08/base` (`ours` = exp049 τ 0.3). 이 러너는 순차 실행이므로 **seed를 둘로 나눠 GPU 2·3에서 두 인스턴스**로 돌린다(예: 1212,1213,1214 / 1215,1216). 단, §0-2에서 pkl이 없었다면 첫 인스턴스의 첫 run(naq 1212)이 끝난 뒤 두 번째 인스턴스를 시작한다.
- 체크포인트: `results/hl/hl08/base/dblp/ours/tau_0.3/seed_1212/best_1shot.pt` = **dblp teacher(enc_b)**, `.../naq/baseline/seed_1212/best_1shot.pt` = enc_n. 경로가 다르면 실제 경로를 기록하고 그것을 쓴다.
- **T13** `T13_dblp_baseline.csv`: arm × {1-shot 평균±std, 5-shot, valid, wall, 메모리}. run 시간·메모리를 CLAUDE.md §9 갱신용으로 기록.
- 커밋 `HL08: dblp baselines`, 태그 `hl08-base`.

## 2. 진단 (학습 없음, GPU 3, 기준선 큐가 끝난 뒤)
`hl/hl010_diag_pdd.py --dataset dblp --ckpt_b <teacher> --ckpt_n <enc_n> --output_dir results/hl/hl08/diag --table_dir reports/HL08_tables` — anchor 2,000개는 seed 1212로 새로 추출. 출력 `P1_judge_rule.csv`, `P2_auroc.csv`, `P3_stratum_base.csv`(파일명 앞에 `dblp_`). R1 기술은 `naq_x10`, `bchoi`, 그리고 §3·§4 arm만(HL04 등 Clothing 전용 행은 제외). 정합성 검사는 HL06과 같이 기록만.

## 3. oracle (4 arm × 5 seed = 20 run)
`hl011 --dataset dblp --node_table ""`로 참고 oracle arm: `g1_p04`, `g1_p10`, `near`, `far`. 결과 `results/hl/hl08/train/dblp/<arm>/seed_<seed>/`.

## 4. 레이블 없음 (5 arm × 5 seed = 25 run)
teacher = §1의 dblp enc_b. arm: `encb_top10`, `encb_strat5`, `encb_ex20`, `choi_far`(HL07 §2 정의; hl011에 없으면 제외), `rand_strat5`.
큐 순서: §4를 §3보다 **앞에**(deadline에 걸려도 레이블 없는 결과가 먼저 남게). 한 큐로 `--deadline 05:00`.

## 5. 표·보고
- **T14** `T14_dblp_train.csv`: arm × {1-shot 평균±std, 5-shot, Δ vs dblp B_Choi(T13), purity, mean_s2, mean_s2rank, pair_s2, 보충 비율}. 참고 행: NaQ·B_Choi(T13).
- 커밋 `HL08: dblp diagnostics, oracle and label-free training`, 태그 `hl08`, push. 보고서 `reports/HL08_dblp_transfer.md`(CLAUDE.md §5 형식; 데이터셋 기본 수치, pkl md5, 고정 에피소드 생성 기록, run 시간·메모리 포함). progress.md 한 줄, 채팅 한 줄.

## 6. 사전 등록 (계산값만)
| 번호 | 예측 | 계산값 |
|---|---|---|
| HLP093 | T13: B_Choi − NaQ (1-shot) > 0 | |
| HLP094 | dblp B4 기본 순도(`rand`) ≤ 0.18 (Clothing 값) | |
| HLP095 | dblp `enc_b` `auroc_all` ≥ 0.70 | |
| HLP096 | `g1_p10` − `g1_p04` ≥ +3.0 (1-shot) | |
| HLP097 | `far` − `near` ≥ +2.0 (1-shot) | |
| HLP098 | `encb_strat5` − `encb_top10` ≥ +1.5 (1-shot) | |
| HLP099 | `encb_strat5` − B_Choi(T13) ≥ +0.5 | |
| HLP100 | `choi_far` − B_Choi(T13) ≥ 0 | |
| HLP101 | `rand_strat5` ≤ B_Choi(T13) | |
| HLP102 | dblp `enc_b` `strat5` 순도 ∈ [0.25, 0.45] | |
