# HL06 — 순도·거리 분리 query 선택: 판정기 × 거리 규칙 채점, R1 기술, oracle 4 arm, 레이블 없는 학습 6 arm (night-auto, 야간 1단계)

작성 2026-10-25 · 이희태 · 브랜치 `hl/anchor` · 태그 `hl06-diag` / `hl06-train` · 서버 gpusystem, **GPU 2·3번**(CLAUDE.md §2), 환경 `NAQ`
데이터셋 Amazon_clothing, τ = 0.3. 기준선 = HL03 B_Choi uniform 82.71 ± 0.31 / 89.91 ± 0.57.
**목표 시각 01:45 KST**(CLAUDE.md §7-1). 이 지시서의 `hl_pdd.py`·`hl010`·`hl011`·`hl000f`는 HL07–HL10이 그대로 재사용하므로 **데이터셋·체크포인트·anchor 파일을 인자로 받게** 만든다(기본값은 Clothing).

## 배경 (스토리 D)

스토리 C는 종결했다(`reports/HL_storyC_closure.md`). 남은 사실을 두 축으로 다시 읽는다.
- 순도 1.0에서: 구조적으로 무작위인 same-class(G1 p=1.0) 89.42 ≈ 먼 same-class(HL05 far) 89.46 ≫ 가까운 same-class(near) 85.49. **근접 비용은 극단적으로 가까운 쪽에서만 난다.**
- 레이블 없는 방법은 G1 순도 곡선보다 아래에 있고, 그 폭이 구조적 근접성 순서와 같다: NaQ(순도 0.60) 80.65, HL04 `comb_top10`(0.53) 79.41, B_Choi(0.42, 필터 + middle) 82.71.
- 어떤 판정기든 **top-k는 정의상 가장 가까운 후보**를 고른다.

가설 H-D: 판정기 선택을 anchor 기준 Â²X 층(stratum) 안으로 제한해 극단적 근접을 피하면, 같은 판정기·같은 teacher의 top-10보다 1-shot이 오르고 B_Choi를 넘는다.

이번 지시서가 하는 일:
1. **진단(학습 없음)**: 기존·새 판정기 12개 × 거리 규칙 7개의 B4 순도와 구조 기술 통계(§1).
2. **R1 기술**: 이미 학습한 arm들(NaQ, B_Choi, HL04, HL03 G1, HL05, p04_choidiff)의 선택 query를 같은 anchor 2,000개에서 같은 통계로 기술(§1-5).
3. **학습 10 arm × 5 seed = 50 run**: oracle 4 arm(R1 분해, far 규칙 상한)과 레이블 없는 6 arm(판정기 사전 고정)(§2).

레이블 사용 범위: §1의 채점·기술 통계, §2의 oracle 4 arm의 query 선택, §2 모든 arm의 사후 query 진단에만. 레이블 없는 판정기·규칙 함수(`hl/hl_pdd.py`)에는 레이블을 인자로 넘기지 않는다.

---

## 0. 시작

1. 브랜치 `hl/anchor`, HEAD가 `1d0a4ce`(HL05: structural near/far same-class oracle) 위의 문서 커밋 `OVERNIGHT: closure doc, CLAUDE.md rev8, HL06-HL10 instructions`인지, `git status`에 추적 파일 변경이 없는지 확인. 아니면 **정지**(야간 모드에서도 멈추는 두 경우 중 하나).
2. `nvidia-smi -i 2,3 --query-gpu=index,memory.used,memory.total --format=csv`와 `nvidia-smi` 프로세스 목록을 `results/hl/hl06/gpu_status_start.txt`에 기록. GPU 판단은 CLAUDE.md §2 규칙대로.
3. 결과 루트 `results/hl/hl06/`, 표 `reports/HL06_tables/`.
4. 재사용: `results/hl/hl03/g0/anchors.npy`(2,000개), 체크포인트 둘(`enc_b` = `results/hl/p0/Amazon_clothing/ours/tau_0.3/seed_1212/best_1shot.pt`, `enc_n` = `.../naq/baseline/seed_1212/best_1shot.pt`), `hl/hl005`·`hl007`·`hl008`·`hl009`·`hl006`의 함수(import), 노드표.

## 1. 공용 모듈 `hl/hl_pdd.py` + 진단 `hl/hl010_diag_pdd.py` (학습 없음, 태그 `hl06-diag`)

### 1-1. 공통 정의 (anchor a마다)
- **B4(a)** = X 코사인 순위 201–1000의 800개 후보(hl005와 같은 순위, 자기 자신 제외, 동률은 stable).
- **s2(a, q)** = Â²X 코사인(`hl_common.structural_scores`의 z2).
- **s2 순위**: B4(a)의 800개를 s2 내림차순으로 정렬한 순위 1–800(동률은 X 순위 순). 층 **T1 = s2 순위 1–160**(구조적으로 가장 가까움), T2 = 161–320, T3 = 321–480, T4 = 481–640, T5 = 641–800.

### 1-2. 거리 규칙 (`hl_pdd.select(rule, score_row, s2rank_row, rng)`, B4 열 인덱스 10개 반환, 레이블 없음)
| 규칙 | 정의 |
|---|---|
| `top10` | B4 전체에서 점수 상위 10 (HL04와 같음) |
| `ex20` | T1을 뺀 640개(s2 순위 161–800)에서 점수 상위 10 |
| `far50` | s2 순위 401–800에서 점수 상위 10 |
| `strat5` | T1–T5 각 층에서 점수 상위 2개씩 |
| `rand` | B4 전체 무작위 10 |
| `rand_ex20` | s2 순위 161–800 무작위 10 |
| `rand_far50` | s2 순위 401–800 무작위 10 |

동률은 X 순위 순(stable). 무작위는 `numpy.random.default_rng(seed)`를 anchor 순서대로 소비(§1 진단은 seed 1212).

### 1-3. 판정기 (레이블 없음, 모두 "클수록 같은 클래스일 것 같다")
기존 8개는 `hl005_diag_distance_purity.pair_scores`와 **같은 정의**: `x`, `ax`, `a2x`, `ppr`, `mv`, `enc_b`, `enc_n`, `comb`(hl005 정의 = mv + enc_b).
새 4개(teacher = `enc_b` 체크포인트, `hl005.encoder_embeddings`로 얻은 행 정규화 임베딩 E):
| 이름 | 정의 |
|---|---|
| `xidf` | X를 TF-IDF로 다시 가중한 코사인. df_j = feature j가 0보다 큰 노드 수(24,919개 전체), idf_j = ln((1+n)/(1+df_j)) + 1, 값 = log1p(count)·idf_j, 행 L2 정규화(0행은 0), 내적. `data.features`는 원시 count이다. |
| `nbr_b` | 임베딩 이웃 평균의 코사인. K(v) = E 코사인 최근접 10개(자기 제외), p(v) = normalize(E_v + Σ_{u∈K(v)} E_u), 점수 = p(a)·p(q). |
| `knnppr_b` | 임베딩 kNN 그래프 위의 PPR. K(v)로 방향 그래프를 만들고 대칭화(A_e = max(K, Kᵀ)), 자기 루프 추가, `hl005.sym_adjacency`와 같은 D^-1/2 정규화. `hl005.ppr_columns`(α 0.15, 30회)로 anchor를 seed로 한 PPR 값이 점수. |
| `clu_b` | k-means 공동 소속. sklearn `KMeans(n_clusters=200, n_init=1, random_state=r)`, r = 0–4, 입력 E(CPU numpy). 점수 = 5회 중 a와 같은 군집에 든 비율 + 1e-3 · cos_E(a, q)(동률 깨기). |

kNN은 24,919 × 24,919 코사인을 2,048행씩 나눠 계산한다. 이 판정기들의 계산 함수는 `hl_pdd.py`에 두고 §2 학습 스크립트가 같은 함수를 import한다.

**정합성 검사**: `hl_pdd`로 계산한 `a2x`·`enc_b`·`ppr`의 B4 점수가 `hl005.pair_scores`의 같은 열과 최대 절대 오차 < 1e-5인지 확인하고 오차를 `P_summary.json`에 기록. 1e-5를 넘으면 **멈추지 않고** 그 사실을 ④에 적고 계속한다(야간 모드). 1e-3을 넘으면 `hl_pdd`의 계산을 `hl005.pair_scores` 호출로 바꿔(결과를 그대로 쓰도록) 다시 돌린다.

**인자**: `hl010`은 `--dataset`, `--anchors`(없으면 seed 1212로 2,000개 새로 추출해 `results/hl/<HL>/anchors.npy`에 저장), `--ckpt_b`, `--ckpt_n`, `--output_dir`, `--table_dir`를 받는다. 모든 노드 수·클래스 수는 데이터에서 읽는다(Clothing 상수 하드코딩 금지).

### 1-4. 채점 (GT는 모든 선택이 끝난 뒤 한 번 읽는다)
판정기 12개 × 규칙 {`top10`, `ex20`, `far50`, `strat5`} + 무작위 규칙 3개 + `choi`(hl007 `select_choi`, B4 안 필터 통과분 X 중앙 10) 각각에 대해, anchor 2,000개의 선택 query로:
- `purity_pooled`, `purity_anchor_mean`
- `mean_x_rank`, `mean_s2`(anchor–query s2 평균), `mean_s2rank`(B4 안 s2 순위 평균, 1–800)
- `mean_encb`(anchor–query cos_E), `pair_s2`(선택 10개 사이 쌍별 s2 평균, 45쌍), `pair_encb`(같은 방식, cos_E)
- `hop1_frac`, `hop2_frac`: anchor와 1-hop, 2-hop 이내인 query 비율(원 그래프, 자기 루프 없음, 대칭화)
- `tp_s2`, `fp_s2`: 선택 중 GT-same / GT-diff만의 `mean_s2`
- `choi_pass_frac`: 선택 중 석휘님 필터(ÂX ≥ 0.3 and Â²X ≥ 0.3) 통과 비율

판정기별 분별력: `auroc_all`(B4 800개), `auroc_ex20`(s2 순위 161–800), `auroc_far50`(401–800) — anchor별 Mann–Whitney AUROC(양·음성이 모두 있는 anchor만) 평균. `hl005.auroc_rows` 사용. 층별 기본 순도(T1–T5 각각의 same-GT 비율)도 기록.

출력: `P1_judge_rule.csv`(judge × rule × 위 통계), `P2_auroc.csv`(judge × {auroc_all, auroc_ex20, auroc_far50}), `P3_stratum_base.csv`(T1–T5 기본 순도, 평균 s2), `P_summary.json`.

### 1-5. R1 기술 `R1_descriptors.csv` (같은 anchor 2,000개, §1-4와 같은 통계)
이미 학습한 arm의 query 선택 규칙을 그 스크립트의 함수를 import해 그대로 재현하고 기술만 한다(학습 없음).
| 행 | 선택 | 출처 함수 |
|---|---|---|
| `naq_x10` | X 순위 1–10 | — |
| `bchoi` | B_Choi 실제 query(전체 노드 대상, X 순위 제한 없음) | `hl_common.mine_queries_for` |
| `hl04_comb_top10`, `hl04_a2x_top10`, `hl04_encn_top10`, `hl04_mv_mid30`, `hl04_rand10` | HL04 G2 arm | `hl008.b4_scores` + `hl008.label_free_queries`(rng = default_rng(1212)) |
| `g1_p04`, `g1_p10` | HL03 G1 p=0.4, 1.0 | `hl006.purity_oracle_queries` |
| `p04_choidiff` | HL04 보조 oracle | `hl008.p04_choidiff_queries` |
| `hl05_near`, `hl05_far` | HL05 oracle | `hl009.structural_oracle_queries` |
| §2의 10 arm | 이번 학습 arm | `hl_pdd` / `hl011` |

B4 밖 query(`naq_x10`, `bchoi`, 보충분)는 `mean_s2rank`를 비우고 나머지 통계는 계산한다. 각 행 옆에 알려진 1-shot / 5-shot(CLAUDE.md §8, HL04 T7, HL05 T8에서 옮김; 새 arm은 §2 뒤 채움)을 둔다.

실행: `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD python hl/hl010_diag_pdd.py --device 0`. 계산 시간과 `max_memory_reserved`를 기록.

커밋 `HL06: purity-distance diagnostics`, 태그 `hl06-diag`.

## 2. 학습 `hl/hl011_pdd_train.py` + 러너 `hl/hl000f_run_hl06.py` (태그 `hl06-train`)

`hl/hl008_distant_query_train.py`를 복사해 만든다(docstring에 출처). anchor는 uniform(hl003과 같은 코드 경로·난수). 바꾸는 곳은 query 선택 한 군데이고, 각 anchor의 query는 한 번 정해 캐시한다(exp049와 같음). 학습·평가·report.json 형식은 hl008 그대로.

**데이터셋 일반화(HL08·HL09가 쓴다)**: `--dataset`, `--teacher`(enc_b 체크포인트), `--teacher_n`(enc_n), `--node_table`(빈 문자열이면 `anchor_pi_bchoi_mean` 진단을 비움), `--x_band lo,hi`(기본 201,1000), `--jobs` 무관. 출력 루트는 `results/hl/<HL##>/train/<dataset>/<arm>/seed_<seed>/`.

**참고 oracle arm(hl011에 넣되 HL06에서는 돌리지 않음, HL08·HL09용)**: `g1_p04`·`g1_p10`(`hl006.purity_oracle_queries` import), `near`·`far`(`hl009.structural_oracle_queries` import; 임베딩 진단은 생략), `rand10`(hl008 규칙). 이 arm들은 Clothing에서는 이미 결과가 있다.

### 2-1. 레이블 없는 arm (6개, 판정기는 사전 고정 — §1 결과로 바꾸지 않는다)
teacher = `enc_b` 체크포인트(seed와 무관하게 seed 1212 하나 고정). 점수·규칙은 `hl_pdd` 함수 그대로.
| arm | 판정기 | 규칙 | 역할 |
|---|---|---|---|
| `encb_top10` | `enc_b` | `top10` | 대조: 같은 teacher의 top-10 |
| `encb_strat5` | `enc_b` | `strat5` | **핵심** |
| `encb_ex20` | `enc_b` | `ex20` | 극단만 제거 |
| `encb_far50` | `enc_b` | `far50` | 먼 절반 |
| `knnppr_strat5` | `knnppr_b` | `strat5` | 전이적 판정기 |
| `knnppr_ex20` | `knnppr_b` | `ex20` | 전이적 판정기 |

### 2-2. oracle arm (4개, GT 사용)
| arm | 정의 | 묻는 것 |
|---|---|---|
| `o_topsame` | B4 GT-same 중 s2 상위 10. 부족하면 B5(1001–3000) GT-same의 s2 상위, 그래도 부족하면 B4 GT-diff의 s2 상위로 채움(`diff_fill` 기록) | 순도 1.0이어도 극단적으로 가까우면 얼마나 떨어지는가 (R1 (a)) |
| `o_combswap` | HL04 `comb_top10`(`hl008` 함수, teacher enc_n 그대로)의 선택에서 GT-same은 두고, GT-diff 각각을 아직 안 뽑힌 B4 GT-diff에서 무작위로 교체(rng = default_rng(seed + 1)) | comb_top10이 낮은 이유가 FP의 종류인가 (R1 (b)) |
| `o_far50_p03` | s2 순위 401–800에서 GT-same 3개 + GT-diff 7개 무작위(rng = default_rng(seed + 1)). GT-same이 부족하면 s2 순위 161–400의 GT-same 무작위, 그래도 부족하면 far50 GT-diff로 채움 | far50 규칙의 순도 0.3 지점 |
| `o_far50_p05` | 같은 방식, GT-same 5 + GT-diff 5 | far50 규칙의 순도 0.5 지점 |

### 2-3. query 진단 (모든 arm, report.json `query_diagnostics`)
§1-4의 통계(`purity`, `mean_x_rank`, `mean_s2`, `mean_s2rank`, `pair_s2`, `hop2_frac`, `choi_pass_frac`, `tp_s2`, `fp_s2`)와 보충 단계별 비율. 레이블 없는 arm에서는 이 진단만 레이블을 읽는다(선택이 끝난 뒤).

### 2-4. 실행
- 먼저 smoke: 10 arm 각각 `--epoch 20 --seed 1212`, 출력 `results/hl/hl06/_smoke/`(표에 넣지 않음, 끝나면 삭제). 한 arm이라도 오류면 한 번 고치고 다시 smoke. 또 실패하면 그 arm만 빼고 진행하고 ④에 적는다.
- 본 실행: 10 arm × seed 1212–1216 = **50 run**. `hl/hl000f_run_hl06.py`(hl000e 복사)에 GPU 슬롯을 추가: `--gpus 2,3 --jobs_per_gpu 3`, 각 작업은 `CUDA_VISIBLE_DEVICES=<슬롯 GPU> ... --device 0`, summary.csv에 `gpu` 열. 그 밖의 인자: `--script hl/hl011_pdd_train.py`, `--arms`, `--seeds`, `--dataset`, `--output_root`, `--extra "<hl011에 그대로 넘길 인자>"`, `--run_timeout_sec`(기본 1200), `--deadline "HH:MM"`(KST; 지나면 새 run을 시작하지 않음). 큐 순서는 seed 바깥, arm 안쪽(seed 1212의 10 arm → 1213 …). 재시작 가능(report.json 있는 run 건너뜀). 이 러너 하나를 HL07–HL10이 모두 쓴다.
- 결과 `results/hl/hl06/train/Amazon_clothing/<arm>/seed_<seed>/`. 시작·끝 GPU 상태를 기록. 이 큐의 `--deadline 01:30`.

### 2-5. 표
- **T10** `T10_train.csv`: arm × {1-shot 평균±std, 5-shot 평균±std, Δ1·Δ5 vs B_Choi(82.71/89.91), purity, mean_x_rank, mean_s2, mean_s2rank, pair_s2, hop2_frac, choi_pass_frac, 보충 비율}. 참고 행(값은 기존 표에서 옮김, 기술 통계는 `R1_descriptors.csv`에서): NaQ, B_Choi, G1 p=0.4, G1 p=1.0, `p04_choidiff`, HL05 near/far, HL04 `comb_top10`, `encn_top10`, `rand10`.
- **T10b** `T10b_train_per_seed.csv`: run별.
- `R1_descriptors.csv`의 새 arm 행에 1-shot / 5-shot을 채워 다시 저장.

커밋 `HL06: purity-distance training`, 태그 `hl06-train`, push.

## 3. 사전 등록 (계산값만; 임계값은 편의 목표치)

§1 진단(B4, anchor 2,000개):
| 번호 | 예측 | 계산값 |
|---|---|---|
| HLP057 | `rand_far50` 순도 ≤ 0.10 (먼 절반의 기본 순도) | |
| HLP058 | `rand_ex20` 순도 ≤ 0.16 | |
| HLP059 | `enc_b` `auroc_far50` ≥ 0.65 (먼 절반 안에서도 분별력이 남는다) | |
| HLP060 | `enc_b` `strat5` 순도 ∈ [0.33, 0.45] | |
| HLP061 | `enc_b` `far50` 순도 ∈ [0.20, 0.33] | |
| HLP062 | **게이트 G3**: 어느 판정기든 `strat5` 또는 `ex20` 순도 ≥ 0.45 | |
| HLP063 | 새 판정기(`xidf`, `nbr_b`, `knnppr_b`, `clu_b`) 중 하나 이상이 `strat5` 순도에서 `enc_b`보다 +0.03 이상 | |
| HLP064 | `xidf` `auroc_far50` ≥ `x` `auroc_far50` + 0.03 | |

§1-5 R1 기술:
| 번호 | 예측 | 계산값 |
|---|---|---|
| HLP065 | `hl04_comb_top10`의 `mean_s2` ≥ 0.70 (HL05 near 0.59보다 더 가깝다) | |
| HLP066 | `hl04_comb_top10`의 `pair_s2` ≥ 0.60, `bchoi`의 `pair_s2` ≤ 0.40 | |
| HLP067 | `hl04_comb_top10`의 `hop2_frac` ≥ 2 × `bchoi`의 `hop2_frac` | |
| HLP068 | `hl04_comb_top10`의 `fp_s2` ≥ `p04_choidiff`의 `fp_s2` | |

§2 oracle:
| 번호 | 예측 | 계산값 |
|---|---|---|
| HLP069 | `o_topsame` 1-shot ≤ HL05 near − 1.0 (= 84.49) | |
| HLP070 | \|`o_combswap` − HL04 `comb_top10`(79.41)\| ≤ 1.0 (FP 종류는 원인이 아니다) | |
| HLP071 | `o_far50_p05` 1-shot ≥ 85.0 | |
| HLP072 | `o_far50_p03` 1-shot ≥ B_Choi + 0.5 (= 83.21) | |

§2 레이블 없음:
| 번호 | 예측 | 계산값 |
|---|---|---|
| HLP073 | `encb_strat5` − B_Choi ≥ +1.0 (1-shot) | |
| HLP074 | `encb_ex20` − B_Choi ≥ +1.0 (1-shot) | |
| HLP075 | `encb_far50` < `encb_strat5` (1-shot) | |
| HLP076 | `knnppr_strat5` ≥ `encb_strat5` (1-shot) | |
| HLP077 | 레이블 없는 6 arm 중 1-shot 최고 arm의 5-shot ≥ B_Choi(89.91) | |
| HLP078 | **핵심**: `encb_strat5` − `encb_top10` ≥ +2.0 (1-shot) | |
| HLP079 | `encb_top10`은 HL04 `encn_top10`(79.94)과 ±1.0 안 | |

**판정 규칙(이희태 적용)**:
- HLP078·HLP073 둘 다 맞음 → H-D 확인. HL07 = 판정기 보강(§1 최고 판정기) + 동적 teacher + 두 번째 데이터셋.
- HLP078만 맞음 → 기제는 확인, 순도 부족. HL07은 판정기 보강 1회(§1 최고 판정기, 손실 기반 FP 억제 포함)로 한정.
- HLP078 틀림 → H-D 기각. Clothing에서 레이블 없는 FP 억제 노선을 닫고 분석 문서를 쓴다.
- HLP070이 틀리면(교체로 ±1.0 밖) R1 해석에 "FP 종류" 단서를 붙인다. HLP069가 틀리면(o_topsame > 84.49) 극단적 근접만으로는 comb_top10의 손실을 설명하지 못한다고 적는다.

## 4. 보고서 `reports/HL06_purity_distance_split.md`

CLAUDE.md §5 형식. 포함: GPU 사용 기록(시작·끝 상태, 실제로 쓴 GPU, run별 gpu 분포), §1 정합성 검사 결과, `P1_judge_rule.csv`의 `strat5`·`ex20`·`far50`·`top10` 행 전체(판정기 12개)와 무작위 규칙·`choi` 행, `P2_auroc.csv`, `P3_stratum_base.csv`, `R1_descriptors.csv`, T10, 사전 등록 표, 실패·경고, 산출물 경로. 보고서를 커밋·push하고 `progress.md`에 한 줄 적은 뒤 **채팅에는 한 줄만**("HL06 끝, 완료 run n/50, 커밋 해시") 쓰고 HL07로 간다. 표는 아침 마스터 보고에 붙인다.
