#!/usr/bin/env bash
set -euo pipefail
TAU="${1:-0.3}"
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
mkdir -p "out/${DATE}"
LOG="out/${DATE}/exp048_tau${TAU}_${TIME}.log"
python -u exp048.py --dataset Amazon_clothing --device 0 \
    --n_way 5 --k_shot 1 --k_shot_test 1 --q_query 10 --q_query_test 8 \
    --lr 1e-4 --latent 256 --epoch 2000 --l2_penalty 1e-4 \
    --num_valid_tasks 50 --num_downstream_tasks 500 --eval_interval 160 \
    --seed 1212 --tau "$TAU" --min_candidate_policy relax > "$LOG" 2>&1 &
echo "exp048 PID=$! tau=$TAU log=$LOG"
