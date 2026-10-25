#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p "out/$(date +%y%m%d)"
LOG_FILE="out/$(date +%y%m%d)/exp069_$(date +%H%M%S).log"
nohup python -u exp069.py \
  --dataset Cora_full \
  --device 0 \
  --seed 1212 \
  --epochs 200 \
  --eval_every 20 \
  --eval_episodes 200 \
  --n_way 5 \
  --q_query 10 \
  --positive_k 10 \
  --ppr_alpha 0.15 \
  --ppr_iterations 40 \
  --min_hop 1 \
  --max_hop 2 \
  --train_anchors 1024 \
  --negative_count 256 \
  --output exp069.json \
  > "$LOG_FILE" 2>&1 &
echo "exp069 PID=$! LOG=$LOG_FILE"
