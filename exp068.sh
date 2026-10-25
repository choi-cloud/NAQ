#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p "out/$(date +%y%m%d)"
LOG_FILE="out/$(date +%y%m%d)/exp068_$(date +%H%M%S).log"
nohup python -u exp068.py \
  --dataset Cora_full \
  --device 0 \
  --seed 1212 \
  --episodes 200 \
  --n_way 5 \
  --q_query 10 \
  --node_k 10 \
  --relation_k 10 \
  --semantic_k 10 \
  --min_hop 1 \
  --max_hop 2 \
  --ppr_alpha 0.15 \
  --ppr_iterations 60 \
  --output exp068.json \
  > "$LOG_FILE" 2>&1 &
echo "exp068 PID=$! LOG=$LOG_FILE"
