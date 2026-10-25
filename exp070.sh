#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p "out/$(date +%y%m%d)"
LOG_FILE="out/$(date +%y%m%d)/exp070_$(date +%H%M%S).log"
nohup python -u exp070.py \
  --dataset Cora_full \
  --device 0 \
  --seed 1212 \
  --episodes 200 \
  --n_way 5 \
  --query_k 10 \
  --percentages 1 5 10 20 50 \
  --checkpoint exp069_sum.pt \
  --output exp070.json \
  > "$LOG_FILE" 2>&1 &
echo "exp070 PID=$! LOG=$LOG_FILE"
