#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p "out/$(date +%y%m%d)"
LOG_FILE="out/$(date +%y%m%d)/exp067_$(date +%H%M%S).log"
nohup python -u exp067.py \
  --dataset Cora_full \
  --device 0 \
  --seed 1212 \
  --k 10 20 40 80 \
  --svd_dim 128 \
  --batch_size 2048 \
  --max_iter 200 \
  --n_init 3 \
  > "$LOG_FILE" 2>&1 &
echo "exp067 PID=$! LOG=$LOG_FILE"
