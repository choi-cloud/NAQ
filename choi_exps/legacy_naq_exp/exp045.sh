#!/usr/bin/env bash
set -euo pipefail
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
LOG_DIR="out/${DATE}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/exp045_${TIME}.log"
python -u exp045.py --dataset Amazon_clothing --device 0 --seed 1212 --num_tasks 200 --n_way 5 --query_count 10 --batch_size 64 --remove_fractions 0,0.5,0.9,0.95,0.99,0.995,0.999 > "$LOG" 2>&1 &
echo "exp045 PID=$! log=$LOG"
