#!/usr/bin/env bash
set -euo pipefail
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
LOG_DIR="out/${DATE}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/exp047_${TIME}.log"
python -u exp047.py --dataset Amazon_clothing --device 0 --seed 1212 --num_tasks 200 --query_count 10 --taus=-0.2,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 > "$LOG" 2>&1 &
echo "exp047 PID=$! log=$LOG"
