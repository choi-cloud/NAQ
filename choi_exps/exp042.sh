#!/usr/bin/env bash
set -euo pipefail
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
LOG_DIR="out/${DATE}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/exp042_${TIME}.log"
python -u exp042.py --dataset Amazon_clothing --device 0 --seed 1212 --num_tasks 200 --n_way 5 --k 10 > "$LOG" 2>&1 &
echo "exp042 PID=$! log=$LOG"
