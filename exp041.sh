#!/usr/bin/env bash
set -euo pipefail
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
LOG_DIR="out/${DATE}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/exp041_${TIME}.log"
python -u exp041.py --dataset Amazon_clothing --device 0 --seed 1212 --epochs 200 --cl_batch_size 1024 --num_tasks 200 --pool_sizes 200,1000 > "$LOG" 2>&1 &
echo "exp041 PID=$! log=$LOG"
