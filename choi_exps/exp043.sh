#!/usr/bin/env bash
set -euo pipefail
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
mkdir -p "out/${DATE}"
LOG="out/${DATE}/exp043_${TIME}.log"
nohup python -u exp043.py --dataset Amazon_clothing --device 0 --seed 1212 --num_tasks 200 --pool_sizes 200,1000 --query_count 10 > "$LOG" 2>&1 &
echo "exp043 PID=$! log=$LOG"
