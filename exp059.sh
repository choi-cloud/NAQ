#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p "out/$(date +%y%m%d)"
LOG_FILE="out/$(date +%y%m%d)/exp059_$(date +%H%M%S).log"
nohup python -u exp059.py \
  --dataset Cora_full \
  --seed 1212 \
  --k 70 > "$LOG_FILE" 2>&1 &
echo "exp059 PID=$! LOG=$LOG_FILE"
