#!/usr/bin/env bash
set -euo pipefail
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT="out/${DATE}"
mkdir -p "$OUT"
LOG="$OUT/exp049_${TIME}.log"
nohup python -u run_exp049.py --device "${DEVICE:-0}" --seed "${SEED:-1212}" --taus "${TAUS:-0.1,0.2,0.3,0.4,0.5,0.6}" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/exp049_${TIME}.pid"
echo "exp049 PID=$PID log=$LOG"
echo "Summary CSV: results/gfsl_benchmark_exp049/summary.csv"
