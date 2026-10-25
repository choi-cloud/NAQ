#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out/$(date +%y%m%d)
log="out/$(date +%y%m%d)/exp072_$(date +%H%M%S).log"
nohup python -u exp072.py --device 0 --epoch 2000 --eval_interval 160 --neg_lambda 0.1 --neg_margin 1.0 "$@" > "$log" 2>&1 &
echo "exp072 PID=$! log=$log"
echo "Follow progress: tail -f $log"
