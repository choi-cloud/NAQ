#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out/$(date +%y%m%d)
log="out/$(date +%y%m%d)/exp071_$(date +%H%M%S).log"
nohup python -u exp071.py --device 0 --epoch 2000 --eval_interval 160 --resume "$@" > "$log" 2>&1 &
echo "exp071 PID=$! log=$log"
echo "Follow progress: tail -f $log"
