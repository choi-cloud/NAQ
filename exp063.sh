#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p "out/$(date +%y%m%d)"
LOG_FILE="out/$(date +%y%m%d)/exp063_$(date +%H%M%S).log"
nohup python -u exp063.py \
  --dataset Cora_full \
  --device 0 \
  --seed 1212 \
  --k 70 \
  --svd_dim 128 \
  --epochs 200 \
  --pairs_per_epoch 12000 \
  --margin_pos_weak 0.5 \
  --margin_weak_strong 0.5 \
  --weak_edge_fraction 0.5 \
  --eval_anchors 200 \
  --eval_every 20 \
  > "$LOG_FILE" 2>&1 &
echo "exp063 PID=$! LOG=$LOG_FILE"
