#!/bin/bash
set -euo pipefail
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp040_${TIME}.log"
python -u exp040.py \
    --dataset Amazon_clothing \
    --device 0 \
    --seed 1212 \
    --steps 1500 \
    --batch_edges 4096 \
    --num_tasks 200 \
    --pool_sizes 200,1000,all \
    --remove_fractions 0.25,0.5,0.75,0.9 \
    --json_path exp040.json \
    > "${LOG_FILE}" 2>&1 &
echo "exp040 running PID=$! log=${LOG_FILE}"
