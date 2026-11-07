#!/bin/bash
set -e
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp032_${TIME}.log"

python -u exp032.py \
    --dataset Amazon_clothing \
    --device 0 \
    --n_way 5 \
    --k_shot 1 \
    --q_query 10 \
    --num_tasks 200 \
    --context ppr \
    --topk 1 3 5 \
    --seed 1212 \
    --json_path exp032.json \
    > "${LOG_FILE}" 2>&1 &

echo "Running exp032 in background"
echo "Log: ${LOG_FILE}"
echo "PID: $!"
