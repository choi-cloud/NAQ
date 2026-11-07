#!/bin/bash
set -e
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp018_${TIME}.log"

python -u exp018.py \
    --dataset Amazon_clothing \
    --n_way 5 \
    --k_shot 1 \
    --q_query 10 \
    --pool_size 1000 \
    --num_episodes 1000 \
    --seed 1212 \
    --json_path exp018.json \
    > "${LOG_FILE}" 2>&1 &

echo "Running exp018 in background"
echo "Log: ${LOG_FILE}"
echo "PID: $!"
