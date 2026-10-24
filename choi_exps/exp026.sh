#!/bin/bash
set -e
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp026_${TIME}.log"

python -u exp026.py \
    --dataset Amazon_clothing \
    --n_way 5 \
    --k_shot_test 1 \
    --q_query_test 8 \
    --num_valid_tasks 50 \
    --num_downstream_tasks 500 \
    --seed 1212 \
    --json_path exp026.json \
    > "${LOG_FILE}" 2>&1 &

echo "Running exp026 fixed episode export in background"
echo "Log: ${LOG_FILE}"
echo "PID: $!"
