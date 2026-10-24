#!/bin/bash
set -e
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp024_${TIME}.log"

python -u exp024.py \
    --dataset Amazon_clothing \
    --checkpoint checkpoints/Amazon_clothing/unsup_NaQ/261009_112926_548403/best.pt \
    --n_way 5 \
    --k_shot_test 1 \
    --q_query_test 8 \
    --latent 256 \
    --num_valid_tasks 50 \
    --num_downstream_tasks 500 \
    --seed 1212 \
    --json_path exp024.json \
    > "${LOG_FILE}" 2>&1 &

echo "Running exp024 in background"
echo "Log: ${LOG_FILE}"
echo "PID: $!"
