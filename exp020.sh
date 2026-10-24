#!/bin/bash
set -e
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp020_${TIME}.log"

python -u exp020.py \
    --dataset Amazon_clothing \
    --ckpt results/Amazon_clothing/exp019/best.pt \
    --num_pseudo_episodes 200 \
    --num_test_tasks 500 \
    --json_path exp020.json \
    > "${LOG_FILE}" 2>&1 &

echo "Running exp020 in background"
echo "Log: ${LOG_FILE}"
echo "PID: $!"
