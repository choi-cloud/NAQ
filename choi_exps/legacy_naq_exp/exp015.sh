#!/bin/bash
set -e
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp015_${TIME}.log"

python -u exp015.py \
    --dataset Amazon_clothing \
    --n_way 5 \
    --k_shot 1 \
    --k_shot_test 1 \
    --query_generation NaQ \
    --type feature \
    --lr 1e-4 \
    --epoch 2000 \
    --temperature 0.5 \
    --json_path exp015.json \
    > "${LOG_FILE}" 2>&1 &

echo "Running exp015 in background"
echo "Log: ${LOG_FILE}"
echo "PID: $!"
