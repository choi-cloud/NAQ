#!/bin/bash
set -e
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp023_${TIME}.log"

python -u exp023.py \
    --dataset Amazon_clothing \
    --n_way 5 \
    --k_shot 1 \
    --k_shot_test 1 \
    --q_query 10 \
    --lr 1e-4 \
    --latent 256 \
    --epoch 2000 \
    --temperature 1.0 \
    --query_generation NaQ \
    --type feature \
    --l2_penalty 1e-4 \
    --num_parts 8 \
    --attn_tau 0.1 \
    --json_path exp023.json \
    > "${LOG_FILE}" 2>&1 &

echo "Running exp023 in background"
echo "Log: ${LOG_FILE}"
echo "PID: $!"
