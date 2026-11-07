#!/bin/bash
set -e
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp019_${TIME}.log"

python -u exp019.py \
    --dataset Amazon_clothing \
    --n_way 5 \
    --k_shot 1 \
    --k_shot_test 1 \
    --q_query 10 \
    --pool_size 1000 \
    --lr 1e-4 \
    --epoch 3000 \
    --temperature 0.5 \
    --ppr_topk 16 \
    --PPR_alpha 0.1 \
    --ppr_steps 20 \
    --attn_lambda 0.1 \
    --json_path exp019.json \
    > "${LOG_FILE}" 2>&1 &

echo "Running exp019 in background"
echo "Log: ${LOG_FILE}"
echo "PID: $!"
