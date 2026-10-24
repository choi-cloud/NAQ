#!/bin/bash
set -e
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp003_${TIME}.log"

python -u exp003.py \
    --dataset Amazon_clothing \
    --n_way 5 \
    --k_shot 1 \
    --k_shot_test 1 \
    --query_generation NaQ \
    --type feature \
    --lr 1e-4 \
    --epoch 2000 \
    --ppr_topk 16 \
    --ppr_steps 20 \
    --PPR_alpha 0.1 \
    --attn_lambda 0.1 \
    --attn_heads 4 \
    > "${LOG_FILE}" 2>&1 &

echo "Running exp003 in background"
echo "Log: ${LOG_FILE}"
echo "PID: $!"
