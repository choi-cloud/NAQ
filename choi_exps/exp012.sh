#!/bin/bash
set -e
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp012_${TIME}.log"

python -u exp012.py \
    --dataset Amazon_clothing \
    --n_way 5 \
    --k_shot 1 \
    --k_shot_test 1 \
    --query_generation NaQ \
    --type feature \
    --lr 1e-4 \
    --epoch 2000 \
    --attn_lambda 0.1 \
    --cl_margin 2.0 \
    --intra_margin 0.5 \
    --intra_weight 0.1 \
    --json_path exp012.json \
    > "${LOG_FILE}" 2>&1 &

echo "Running exp012 in background"
echo "Log: ${LOG_FILE}"
echo "PID: $!"
