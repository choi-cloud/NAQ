#!/bin/bash
set -e
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp008_${TIME}.log"

python -u exp008.py \
    --dataset Amazon_clothing \
    --n_way 5 \
    --k_shot 1 \
    --k_shot_test 1 \
    --query_generation NaQ \
    --type feature \
    --lr 1e-4 \
    --epoch 2000 \
    --cl_margin 2.0 \
    --cl_pos_weight 1.0 \
    --cl_neg_weight 1.0 \
    --json_path exp008.json \
    > "${LOG_FILE}" 2>&1 &

echo "Running exp008 in background"
echo "Log: ${LOG_FILE}"
echo "PID: $!"
