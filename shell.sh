#!/bin/bash

DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S%3N")

OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"

LOG_FILE="${OUT_DIR}/${TIME}.log"

python -u unsup_protonet.py \
    --dataset Amazon_clothing \
    --n_way 5 \
    --k_shot_test 1 \
    --q_query 10 \
    --query_generation NaQ \
    --type feature \
    --lr 1e-4 \
    --epoch 2000 > "${LOG_FILE}" 2>&1 &

echo "Running NAQ-Feat in background"
echo "Log: ${LOG_FILE}"
echo "PID: $!"
