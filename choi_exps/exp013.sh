#!/bin/bash
set -e
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp013_${TIME}.log"

python -u exp013.py \
    --dataset Amazon_clothing \
    --n_way 5 \
    --k_shot 1 \
    --q_query 10 \
    --num_supports 1000 \
    --context_topk 16 \
    --ppr_steps 20 \
    --PPR_alpha 0.1 \
    --seed 1212 \
    --json_path exp013.json \
    > "${LOG_FILE}" 2>&1 &

echo "Running exp013 1-shot coverage analysis in background"
echo "Log: ${LOG_FILE}"
echo "PID: $!"
