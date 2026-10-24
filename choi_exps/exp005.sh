#!/bin/bash
set -e
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp005_${TIME}.log"

python -u exp005.py \
    --dataset Amazon_clothing \
    --ppr_topk 16 \
    --ppr_steps 20 \
    --PPR_alpha 0.1 \
    --num_supports 1000 \
    --seed 1212 \
    --json_path exp005.json \
    > "${LOG_FILE}" 2>&1 &

echo "Running exp005 analysis in background"
echo "Log: ${LOG_FILE}"
echo "PID: $!"
