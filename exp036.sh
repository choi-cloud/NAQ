#!/bin/bash
set -e
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp036_${TIME}.log"

python -u exp036.py \
    --dataset Amazon_clothing \
    --device 0 \
    --seed 1212 \
    --n_way 5 \
    --k_shot 1 \
    --q_query 10 \
    --num_tasks 200 \
    --topk 20 \
    --sample_episodes 5 \
    --sample_query_limit 10 \
    --embedding_path results/Amazon_clothing/exp035/embeddings.pt \
    --json_path exp036.json \
    --output_dir results/Amazon_clothing/exp036 \
    > "${LOG_FILE}" 2>&1 &

echo "Running exp036 in background"
echo "Log: ${LOG_FILE}"
echo "PID: $!"
