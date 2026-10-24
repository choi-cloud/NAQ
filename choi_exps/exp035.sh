#!/bin/bash
set -e
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp035_${TIME}.log"

python -u exp035.py \
    --dataset Amazon_clothing \
    --device 0 \
    --seed 1212 \
    --latent 256 \
    --hidden 512 \
    --epoch 2000 \
    --lr 1e-3 \
    --batch_edges 8192 \
    --valid_ratio 0.1 \
    --eval_interval 100 \
    --patience 10 \
    --output_dir results/Amazon_clothing/exp035 \
    --json_path exp035.json \
    > "${LOG_FILE}" 2>&1 &

echo "Running exp035 in background"
echo "Log: ${LOG_FILE}"
echo "PID: $!"
