#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp054_${TIME}.log"

python -u exp054.py \
    --dataset Cora_full \
    --n_way 5 \
    --k_shot 1 \
    --k_shot_test 5 \
    --q_query 10 \
    --q_query_test 8 \
    --lr 1e-4 \
    --latent 256 \
    --epoch 2000 \
    --eval_interval 160 \
    --seed 1212 \
    > "${LOG_FILE}" 2>&1 &

echo "exp054 started | PID: $! | LOG: ${LOG_FILE}"
