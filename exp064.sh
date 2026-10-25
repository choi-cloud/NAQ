#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp064_${TIME}.log"

python -u exp064.py \
    --dataset Cora_full \
    --n_way 5 \
    --k_shot 1 \
    --shots 1 5 \
    --q_query 10 \
    --q_query_test 8 \
    --epoch 2000 \
    --eval_interval 160 \
    --pretrain_checkpoint exp063.pt \
    --candidate_cache 15 \
    --pair_batch 8192 \
    --seed 1212 \
    > "${LOG_FILE}" 2>&1 &

echo "exp064 started | PID: $! | LOG: ${LOG_FILE}"
