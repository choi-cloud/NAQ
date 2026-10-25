#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp058_${TIME}.log"

python -u exp058.py \
    --dataset Cora_full \
    --num_episodes 200 \
    --full_anchors 50 \
    --pool_sizes 20 50 100 \
    --seed 1212 \
    > "${LOG_FILE}" 2>&1 &

echo "exp058 started | PID: $! | LOG: ${LOG_FILE}"
