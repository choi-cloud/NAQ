#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp053_${TIME}.log"

python -u exp053.py \
    --dataset Cora_full \
    --seed 1212 \
    --n_way 5 \
    --num_episodes 2000 \
    --diagnostic_interval 160 \
    > "${LOG_FILE}" 2>&1 &

echo "exp053 started | PID: $! | LOG: ${LOG_FILE}"
