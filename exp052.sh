#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp052_${TIME}.log"

python -u exp052.py \
    --dataset Cora_full \
    --seed 1212 \
    > "${LOG_FILE}" 2>&1 &

echo "exp052 started | PID: $! | LOG: ${LOG_FILE}"
