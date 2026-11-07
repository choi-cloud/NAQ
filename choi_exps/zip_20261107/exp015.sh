#!/usr/bin/env bash
set -euo pipefail

mkdir -p out/exp015
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-2000}"
TEMPERATURE="${TEMPERATURE:-0.5}"
LAMBDA_NEIGHBOR="${LAMBDA_NEIGHBOR:-0.1}"
MAX_NEIGHBORS="${MAX_NEIGHBORS:-0}"

for DATASET in Cora_full Amazon_clothing; do
    echo "===== exp015: ${DATASET}, max_neighbors=${MAX_NEIGHBORS} ====="
    python -u exp015.py \
        --dataset "$DATASET" \
        --device "$DEVICE" \
        --n_way 5 \
        --k_shot 1 \
        --k_shot_test 5 \
        --query_generation NaQ \
        --type feature \
        --q_query 10 \
        --q_query_test 8 \
        --lr 1e-4 \
        --epoch "$EPOCHS" \
        --temperature "$TEMPERATURE" \
        --lambda_neighbor "$LAMBDA_NEIGHBOR" \
        --max_neighbors "$MAX_NEIGHBORS" \
        2>&1 | tee "out/exp015/exp015_${DATASET}.log"
done
