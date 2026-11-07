#!/usr/bin/env bash
set -euo pipefail

mkdir -p out/exp012
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-2000}"
TEMPERATURE="${TEMPERATURE:-0.5}"
ATTENTION_TEMPERATURE="${ATTENTION_TEMPERATURE:-1.0}"
MAX_NEIGHBORS="${MAX_NEIGHBORS:-10}"

for DATASET in Cora_full Amazon_clothing; do
    echo "===== exp012: ${DATASET}, max_neighbors=${MAX_NEIGHBORS} ====="
    python -u exp012.py \
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
        --attention_temperature "$ATTENTION_TEMPERATURE" \
        --max_neighbors "$MAX_NEIGHBORS" \
        2>&1 | tee "out/exp012/exp012_${DATASET}.log"
done
