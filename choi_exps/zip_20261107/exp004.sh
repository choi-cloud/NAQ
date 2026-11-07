#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="out/exp004"
mkdir -p "$OUT_DIR"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-400}"
EDGE_DROP="${EDGE_DROP:-0.2}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-0.001}"
EVAL_EVERY="${EVAL_EVERY:-50}"
MOMENTUM="${MOMENTUM:-0.99}"

for DATASET in Cora_full Amazon_clothing; do
    echo "===== exp004: BGRL pretraining ${DATASET} ====="
    python -u exp004.py \
        --dataset "$DATASET" \
        --device "$DEVICE" \
        --epoch "$EPOCHS" \
        --latent 256 \
        --lr "$LR" \
        --edge_drop "$EDGE_DROP" \
        --batch_size "$BATCH_SIZE" \
        --momentum "$MOMENTUM" \
        --eval_every "$EVAL_EVERY" \
        2>&1 | tee "$OUT_DIR/exp004_${DATASET}.log"
done
