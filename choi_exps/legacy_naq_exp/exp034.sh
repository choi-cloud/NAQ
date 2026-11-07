#!/bin/bash
set -e
DATE=$(date +"%y%m%d")
TIME=$(date +"%H%M%S")
OUT_DIR="out/${DATE}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/exp034_${TIME}.log"

python -u exp034.py \
    --dataset Amazon_clothing \
    --device 0 \
    --n_way 5 \
    --k_shot 1 \
    --k_shot_test 1 \
    --q_query 10 \
    --q_query_test 8 \
    --lr 1e-4 \
    --latent 256 \
    --epoch 2000 \
    --query_generation NaQ \
    --type feature \
    --l2_penalty 1e-4 \
    --num_valid_tasks 50 \
    --num_downstream_tasks 500 \
    --eval_interval 160 \
    --attn_tau 0.1 \
    --attn_dim 256 \
    --correction_lambda 0.1 \
    --ppr_topk 20 \
    --ppr_alpha 0.15 \
    --ppr_iter 60 \
    --ppr_tol 1e-7 \
    --feature_prior_lambda 1.0 \
    --seed 1212 \
    --json_path exp034.json \
    > "${LOG_FILE}" 2>&1 &

echo "Running exp034 in background"
echo "Log: ${LOG_FILE}"
echo "PID: $!"
