#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
DATE=$(date +"%y%m%d")
mkdir -p "out/${DATE}"
LOG="out/${DATE}/exp050_$(date +"%H%M%S").log"

python -u exp050.py --dataset Cora_full --n_way 5 --k_shot 1 --k_shot_test 5 --q_query 10 --q_query_test 8 --epoch 2000 --seed 1212 > "${LOG}" 2>&1 &
