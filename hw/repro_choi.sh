#!/usr/bin/env bash
# Run a frozen Choi script exactly as its .sh specifies (reproduction only).
#   hw/repro_choi.sh <series> <exp> <dataset> [EPOCHS]
# The `python -u <exp>.py ...` argument block and the VAR="${VAR:-default}" lines
# are read from choi_exps/<series>/<exp>.sh; only DATASET (and EPOCHS if given)
# are substituted. Output JSON is copied to results/hw/repro/<series>_<exp>/.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "usage: $0 <series> <exp> <dataset> [EPOCHS]" >&2
    exit 2
fi
SERIES="$1"; EXP_NAME="$2"; DATASET="$3"; EPOCHS_OVERRIDE="${4:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SH="choi_exps/${SERIES}/${EXP_NAME}.sh"
PY="choi_exps/${SERIES}/${EXP_NAME}.py"
[[ -f "$SH" && -f "$PY" ]] || { echo "missing $SH or $PY" >&2; exit 2; }

# 1) Defaults exactly as written in the .sh (lines like FOO="${FOO:-value}").
unset DEVICE EPOCHS TEMPERATURE LAMBDA_NEIGHBOR MAX_NEIGHBORS ATTENTION_TEMPERATURE BGRL_DIR
eval "$(grep -E '^[A-Z_]+="\$\{[A-Z_]+:-[^}]*\}"$' "$SH")"
if [[ -n "$EPOCHS_OVERRIDE" ]]; then EPOCHS="$EPOCHS_OVERRIDE"; fi

# 2) Argument block after `python -u <exp>.py`, up to the redirection line.
ARGS_TEXT="$(awk -v py="${EXP_NAME}.py" '
    $0 ~ "python -u " py { on=1; next }
    on && /2>&1/ { exit }
    on { sub(/\\[[:space:]]*$/, ""); print }' "$SH" | tr '\n' ' ')"
eval "set -- $ARGS_TEXT"

mkdir -p "out/${EXP_NAME}"
SUFFIX=""
[[ -n "$EPOCHS_OVERRIDE" ]] && SUFFIX="_smoke"
LOG="out/${EXP_NAME}/${EXP_NAME}_${DATASET}${SUFFIX}.log"
echo "===== repro ${SERIES}/${EXP_NAME}: ${DATASET} EPOCHS=${EPOCHS} =====" | tee "$LOG"
echo "args: $*" | tee -a "$LOG"

PYTHONPATH=. conda run --no-capture-output -n taskns python -u "$PY" "$@" 2>&1 | tee -a "$LOG"

SRC="out/${EXP_NAME}/${EXP_NAME}_${DATASET}.json"
DST_DIR="results/hw/repro/${SERIES}_${EXP_NAME}"
mkdir -p "$DST_DIR"
cp "$SRC" "${DST_DIR}/${DATASET}${SUFFIX}.json"
echo "copied $SRC -> ${DST_DIR}/${DATASET}${SUFFIX}.json"
