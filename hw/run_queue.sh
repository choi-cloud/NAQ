#!/usr/bin/env bash
# Run a list of hw.run jobs with N workers sharing a flock-protected queue.
#   [GPUS=0,1,...] [WORKERS=N] [HW_RESULTS_ROOT=results/hw] hw/run_queue.sh <runlist.txt> [WORKERS]
# GPUS: comma list of device indices (default 0). Worker i (1-based) runs with
#   --device GPUS[(i-1) mod len(GPUS)].
# WORKERS: positional arg > env WORKERS > len(GPUS). seio legacy call (2 workers on
#   device 0) = GPUS=0 WORKERS=2 (or "hw/run_queue.sh list 2").
# runlist: one "arm,dataset,seed" per line ('#' comments and blank lines ignored).
# - skips runs whose metrics.json already has status == "completed"
# - retries a failed run once (exit != 0 and status not "nan"); status "nan" is not retried
# Log: out/hw_queue/<runlist basename>.log
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 <runlist.txt> [WORKERS]" >&2
    exit 2
fi
LIST="$(realpath "$1")"
IFS=, read -r -a GPU_LIST <<< "${GPUS:-0}"
WORKERS="${2:-${WORKERS:-${#GPU_LIST[@]}}}"
RESULTS_ROOT="${HW_RESULTS_ROOT:-results/hw}"
export HW_RESULTS_ROOT="$RESULTS_ROOT"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/h2tae/.conda/envs/taskns/bin/python}"

NAME="$(basename "$LIST" .txt)"
QDIR="out/hw_queue/${NAME}"
mkdir -p "$QDIR"
LOG="out/hw_queue/${NAME}.log"
grep -vE '^\s*(#|$)' "$LIST" > "${QDIR}/jobs"
echo 0 > "${QDIR}/next"
TOTAL=$(wc -l < "${QDIR}/jobs")

log() { (flock 9; echo "$(date -u +%FT%T) $*" >> "$LOG") 9>"${QDIR}/log.lock"; }

status_of() {  # arm ds seed -> status or "missing"
    local f="${RESULTS_ROOT}/$1/$2/seed$3/metrics.json"
    [[ -f "$f" ]] || { echo missing; return; }
    "$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1])).get('status'))" "$f"
}

worker() {
    local wid="$1" idx line arm ds seed st rc attempt
    local dev="${GPU_LIST[$(( (wid - 1) % ${#GPU_LIST[@]} ))]}"
    while true; do
        idx=$( (flock 8; i=$(cat "${QDIR}/next"); echo $((i + 1)) > "${QDIR}/next"; echo "$i") 8>"${QDIR}/queue.lock")
        (( idx < TOTAL )) || break
        line=$(sed -n "$((idx + 1))p" "${QDIR}/jobs")
        IFS=, read -r arm ds seed <<< "$line"
        st=$(status_of "$arm" "$ds" "$seed")
        if [[ "$st" == "completed" ]]; then
            log "w$wid skip   $arm,$ds,$seed (completed)"
            continue
        fi
        for attempt in 1 2; do
            log "w$wid start  $arm,$ds,$seed attempt=$attempt device=$dev"
            rc=0
            PYTHONPATH=. "$PYTHON" -m hw.run --arm "$arm" --dataset "$ds" --seed "$seed" --device "$dev" \
                > "${QDIR}/${arm}_${ds}_${seed}.out" 2> "${QDIR}/${arm}_${ds}_${seed}.err" || rc=$?
            st=$(status_of "$arm" "$ds" "$seed")
            log "w$wid end    $arm,$ds,$seed rc=$rc status=$st"
            if [[ "$st" == "completed" || "$st" == "nan" ]]; then
                break
            fi
        done
    done
}

log "queue $NAME: $TOTAL jobs, $WORKERS workers, GPUS=${GPUS:-0}, results=$RESULTS_ROOT"
for w in $(seq 1 "$WORKERS"); do
    worker "$w" &
done
wait
log "queue $NAME: done"
