#!/usr/bin/env bash
# Run a list of hw.run jobs with N workers sharing a flock-protected queue.
#   hw/run_queue.sh <runlist.txt> [WORKERS=2]
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
WORKERS="${2:-2}"
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
    local f="results/hw/$1/$2/seed$3/metrics.json"
    [[ -f "$f" ]] || { echo missing; return; }
    "$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1])).get('status'))" "$f"
}

worker() {
    local wid="$1" idx line arm ds seed st rc attempt
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
            log "w$wid start  $arm,$ds,$seed attempt=$attempt"
            rc=0
            PYTHONPATH=. "$PYTHON" -m hw.run --arm "$arm" --dataset "$ds" --seed "$seed" \
                > "${QDIR}/${arm}_${ds}_${seed}.out" 2> "${QDIR}/${arm}_${ds}_${seed}.err" || rc=$?
            st=$(status_of "$arm" "$ds" "$seed")
            log "w$wid end    $arm,$ds,$seed rc=$rc status=$st"
            if [[ "$st" == "completed" || "$st" == "nan" ]]; then
                break
            fi
        done
    done
}

log "queue $NAME: $TOTAL jobs, $WORKERS workers"
for w in $(seq 1 "$WORKERS"); do
    worker "$w" &
done
wait
log "queue $NAME: done"
