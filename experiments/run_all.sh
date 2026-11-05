#!/usr/bin/env bash
# Runs every experiment in sequence; each script skips cached points in results/ and stops
# starting new points once BUDGET seconds have elapsed (so the queue can be resumed).
set -u
cd "$(dirname "$0")"
export DEADLINE_TS=$(( $(date +%s) + ${BUDGET:-100000} ))
LOG=../results/logs; mkdir -p "$LOG"
run() { echo "=== $(date +%T) $*" >> "$LOG/run_all.log"; python "$@" 2>&1 | tee -a "$LOG/$(basename "$1" .py).log"; }
run e0_haar_check.py --n-list 2 3 4 5 6 7 8 --samples 400
run e1_variance_vs_n.py --n-list 2 4 6 8 10 12 14 --L-list 2 5 10 20 50 --samples 300
run e1_variance_vs_n.py --n-list 16 --L-list 2 5 10 --samples 200
run e1_variance_vs_n.py --n-list 16 --L-list 20 50 --samples 100
run e2_local_vs_global.py --n-list 4 6 8 10 12 14 --L-list 1 2 4 8 16 --samples 200
run e3_initialisation.py --n-list 4 6 8 10 12 14 --samples 200
run e6_qaoa.py --part var --n-list 4 6 8 10 12 14 16 --p-list 1 2 4 8 --samples 200
run e6_qaoa.py --part ratio
run e7_dla.py
run e5_training.py --n-list 4 6 8 10 12 14 --seeds 16 --steps 250
run e5_training.py --n-list 4 6 8 10 12 14 --seeds 12 --steps 250 --shots 10000
run e1_variance_vs_n.py --exp e4 --n-list 12 --L-list 1 2 3 4 6 8 12 16 24 32 50 75 100 --samples 300
[ "$(date +%s)" -lt "$DEADLINE_TS" ] && echo "=== $(date +%T) ALL DONE" | tee -a "$LOG/run_all.log"
