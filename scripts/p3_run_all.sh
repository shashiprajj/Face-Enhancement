#!/usr/bin/env bash
# Run configurations sequentially over one frozen population (P3-T3/P3-T4). Each is resumable.
# Usage: scripts/p3_run_all.sh <trial|main> [config ids...]   (default: all configurations in protocol order)
set -uo pipefail
POP="$1"; shift
CFGS=("$@"); [ ${#CFGS[@]} -eq 0 ] && CFGS=(B0 B1 B2 B2c A3 A5 A1 A2 A4)
mkdir -p logs/phase3
for c in "${CFGS[@]}"; do
  echo "=== $(date -u +%FT%TZ) start $c/$POP"
  for attempt in 1 2 3 4 5; do   # resume until every pair has a record (a broken worker pool exits early)
    "$(dirname "$0")/dock.sh" python scripts/p3_run_inference.py --config-id "$c" --population "$POP" \
        >> "logs/phase3/run_${POP}_${c}.log" 2>&1
    tail -1 "logs/phase3/run_${POP}_${c}.log" | grep -q '"n_missing": 0, *"infer' && \
      ! tail -1 "logs/phase3/run_${POP}_${c}.log" | grep -q '"n_failed": [1-9]' && break
    echo "    attempt $attempt: pairs missing or failed; resuming"
  done
  echo "=== $(date -u +%FT%TZ) end $c/$POP $(tail -1 logs/phase3/run_${POP}_${c}.log)"
done
