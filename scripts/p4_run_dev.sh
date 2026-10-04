#!/usr/bin/env bash
# P4-T6/T7 — run every Phase 3 configuration (unchanged adapters/weights/geometry) over the frozen SYNTHETIC dev
# calibration population `p4dev`. Resumable: completed pairs (OK record + output PNG) are never recomputed.
# Usage: scripts/p4_run_dev.sh [config ids...]
#
# Workers are conservative (memory pressure on the shared Docker VM, 2026-10-03): threads per worker stay at 2 —
# the only setting that affects pixels (P3-T10: outputs reproduce exactly at a fixed thread count); the worker count
# only changes wall time. Override per run with P4_WORKERS / P4_WORKERS_A2. Every change is logged below.
set -uo pipefail
POP=p4dev; ROOT=data/manifests/phase4
CFGS=("$@"); [ ${#CFGS[@]} -eq 0 ] && CFGS=(B0 B1 B2 B2c A3 A5 A1 A4 A2)
mkdir -p logs/phase4
for c in "${CFGS[@]}"; do
  W="${P4_WORKERS:-2}"; [ "$c" = "A2" ] && W="${P4_WORKERS_A2:-1}"   # VQFR ≈ 3.3 GB per process
  echo "=== $(date -u +%FT%TZ) start $c/$POP workers=$W threads=2"
  for attempt in 1 2 3 4 5; do
    "$(dirname "$0")/dock.sh" python scripts/p3_run_inference.py --config-id "$c" --population "$POP" \
        --root "$ROOT" --outputs-root experiments/phase4 --workers "$W" --threads 2 \
        >> "logs/phase4/run_${POP}_${c}.log" 2>&1
    tail -1 "logs/phase4/run_${POP}_${c}.log" | grep -q '"n_missing": 0, *"infer' && \
      ! tail -1 "logs/phase4/run_${POP}_${c}.log" | grep -q '"n_failed": [1-9]' && break
    echo "    attempt $attempt: pairs missing or failed; resuming"
  done
  echo "=== $(date -u +%FT%TZ) end $c/$POP $(tail -1 logs/phase4/run_${POP}_${c}.log)"
done
