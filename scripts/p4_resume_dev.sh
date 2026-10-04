#!/usr/bin/env bash
# P4-INFRA-RESUME — resume p4dev inference from the existing checkpoints with measured, safe worker counts.
# Same experiment: p3_run_inference.py, same population/adapters/weights/geometry, threads per worker = 2 (as every
# completed record). Completed pairs (OK record + output PNG) are skipped by the runner. Only the worker count changes.
# A2 watchdog: if the Docker VM's MemAvailable drops below MIN_AVAIL_KB, the A2 container is stopped gracefully
# (flushed records are kept) and A2 resumes with 1 worker.
set -uo pipefail
cd "$(dirname "$0")/.."
POP=p4dev; ROOT=data/manifests/phase4; LOG=logs/phase4
MIN_AVAIL_KB=${MIN_AVAIL_KB:-1500000}
declare -A W=( [A5]=5 [A1]=4 [A4]=4 [A2]=${A2_WORKERS:-2} )
ORDER=(${@:-A5 A1 A4 A2})
mkdir -p "$LOG"

run_cfg() {  # $1 config, $2 workers; returns 0 when every pair has an OK record
  local c=$1 w=$2 name="fes-p4dev-$1"
  echo "=== $(date -u +%FT%TZ) start $c workers=$w threads=2"
  local part="$ROOT/runs/$POP/$c.partial.jsonl" n=0
  [ -f "$part" ] && n=$(grep -c '"OK"' "$part")
  echo "$(date +%s) $n" > "$ROOT/runs/$POP/$c.resume_start"
  MSYS_NO_PATHCONV=1 docker run --rm -i --name "$name" -e FES_GIT_HEAD="$(git rev-parse HEAD)-dirty" \
    -v "$(pwd -W 2>/dev/null || pwd):/workspace" -w /workspace fes-dev:cpu-p2 \
    python scripts/p3_run_inference.py --config-id "$c" --population "$POP" --root "$ROOT" \
      --outputs-root experiments/phase4 --workers "$w" --threads 2 >> "$LOG/run_${POP}_${c}.log" 2>&1 &
  local pid=$!
  if [ "$c" = "A2" ]; then
    while kill -0 $pid 2>/dev/null; do
      avail=$(docker exec "$name" sh -c "grep MemAvailable /proc/meminfo | awk '{print \$2}'" 2>/dev/null || echo 99999999)
      if [ "${avail:-99999999}" -lt "$MIN_AVAIL_KB" ] && [ "$w" -gt 1 ]; then
        echo "    $(date -u +%FT%TZ) A2 memory pressure (MemAvailable ${avail} kB) → stopping, resume with 1 worker"
        docker stop -t 120 "$name" >/dev/null 2>&1; wait $pid; return 2
      fi
      sleep 30
    done
  fi
  wait $pid
  tail -1 "$LOG/run_${POP}_${c}.log" | grep -q '"n_missing": 0, *"infer' && \
    ! tail -1 "$LOG/run_${POP}_${c}.log" | grep -q '"n_failed": [1-9]'
}

for c in "${ORDER[@]}"; do
  w=${W[$c]}
  for attempt in 1 2 3 4 5 6; do
    run_cfg "$c" "$w"; rc=$?
    [ $rc -eq 0 ] && break
    [ $rc -eq 2 ] && w=1
    echo "    attempt $attempt rc=$rc: pairs missing/failed or memory stop; resuming from checkpoint (workers=$w)"
  done
  echo "=== $(date -u +%FT%TZ) end $c $(tail -1 $LOG/run_${POP}_${c}.log | cut -c1-120)"
done
