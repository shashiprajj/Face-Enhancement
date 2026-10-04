#!/usr/bin/env bash
# P4-INFRA-RESUME monitor: every INTERVAL s append progress + resources to logs/phase4/resume_monitor.log.
cd "$(dirname "$0")/.."
INTERVAL=${INTERVAL:-600}
while true; do
  {
    echo "--- $(date -u +%FT%TZ)"
    python - <<'PY'
import json, os, time
exp = 896
for c in ["A5", "A1", "A4", "A2"]:
    f = f"data/manifests/phase4/runs/p4dev/{c}.partial.jsonl"
    if not os.path.exists(f):
        print(f"{c}: 0/{exp} (not started)"); continue
    rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    ok = len({r["pair_id"] for r in rows if r["status"] == "OK"}); fail = sum(r["status"] != "OK" for r in rows)
    m = os.path.getmtime(f); age = time.time() - m
    st = f"data/manifests/phase4/runs/p4dev/{c}.resume_start"
    rate = eta = ""
    if os.path.exists(st):
        t0, n0 = map(float, open(st).read().split())
        el = time.time() - t0
        if ok > n0 and el > 0:
            r = (ok - n0) / el * 3600; rate = f" {r:.0f}/h"; eta = f" ETA {(exp - ok) / r:.1f} h"
    print(f"{c}: {ok}/{exp} ({100*ok/exp:.1f}%) failed={fail}{rate}{eta} last_write={age:.0f}s ago")
PY
    docker stats --no-stream --format '{{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}}' | grep fes- || true
    docker run --rm fes-dev:cpu-p2 sh -c "grep MemAvailable /proc/meminfo; cat /proc/loadavg" 2>/dev/null
  } >> logs/phase4/resume_monitor.log 2>&1
  sleep "$INTERVAL"
done
