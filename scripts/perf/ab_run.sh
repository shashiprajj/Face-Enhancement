#!/usr/bin/env bash
# PERF Step 1 / Step 2: one paired A/B run. Two services from the SAME image (fes-dev:cpu-p5, PyTorch) with one or two
# settings changed are started side by side; scripts/perf/service_ab.py sends every image to one, then the other.
#   scripts/perf/ab_run.sh <pair> <labelA> "<env A>" <labelB> "<env B>" [sets]
#   e.g. scripts/perf/ab_run.sh step1 baseline "" parallel "FES_BEST_PARALLEL=1"
# Per service: memory after load and at the end (per process, with peaks), CPU seconds used, OOM / restart state.
set -euo pipefail
PAIR="$1"; LA="$2"; EA="$3"; LB="$4"; EB="$5"; SETS="${6:-real200,archive100,known73}"
IMAGE=fes-dev:cpu-p5; NET=fes-bench
ROOT="$(cd "$(dirname "$0")/../.." && pwd -W 2>/dev/null || pwd)"
OUT="experiments/perf/outputs/${PAIR}"
export MSYS_NO_PATHCONV=1
docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null
mkdir -p "${ROOT}/${OUT}"
docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}' > "${ROOT}/${OUT}/host_before.txt" || true

start() {  # label, env string
  local name="fes-ab-$1" envs=()
  for kv in $2; do envs+=(-e "$kv"); done
  docker rm -f "$name" >/dev/null 2>&1 || true
  mkdir -p "${ROOT}/${OUT}/$1"
  docker run -d --name "$name" --network "$NET" -e PYTHONPATH=src -e FES_EXPERIMENTAL_PROFILES=r6,r6_ft "${envs[@]}" \
    -v "${ROOT}:/workspace" -v "${ROOT}/models:/workspace/models:ro" -w /workspace \
    "$IMAGE" uvicorn fes_service.app:app --host 0.0.0.0 --port 8080 >/dev/null
}
wait_ready() {
  local name="fes-ab-$1"
  for _ in $(seq 1 600); do
    CODE=$(docker exec "$name" python -c "import urllib.request as u
try: print(u.urlopen('http://127.0.0.1:8080/ready',timeout=5).status)
except Exception as e: print(getattr(e,'code',0))" 2>/dev/null || echo 0)
    [ "$CODE" = "200" ] && return 0
    [ "$(docker inspect -f '{{.State.Running}}' "$name")" = "true" ] || break
    sleep 1
  done
  echo "$name did not become ready"; docker logs "$name" 2>&1 | tail -15; return 1
}
snap() { docker exec "fes-ab-$1" python scripts/r6_onnx/proc_mem.py > "${ROOT}/${OUT}/$1/$2.json"; }

start "$LA" "$EA"; wait_ready "$LA"
start "$LB" "$EB"; wait_ready "$LB"
snap "$LA" mem_loaded; snap "$LB" mem_loaded

docker run --rm --network "$NET" -e PYTHONPATH=src -v "${ROOT}:/workspace" -w /workspace "$IMAGE" \
  python scripts/perf/service_ab.py run --pair "$PAIR" --a "${LA}=http://fes-ab-${LA}:8080" --b "${LB}=http://fes-ab-${LB}:8080" --sets "$SETS"

for L in "$LA" "$LB"; do
  snap "$L" mem_end
  docker inspect -f '{"oom_killed": {{.State.OOMKilled}}, "restart_count": {{.RestartCount}}, "running": {{.State.Running}}}' "fes-ab-$L" > "${ROOT}/${OUT}/$L/container_end.json"
done
docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}' > "${ROOT}/${OUT}/host_after.txt" || true
docker rm -f "fes-ab-$LA" "fes-ab-$LB" >/dev/null
echo "pair ${PAIR} complete"
