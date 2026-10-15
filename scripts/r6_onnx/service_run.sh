#!/usr/bin/env bash
# R6-ONNX-INTEGRATION: one full-service validation run for ONE backend (sequential runs; one service at a time).
#   scripts/r6_onnx/service_run.sh torch|onnx [sets]
# Starts the service from fes-dev:cpu-p5-onnx (both backends use the SAME image; only FES_NAFNET_BACKEND differs),
# measures cold start (container start → /ready), memory after load and at the end (per process, with peaks), runs
# scripts/r6_onnx/service_eval.py against it from a second container on a private Docker network, and stops it.
set -euo pipefail
BACKEND="$1"; SETS="${2:-real200,archive100}"
IMAGE=fes-dev:cpu-p5-onnx; NET=fes-bench; SVC=fes-svc-bench
ROOT="$(cd "$(dirname "$0")/../.." && pwd -W 2>/dev/null || pwd)"
OUT="experiments/r6_onnx/outputs/service/${BACKEND}"
export MSYS_NO_PATHCONV=1
mkdir -p "${ROOT}/${OUT}"
docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null
docker rm -f "$SVC" >/dev/null 2>&1 || true
docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}' > "${ROOT}/${OUT}/host_before.txt" || true

T0=$(date +%s.%N)
docker run -d --name "$SVC" --network "$NET" -e PYTHONPATH=src -e FES_EXPERIMENTAL_PROFILES=r6,r6_ft \
  -e FES_NAFNET_BACKEND="$BACKEND" -v "${ROOT}:/workspace" -v "${ROOT}/models:/workspace/models:ro" -w /workspace \
  "$IMAGE" uvicorn fes_service.app:app --host 0.0.0.0 --port 8080 >/dev/null
READY=""
for _ in $(seq 1 600); do
  CODE=$(docker exec "$SVC" python -c "import urllib.request as u,sys
try: print(u.urlopen('http://127.0.0.1:8080/ready',timeout=5).status)
except Exception as e: print(getattr(e,'code',0))" 2>/dev/null || echo 0)
  if [ "$CODE" = "200" ]; then READY=1; break; fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$SVC")" != "true" ]; then break; fi
  sleep 1
done
T1=$(date +%s.%N)
if [ -z "$READY" ]; then echo "service did not become ready"; docker logs "$SVC" 2>&1 | tail -20; docker rm -f "$SVC" >/dev/null; exit 1; fi
python -c "print(round($T1-$T0,1))" > "${ROOT}/${OUT}/cold_start_s.txt"
docker exec "$SVC" python scripts/r6_onnx/proc_mem.py > "${ROOT}/${OUT}/mem_loaded.json"

docker run --rm --network "$NET" -e PYTHONPATH=src -v "${ROOT}:/workspace" -w /workspace "$IMAGE" \
  python scripts/r6_onnx/service_eval.py run --backend "$BACKEND" --url "http://${SVC}:8080" --sets "$SETS"

docker exec "$SVC" python scripts/r6_onnx/proc_mem.py > "${ROOT}/${OUT}/mem_end.json"
docker inspect -f '{"oom_killed": {{.State.OOMKilled}}, "restart_count": {{.RestartCount}}, "running": {{.State.Running}}}' "$SVC" > "${ROOT}/${OUT}/container_end.json"
docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}' > "${ROOT}/${OUT}/host_after.txt" || true
docker rm -f "$SVC" >/dev/null
echo "run ${BACKEND} complete: cold start $(cat "${ROOT}/${OUT}/cold_start_s.txt") s"
