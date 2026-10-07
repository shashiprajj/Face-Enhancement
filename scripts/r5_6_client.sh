#!/usr/bin/env bash
# R5-6: run the evaluation client in a container on the fes-r56 network (services fes-r56-s1..s3).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -W 2>/dev/null || pwd)"
MSYS_NO_PATHCONV=1 exec docker run --rm -i --network fes-r56 -e PYTHONPATH=src \
  -e FES_GIT_HEAD="$(git -C "$(dirname "$0")/.." rev-parse HEAD)" -v "${ROOT}:/workspace" -w /workspace \
  fes-dev:cpu-p5 python scripts/r5_6_service_eval.py "$@"
