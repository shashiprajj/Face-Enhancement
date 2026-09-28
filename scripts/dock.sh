#!/usr/bin/env bash
# Run a command inside the pinned CPU dev image with the repo mounted at /workspace.
# Usage: scripts/dock.sh python scripts/p2_fairface_verify.py ...
# (Required on this host: Windows Smart App Control blocks PyTorch DLLs outside the container.)
set -euo pipefail
IMAGE="${FES_IMAGE:-fes-dev:cpu-p2}"
ROOT="$(cd "$(dirname "$0")/.." && pwd -W 2>/dev/null || pwd)"
HEAD_SHA="$(git -C "$(dirname "$0")/.." rev-parse HEAD 2>/dev/null || echo unknown)"
DIRTY="$(git -C "$(dirname "$0")/.." status --porcelain 2>/dev/null | grep -qv '^??' && echo -dirty || true)"
MSYS_NO_PATHCONV=1 exec docker run --rm -i -e FES_GIT_HEAD="${HEAD_SHA}${DIRTY}" -v "${ROOT}:/workspace" -w /workspace "${IMAGE}" "$@"
