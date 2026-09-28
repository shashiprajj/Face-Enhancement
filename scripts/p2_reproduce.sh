#!/usr/bin/env bash
# Rebuild the complete Phase 2 benchmark-data foundation from the immutable raw FairFace archive.
# Every step is config-driven, deterministic and idempotent (existing outputs are hash-verified, never overwritten).
set -euo pipefail
cd "$(dirname "$0")/.."
D=scripts/dock.sh
HOSTPY="${HOSTPY:-.venv/Scripts/python.exe}"   # stdlib-only extraction runs on the host (fast native file I/O)

PYTHONPATH=src "$HOSTPY" scripts/p2_fairface_prepare.py --stage extract          # P2-T2 working copy
$D python scripts/p2_fairface_prepare.py --stage analyze                          # P2-T2 verification + manifest
$D python scripts/p2_build_references.py --config configs/phase2/references.yaml  # P2-T3 clean references
$D python scripts/p2_degrade.py --config configs/phase2/degrade_pilot.yaml        # P2-T4 pilot
$D python scripts/p2_build_splits.py --config configs/phase2/benchmark_v1.yaml    # P2-T7 splits + leakage screen
$D python scripts/p2_degrade.py --config configs/phase2/degrade_fes-syn-v1_dev_pilot.yaml
$D python scripts/p2_degrade.py --config configs/phase2/degrade_fes-syn-v1_test_core.yaml
$D python -m pytest -q -p no:cacheprovider tests                                  # P2-T6 contract tests
$D python scripts/p2_validate_benchmark.py --config configs/phase2/benchmark_v1.yaml --repro-all   # P2-T8
