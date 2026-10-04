# P4-INFRA-RESUME — execution log (same experiment; no protocol change)

Unchanged: population `p4dev` (`8a257f44…`), adapters, weights, geometry, `p3-protocol-1.0.1`, threads per worker = 2,
output naming, evaluators, scoring. Only the worker count and the host's other workloads changed.

## Phase 1 — state before stopping (2026-10-03 10:40 UTC)

| Config | Records | OK with output | Duplicates | Failed | Malformed | Run manifest |
|---|---|---|---|---|---|---|
| B0, B1, B2, B2c, A3 | 896 each | 896 each | 0 | 0 | 0 | written |
| A5 | 210 | 210 | 0 | 0 | 0 | — (in progress) |
| A1, A4, A2 | 0 | — | — | — | — | not started |

Running: driver `p4_run_dev.sh A5 A1 A4 A2` (2 workers; A2 1), container `quirky_chandrasekhar` (A5, ≈ 3.0 GB, ≈ 390 % CPU).
Host: Docker VM 16 CPU / 15.3 GB. WelOrbit (12 containers) ≈ 10 GB RAM, celery ≈ 400 % CPU.
Estimate at this configuration: **≈ 28–30 h** remaining (incl. scoring + T6–T8).

## Phase 2 — safe stop (10:41 UTC)

Driver killed first (no relaunch), then `docker stop -t 90` (graceful). Exit 143 = SIGTERM as expected.
After stop: A5 246 records, 246 OK with output, 0 malformed; completed run manifests byte-identical to before.

## Phase 3 — temporary resource release (10:43 UTC)

`docker stop -t 60` (no rm, no prune, no data/code change) for the 12 containers in
`logs/phase4/welorbit_stopped_containers.txt` (project `welorbit`, `D:\WelOrbit`; restart policy unless-stopped →
they stay stopped until `docker start`). Note: `welorbit-celery-1` did not exit within 60 s and was SIGKILLed by
Docker (exit 137) — an in-flight Celery task, if any, was interrupted; container and data intact.
After: 16 CPU, MemAvailable 14.3 GB of 15.3 GB.

**Restore after the run:** `xargs docker start < logs/phase4/welorbit_stopped_containers.txt`

## Phase 4 — worker counts (measured basis: Phase 3 peak RSS per process)

| Model | Workers | Threads | Peak RSS/process | Est. peak |
|---|---|---|---|---|
| A5 | 5 | 2 | 2.1 GB | ≈ 10.5 GB |
| A1 | 4 | 2 | 1.3 GB | ≈ 5.2 GB |
| A4 | 4 | 2 | 1.3 GB | ≈ 5.2 GB |
| A2 | 2 (→ 1 automatically if VM MemAvailable < 1.5 GB) | 2 | 3.3 GB | ≈ 6.6 GB |

## Phase 5–6 — resume + monitoring

Driver `scripts/p4_resume_dev.sh` (started 10:45 UTC), monitor `scripts/p4_monitor_dev.sh` →
`logs/phase4/resume_monitor.log` (every 10 min: completed/expected, %, rate, ETA, failures, container CPU/RAM,
VM MemAvailable, load). Completed pairs are skipped by the runner (OK record + PNG present).

## Progress checkpoint (2026-10-03T18:12:27Z)

- Complete (896/896, 0 failed) and scored by SFace + dlib: B0 B1 B2 B2c A1 A3 A4 A5.
- A2 (VQFR v1) running in container fes-p4dev-A2, 1 worker × 2 threads, 45/896, ~64 pairs/h.
- WelOrbit restarted externally at 11:26 UTC; user decision at 17:30 UTC: keep WelOrbit running, run A2 in parallel (1 worker).
- Claude Code background shells were stopped under host memory pressure 3× (driver/monitor shells only); containers continued; no records lost.

## Remaining steps (in order)
1. When A2 finishes: score A2 (p3_evaluate.py + p4_evaluate_e07.py, --configs A2, p4dev).
2. scripts/p4_validate_dev_runs.py must pass; commit infra/resume task.
3. T6: p4_gate_calibrate.py → commit frozen configs/phase4/t6_gate.json → p4_gate_evaluate.py (main, once) → report + commit.
4. T7: p4_route_calibrate.py → p4_router_validate.py → commit → p4_route_evaluate.py (main, once) → report + commit.
5. T8 final report + gate matrix + Phase 5 scope → commit. Restart WelOrbit only if it was stopped.

## Later events (2026-10-03/04)
- 2026-10-03 19:00 UTC → 2026-10-04 06:28 UTC: laptop in Modern Standby after switching to battery (no compute; no records lost).
- 2026-10-04 06:36 UTC: A2 stopped at checkpoint (142 OK), resumed with 2 workers (A4 finished; VQFR steady ≈ 1.1 GB/worker).
- 2026-10-04 08:31 UTC: user request — the 4 WelOrbit containers then running were stopped (`logs/phase4/welorbit_stopped_containers_2.txt`);
  A2 stopped at checkpoint (481 OK) and resumed with 4 workers. The other 8 WelOrbit containers were already exited at that time (not stopped by this task).
- 2026-10-04 ~09:35 UTC: A2 896/896. WelOrbit: the 4 stopped containers restarted with `docker start`.

## Phase 7 — completion validation (`scripts/p4_validate_dev_runs.py` → `validation.json`)
All 9 configurations: 896/896 rows, 0 missing, 0 extra, 0 duplicates, all OK, required metadata present, every output
exists with file SHA-256 and pixel SHA-256 equal to the record, 224×224×3 uint8 PNG, 0 malformed / 0 conflicting
checkpoint records, run.json counts complete, population hash = frozen p4dev manifest. **all_ok = true.**

## Phase 8 — dev scoring
SFace (E01) and dlib (E07) metrics for all 9 configurations (`data/manifests/phase4/metrics/p4dev/`,
`metrics_e07/p4dev/`); analysis `data/manifests/phase4/results/p4dev/` (ΔID per evaluator with clustered CIs,
rescues/flips, confirmed benefit/harm, by family and severity, evaluator agreement). A3 re-score hashes equal the
earlier score (`89b7f755…` E01, `de2158c8…` E07) → scoring reproducible.

## Runtime summary
| | Before (2 workers, WelOrbit up) | After optimisation |
|---|---|---|
| Estimate at 10:40 UTC 10-03 | ≈ 28–30 h remaining | — |
| A5 throughput | ≈ 336 pairs/h | ≈ 825–880 pairs/h (5 workers, WelOrbit stopped) |
| A1 | — | ≈ 840 pairs/h (4 workers) |
| A4 | — | ≈ 180–195 pairs/h (4 workers; WelOrbit restarted externally) |
| A2 | ≈ 64 pairs/h (1 worker) | ≈ 120–150 (2 workers) → ≈ 420 pairs/h (4 workers, WelOrbit stopped) |
Measured compute time for the remaining inference ≈ 12 h wall excluding the 11.5 h standby, vs the 28–30 h estimate.
Final worker configuration: A5 5, A1 4, A4 4, A2 1 → 2 → 4; threads per worker 2 throughout (protocol unchanged).
