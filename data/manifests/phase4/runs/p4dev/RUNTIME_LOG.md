# p4dev inference — runtime log (no protocol change)

Protocol: p3-protocol-1.0.1 adapters/weights/geometry, unchanged. Threads per worker = 2 throughout.

| UTC | Event |
|---|---|
| 2026-10-03 06:04 | Start, 4 workers × 2 threads (A2 planned at 2). B0/B1/B2/B2c 896/896 each. |
| 2026-10-03 ~09:40 | Driver shell stopped by the host (system memory pressure). The in-flight A3 container and the orphaned loop continued; A3 finished 896/896, 0 failed. |
| 2026-10-03 10:08 | Orphaned A5 run (4 workers) stopped deliberately at 31/896 to lower memory (shared VM ≈ 15.3 GB, welorbit ≈ 7 GB). |
| 2026-10-03 10:10 | Resumed with conservative workers: 2 (A2: 1), threads 2. Completed pairs are not recomputed. Worker count affects wall time only. |
