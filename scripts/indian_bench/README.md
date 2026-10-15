# Indian-expression FES benchmark

Benchmark code only. It reads the source dataset, calls the running Face Enhancement Service over HTTP and writes
everything under `Dataset-and-code/fes_indian_expression_benchmark/` (git-ignored). It changes no service code,
weights, thresholds, routing or Docker configuration.

Run on the host venv (numpy / OpenCV / Pillow / scikit-image only; no torch needed):

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py scripts\indian_bench\inventory.py           # 1  read-only inventory        → manifests/source_inventory.csv
& $py scripts\indian_bench\select_originals.py    # 2  seeded 50 % per folder     → selected_originals.csv, holdout_originals.csv
& $py scripts\indian_bench\degrade.py --pilot     # 3  pilot (3 originals per expression)
& $py scripts\indian_bench\enhance.py --pilot     # 5  needs the service ready on config.service.url
& $py scripts\indian_bench\compare.py --pilot     # 6+7 comparisons + metrics
& $py scripts\indian_bench\sheets.py --pilot
& $py scripts\indian_bench\verify.py --pilot
& $py scripts\indian_bench\report.py --pilot
# full scope: the same commands without --pilot
```

Every step is resumable: run the same command again and it continues where it stopped.

`config.json`
- `seed` — selection and every random degradation derive from it.
- `scope_per_expression` — how many selected originals per expression are processed (selection order).
  `50` = 400 originals; `null` = the whole 50 % selection (1,774). Raising it continues the same benchmark.
- `matrix` / `matrix_tiers` — the degradation matrix (`degradation_matrix.json`): 23 types, 79 variants per original.
- `service` — URL, endpoint (`/v3/enhance`, the service's single-output route), timeout.

Service used: container `fes-ui` (`docker start fes-ui`, then wait for `GET /ready` = 200). The exact state that was
benchmarked is recorded in `manifests/service_snapshot.json`.
