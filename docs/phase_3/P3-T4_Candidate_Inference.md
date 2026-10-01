# P3-T4 — Candidate Inference Benchmark

| Field | Value |
|---|---|
| Phase / Task | Phase 3 / P3-T4 |
| Status | Complete — every configuration has 896/896 outputs on `main` and 128/128 on `trial` |
| Date | 2026-10-01 |
| Runner | `scripts/p3_run_inference.py` (resumable, process pool); driver `scripts/p3_run_all.sh`; resume driver `logs/phase3_resume_A2_A4.sh` |
| Output manifests | `data/manifests/phase3/runs/<population>/<config>.csv.gz` + `<config>_run.json` + `<config>.partial.jsonl` |
| Outputs | `experiments/phase3/outputs/<population>/<config>/<cell>/<ref_id>.png` (**git-ignored**; only manifests are tracked) |
| Logs | `logs/phase3/run_<population>_<config>.log`, `logs/phase3/run_all_main.log`, `logs/phase3_main_driver.log` (git-ignored) |

---

## 1. Procedure

1. **Trial first** (dev data, 4 references × 32 cells = 128 pairs) for all 9 configurations. Outputs were checked for contract, determinism and visual plausibility before any main run (§3).
2. **Main run** (test data, 28 references × 32 cells = 896 pairs) in protocol order B0 → B1 → B2 → B2c → A3 → A5 → A1 → A2 → A4.
3. **Identical inputs for every configuration.** One population manifest (`population_main.csv.gz`, SHA-256 `840ee52d…`), with every degraded file re-hashed before inference.
4. **No substitution.** Each configuration's weights are SHA-256-verified at every load. The manifests record `config_id`, load reports, git head, and per-output file and pixel SHA-256.
5. **Failures are records, not deletions.** A pair whose inference raises is written as `FAILED` and retried on resume. A pair never attempted is `MISSING`. Aggregates state n.

## 2. Results — Completeness

| Config | Main OK / failed / missing | Main manifest SHA-256 | Trial OK |
|---|---|---|---|
| B0 | 896 / 0 / 0 | `76ecc3eb…` | 128 |
| B1 | 896 / 0 / 0 | `962cce84…` | 128 |
| B2 | 896 / 0 / 0 | `20f244f6…` | 128 |
| B2c | 896 / 0 / 0 | `e45177bf…` | 128 |
| A1 RestoreFormer++ | 896 / 0 / 0 | `292d167b…` | 128 |
| A2 VQFR v1 | 896 / 0 / 0 | `afd45a85…` | 128 |
| A3 Real-ESRGAN | 896 / 0 / 0 | `feb0633a…` | 128 |
| A4 SwinIR-L | 896 / 0 / 0 | `ca6d579b…` | 128 |
| A5 FBCNN → NAFNet → Real-ESRGAN | 896 / 0 / 0 | `92fbd2a4…` | 128 |

**No model failed on any input.** Every `FAILED` attempt during the run came from the infrastructure (§4), was retried, and ended `OK`.

Per-image time inside a worker **under load** is recorded in each `_run.json` (main p50: A1 17 s, A3 29 s, A2 37 s, A5 42 s, A4 106 s). This is **not** a performance measurement, because workers run concurrently with background load. The controlled measurement is P3-T8.

## 3. Trial Validation (dev data; never reported as a result)

| Check | Outcome |
|---|---|
| Output contract (224×224×3 uint8, E-frame) | ✅ All configurations |
| Byte-identical repeat at a fixed thread count | ✅ P3-T2 `adapters` section; re-checked in P3-T10 |
| Metrics re-scored | ✅ Byte-identical metric files |
| Aggregation + tables end to end | ✅ `results/trial/tables.md` |
| Visual review of contact sheets | No harness bug found. Two real model behaviours were recorded for P3-T7: (a) on soft FairFace references, RestoreFormer++ synthesises a sharper but different-looking face (it even changed gaze), which PSNR punishes heavily on mild inputs; (b) A5 turns 14-px inputs into colour noise, because its first two stages reduce such an image to a 2×2 or 1×1 grid internally. |

No protocol change was needed after the trial. The only change before the main run was runtime-only (P3-T1 §11, `p3-protocol-1.0.1`: 4 workers × 2 threads).

## 4. Incidents During Execution (all infrastructure; none changed an output)

| # | When (UTC) | What happened | Handling | Effect on results |
|---|---|---|---|---|
| I1 | 09-29/30 | **A2 trial hung** for about 12 h at 119/128. A worker died during weight loading, and the old pool waited forever. | The runner now fails fast on a broken pool, the driver resumes up to 5 times, and completed records are kept | None |
| I2 | 09-29/30 | **ENOMEM reading checkpoints** through the Windows bind mount while the Docker VM was short of memory (other host containers left about 2 GB free) | Weights are copied locally in chunks with the SHA-256 checked on the way, workers initialise one at a time, and LPIPS weights use the same path | None (content is still SHA-verified) |
| I3 | 09-30 ≈09:03 | **Docker engine stopped** during A3/main at 647/896 | Docker restarted, and the runner resumed from the records | None (resumed pairs are new, finished pairs untouched) |
| I4 | 09-30 18:02–18:43 | **A2/main OOM-killed 4×** at 4 workers. VQFR needs about 3 GB per worker, and with host containers the VM's 15 GB was exceeded. The kernel `oom-killer` was confirmed in `dmesg`. | Resumed at **2 workers × 4 threads** (same total threads, about half the memory). No OOM after that. | See §5 |
| I5 | 09-30/10-01 | **Host slept** (overnight sleep, Modern Standby on battery, lid closed): 19:43–02:43Z, 03:18–04:47Z, 06:53–07:03Z, 07:20–07:39Z, 12:11–13:46Z | Inference simply paused. A keep-awake helper was added during the run. | None |
| I6 | 10-01 | **9 A2 trial pairs** were still missing from I1 | Run on 10-01 at 2 workers × 2 threads → 128/128 | None |

The resume rule (P3-T1 §9) guarantees that a restart never rewrites an existing OK output.

## 5. Runtime Deviations Recorded (runtime only; protocol unchanged)

| Config | Population | Runtime actually used |
|---|---|---|
| A2 | main | First 122 pairs at 4 workers × 2 threads, remaining 774 at 2 workers × 4 threads (I4) |
| A2 | trial | 4×2 for the first 119 pairs, 2×2 for the last 9 |
| B0–B2c, A3 | trial | Various (3×4, 2×1). These are trial only, from before the 1.0.1 runtime change. |
| all others on main | main | 4 workers × 2 threads |

**Why this matters:** PyTorch CPU kernels can produce bit-level differences when the intra-op thread count changes. P3-T1 guarantees determinism **at a fixed thread count**, not across thread counts. P3-T10 re-runs a sample of A2 pairs at both thread counts and quantifies the difference (`data/manifests/phase3/repro/`).

## 6. Wall Time (for planning, not performance)

| Config | Main wall time actually spent computing |
|---|---|
| B0 / B1 / B2 / B2c | 16 s / 17 s / 47 s / 45 s |
| A3 | ≈ 2.7 h |
| A5 | ≈ 2.2 h (after subtracting about 2.5 h of host sleep) |
| A1 | 1.1 h |
| A2 | ≈ 4.5 h (after OOM restarts and about 9.5 h of host sleep) |
| A4 | ≈ 5.7 h (after subtracting about 1.6 h with the lid closed) |

## 7. Done-When Check

Every runnable candidate (all 5) and every baseline has outputs for the **same** 896 controlled pairs, with complete manifests, zero failed or missing pairs, and every incident recorded. ✅
