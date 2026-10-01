# P3-T8 — CPU Performance & Resource Benchmark

| Field | Value |
|---|---|
| Phase / Task | Phase 3 / P3-T8 |
| Status | Complete — **CPU only; GPU benchmarking remains pending (G-1)** |
| Date | 2026-10-01 (15:15–15:31 UTC, re-check 15:42 UTC) |
| Script | `scripts/p3_cpu_benchmark.py` |
| Results | `data/manifests/phase3/perf/cpu_benchmark_t2.json` (all 9 configurations), `cpu_benchmark_t2_recheck_A3_A5.json` (re-check) |

---

## 1. Environment Record

| Item | Value |
|---|---|
| CPU | Intel Core Ultra 7 255H (16 logical CPUs visible to the Docker VM) |
| Host | Windows 11 laptop. Docker Desktop, WSL2 kernel 6.18.33.1. **Plugged in** during the measurement. |
| Image | `fes-dev:cpu-p2` — Python 3.11.16, torch 2.14.0+cpu, OpenCV 5.0.0, numpy 2.4.6 |
| Precision / batch | fp32, batch 1 |
| Threads | `torch.set_num_threads(2)` (the runtime per-worker setting), `cv2.setNumThreads(1)` |
| Code | git `b14cef3` + uncommitted Phase 3 code (committed with this phase) |
| Background load | No other container was running: the user's other workloads were stopped for the measurement. VM load average recorded per configuration (1-min 1.8–4.5, from the benchmark itself and residual earlier jobs). |

## 2. Method (one consistent methodology for every configuration)

1. Each configuration runs in a **fresh spawned process**, sequentially. Nothing else of the harness runs at the same time.
2. **Cold start** = network construction plus safe weight loading (`weights_only=True`). The SHA-256 check is timed separately.
3. For each input size: **1 warm-up call (excluded)**, then **5 timed calls** of `adapter.enhance()` on the same real degraded image (main reference `ff-val-10081`). The timed region is end to end: geometry, model and resampling, uint8 in to uint8 out.
4. **Peak RSS** (`ru_maxrss`) of the process after all calls.
5. Input sizes:
   - face models: 224 and 56 (they always run at 512×512 internally);
   - general SR, chain and baselines: 224, 112, 56, 28 and 14.

## 3. Results — Warm Latency per Image (median of 5, seconds; 2 threads)

| Config | 224 px | 112 px | 56 px | 28 px | 14 px | Spread (max − min) at 224 |
|---|---|---|---|---|---|---|
| B0 / B1 | < 0.001 | < 0.001 | < 0.001 | < 0.001 | < 0.001 | — |
| B2 / B2c | 0.053 | 0.053 / 0.055 | 0.056 / 0.054 | 0.052 / 0.055 | 0.054 / 0.053 | 0.005 |
| **A1** RestoreFormer++ | **5.50** | — | 5.56 | — | — | 0.65 |
| **A2** VQFR v1 | **20.06** | — | 20.15 | — | — | 1.47 |
| **A3** Real-ESRGAN | **12.63** (re-check 11.85) | 2.61 (2.75) | 0.76 (0.66) | 0.23 (0.18) | 0.067 (0.065) | 0.71 |
| **A4** SwinIR-L | **49.36** | 10.17 | 2.27 | 0.68 | 0.22 | 1.51 |
| **A5** FBCNN → NAFNet → R-ESRGAN | **12.71** (re-check 13.25) | 3.07 (3.32) | 0.80 (0.87) | 0.31 (0.30) | 0.11 (0.14) | 1.82 |

Single-process throughput is the inverse of the median: at 224 px that is A1 0.18 img/s, A2 0.050, A3 0.079, A4 0.020 and A5 0.079.

**Run-to-run reproducibility.** The A3 and A5 re-check on an idle machine differs from the first run by −6 % and +4 % at 224 px. During the first A3 measurement, light host-side analysis was running. Differences below about ±6 % between configurations are therefore not meaningful.

## 4. Results — Load Time and Memory

| Config | Cold load (s) | SHA-256 check (s) | RSS after load (MB) | **Peak RSS (MB)** |
|---|---|---|---|---|
| B0–B2c | 0 | 0 | 243 | 244–253 |
| A1 | 2.1 | 1.8 | 806 | **1,344** |
| A2 | 3.6 | 1.4 | 922 | **3,300** |
| A3 | 1.7 | 0.5 | 465 | **1,172** |
| A4 | 1.3 | 2.1 | 525 | **1,258** |
| A5 | 8.7 | 8.3 | 1,498 | **2,108** |

**A2's 3.3 GB peak per process explains the main-run OOM kills at 4 workers (P3-T4 I4):** 4 × 3.3 GB plus other workloads exceeded the VM's 15 GB.

## 5. Observations (descriptive; TH-8 targets are TBD)

1. **Face models cost the same at any input size.** They always process 512×512, so A1 takes 5.5 s and A2 20 s per face whether the input is 14 or 224 px.
2. **General-SR cost scales with input area.** A ×4 model on a 224-px input produces 896×896. A4 takes 49 s at 224 px but 0.22 s at 14 px. Since the evaluation frame is 224 px, downscaling large inputs before ×4 SR would change this, but that would be a different configuration and was **not** tested.
3. **A5's extra stages are cheap relative to its SR stage** (+0.1 to +1.4 s over A3), but it needs 1.8× A3's memory and 8.7 s cold load (three networks).
4. Approximate ordering at 224 px, 2 threads: B2 (0.05 s) ≪ A1 (5.5 s) < A3 ≈ A5 (12–13 s) < A2 (20 s) < A4 (49 s).
5. Latency under concurrent load during the main run was higher (`_run.json`, e.g. A4 p50 106 s at 4 × 2 workers). It is **not** a performance figure.

## 6. Not Measured / Not Claimed

- **No GPU numbers, and no extrapolation from CPU to GPU.** The relative ordering on a GPU can differ, because attention-heavy (A4) and VQ/transformer (A1/A2) models scale differently on GPUs. GPU benchmarking remains pending (G-1).
- No P95 latency under service load, no multi-face images, no batching, no fp16/int8, and no optimised runtimes (ONNX, TensorRT, OpenVINO). These belong to a later phase.
- No comparison against TH-8 (P95 latency, throughput and memory targets are TBD, P0-T5).

## 7. Done-When Check

CPU latency (warm and cold), throughput and peak memory are measured for every configuration under one recorded methodology and environment, with a reproducibility re-check. They are reported without GPU extrapolation, and GPU benchmarking is recorded as pending. ✅
