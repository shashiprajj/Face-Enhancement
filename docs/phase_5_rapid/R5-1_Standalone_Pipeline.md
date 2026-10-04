# R5-1 — Standalone Working Enhancement Pipeline

| Field | Value |
|---|---|
| Phase / task | 5 Rapid Validation / R5-1 |
| Date | 2026-10-04 |
| Baseline | Phase 4 final `7aa8b29` (YELLOW). T6 seal `b8ee2c43…`, T7 seal `2f026f22…`, both loaded and verified at start-up. |
| Code | `src/fes_service/{pipeline,worker,models,app}.py` |
| Image | `fes-dev:cpu-p5` = `cpu-p4` + hash-pinned FastAPI layer (`requirements-p5.txt`, `docker/Dockerfile.cpu-p5`). Licences MIT / BSD / Apache-2.0. |
| Tests | `tests/test_r5_service.py`: 23 pass, plus 1 opt-in real-A4 test |
| E2E evidence | `data/manifests/phase5_rapid/r5_1/{e2e.json, requests.csv.gz}` (`scripts/r5_service_e2e.py`) |
| Status | **Research candidate.** Not production-approved. Gate/router thresholds are synthetic-calibrated. A4 weights and YuNet are licence REVIEW_REQUIRED. |

**Acceptance met.** A poor-quality input goes through the standalone service. It deterministically returns either the original or an A4-enhanced face, according to the frozen policy. Every failure mode tested falls back to the original.

## 1. What was built

```
bytes → validation (size/bytes/pixels/aspect, header-checked before decode)
      → detection (existing FES contract, YuNet, largest face, multiple_faces warning)
      → face chip (FairFace/dlib framing, integer-pixel crop, native scale; or caller-supplied chip)
      → p4-quality features → FROZEN T6 gate → FROZEN T7 table → ORIGINAL | A4
      → output guard (frozen router: shape/dtype, std ≥ 1, face present on output) → response
```

- **Frozen logic reused, not rewritten.**
  - `fes_bench.gate.load_rules` and `fes_bench.routing.load_routing` (seal checks) and `routing.Router` (fallbacks + output guard) are used unchanged.
  - The pipeline refuses to start if the table routes anything other than A4.
- **A4:** the existing Phase 3/4 adapter, loaded from `models/M24_swinir` with its SHA-256 verified. It runs on CPU only (`CUDA_VISIBLE_DEVICES=""`, 2 torch threads) and makes no network calls.
- **Process isolation.**
  - A4 runs in one spawned child process.
  - On timeout (default 120 s) the child is **killed**, not abandoned. It restarts on the next request.
  - Child death by signal (OOM-kill) → `RESOURCE_FAILURE`; model exception → `MODEL_FAILURE`.
  - This closes the Phase 4 limitation where a thread timeout could not kill a hung model.
- **Input modes:**
  - `face_chip`: the exact Phase 4 contract, a square chip ≤ 224 px.
  - `full_image`: any photo, cropped by detection.
- **Outputs:**
  - `ENHANCED`: the A4 face chip (224×224 PNG).
  - `ORIGINAL`: the **input bytes exactly as received**.
  - `REJECTED`: HTTP 422, undecodable or out-of-limit input.
- **Metadata.** Returned: status, T6 decision/reason, bucket, route/used, fallback, escalate, warnings, face count, chip size, 7 quality numbers, timings. **Not returned:** landmarks, boxes, embeddings. No image is stored or logged.
- **Endpoints:**
  - `GET /health`: liveness.
  - `GET /ready`: seals verified, detector loaded, A4 worker loaded and warmed up; otherwise 503 with the reason.
  - `POST /v1/enhance`.
- **Concurrency.** One request is processed at a time (YuNet and the router are stateful). At most 4 may wait; beyond that the response is **503 BUSY** with `Retry-After` instead of an unbounded queue.

## 2. Required safety behaviour (verified)

| Case | Result | Evidence |
|---|---|---|
| Good image | ORIGINAL (`DO_NOT_ENHANCE`), A4 not called | test + e2e `DNE_mild` |
| Unsafe/uncertain (no face, too small, low score, bad pose) | ORIGINAL + `escalate` | test + e2e `UNSAFE` |
| Approved degradation (NOISE/BLUR) | A4 | e2e: 4/4 ENHANCED, pixels equal Phase 4 |
| Gated but other bucket (PHOTOMETRIC, COMPRESS) | ORIGINAL | e2e `ENHANCE_to_original` |
| A4 exception / OOM-kill / invalid output / hang | ORIGINAL with `MODEL_FAILURE` / `RESOURCE_FAILURE` / `INVALID_OUTPUT` / `TIMEOUT`. Hung or dead worker killed and restarted. | `test_a4_failures_fall_back_to_original` (real child processes) |
| Detection failure / no face in photo | ORIGINAL + escalate | test + e2e `full_no_face` |
| Multiple faces | Largest face processed; `multiple_faces` warning; count returned | test + e2e `full_two_faces` |
| Corrupt / empty / tiny / extreme aspect / oversize bytes / decompression bomb | REJECTED (422) | tests + e2e `corrupt_file` |
| Model cannot load | `/ready` 503 and `/v1/enhance` 503 (never a half-working service) | `test_api_not_ready_when_model_cannot_load` |

## 3. End-to-end results (real A4, HTTP, CPU)

| Check | Result |
|---|---|
| Policy equivalence: all 896 Phase 4 dev chips through the service (`face_chip`) vs frozen decision log | **896 / 896 identical** (decision, reason, bucket, route) |
| A4 outputs vs Phase 4 run manifest | pixel-identical (4/4) |
| Determinism (each request sent twice) | 13/13 identical metadata + bytes |
| ORIGINAL returns the input bytes unchanged | 8/8 |
| Unexpected fallbacks | 0 |

| Runtime (shared laptop CPU; WelOrbit containers running) | Value |
|---|---|
| A4 request (224-px chip, 2 threads) | p50 **71 s**, max 81 s |
| ORIGINAL / REJECTED request | p50 15–20 ms, max 40 ms |
| Cold start (worker load + warm-up) | 6.2 s |
| Peak RSS | service 187 MB, A4 worker **1.25 GB** |

CPU latency was not optimised (out of scope). A4 at about 70 s per face rules out interactive use on this hardware. It is acceptable for an offline or batch evaluation.

## 4. Defects found and fixed in this task

1. **UNSAFE inputs returned HTTP 500.** No detection means NaN quality values, which are not valid JSON. Fixed: non-finite → `null`. Regression test added.
2. **Full-image crop erased the gate's evidence.**
   - Cause: the first crop version used a sub-pixel similarity warp. Bilinear interpolation low-passes noise and sharpness, which are exactly what T6 reads.
   - Fix: the crop is now an integer-pixel cut at native scale. A test proves every chip pixel is an original source pixel.

## 5. Known limitation carried into R5-2 (important)

The frozen gate was calibrated on tight FairFace chips (`face_chip` mode, reproduced 896/896). For `full_image` input:
- The detected framing varies by about ±10 % (chip side 201–272 px for a 224-px face).
- Surrounding pixels enter the chip.

On synthetic chips pasted into constructed canvases:
- The router stayed **safe**: 1–2 new A4 routes out of 896.
- It became **far more conservative**: most Phase 4 A4 routes became ORIGINAL.

The canvases were not realistic (flat or mirrored surroundings), so this is not a measurement of real photos. Real-world inputs in R5-2 will show whether real photos reach A4 often enough. If they don't, a targeted framing fix belongs to R5-3. The frozen thresholds stay unchanged.

## 6. Run

```bash
docker build -f docker/Dockerfile.cpu-p5 -t fes-dev:cpu-p5 .
# service (models mounted read-only; port 8080)
MSYS_NO_PATHCONV=1 docker run --rm -p 8080:8080 -v "$(pwd -W):/workspace" -v "$(pwd -W)/models:/workspace/models:ro" \
  -w /workspace -e PYTHONPATH=src fes-dev:cpu-p5 uvicorn fes_service.app:app --host 0.0.0.0 --port 8080
curl -F image=@face.jpg -F input_mode=full_image http://localhost:8080/v1/enhance
# tests / e2e
FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python -m pytest -q tests/test_r5_service.py      # FES_RUN_A4=1 adds the real-A4 test
FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_service_e2e.py --per-group 2
```

## 7. Validation status

- `tests/test_r5_service.py`: **23 passed, 1 skipped** (opt-in real-A4 test; the same check passed in the e2e run, A4 pixels equal Phase 4).
- E2E: all 5 checks true (§3).
- The full repository suite was started in `fes-dev:cpu-p5` but **did not complete**: the host ran critically low on memory (WelOrbit containers ≈ 10 GB of the Docker VM) and the run was stopped. No test failed; the suite simply has no result for this commit. Re-run it when memory allows: `FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python -m pytest -q`. No frozen Phase 4 module was modified (all new code is in `src/fes_service/`).

## 8. Not done (by design)

- No A1/A2/A5.
- No change to T6/T7 or the quality model.
- No new benchmark.
- No Feature-Match integration.
- No CPU optimisation.
- No production claim.
