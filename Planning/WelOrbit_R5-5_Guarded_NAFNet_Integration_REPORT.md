# R5-5: Guarded NAFNet-GoPro Integration — Report

Date: 2026-10-07 · Task: `Tasks/Face_Enhancement_Phase_5_Rapid_Validation/R5-5_Guarded_NAFNet_Integration_PROMPT.txt`

## 1. Executive summary

NAFNet-GoPro width-64 now runs inside the real FastAPI service as a **guarded deblurring route**:

input → validation → detection/alignment → quality features → **NAFNet input gate** → NAFNet-GoPro (isolated worker) → **artifact guard** → NAFNet output, or the ORIGINAL at any failing step.

- The real model runs through the real service. This was checked with the live uvicorn service (§14.4), not with a stub.
- NAFNet is **not** always-on. On the DEV population, 47 of 1,120 inputs (4.2%) are eligible. All of them are strong, low-noise blur.
- Every failure path returns the ORIGINAL, and each path is tested: timeout, crash/OOM, exception, load failure, invalid output, artifact-guard failure, and an exception inside the guard.
- **A4/SwinIR-L is unchanged and still selectable.** Use `route_mode=a4` per request or `FES_ROUTE_MODE=a4` per service. In `auto` mode, A4 is kept only for the one route with evidence behind it (strong noise).
- The artifact guard works on real failures. On DEV strong blur, real NAFNet destroyed 3 of 47 faces (rainbow mosaic, checkerboard tint). The guard caught all 3 and passed the 44 clean outputs.
- Tests: **253 passed / 5 skipped / 0 failed** (full suite in Docker), **+4/4** real-NAFNet tests (`FES_RUN_NAFNET=1`), **21/21** UI-logic tests (node).

**Verdict: GREEN.** The implementation is ready for service-level validation (R5-6). Two limitations matter for R5-6 and are listed in §16 and §18. Neither is a safety gap.

## 2. Baseline commit

`c801a84` (R5-ARCHIVE). The working tree also held unrelated uncommitted changes. These were left untouched and **not** committed:

- `data/manifests/phase5_rapid/metrics*/main/identity_calibration.json`
- the old-plan 5.1 baseline-freeze files: `configs/phase5/baseline_manifest.json`, `scripts/p5_baseline_freeze.py`, `tests/test_p5_baseline.py`, `docs/phase_5/`
- `Tasks/Face_Enhancement_Phase_5/`

## 3. Final commit

This report is committed with the R5-5 work, in `feat(fes): add guarded NAFNet GoPro blur route` on `main`. Not pushed.

## 4. Exact NAFNet checkpoint and path

| Field | Value |
|---|---|
| Path | `models/M28_nafnet/NAFNet-GoPro-width64.pth` (git-ignored; 271,778,961 bytes) |
| Asset / candidate | A06 / M28 (Phase 1 acquisition manifest) |
| Architecture | `NAFNetLocal(img_channel=3, width=64, middle_blk_num=1, enc_blk_nums=[1,1,1,28], dec_blk_nums=[1,1,1,1])`, the official `options/test/GoPro/NAFNet-width64.yml` @2b4af71, vendored in `third_party/nafnet` |
| Load | `torch.load(weights_only=True)`, key `params`, **strict**: 0 missing and 0 unexpected keys, 664 tensors, 67,888,835 parameters |
| Registry | `fes_bench.adapters.WEIGHTS["M28-GoPro"]` (unchanged). The service loads it only through `fes_service.models.load_nafnet` / `nafnet_factory`. |

The SIDD variant in the same folder is **not** substitutable. Loading it under the GoPro registry entry fails the checksum check, and a test covers this.

## 5. Hash

SHA-256 `329d3ab4077b8d6b7ff61de376e483714667960bf85be027bf4335cda701196f`. It is verified while the file streams to container-local storage (`archs.staged_copy`) on every load. A mismatch raises `checksum mismatch while staging …`, and a missing file raises `FileNotFoundError: NAFNet-GoPro checkpoint missing: …`. The hash is also recorded in the sealed route config (`model_asset`).

## 6. Licence and provenance evidence

- `docs/phase_1/model_acquisition_manifest.csv`, row A06:
  - source: the official Google Drive file `1S0PVRbyTakYY9a82kujgZLbMihfNBLfC`, linked from the README model table of `megvii-research/NAFNet` @2b4af71
  - downloaded 2026-09-28
  - licence **MIT** (repo; copy at `models/M28_nafnet/LICENSE`, "Copyright (c) 2022 megvii-model")
  - status: acquisition/eval **ELIGIBLE**, production **REVIEW_REQUIRED**
- No checksum is published upstream. The hash above is the hash at acquisition.
- `data/manifests/phase3/model_validation_report.json` records `M28-GoPro: VALIDATED` (strict, finite outputs).
- **No production approval is claimed.** No licence reviewer is named (stakeholder status 2026-09-28). The archive dataset's provenance is still unresolved. Nothing was trained or fine-tuned.

## 7. Files changed

| File | Change |
|---|---|
| `src/fes_service/nafnet_route.py` | **new**: texture features, input gate `decide()`, `ArtifactGuardResult` + `artifact_guard()`, synthetic risk generators for calibration and tests |
| `configs/phase5/r5_nafnet_route.json` | **new**: sealed thresholds `p5-nafnet-route-1.0.0` (seal `eb6fe101…c8b9`) |
| `scripts/r5_nafnet_calibrate.py` | **new**: derives and seals the config from DEV data; NAFNet DEV probe |
| `scripts/r5_service_validate.py` | **new**: R5-6 runner (folder → real pipeline → per-input records) |
| `src/fes_service/models.py` | `load_nafnet`, `nafnet_factory`; NAFNet test doubles (ok / rainbow / bad shape) |
| `src/fes_service/pipeline.py` | `route_mode` (a4 / nafnet / auto); `_nafnet_path`; metadata fields; `validation_record()`. The A4 code moved unchanged into `_a4_path`. |
| `src/fes_service/app.py` | NAFNet `IsolatedModel`, `FES_ROUTE_MODE` / `FES_NAFNET_*`, `route_mode` form field, `/ready` NAFNet fields |
| `src/fes_service/ui/fes_ui.js` | real model / route / artifact metadata in the result view (logic only) |
| `tests/test_r5_nafnet.py` | **new**: 59 tests |
| `tests/test_r5_service.py`, `tests/test_r5_ui.py` | API fixtures also stub the NAFNet worker (1 line each) |
| `tests/ui/fes_ui.test.mjs` | +4 NAFNet / A4 view tests |
| `data/manifests/phase5_rapid/nafnet_route/calibration.json` | calibration and probe evidence |
| `data/manifests/phase5_rapid/service_validation/r5_5_smoke_dev*` | runner smoke test (4 DEV images) |

## 8. Architecture and route flow

```
POST /v1/enhance (image, input_mode, return_image, force_a4, route_mode="" → FES_ROUTE_MODE)
  decode/limits ─ invalid → REJECTED (422)
  detect (YuNet) + landmarks plausible ─ fail → ORIGINAL (UNSAFE)
  face chip (FairFace framing, native ≤ 224 px)
  ├─ route_mode=a4 or force_a4 → R5-1/R5-3 path, unchanged (rwgate|t6 → A4 | ORIGINAL)
  └─ route_mode=nafnet|auto
       rw features (p5-rwfeat-1.0.0) + texture features (p5-nafnet-feat-1.0.0)
       NAFNet gate ─ ineligible ─┬ auto and rwgate says strong_noise → A4 path (retained evidence-backed route)
                                 └ otherwise → ORIGINAL (reason NAFNET_REJECT_*)
       eligible → NAFNet-GoPro (IsolatedModel: own process, loaded once, hard timeout, killed and restarted on hang/crash)
              ─ TIMEOUT / RESOURCE_FAILURE / MODEL_FAILURE → ORIGINAL
       artifact guard ─ FAIL (or guard exception) → ORIGINAL (fallback ARTIFACT_GUARD)
       PASS → NAFNet output (native-size face chip, PNG; greyscale input → greyscale output)
```

- `auto` never calls both models. A request reaches A4 only when NAFNet is ineligible and the rwgate route is strong noise. These are disjoint: NAFNet requires noise σ ≤ 1.08, and strong noise is σ ≥ 9.84.
- The rwgate **strong_blur → A4** route is not used in `auto`, because R5-ARCHIVE measured no blur benefit for A4 (SFace Δ −0.014).

**Service defaults.** `FES_GATE=rw`, `FES_ROUTE_MODE=auto`, `FES_NAFNET_TIMEOUT_S=60`, `FES_NAFNET_THREADS=4`. To roll back to the exact pre-R5-5 behaviour, set `FES_ROUTE_MODE=a4`. With `FES_GATE=t6`, the service stays in a4 mode.

## 9. Input-gate rules (`p5-nafnet-route-1.0.0`)

Rules are evaluated in this order. The first match wins, and every outcome except the last returns the ORIGINAL. If any required feature is NaN, the result is `NAFNET_REJECT_MISSING_FEATURE:<keys>`.

| # | Reason | Rule | Threshold | Rationale (DEV only, no outcome data for NAFNet) |
|---|---|---|---|---|
| 1 | `NAFNET_REJECT_NO_FACE` / `_LOW_DETECTION` / `_UNSAFE_ALIGNMENT` | T6 safety rules, unchanged | det score ≥ 0.85, \|yaw\| ≤ 0.25 | Frozen Phase 4 rules |
| 2 | `NAFNET_REJECT_LOW_FACE_SIZE` | IOD < `iod_min` | 35.347 px | = rwgate `iod_min_a4` (smallest DEV size tier with a net enhancement benefit). NAFNet's own evidence is at IOD ≈ 46 px (archive 128-px chips). |
| 3 | `NAFNET_REJECT_PHOTOMETRIC` | luma < 47.13 or RMS contrast < 25.38 | rwgate values | Clean DEV reference quantiles (T7) |
| 4 | `NAFNET_REJECT_COMPRESSION` | blockiness > 1.523 | rwgate value | Clean DEV reference quantile (T7) |
| 5 | `NAFNET_REJECT_PIXELATION` | gradient-profile periodicity > 0.6255 | new | 1.1 × the maximum over DEV inputs passing every other rule |
| 6 | `NAFNET_REJECT_HALFTONE` | HF spectral-peak share > 0.2736 **and** HF energy > 0.0999 | new | 1.1 × the same DEV maxima |
| 7 | `NAFNET_REJECT_HIGH_NOISE` | noise σ > `noise_max` | 1.0835 | = rwgate `clean_noise_hi` (98% quantile of clean DEV refs): low noise means no more grain than a clean photo |
| 8 | `NAFNET_REJECT_NOT_BLUR` | lapd1_c ≥ 0.005579 | rwgate `clean_lapd1_lo` | 2% quantile of clean DEV refs: sharp/clean-looking, or a non-blur degradation |
| 9 | `NAFNET_REJECT_MILD_BLUR` | 0.002617 < lapd1_c < 0.005579 | | Between "strong" and "clean" |
| 10 | `NAFNET_ELIGIBLE_STRONG_LOW_NOISE_BLUR` | lapd1_c ≤ 0.002617 | rwgate `blur_max` | Existing strong-blur boundary |

- **There is no lapd1 floor**, by design (R5-ARCHIVE §8): this route covers the extreme blur that the A4 policy rejected. Recoverability is still limited by detection score ≥ 0.85, landmark plausibility and face size.
- Blur alone never qualifies. A blurred chip with σ=6 grain added keeps lapd1 ≤ blur_max but is rejected as `HIGH_NOISE` (test).
- Thresholds were **not** tuned on the archive test/ztest, the 200 real photos, or the real-person data.

**DEV outcome over 1,120 inputs** (p4dev 896 + r5gap 224):

| Outcome | n |
|---|---|
| ELIGIBLE | 47 (gaussblur extreme 42, motionblur extreme 4, motionblur severe 1) |
| NOT_BLUR | 302 |
| HIGH_NOISE | 154 |
| NO_FACE | 154 |
| PHOTOMETRIC | 141 |
| LOW_FACE_SIZE | 94 |
| COMPRESSION | 73 |
| MILD_BLUR | 72 |
| UNSAFE_ALIGNMENT | 50 |
| LOW_DETECTION | 33 |

**Printed-image risks.** These are synthetic versions of 28 clean DEV references: pixelated ×3/×4/×6/×8 and 45° dot screens with period 3/4/6/8.

- None of the 224 reaches the strong-blur zone (lapd1 ≥ 0.004 for pixelation, ≥ 0.016 for halftone). All 112 halftones fail face detection.
- The texture rules on their own flag 112/112 pixelated images and 100/112 halftones (mostly through the pixelation rule).
- The dedicated halftone rule alone flags only 6/112. It is kept as a second line of defence. In practice, halftone protection comes from the blur requirement, detection and the pixelation rule (see §16).

## 10. Artifact-guard rules

`artifact_guard(input_chip, output, cfg, face_check, input_is_gray) → ArtifactGuardResult(passed, reasons, metrics)`

| Reason | Check | Threshold |
|---|---|---|
| `ARTIFACT_NUMERIC_INVALID` | not an array, dtype ≠ uint8, NaN/Inf, shape ≠ input, near-constant: **terminal** | — |
| `ARTIFACT_NUMERIC_INVALID` | channel collapsed, or a channel mean shifted: reported, other checks still run | shift > 25 grey levels |
| `ARTIFACT_FACE_INVALID` | YuNet finds no face on the output | — |
| `ARTIFACT_CHROMA_P99` | 99th percentile of the Lab (a, b) distance, output vs input chip | > 10.0 (the R5-ARCHIVE artifact signal; configurable) |
| `ARTIFACT_HIGH_FREQ_COLOR` | HF chroma / HF luma energy, residual vs G(σ=1.5): absolute level **and** growth over the input | > 0.2709 (1.5 × the 99.5% quantile of clean DEV refs) and > 3 × the input's ratio |
| `ARTIFACT_GRAYSCALE_VIOLATION` | greyscale input: p99 chroma magnitude of the output | > 6.0 Lab units |

- If the guard itself raises an exception, the result counts as FAIL.
- When a greyscale input passes, the output is converted back to greyscale (`grayscale_preserved`), as on the A4 path.

**Real-model evidence (DEV, diagnostic only; no threshold was changed afterwards).** On the 47 DEV strong-blur inputs, NAFNet produced 44 PASS and 3 FAIL:

- 28962@168: FACE + CHROMA (p99 145) + HF
- 30710@168: NUMERIC (channel shift 25.6) + FACE + CHROMA + HF
- 82760@168: CHROMA (p99 61.5)

A visual check confirmed that all 3 failures were destroyed faces and the passing outputs were clean. The git-ignored sheet is at `experiments/phase5_rapid/outputs/nafnet_route/dev_guard_fails.png`.

## 11. A4 compatibility strategy

- The A4 model code, checkpoint handling, rwgate / T6 / T7 configs and their seals are **unchanged**. The A4 branch of `_process` moved verbatim into `_a4_path`; the one addition is the new `model` field.
- `route_mode=a4` is the pre-R5-5 behaviour: all previous tests pass unchanged, including the Phase 4 pixel-match path and `force_a4`. `force_a4` always takes the A4 path.
- In a running service, A4 can be called per request with `route_mode=a4`. Live check: lazy load followed by an A4 enhancement, 126 s including the load.
- The `EnhancementPipeline` constructor defaults to `route_mode="a4"`, so existing callers and scripts behave exactly as before.

## 12. API changes (additive; existing fields keep their meaning)

- **Request:** optional form field `route_mode` (`""` = service default, `a4`, `nafnet`, `auto`). An unknown value, or nafnet/auto on a service without NAFNet, returns 422 REJECTED.
- **Response:** new fields `route_mode`, `model` (the model actually invoked: `NAFNET_GOPRO`, `A4_SWINIR_L` or null), `artifact_check` (`PASS` / `FAIL` / `NOT_RUN`), and `nafnet` (`{eligible, reason, decision, route: STRONG_LOW_NOISE_BLUR|NAFNET_INELIGIBLE, route_version, infer_s, artifact: {passed, reasons, metrics}}`).
- **NAFNet route values in existing fields:**
  - `used` = `NAFNET_GOPRO`
  - `route` = `NAFNET_GOPRO` (eligible) or `B0`
  - `reason` = `NAFNET_*`
  - `fallback` adds `ARTIFACT_GUARD`; `error` lists the failed checks
  - `output` = `nafnet_face_chip_native`
  - `quality` adds `pixelation`, `hf_peak_frac`, `hf_energy`
- **`/ready`:** adds `route_mode`, `nafnet_model` (weights, SHA-256, params, load time), `nafnet_worker_alive`, `nafnet_worker_restarts`, `seals.nafnet_route`.
- **Version:** `SERVICE_VERSION` is now `fes-service-r5-0.2.0`.

## 13. UI changes (`fes_ui.js` logic only; no layout, HTML or CSS change)

- **Titles and labels** come from the backend `used` / `model` fields:
  - "Enhanced — NAFNet-GoPro (deblur)", with the right-hand label "NAFNet-GoPro (guarded blur route)"
  - "Enhanced (A4)", with "A4 (gate-approved)" or "A4 (forced)"
  - ORIGINAL
- **Artifact-guard fallback:** the message lists the real failed checks.
- **Details panel adds:** Route mode, Model invoked ("none" when no model ran), NAFNet route (eligible/ineligible + reason), Artifact check, Artifact reasons, Chroma change p99, HF colour ratio, NAFNet time.
- **Explanations:** plain-language text for every `NAFNET_*` reason. An unknown reason is shown raw, as before.

## 14. Tests and results

1. **Full suite, Docker `fes-dev:cpu-p5`** (`pytest tests --ignore=tests/test_p5_baseline.py`): **253 passed, 5 skipped, 0 failed** (254 s).
   - Skipped: 3 real-NAFNet tests (opt-in), the real-A4 pixel test (opt-in, unchanged), and the node test (no node in the image).
   - `test_p5_baseline.py` was excluded because it belongs to the uncommitted old plan.
2. **Real NAFNet** (`FES_RUN_NAFNET=1 … pytest tests/test_r5_nafnet.py -k real`): **4/4 passed**:
   - strict offline load with network `connect` blocked
   - missing file, wrong hash and the SIDD file all fail clearly
   - 3 repeated pipeline runs are bit-identical
   - no worker restart, and the same worker PID across requests
3. **UI logic** (`node --test tests/ui/fes_ui.test.mjs`): **21/21 passed**.
4. **Live service** (uvicorn in a throwaway container, `FES_ROUTE_MODE=nafnet`, port 8081):
   - `/ready` 200 with the real NAFNet info
   - DEV strong-blur chip: ENHANCED, NAFNET_GOPRO, ELIGIBLE, PASS, 2.16 s
   - clean reference: ORIGINAL, NOT_BLUR
   - `route_mode=a4`: ENHANCED, A4_SWINIR_L
   - `/ui/` 200
5. **R5-6 runner smoke test** (4 DEV images, real workers, auto):
   - 2 ENHANCED by NAFNet (PASS)
   - 1 NOT_BLUR
   - 1 LOW_DETECTION
   - 0 fallbacks, 0 restarts

**Coverage map against the prompt:**

| Area | Tests |
|---|---|
| Model loading | `test_real_checkpoint_loads_strict_and_offline`, `…missing_or_wrong_checkpoint…` |
| Input gate | strong low-noise blur eligible; high-noise blur, clean, mild and pure noise rejected; pixelated (also blurred+pixelated) rejected; halftone rejected; low face size; no face / low detection / NaN score / yaw / NaN pose; each rule individually; missing features; determinism |
| Artifact guard | normal pass; rainbow → CHROMA + HF; HF colour alone → HF only; 5 invalid-output kinds; channel corruption; no face; greyscale tint |
| Routing | eligible → NAFNet only; ineligible → NAFNet not called; auto keeps A4 only for strong noise; A4 per request and `force_a4`; bad / unconfigured route modes; 4 worker failure modes → ORIGINAL (with restart after timeout/crash); artifact failures → ORIGINAL; guard exception → ORIGINAL; greyscale preserved; full_image mode |
| Regression | all R5-1 / R5-3 / R5-UI / P4 tests unchanged except two 1-line fixture additions |
| Determinism / performance | stable routes and metadata; bit-identical real outputs; load once per worker |

## 15. Runtime and memory

The Docker VM was shared with the running WelOrbit stack during all timings.

| Measure | Value |
|---|---|
| NAFNet load (stage + SHA-256 + strict load) | 8–21 s, once per worker at startup |
| Inference, 224-px chip, 4 threads | 2.2–3.5 s |
| Inference, DEV probe in-process | mean 1.44 s (112–224 px chips); 128-px chips ≈ 0.9 s (R5-ARCHIVE) |
| Gate + guard overhead | ~30–120 ms |
| Service memory, nafnet mode | 0.6 GB |
| Service memory, NAFNet + A4 workers | 1.2 GB |
| A4 per face (unchanged) | ~70 s |

## 16. Known limitations

1. **Black padding in full_image mode on tight crops.** When a `full_image` upload is itself a tight face crop, the FairFace-framed chip extends past the image and is padded with black. The padding edges inflate lapd1, so the gate says `NOT_BLUR`. Observed on the live check: the same DEV blur chip was ENHANCED in `face_chip` mode and ORIGINAL in `full_image` mode. The result is safe (ORIGINAL), but it lowers coverage for cropped inputs such as the 128-px archive images. This is the same effect as the R5-1 note that full_image cropping makes the gate conservative. Not fixed here: it would change features the A4 gate also uses.
2. **Sensitivity of the dedicated halftone rule** is low on its own (6/112 synthetic screens). Halftones are actually rejected by the blur requirement (no halftone reaches lapd1 ≤ blur_max), by detection (0/112 detected) and by the pixelation rule (100/112). Real printed/engraved photos have not yet been tested through the service; that is R5-6.
3. **The guard catches failures but does not prevent them.** NAFNet failed on 3 of 47 DEV strong-blur inputs, all at 168-px chips. All 3 were caught. The real false-negative rate is unknown until R5-6.
4. **Conservative eligibility.** "Strong blur" reuses the rwgate boundary (DEV gaussblur-extreme territory), so moderate blur goes to ORIGINAL even though R5-ARCHIVE saw NAFNet gains across blur levels. Widening it needs DEV evidence; the archive `val` split is untouched and could serve.
5. **Synthetic-only benefit evidence.** NAFNet's identity gain is measured only on synthetic blur with SFace/dlib proxies, not Feature-Match.
6. **Output format.** The output is the deblurred face chip at native size, not the full photo with the face pasted back, which is consistent with the existing A4 contract.

## 17. Deliberately unchanged areas

- A4 model code, adapters and weights registry
- rwgate / T6 / T7 configs and seals
- the worker isolation mechanism (reused as is)
- the UI layout, HTML and CSS
- Feature-Match: not touched
- GFPGAN / CodeFormer: not touched
- no training, fine-tuning or archive training
- no WelOrbit change
- the R5-ARCHIVE data and outputs
- the old-plan uncommitted files

## 18. R5-6 readiness: READY

- **Runner:** `scripts/r5_service_validate.py <dir> --name <run> --route-mode auto|nafnet|a4 --input-mode full_image|face_chip [--save-outputs]` sends any folder through the real pipeline and real workers.
- **Records** come from `Result.validation_record()`: input id, route mode, status, decision, reason, route, used, model, fallback, NAFNet eligibility / reason / inference time, artifact check, reasons and all metrics, output validity, quality features, total ms.
- **Decide before R5-6:**
  - which **input mode** to use for the archive's 128-px crops (see limitation 1). `face_chip` mode matches the R5-ARCHIVE N1 evaluation; `full_image` shows the real-upload behaviour. Running both is recommended.
  - for the 200 real photos, use `full_image`.

## 19. Final verdict

**GREEN.** The implementation is complete. The focused and full suites pass, and the fail-safe behaviour is verified with fake and real models and through the live service. NAFNet is gated, guarded and reversible, and A4 is still available. Next step: **R5-6, service-level validation** (archive test/ztest plus the 200 real photos through the actual service). Production approval is **not** claimed.
