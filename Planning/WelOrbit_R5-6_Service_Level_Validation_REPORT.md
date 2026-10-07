# R5-6: Service-Level Validation and Safety Certification — Report

Date: 2026-10-07 · Task: `Tasks/Face_Enhancement_Phase_5_Rapid_Validation/R5-6_Service_Level_Validation_PROMPT.txt`
Evaluation only. No threshold, gate rule, guard rule, routing, model, preprocessing, API or UI was changed.

## 1. Executive summary

The R5-5 build was validated through the **real FastAPI service**: 3 unmodified uvicorn containers in their default configuration (`FES_GATE=rw`, `FES_ROUTE_MODE=auto`), all requests sent over HTTP. In total, 39,080 requests were sent.

| Dataset | Mode | N | NAFNet out | A4 | ORIGINAL | Guard block (fallback) | HTTP 200 / valid output | Key result |
|---|---|---|---|---|---|---|---|---|
| Archive TEST | face_chip | 19,290 | 7,393 | 0 | 11,897 | 754 | 100% / 100% | NAFNet ΔSFace **+0.149** [0.146, 0.152], Δdlib +0.018; harmed 0.1%; 0 artifacts returned |
| Archive TEST | full_image | 19,290 | 5,654 | 0 | 13,636 | 115 | 100% / 100% | NAFNet ΔSFace +0.053 [0.051, 0.055]; harmed 0.5%; 0 artifacts returned |
| Archive ztest | face_chip | 100 | 0 | 0 | 100 | 0 | 100% / 100% | grainy blur fully contained (87 HIGH_NOISE); 0 flips, 0 new false accepts |
| Archive ztest | full_image | 100 | 0 | 0 | 100 | 0 | 100% / 100% | same (81 HIGH_NOISE) |
| 200 real photos | full_image, auto | 200 | 4 | 2 | 194 | 0 | 100% / 100% | no destructive output; all 11 faces destroyed in R5-ARCHIVE kept off NAFNet |

- **Safety: contained.**
  - NAFNet never reached grainy blur: 0/200 ztest attempts, and all 14 ztest inputs where always-on NAFNet produced artifacts returned ORIGINAL.
  - Every destroyed archive output that NAFNet produced was blocked by the guard.
  - No unsafe output was returned on any dataset.
  - There were 0 worker restarts, 0 timeouts or OOMs, and 0 failed requests.
- **Benefit: reproduced.** In face_chip mode, the service reproduces the R5-ARCHIVE NAFNet gain exactly. On the matched stage-2 pairs, the service's outputs give ΔSFace +0.1622, identical to the offline N1 run. Whole policy vs ORIGINAL: SFace 0.492 → 0.549.
- **Two material coverage/behaviour issues** (they reduce the benefit; they are not safety defects):
  1. **The artifact guard is over-strict.** It blocked 754 face_chip outputs (9.3% of NAFNet runs). Visual inspection and the matched N1 data show most of these are good deblurs with chroma p99 just above 10: median 11.7, and only 7–9 exceed 30. The R5-ARCHIVE N1 run scored the matched blocked pairs at ΔSFace +0.20 with 0 harmed.
  2. **full_image mode is much less effective than face_chip** on the archive crops. It sends 29% fewer inputs to NAFNet, and the gain per enhanced face is about ⅓ (+0.053 vs +0.149). **Black padding is NOT the main cause:** 96.6% of full_image enhanced chips had no padding. The cause is the service's own FairFace re-framing of a 128-px crop, which changes the blur and face-size measurements and the face scale NAFNet sees.

**Verdict: YELLOW** (§17). The service is stable and safe, and the archive benefit is real. The guard calibration and the full_image coverage gap are material and need a focused R5-7 before the real-person Feature-Match validation.

## 2. Baseline and reproducibility

| Item | Value |
|---|---|
| Code commit evaluated | `9f55f333c1dddff8fea1016fa9c65ab06796dd18` (R5-5). `src/` and `configs/` unchanged (verified with `git diff HEAD -- src configs` before the runs). |
| Evaluation scripts | `scripts/r5_6_service_eval.py`, `scripts/r5_6_analyze.py`, `scripts/r5_6_client.sh` (committed with this report) |
| Container image | `fes-dev:cpu-p5` (`sha256:403e0a63ea01…`), CPU only, `CUDA_VISIBLE_DEVICES=""` |
| Service | `uvicorn fes_service.app:app`, 3 replicas (`fes-r56-s1..3`) on a private Docker network; client in a 4th container |
| Service version | `fes-service-r5-0.2.0` |
| Service configuration | defaults: gate rw, route_mode auto, NAFNet timeout 60 s, NAFNet 4 torch threads, A4 timeout 120 s, A4 2 threads, max_inflight 4, warm-up on |
| NAFNet | `models/M28_nafnet/NAFNet-GoPro-width64.pth`, SHA-256 `329d3ab4…701196f`, 67,888,835 params, megvii-research @2b4af71, MIT; production REVIEW_REQUIRED |
| Seals (reported by `/ready` of every replica) | nafnet_route `eb6fe101…`, rw_gate `78837a8d…`, t6 `b8ee2c43…`, t7 `2f026f22…` |
| Datasets | Archive TEST: all 19,290 pairs of `Dataset-and-code/archive` test (sorted names, same as R5-ARCHIVE stage 1). ztest: all 100. Real: the 200 CC0 photos in `Dataset-and-code/dataset`. Matched comparisons reuse the R5-ARCHIVE stage-2 sample (300 test pairs, seed 0). |
| Requests | original file bytes, `return_image=true`, `route_mode` not sent (the service default `auto` applies) |
| Execution | 2026-10-07, start 06:27 UTC (services ready) |

The archive dataset's provenance and licence are still unknown. It was used for evaluation only; no training of any kind was done.

## 3. Frozen configuration

Nothing listed in the task's HARD FREEZE was changed: weights, architecture, NAFNet thresholds, guard thresholds, rwgate, YuNet, A4, route policy, API, UI, pre- and post-processing.

**No blocking implementation defect was found.** Two defects were in my own evaluation client, not in the service:

1. **Round-robin over-subscription.** The first client version could put more than 4 requests on one replica. The service correctly answered 503 BUSY (its documented `max_inflight` backpressure).
2. **A missed kill.** A kill command missed one client container (`docker ps` truncates the command, so my grep never matched), and two clients briefly ran at once.

**Remedy:**
- Each client thread was pinned to one replica.
- All BUSY rows and 28 garbled log lines were dropped, and those inputs were re-requested.
- The 936 face_chip records written during the overlap window are tagged `client_run=overlap_window`. They are excluded from latency statistics but kept for routing, because routing is deterministic.
- Every final record is one HTTP 200 response.

## 4. Archive TEST — face_chip (primary)

| Metric | Value |
|---|---|
| N / HTTP 200 / valid output | 19,290 / 100% / 100% (11,897 ORIGINAL byte-identical to the input; 7,393 valid PNGs) |
| NAFNet eligible = executed | 8,147 (42.2%) |
| NAFNet output returned | 7,393 |
| Guard FAIL → ORIGINAL | 754 (751 CHROMA_P99 only; 3 multi-reason destroyed faces) |
| A4 | 0 |
| UNSAFE / uncertain | 2,272 |

**Gate reasons:**

| Reason | n |
|---|---|
| ELIGIBLE | 8,147 |
| MILD_BLUR | 4,035 |
| LOW_FACE_SIZE | 3,646 |
| UNSAFE_ALIGNMENT | 1,112 |
| LOW_DETECTION | 955 |
| NOT_BLUR | 916 |
| PHOTOMETRIC | 267 |
| NO_FACE | 205 |
| HIGH_NOISE | 5 |
| HALFTONE | 2 |

**Identity** (R5-ARCHIVE methodology: GT landmarks, cosine to the sharp GT, Δ vs B0, bootstrap 95% CI; "clearly" = both evaluators beyond ±0.02):

| Subset | n | SFace B0 → out | ΔSFace (95% CI) | Δdlib (95% CI) | Clearly improved | Clearly harmed | Artifact returned | Face on output |
|---|---|---|---|---|---|---|---|---|
| NAFNet-enhanced | 7,393 | 0.461 → 0.611 | **+0.149** [0.146, 0.152] | **+0.018** [0.018, 0.019] | 2,884 (39.0%) | 6 (0.1%) | 0.0% | 100% |
| Whole policy vs ORIGINAL | 19,290 | 0.492 → 0.549 | +0.057 [0.056, 0.059] | +0.007 [0.007, 0.007] | | | | |

**Matched comparison** (R5-ARCHIVE stage-2 sample, 300 pairs, outputs from the earlier always-on runs):

| Service group | n | Always-on N1 ΔSFace | Always-on A4 ΔSFace | Service output ΔSFace |
|---|---|---|---|---|
| Routed to NAFNet | 114 | +0.162 [0.139, 0.186], 48 improved / 0 harmed | −0.015 | **+0.162** (identical: same model, same chip) |
| Eligible but guard-blocked | 10 | +0.204 [0.10, 0.34], 6 improved / 0 harmed | −0.019 | 0 (ORIGINAL) |
| Not eligible | 176 | +0.160, 79 improved / 0 harmed | −0.014 | 0 (ORIGINAL) |

- The one sample pair that R5-ARCHIVE flagged as an N1 artifact was eligible and **was blocked by the guard** in the service.
- Latency (clean window, 18,354 requests): ORIGINAL median 13 ms server / 28 ms client. NAFNet inference median 0.51 s, p95 3.4 s, max 14.6 s (the max was during the shared-CPU phase). NAFNet request median 0.53 s server.

## 5. Archive TEST — full_image

| Metric | Value |
|---|---|
| N / HTTP 200 / valid output | 19,290 / 100% / 100% |
| NAFNet executed | 5,769 (29.9%) |
| NAFNet output returned | 5,654 |
| Guard FAIL | 115 (101 CHROMA_P99 only; 14 multi-reason) |
| A4 | 0 |
| UNSAFE | 2,408 |

**Gate reasons:**

| Reason | n |
|---|---|
| ELIGIBLE | 5,769 |
| NOT_BLUR | 3,518 |
| MILD_BLUR | 3,237 |
| LOW_FACE_SIZE | 3,135 |
| LOW_DETECTION | 953 |
| UNSAFE_ALIGNMENT | 926 |
| COMPRESSION | 866 |
| NO_FACE | 324 |

**Identity.** Every full_image output was pasted back into the input frame. The chip geometry was recomputed and **verified pixel-exact against the chip the service returned (5,654/5,654)**.

| Subset | n | SFace B0 → out | ΔSFace | Δdlib | Clearly improved | Clearly harmed |
|---|---|---|---|---|---|---|
| NAFNet-enhanced | 5,654 | 0.459 → 0.512 | +0.053 [0.051, 0.055] | +0.008 [0.007, 0.008] | 806 (14.3%) | 28 (0.5%) |
| Whole policy | 19,290 | 0.492 → 0.508 | +0.016 [0.015, 0.016] | +0.002 | | |

On the matched sample, the 74 pairs that full_image routed to NAFNet would gain +0.164 with face_chip-style NAFNet (N1), but the service's full_image outputs gain only +0.062 [0.041, 0.085]. On the same images, NAFNet applied to the service's re-framed full_image chip helps far less than NAFNet applied to the archive crop. Latency: NAFNet inference median 0.57 s, p95 1.34 s.

## 6. face_chip vs full_image (archive TEST)

| Metric | face_chip | full_image |
|---|---|---|
| Total | 19,290 | 19,290 |
| NAFNet eligible | 8,147 | 5,769 (−29.2%) |
| NAFNet executed | 8,147 | 5,769 |
| NAFNet output returned | 7,393 | 5,654 (−23.5%) |
| A4 executed | 0 | 0 |
| ORIGINAL | 11,897 | 13,636 |
| UNSAFE / uncertain | 2,272 | 2,408 |
| Artifact failure | 754 (9.3% of runs) | 115 (2.0%) |
| Fallback (all ARTIFACT_GUARD) | 754 | 115 |
| Success rate (HTTP 200, valid) | 100% | 100% |
| Median client latency | 0.49 s | 0.34 s |
| ΔSFace on enhanced | +0.149 | +0.053 |
| Whole-policy ΔSFace | +0.057 | +0.016 |

**What happens to the 8,147 face_chip-eligible inputs in full_image mode:**

| full_image outcome | n |
|---|---|
| Still eligible | 3,943 |
| NOT_BLUR | 1,366 |
| MILD_BLUR | 1,185 |
| LOW_FACE_SIZE | 789 |
| COMPRESSION | 470 |
| LOW_DETECTION | 156 |
| UNSAFE_ALIGNMENT | 154 |
| PHOTOMETRIC | 66 |
| HALFTONE | 9 |
| NO_FACE | 9 |

A further 1,826 inputs are eligible only in full_image mode.

**Black padding.** 5,460 of 5,654 enhanced full_image chips (96.6%) had zero padding. The coverage loss comes mainly from the FairFace re-framing: the chip is a different crop or scale of the 128-px image, so lapd1, noise, IOD and blockiness change. Some NOT_BLUR outcomes are padding-driven, but padding explains only part of the loss. **Black padding is real but not the dominant factor.** Crop logic was not changed (R5-7 item).

## 7. ztest (100 identities, grainy blur, median noise σ 1.75)

- **face_chip:** 0 NAFNet attempts. Reasons: HIGH_NOISE 87, UNSAFE_ALIGNMENT 10, LOW_FACE_SIZE 2, PHOTOMETRIC 1.
- **full_image:** 0 NAFNet attempts. HIGH_NOISE 81.
- A4 was used 0 times, and there were 0 fallbacks.
- **R5-ARCHIVE cross-check.** All 14 ztest inputs on which always-on NAFNet produced colour artifacts returned ORIGINAL:
  - face_chip: 11 HIGH_NOISE, 2 alignment, 1 size
  - full_image: 13 HIGH_NOISE, 1 alignment
- **Verification** (FAR 1e-2 thresholds from clean GT impostor pairs): the service equals B0 exactly. SFace TAR 1.00, dlib TAR 1.00, 0 flips, 0 new false accepts.
- **Visual inspection:** there are no NAFNet ztest outputs to inspect.

**Grainy and high-noise blur is fully kept away from NAFNet.**

## 8. 200 real photos (full_image, auto)

| Outcome | n |
|---|---|
| NAFNet (guard PASS) | 4 |
| A4 (strong_noise) | 2 |
| ORIGINAL | 194 |
| of which UNSAFE / uncertain | 75 |
| Artifact failures | 0 |
| Fallbacks | 0 |
| Failed requests | 0 |
| Worker restarts or crashes | 0 |

**Reasons:**

| Reason | n |
|---|---|
| LOW_FACE_SIZE | 52 |
| NOT_BLUR | 36 |
| no_valid_face (detector) | 32 |
| HIGH_NOISE | 25 |
| LOW_DETECTION | 17 |
| NO_FACE | 15 |
| UNSAFE_ALIGNMENT | 11 |
| ELIGIBLE | 4 |
| PIXELATION | 3 |
| MILD_BLUR | 2 |
| strong_noise → A4 | 2 |
| COMPRESSION | 1 |

**R5-ARCHIVE cross-check.** The 11 real photos that always-on NAFNet destroyed:

| Photo(s) | Service outcome |
|---|---|
| img140 | COMPRESSION |
| img25, img20 | HIGH_NOISE |
| img104, img60 | NO_FACE |
| img196 | LOW_FACE_SIZE |
| img113, img171 | LOW_DETECTION |
| img146 | UNSAFE_ALIGNMENT |
| img165, img72 | routed to **A4** (strong_noise), not NAFNet |

No identity ground truth exists for these photos, so no identity claim is made.

## 9. Visual safety

I inspected every real-photo enhanced output, plus samples of the archive outputs. The inspection sheets are git-ignored, under `experiments/phase5_rapid/outputs/r5_6/`.

| Input | Route | Finding |
|---|---|---|
| real/img147 | NAFNet | Extreme defocus. Slightly darker eyes and lip edges; no artifacts; no destruction. |
| real/img148 | NAFNet | Extreme defocus. Similar mild sharpening of the eyes; no colour shift. |
| real/img153 | NAFNet | Artistic double-exposure / motion ghosting, greyscale. Almost unchanged; stays greyscale. |
| real/img154 | NAFNet | Motion streaks, greyscale. Slightly crisper streaks; no artifacts; stays greyscale. |
| real/img165 | A4 | **Halftone-printed** photo, measured as strong noise (σ 20.6). A4 smooths the dot screen into a plausible face. Not destructive, but this is A4 reconstructing a printed image. |
| real/img72 | A4 | Grainy vintage print (σ 14.3). Smoother, plausible, slightly higher contrast; no visible identity change. |
| archive, 6 "clearly harmed" (face_chip) | NAFNet | Subtle warping, ringing or smearing on extreme blur (worst: an elderly face, ΔSFace −0.19). No colour artifacts or destroyed faces. |
| archive, random enhanced sample | NAFNet | Natural, visible deblurring; no artifacts. |
| archive, guard-blocked sample (13) | (blocked) | 10 random chroma-only blocks (p99 10.1–15.5) are **good deblurs**: false positives. The 3 multi-reason blocks are **destroyed faces** (rainbow pixel blocks, p99 ≈ 120): true positives. |

No black padding, crop misalignment, clipping, or greyscale corruption was seen in any returned output. ORIGINAL outputs are byte-identical to the input, checked for every request.

## 10. Artifact guard

- **Every block reproduced:** re-running NAFNet in-process on every blocked input gave the same guard reasons (754/754 face_chip, 115/115 full_image).
- **True positives:** every destroyed output was caught (multi-reason, chroma ≈ 120).
- **No artifact was returned:** 0% of returned outputs exceed the R5-ARCHIVE artifact signal, and all returned outputs have chroma p99 ≤ 10 by construction (face_chip median 5.4, full_image median 3.2).
- **Over-strict:**

| Blocked outputs by chroma p99 | face_chip (754) | full_image (115) |
|---|---|---|
| > 15 | 105 | 34 |
| > 20 | 21 | 29 |
| > 30 | 9 | 23 |
| > 50 | 7 | 17 |
| multi-reason | 3 | 14 |

In face_chip mode, roughly 650–700 of the 754 blocks are legitimate deblurs whose colour edges sharpen past the 10-unit chroma limit. The cost is lost benefit (the matched blocked pairs would gain about +0.20 SFace), not a safety risk. The threshold was **not** changed (R5-7).

## 11. A4 routing (auto)

- **Archive:** 0 A4 invocations across 38,780 archive requests. There is no blur → A4 route in auto, as designed.
- **Real photos:** 2 invocations (img165, img72), both for `strong_noise`. A4 is never invoked for blur.
- **Request latency:** 35.3 s and 66.8 s. No timeout or failure.
- **A4 stays in its strong-noise role,** but both real invocations were **printed photos**: a halftone screen and a print grain that the noise estimator reads as noise. The NAFNet route's pixelation and halftone rules do not apply to the A4 route. This is worth a small, separate A4 investigation, but A4 is rare (1%) and the 2 outputs were not destructive. A4 was not removed.

## 12. Runtime and resources

| Measure | Value |
|---|---|
| Startup to ready | ~60–75 s per replica (A4 load 6–9 s, NAFNet load 16–27 s, warm-up) |
| NAFNet inference, 128-px chips, CPU free | median 0.51–0.57 s, p95 1.3–3.4 s |
| NAFNet inference, real photos (224-px chips; WelOrbit was running) | median 4.7 s, max 6.3 s |
| A4 request | 35–67 s |
| ORIGINAL request | 13–35 ms server |
| Memory per replica (both workers loaded) | 0.70–1.3 GB |
| Throughput | 3 replicas × 4 threads: ~13 requests/s on the archive with the CPU free; ~1.3 requests/s while WelOrbit was running (CPU contention; the user stopped WelOrbit mid-run) |
| Timeouts / OOM / worker restarts / crashes / failed requests | 0 / 0 / 0 / 0 / 0 (`/ready` restart counters 0 on all replicas at the end) |
| Network / model downloads | none (offline; weights from the local mount, SHA-verified) |

Latencies measured while WelOrbit was running are not representative of an unshared CPU.

## 13. Failures and fallbacks

- There were no service failures. All fallbacks were `ARTIFACT_GUARD`: 754 face_chip and 115 full_image on the archive, 0 on ztest and real photos.
- There were no TIMEOUT, MODEL_FAILURE, RESOURCE_FAILURE or INVALID_OUTPUT fallbacks.
- BUSY responses occurred only from the first client's over-subscription (§3). This was the correct backpressure, and the rows were re-requested.

## 14. Identity proxy analysis

These are SFace/dlib **proxies against the sharp GT of the same photo**, on synthetic blur. They are **not Feature-Match** and not cross-photo recognition.

- **Archive (face_chip):** the guarded route keeps the full NAFNet gain on the faces it enhances (+0.149, with 39% clearly improved and 0.1% clearly harmed). On the matched pairs it equals the offline N1 run.
- **Archive (full_image):** the gain is real but about ⅓ of that.
- **ztest:** identity is unchanged, because nothing was enhanced.
- **Real photos:** no ground truth, so no claim.
- **No Feature-Match benefit is claimed.**

## 15. Limitations

- The archive is synthetic blur with unknown provenance and licence. Identity is measured by proxies.
- Only 4 real photos reached NAFNet, so real-photo safety evidence for the NAFNet route is small.
- The real-photo set has no same-person references.
- Part of the runtime data was collected under CPU contention.
- 936 face_chip latency records come from the client-overlap window and are excluded from latency statistics.

## 16. R5-7 issues (quantified; nothing tuned here)

1. **Artifact-guard chroma threshold is over-strict.** 9.3% of face_chip NAFNet outputs are blocked, about 90% of them good deblurs; true destruction sits at chroma ≈ 120. Recalibrate on **DEV data only**: the untouched archive `val` split plus the R5-5 DEV probe. Consider a threshold or a relative measure that separates 10–16 (edge sharpening) from 60+ (destruction).
2. **full_image framing loses coverage and benefit.** −29% eligibility and ⅓ of the gain on the archive crops. Investigate how the FairFace re-crop of small or tight inputs affects the gate features and the NAFNet input scale. Black padding is only a minor part.
3. **(Minor) The A4 strong_noise route admits printed photos** (halftone / print grain), 2/200 real photos. Consider applying the NAFNet route's texture rules to the A4 route as well, or document it as accepted.

## 17. Final verdict: **YELLOW**

| GREEN criterion | Status |
|---|---|
| Stable service | ✅ 39,080 requests: 0 failures, 0 restarts, 0 timeouts |
| Routing behaves as designed | ✅ |
| Unsafe NAFNet cases contained | ✅ ztest 0/100; all 11 earlier real-photo destructions kept off NAFNet |
| Artifact guard works | ✅ for safety (all destroyed outputs caught, 0 artifacts returned) — ⚠ over-blocks good outputs |
| Archive service path preserves meaningful NAFNet benefit | ✅ face_chip (+0.149, equal to offline) — ⚠ full_image only +0.053 |
| No unacceptable destructive real-photo behaviour | ✅ |
| No blocking safety defect | ✅ |

There is **no safety defect**. YELLOW rather than GREEN because two material coverage/behaviour issues (§16, items 1–2) cut the delivered benefit by up to two-thirds in the realistic full_image mode, and they need a focused R5-7. This verdict is not tuned to the test results, and no production or commercial approval is claimed: licensing remains REVIEW_REQUIRED.

## 18. Answers and next-task recommendation

1. **Is NAFNet safe enough to continue?** Yes. It is gated and guarded, no unsafe output was returned, and every destroyed output was blocked.
2. **Does the real service reproduce the archive benefit?** Yes in face_chip mode (identical: +0.162 on the matched pairs, +0.149 over the whole test split). Only partly in full_image mode (+0.053).
3. **Does the gate contain grainy or high-noise blur?** Yes, completely: 0/200 ztest NAFNet attempts, and all 14 earlier artifacts returned ORIGINAL.
4. **How much does full_image reduce coverage?** −29% NAFNet eligibility and −24% enhanced outputs, with about ⅓ of the per-face gain. Whole-policy ΔSFace drops from +0.057 to +0.016.
5. **Is black padding material?** Not much: 96.6% of full_image enhanced chips have none. The re-framing itself is the main factor.
6. **How often is A4 used?** 0 of 38,780 archive requests; 2 of 200 real photos (1%), both strong noise.
7. **Does A4 need separate investigation?** Small, low priority: its noise route admits printed photos.
8. **Is the service ready for actual Feature-Match validation?** Operationally yes: stable, safe, and metadata complete. It is still blocked on Feature-Match access and ≥ 20 real identities (R5-4). It is better done after R5-7 item 1, so that real people are not routed through a guard that throws away ~9% of the benefit.
9. **Smallest justified R5-7:** recalibrate the artifact-guard chroma rule on **DEV data only** (archive `val` + the DEV probe; a single sealed config bump), and measure the full_image framing effect on the gate features. Re-run the R5-6 harness (`scripts/r5_6_service_eval.py`) unchanged to confirm. After that comes Feature-Match validation: ≥ 20 people, clear references, 1–3 degraded images per person, the real Feature-Match model and its real threshold.
