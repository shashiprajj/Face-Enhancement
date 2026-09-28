# Face Enhancement Service — Phase 0 Master Specification

| Field | Value |
|---|---|
| Document | Authoritative Phase 0 specification |
| Version | 1.0 |
| Date | 2026-09-28 |
| Sources | P0-T1, P0-T2, P0-T3, P0-T4, P0-T5 (all in `docs/phase_0/`) |
| Precedence | Where this document and a source document differ, **this document governs**, and every difference is listed in §2 (Conflict Register). |

---

## 1. Product Objective

The **Face Enhancement Service (FES)** is a standalone service that takes an image containing a human face and returns a **faithfully restored face image plus quality metadata**. It targets low resolution, blur, noise, compression artifacts and (where it can be done safely) poor lighting and contrast.

FES is an **image enhancement/restoration service**. It is **not** a recognition, matching, or verification service and never makes identity decisions. *(P0-T1 §3)*

**Governing principle:** faithful restoration over visual attractiveness. Identity information must be preserved, never invented. *(P0-T3 §1, §6)*

## 2. Conflict & Overlap Register

All earlier decisions stand except where resolved below. No earlier decision was changed silently.

| # | Documents | Conflict / overlap | Resolution |
|---|---|---|---|
| C1 | P0-T1 X3 vs P0-T4 §3B/§5 | T1 puts embedding generation out of scope; T4 uses face embeddings to measure identity preservation. | **No conflict in substance.** Embeddings are used **only inside the offline evaluation harness**. They are never part of the FES runtime, API, or output. T1 X3 already allowed this. |
| C2 | P0-T2 §2.6 (crop mode, 0 faces → process whole crop) vs P0-T3 E1 ("never fabricate a face") | A face-restoration model run on a crop with **no face in it** could invent a face, which would break E1. | **Refinement of P0-T2.** In `face_crop` mode, FES first runs the face detector at a relaxed threshold (TBD). If a face is detected but landmarks fail → process unaligned with warning `no_landmarks_found` (T2 behavior kept). If **no face is detected at all** → reject with `NO_FACE_DETECTED`. |
| C3 | P0-T2 §2.4 ("no model-specific dimensions fixed") vs P0-T3 §2 (pixel ranges for DG1 severity) | T3 gives pixel sizes for face-resolution severities. | **No conflict.** T3's pixel values are *starting points for building synthetic test data* and are explicitly subject to calibration (T4 TH-6). They are not service input requirements. |
| C4 | P0-T2 §4.2 / P0-T3 R-D8 (determinism) vs possible stochastic (e.g., diffusion) candidates | Some candidate model families are not deterministic by default. | Determinism is a **requirement** for the service: stochastic models must run with a fixed seed per request (seed derived from content hash + version), or be marked non-compliant with R-D8. This counts in model selection (T4 E7). |
| C5 | P0-T1 A3 (GPU primary, CPU fallback possible) vs P0-T5 §4 (must be runnable on CPU for dev) | T5 adds a CPU-runnability requirement. | **Both hold.** The dev machine has no NVIDIA GPU, so candidates must run on CPU **for development and testing**. *Production* CPU serving remains optional (T5 Q5-3). |
| C6 | P0-T2 error `ENHANCEMENT_NOT_RECOMMENDED` vs P0-T3 §3 (Extreme: flag **or** decline) | Two possible behaviors for extreme inputs. | **Deferred, bounded:** both are allowed by the contract; the choice (D-6) is made in Phase 1 from evaluation data (T4 TH-7). Silent hallucinated output is never allowed (U8). |
| C7 | P0-T1 S6 (lighting/contrast "conditional") vs P0-T3 DG6–DG7 and P0-T4 B2 (CLAHE baseline) | Unclear whether lighting correction is a product capability or only a baseline. | Lighting/contrast correction is **optional in v1**. It is included only if Phase 1 shows it improves results without affecting identity (U3: no skin-tone shift). Open question Q-3 remains. |
| C8 | P0-T2 metadata `input_quality_score`/`fidelity_risk` vs P0-T4 §4 (limits of no-reference IQA) | Runtime quality scores are proposed, but T4 notes no-reference metrics are unreliable. | Runtime scores are **"should have"** and advisory only. They are included only after they are checked against human review in Phase 1. They must never be presented as identity confidence. |

## 3. Scope

**In scope** *(P0-T1 §4)*: face restoration (core); low-resolution/super-resolution, blur (defocus and motion), noise, and compression-artifact reduction (core); lighting/contrast (conditional, C7); face detection, cropping, and alignment (supporting); input/output quality assessment; standardized output; HTTP API (**later phase**).

**Out of scope** *(P0-T1 §5)*: recognition/matching; 1:1 and 1:N verification; embeddings as output; vector/DB storage; PAN/KYC/document processing; **Feature-Match integration**; client-specific rules; training a foundation model from scratch; liveness/deepfake detection; attribute estimation; beautification, stylization, or face swap; video; general non-face restoration as a goal.

**Feature-Match boundary:** Feature-Match is external. It was not inspected, imported, or depended on. It may become a consumer through the generic versioned API only. *(P0-T1 §7, P0-T5 §9)*

## 4. Input Requirements *(P0-T2 §2, refined by C2)*

- **One still image per request.** Accepts **both full images and face crops** (`input_mode`: `auto` (default), `full_image`, `face_crop`).
- **Formats:** JPEG, PNG, WebP (initial); BMP (candidate); TIFF and HEIC (deferred); GIF, animated formats, and RAW (no). Format is detected by magic bytes.
- **Encoding:** decoded to 8-bit sRGB RGB. EXIF orientation is applied. ICC profiles are converted. Grayscale is accepted and never colorized. CMYK is converted or rejected.
- **Limits:** encoded file size, decoded pixel count, maximum and minimum side length, and minimum face size are all **TBD** (D-1, D-3). Limits are enforced before full decode.
- **Face count:** default `largest`. Other options: `all` (up to a max, TBD) and `index`. 0 faces → `NO_FACE_DETECTED` (crop-mode rule in C2).

## 5. Output Requirements *(P0-T2 §4–§6)*

- **Default:** aligned, restored face crop at a standard size (TBD by model). **Optional:** full image with the restored face pasted back, or both.
- PNG by default (JPEG/WebP optional); 8-bit sRGB; **all EXIF/metadata stripped**.
- **Metadata (must have):** `request_id`, `status`, service/API version, model name/version, face counts, per-face bbox/confidence/size category, `warnings[]`, processing time. **Should have:** input/output quality scores, degradation estimate, dimensions, format, fidelity risk (advisory, C8).
- **Never returned:** embeddings, identity scores, match results, consumer-specific fields.
- **Errors:** client (`INVALID_INPUT`, `UNSUPPORTED_FORMAT`, `UNSUPPORTED_COLOR_MODE`, `FILE_TOO_LARGE`, `IMAGE_TOO_LARGE/SMALL`, `DECODE_ERROR`, `INVALID_FACE_INDEX`); content (`NO_FACE_DETECTED`, `FACE_TOO_SMALL`, `ENHANCEMENT_NOT_RECOMMENDED`); server (`PROCESSING_ERROR`, `TIMEOUT`, `RESOURCE_EXHAUSTED`, `SERVICE_UNAVAILABLE`).
- **Processing flow:** validate → decode/normalize → input QA → detect + landmarks → select face(s) → align/crop → restore → output QA and guardrails → compose/encode. *(ASCII diagram in P0-T2 §3)*

## 6. Degradation Requirements *(P0-T3 §2–§3)*

| Degradation | Primary target | Secondary | Not targeted |
|---|---|---|---|
| Low resolution, defocus blur, noise, JPEG artifacts, combined | Mild–Moderate | Severe | Extreme (flag or decline) |
| Motion blur | Mild–Moderate | Severe (best effort) | Extreme |
| Poor lighting, low contrast | Conditional (C7) | — | — |

Edge cases *(P0-T3 §8)*:
- **No face:** reject.
- **Multiple faces:** handled per face, following the face-selection policy.
- **Tiny face:** warn or reject.
- **Extreme degradation:** warn or decline; never silent.
- **Occlusion:** keep the occluder; do not invent the hidden features.
- **Unusual pose:** process and warn.
- **Already high-quality input:** change it as little as possible.

## 7. Identity-Preservation Principle *(P0-T3 §5–§6)*

1. Identity preservation is a **hard gate**. A model that fails it is ineligible, whatever its visual quality.
2. **Critical defects:** identity change (U1), fabricated facial details (U2), and attribute shifts such as age, gender presentation, ethnicity, or skin tone (U3). **High-severity defects:** beautification, changed expression, filled-in occlusions, silent hallucination, and demographic unfairness.
3. Identity is measured with **≥ 2 independent, test-only face-identity evaluators**. Feature-Match is never used for this.
4. Where a model has a fidelity/quality trade-off, the default favors fidelity.
5. Quality and identity scores are **reported separately**. They are never combined into one score.

## 8. Evaluation Methodology *(P0-T4)*

- **Dimensions:** A quality · B identity (gate) · C performance · D reliability. Candidates are ranked lexicographically: B → D → C → A.
- **Data:** synthetic degradations (seeded, versioned pipeline), real-world degraded images, identity-paired sets, an edge-case suite, and fairness strata. Test data must not overlap the candidates' training data (e.g., not FFHQ only).
- **Metrics:**
  - Quality: PSNR, SSIM, LPIPS (with DISTS, FID/KID, and no-reference IQA as candidates) plus blind human review.
  - Identity: ID similarity, ID gain, identity-drop rate, FAR change, attribute consistency.
  - Performance: latency, P95/P99, throughput, memory/VRAM, cold start.
  - Reliability: failure, timeout, and OOM rates; determinism; soak stability.
- **Baselines:** B0 degraded input, B1 classical upscaling, B2 classical restoration, B3 general learned SR, B4 face-specific candidates.
- **Fairness of comparison:** shared pre/post-processing, validation/test splits, same hardware, bootstrap confidence intervals, paired tests, stratified reporting, blind review, full reproducibility records.
- **Thresholds:** all are TBD (TH-1 … TH-11). They are set from noise floors and baselines, and fixed **before** candidate results are compared.

## 9. Performance Requirements *(P0-T4 §3C, P0-T5 §5)*

- Measure per-stage and end-to-end latency (mean, P95, P99), throughput, peak RAM/VRAM, and cold start, across face-size × batch × concurrency × precision × runtime.
- **Working goal (non-binding):** "interactive" synchronous latency on a GPU. CPU is expected to take several seconds or more per face.
- **Final targets are TBD** (TH-8). They depend on consumer SLAs (Q-1) and measurements on the production-class GPU (Q-2).

## 10. Reliability Requirements *(P0-T4 §3D, §8)*

- Every valid request ends in a defined status. Unhandled crashes count as failures.
- Overload degrades gracefully to `RESOURCE_EXHAUSTED`, not to crashes.
- Output is deterministic (C4).
- Fuzz, load, and soak tests must pass. Failure and timeout rates are TBD (TH-9).
- Warnings (`extreme_degradation`, `face_too_small`, etc.) must actually be issued when they apply. Their precision is measured.

## 11. Hardware / Deployment Constraints *(P0-T5 §1–§4)*

- **Dev machine (fact):** Windows 11, Python 3.11.0, Intel Core Ultra 7 255H (16C), ≈ 31.4 GB RAM, **Intel Arc 140T iGPU only — no NVIDIA GPU, no CUDA, no dedicated VRAM**, ≈ 229 GB free on D:, Docker 29.8.0 via WSL2. Not a git repository.
- **Targets:** local dev (CPU/iGPU), Docker (Linux containers), NVIDIA GPU server (assumed primary production target), and optional CPU fallback.
- **Requirements:**
  - Candidates must run on CPU for development (C5).
  - Candidates should not depend only on custom CUDA kernels.
  - Memory per worker must be bounded.
  - Local timings are not representative of production.

## 12. Licensing / Commercial Constraints *(P0-T5 §6)*

Licensing is a **hard selection gate**. For every candidate, check code, weights, training-data inheritance, detector/landmark models, test-only identity evaluators, third-party runtime dependencies, and redistribution in container images, each from the primary license texts. Missing evidence makes a candidate ineligible. Many open face-restoration and face-analysis components are non-commercial or research-only, so each one must be verified individually.

## 13. Security / Operational Constraints *(P0-T5 §7–§8)*

- **Input handling:** size limits applied before decode, magic-byte format validation, patched codecs, and a non-root container.
- **Privacy:** in-memory processing; temp files (if any) private and always deleted; **no retention by default**; EXIF/GPS stripped; TLS in transit.
- **Logging:** logs never contain image content or base64 payloads. Structured logs carry `request_id`, timings, status, and model version.
- **Monitoring:** latency histograms, error rates by category, queue depth, and GPU/CPU/memory usage.
- **Model files:** safe weight formats (safetensors/ONNX) preferred; checksummed weights; pinned dependencies; SBOM; image scanning.
- **Access control and abuse limits:** authentication, rate limits, bounded queues, timeouts.
- **Provenance:** outputs are marked as enhanced in the metadata; watermarking is an open question.

## 14. Deferred Decisions

| ID | Decision | Target phase | Source |
|---|---|---|---|
| D-1 | File-size, pixel, and side-length limits | API phase | T2 D2-1 |
| D-2 | Standard output face size | Phase 1 | T2 D2-2 |
| D-3 | Minimum face size, and reject vs. warn | Phase 1 | T2 D2-3, T4 TH-6 |
| D-4 | Detector, confidence thresholds, auto-mode heuristic, crop-mode relaxed threshold (C2) | Phase 1/2 | T2 D2-4 |
| D-5 | Max faces per request | API phase | T2 D2-5 |
| D-6 | Extreme degradation: flag vs. decline | Phase 1 | T3 Q3-1, C6 |
| D-7 | Lighting/contrast correction inclusion and method | Phase 1 | T3 Q3-2, C7 |
| D-8 | Exposing a fidelity/quality option to consumers | Phase 1/2 | T2 D2-10, T3 Q3-3 |
| D-9 | Runtime quality-score / fidelity-risk methods | Phase 1 | T2 D2-9, C8 |
| D-10 | Transport, sync/async, batch endpoint, HTTP status mapping | API phase | T2 D2-6 |
| D-11 | HEIC/TIFF, alpha, 16-bit/grayscale output preservation | Later | T2 D2-7, D2-8 |
| D-12 | All numeric thresholds TH-1 … TH-11 | Phase 1 | T4 §10 |
| D-13 | Production GPU class, runtime/export format, precision | Phase 1/2 | T5 |
| D-14 | Production CPU fallback: yes/no | Phase 1 | T5 Q5-3 |
| D-15 | Model choice | Phase 1 | — |

## 15. Assumptions

| ID | Assumption | Confirmed? |
|---|---|---|
| A-1 | Suitable pretrained restoration models exist | No — Phase 1 survey |
| A-2 | Still images only, one image per request | Proposed |
| A-3 | Containerized deployment; NVIDIA GPU server as primary production target | **Not confirmed** |
| A-4 | Commercial use intended → licenses must permit commercial use | **Not confirmed** (treated as requirement) |
| A-5 | Human faces only | Proposed |
| A-6 | Face images are privacy-sensitive; no retention | Treated as requirement |
| A-7 | Linux production OS | Not confirmed |
| A-8 | No prior code to preserve (repo holds only Phase 0 material) | Fact |

## 16. Risks

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R-1 | **Identity hallucination** by generative restoration models | High | Critical | Identity gate; ≥ 2 evaluators; human review; fidelity-first defaults; extreme-input guardrails |
| R-2 | **No commercially licensed model** of adequate quality | Medium | High | License check first; consider permissively licensed alternatives, classical pipelines, or (later) fine-tuning a permissive architecture on licensed data |
| R-3 | **No NVIDIA GPU available** for Phase 1 benchmarking | High (currently true) | Medium | Secure GPU access early (Q-2); do quality/identity work on CPU in the meantime |
| R-4 | Evaluator bias / metric gaming | Medium | High | Evaluators from different families; exclude evaluators matching a candidate's training loss; human review |
| R-5 | Synthetic-to-real domain gap | High | Medium | Mandatory real-world test set |
| R-6 | Demographic performance disparity | Medium | High | Stratified evaluation; fairness gate |
| R-7 | Privacy/regulatory non-compliance | Medium | High | No retention, no image logging; legal review (Q-5) |
| R-8 | Consumer misuse of enhanced output as original evidence | Medium | High | `enhanced: true` metadata; provenance/watermark decision (Q-6); consumer guidance |
| R-9 | Latency/cost too high for consumer SLAs | Medium | Medium | Measure early; export/FP16 options; smaller models |
| R-10 | Scope creep toward recognition/Feature-Match coupling | Low | High | Hard boundary in §3; contract review |
| R-11 | Test-set contamination with candidates' training data | Medium | Medium | Dataset provenance checks (T4 §2) |

## 17. Consolidated Open Questions

| ID | Question | Owner | Blocks |
|---|---|---|---|
| Q-1 | First consumers, volume, latency SLA | Product | Performance targets |
| Q-2 | Access to an NVIDIA GPU (class/VRAM/CUDA) for Phase 1 | Infra | Performance evaluation |
| Q-3 | Lighting/contrast correction wanted? | Product | D-7 |
| Q-4 | Is commercial use confirmed? Who does license review? | Product / Legal | Candidate shortlisting |
| Q-5 | Applicable data-protection obligations and data residency | Legal / Compliance | Production |
| Q-6 | Must enhanced output carry a watermark/provenance marker? | Product / Compliance | API phase |
| Q-7 | Human-review panel availability | Product | Human evaluation |
| Q-8 | Evaluation datasets with suitable licenses/consent; fairness attributes | Data / Legal | Phase 1 evaluation |
| Q-9 | Fine-tuning allowed later, and in-domain data available? | Product / Data | Later phases |
| Q-10 | Is CPU production fallback needed? | Product | D-14 |

## 18. Phase 1 Prerequisites

Phase 1 = candidate survey, license screening, evaluation harness, and benchmarking. It should start with:

1. **Sign-off** on this specification by the product owner. In particular, confirm A-3 and A-4 and the in/out-of-scope lists.
2. **Version control:** `git init` the repository and commit the Phase 0 docs as the baseline.
3. **Licensing reviewer** named, and the commercial-use requirement confirmed (Q-4).
4. **GPU access plan** (Q-2). Phase 1 can start without it, but final model selection cannot.
5. **Evaluation data plan:** candidate datasets with license and consent status, including a real-world degraded set (Q-8).
6. **Project-local Python environment** (no reliance on global packages). Downloading models and datasets becomes permitted in Phase 1 only after the license screen.
7. **Evaluation plan document**, with thresholds methodology, repeated-run counts, and α, fixed **before** candidate results (T4 TH-11).
