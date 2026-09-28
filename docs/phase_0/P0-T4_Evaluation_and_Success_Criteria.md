# P0-T4 — Evaluation & Success Criteria

| Field | Value |
|---|---|
| Phase / Task | Phase 0 / P0-T4 |
| Status | Complete |
| Date | 2026-09-28 |
| Depends on | P0-T1, P0-T2, P0-T3 (reviewed) |
| Nature | Evaluation design only — no benchmarks implemented, no models downloaded. **No final numeric thresholds invented**; all are TBD with a defined method. |

---

## 1. Evaluation Dimensions

| Dim | Name | Question answered | Gating? |
|---|---|---|---|
| **A** | Image / visual quality | Is the output measurably better and free of artifacts? | Yes (must beat baselines) |
| **B** | Identity preservation | Does the output keep the same person — and not become someone else? | **Yes — hard gate** (P0-T3 §6) |
| **C** | Inference / system performance | Is it fast and resource-efficient enough for the deployment target? | Yes (must meet P0-T5 targets once set) |
| **D** | Operational reliability | Does it fail rarely, predictably, and safely? | Yes |

Dimensions are reported **separately**. No composite score allows one dimension to compensate for failure in another. Model ranking uses a *lexicographic* rule: pass B → pass D → meet C → best A (see §9).

## 2. Evaluation Data (design)

No datasets are downloaded in Phase 0. Phase 1 will assemble:

| Set | Purpose | Ground truth? |
|---|---|---|
| **S-SYN (synthetic)** | High-quality face images degraded with controlled DG1–DG8 at each severity (P0-T3 §2). | Yes → full-reference metrics |
| **S-REAL (real-world degraded)** | Naturally degraded faces (old photos, low-light, CCTV-like, messaging recompression). | Usually no → no-reference + identity vs other images of same person |
| **S-ID (identity pairs)** | Multiple images per identity, so one clean image can serve as identity reference for another degraded image. | Identity labels |
| **S-EDGE** | P0-T3 edge cases (no face, multi-face, tiny, occluded, profile, grayscale). | Behavioral |
| **S-FAIR** | Stratified subsets by skin tone / gender / age group. | Yes |

Data requirements: license must permit use for evaluation (and commercial if applicable); consent/privacy compliance; held-out from any model training data where known (to avoid contamination — many restoration models were trained on FFHQ, so FFHQ-derived test data is **biased in their favor** and must not be the only test set).

Degradation pipeline must be **fixed, versioned, and seeded** so every model sees identical inputs.

## 3. Candidate Metrics

### A. Image / visual quality

| Metric | Type | What it measures | Use |
|---|---|---|---|
| **PSNR** | Full-reference | Pixel-level fidelity | S-SYN |
| **SSIM** (and MS-SSIM) | Full-reference | Structural similarity | S-SYN |
| **LPIPS** | Full-reference, learned | Perceptual similarity to GT | S-SYN (primary perceptual FR metric) |
| **DISTS** (candidate) | Full-reference, learned | Structure + texture similarity | S-SYN, complementary |
| **FID / KID** (candidate) | Distribution-level | Realism of output distribution | S-SYN, S-REAL (needs many samples) |
| **NIQE / BRISQUE / MUSIQ / CLIP-IQA** (candidates) | No-reference | Estimated naturalness/quality | S-REAL; also for runtime input/output QA (P0-T2 §5) |
| **Landmark distance** (candidate) | Geometry | Landmark shift between GT and output | S-SYN; detects geometry changes (R-D2) |
| **Human review (blind MOS / pairwise)** | Subjective | Artifacts, fabrication, naturalness | Curated subset; required for U2/U4/U5/U6 |

### B. Identity preservation

| Metric | Definition |
|---|---|
| **ID similarity to GT** | cos-sim(emb(output), emb(ground truth)) — S-SYN |
| **ID gain** | cos-sim(output, GT) − cos-sim(degraded input, GT). Must be ≥ 0 on average and rarely strongly negative. |
| **ID similarity to reference** | cos-sim(output, other clean image of same identity) — S-ID / S-REAL |
| **Verification-style rates** | On S-ID: TAR at fixed FAR, comparing *degraded* vs *enhanced* inputs. Enhancement must not increase FAR (impostor pairs becoming more similar = hallucination toward a "generic" face). |
| **Identity-drop rate** | % of samples whose ID similarity drops by more than δ (δ TBD) vs input. Catches rare catastrophic identity swaps that averages hide. |
| **Attribute consistency** (test-only) | Change in estimated age/skin tone between GT and output. |

### C. Inference / system performance

| Metric | Notes |
|---|---|
| **Latency** (mean, P50) | Per stage and end-to-end, per face-size category |
| **P95 / P99 latency** | Tail latency is the operational target |
| **Throughput** | Images/s at steady state for a given concurrency / batch size |
| **Memory / VRAM** | Peak and steady-state; model load footprint |
| **Cold-start time** | Model load + warm-up |
| **Cost per 1k images** | Derived from hardware cost and throughput |

### D. Operational reliability

| Metric | Notes |
|---|---|
| **Failure rate** | % of valid requests returning server errors |
| **Error-category distribution** | Client vs content vs server (P0-T2 §6) |
| **Timeout rate / OOM rate** | Under nominal and stress load |
| **Warning precision** | Are `extreme_degradation` / `face_too_small` warnings issued when they should be? |
| **Determinism** | Same input → same output (R-D8) |
| **Soak stability** | Memory growth / crash over long runs |
| **Recovery** | Behavior after OOM / restart |

## 4. Metric Limitations

| Metric | Limitation | Mitigation |
|---|---|---|
| PSNR | Rewards blurry averages; penalizes realistic texture; poorly correlated with perception. | Never used alone; paired with LPIPS + human review. |
| SSIM | Similar bias toward smooth outputs; insensitive to some hallucinations. | Same as above. |
| LPIPS / DISTS | Learned on generic images; may not penalize identity change; can be gamed. | Pair with identity metrics. |
| FID / KID | Needs large samples; measures distribution realism, not per-image fidelity; a model can have great FID while changing identities. | Distribution-level only, never gating alone. |
| No-reference IQA | Often favors over-sharpened/"GAN-looking" outputs; domain-sensitive. | Use for real-world sets and runtime QA only; validate against human review. |
| Identity embeddings | **Evaluator bias**: embeddings can be fooled — generative models may produce outputs that score well on one recognizer while looking different to humans. Also demographic bias of the evaluator itself. | Use ≥ 2 independent evaluators from different model families; include human review; check stratified results. Never use the evaluator also used in training the candidate (e.g., if a candidate uses an ArcFace identity loss, ArcFace-based evaluation is biased in its favor). |
| Synthetic degradations | Real degradations differ (domain gap). | Mandatory S-REAL evaluation. |
| Latency | Hardware-, driver-, and batch-dependent. | Fixed reference hardware, warm-up, documented config. |
| Human review | Subjective, costly, small samples. | Blind, randomized, multiple raters, inter-rater agreement reported. |

## 5. Identity-Preservation Approach (independent of Feature-Match)

1. **Independent evaluator(s)**: use publicly available, commercially usable face-recognition models *only inside the evaluation harness*. Feature-Match is **not** used, imported, or referenced. The evaluator is a test tool, not part of the FES runtime or API.
2. **At least two evaluators** from different architectures/training sets, to reduce single-evaluator bias and training-loss leakage (§4).
3. **Evaluator licensing** must allow the intended evaluation use (checked in Phase 1; P0-T5 §5).
4. **Protocols**:
   - S-SYN: ID gain vs GT (§3B).
   - S-ID / S-REAL: similarity to a separate clean image of same identity; genuine-vs-impostor distributions for degraded vs enhanced.
   - Report mean, median, 5th percentile, identity-drop rate, and FAR change.
5. **Human confirmation**: blind raters answer "same person?" for (GT, output) pairs on a sample, especially for failures flagged by metrics.
6. **Fairness**: all identity metrics reported per demographic stratum (S-FAIR).

## 6. Baselines

| ID | Baseline | Purpose |
|---|---|---|
| B0 | **Degraded input (identity transform)** | Minimum bar: candidates must beat "do nothing" on quality without losing identity. |
| B1 | **Classical upscaling** (bicubic / Lanczos) | Low-resolution baseline. |
| B2 | **Classical restoration** (e.g., unsharp mask, non-local-means / bilateral denoising, JPEG deblocking filter, CLAHE for contrast) | Shows whether a learned model is actually needed per degradation type. |
| B3 | **General (non-face) learned super-resolution / restoration** | Tests whether face-specific priors add value. |
| B4 | **Face-specific restoration candidates** | The candidates under evaluation (surveyed in Phase 1; typical families: codebook/prior-based, GAN-prior-based, diffusion-based, transformer-based). Selection criteria include licensing (P0-T5). |

Oracle (upper reference, not a baseline to beat): the clean ground truth.

## 7. Fair Comparison Methodology

1. **Identical inputs**: same versioned, seeded degradation pipeline and same test splits for all models.
2. **Identical pre/post-processing**: same detector, alignment, crop size policy, and paste-back — so differences come from the restoration model only. (A second track may evaluate each model's native pipeline, reported separately.)
3. **Default settings first**, then at most a documented, small set of tunings per model (e.g., fidelity weight), tuned on a **validation split**, reported on a **separate test split**.
4. **No test-set contamination**: exclude datasets known to be in a candidate's training data from its headline results, or report them separately.
5. **Same hardware and software stack** for performance numbers; warm-up runs excluded; ≥ N repeated runs (N TBD) with variance reported.
6. **Statistical rigor**: report confidence intervals (bootstrap) and use paired significance tests; differences within noise are treated as ties.
7. **Stratified reporting**: by degradation type, severity, face-size category, pose, and demographic group.
8. **Blind human review**: raters don't know which model produced which output; order randomized.
9. **Reproducibility**: record model version/commit, weights checksum, config, seeds, library versions.

## 8. Performance and Reliability Approach

**Performance**

- Measure on (a) the current dev machine (CPU / Intel iGPU; see P0-T5) for feasibility, and (b) the target GPU class once available.
- Test matrix: face-size categories × batch size × concurrency × precision (FP32/FP16/INT8 where supported) × runtime (native framework vs exported, e.g., ONNX/OpenVINO/TensorRT — options only, not decisions).
- Report per stage: decode, detection, alignment, restoration, paste-back, encode.

**Reliability**

- Fuzz tests with corrupt/truncated/oversized/decompression-bomb images (P0-T2 §6).
- Load test to saturation; verify graceful `RESOURCE_EXHAUSTED` rather than crashes.
- Soak test (duration TBD, e.g., several hours) to detect leaks.
- Edge-case suite S-EDGE for correct error/warning behavior.

## 9. Model-Selection Evidence Requirements

A model may be proposed as **production candidate** only with all of the following evidence:

| # | Evidence | Dimension |
|---|---|---|
| E1 | License review: code, weights, and training-data terms permit intended commercial use (P0-T5 §5). **Missing = ineligible.** | Gate |
| E2 | Identity: ID gain ≥ 0 (mean) with CI, identity-drop rate ≤ TBD, no FAR increase — on S-SYN **and** S-ID/S-REAL, with ≥ 2 evaluators. | B |
| E3 | Fairness: no demographic stratum significantly worse than overall (TBD margin). | B |
| E4 | Quality: statistically significant improvement over B0 and the best classical baseline (B1/B2) on primary-target severities; not worse than B0 on mild inputs. | A |
| E5 | Human review: fabrication/beautification/artifact report rates ≤ TBD on blind review. | A/B |
| E6 | Performance: meets P95 latency, throughput, and memory targets (TBD, P0-T5) on target hardware. | C |
| E7 | Reliability: failure rate ≤ TBD; passes fuzz, load, and soak tests; deterministic. | D |
| E8 | Reproducible evaluation report with all configs, seeds, and versions. | All |

Ranking among eligible candidates: prefer higher identity preservation → then quality → then efficiency/cost. A small quality gain never outweighs an identity-preservation loss.

## 10. TBD Thresholds and How They Will Be Determined

| TBD | Description | Method to establish |
|---|---|---|
| TH-1 | Min acceptable mean ID gain & identity-drop δ / max drop rate | Measure evaluator noise floor (same-image re-encoding, genuine-pair variance) in Phase 1; set δ above noise; set max drop rate with product input. |
| TH-2 | Max FAR change | Evaluate genuine/impostor distributions for B0; allowed change = within CI of B0. |
| TH-3 | Min quality improvement over baselines (LPIPS etc.) | Statistical significance vs B0/B2 on primary-target severities (paired test, α TBD, e.g., 0.05). |
| TH-4 | Fairness margin | Compare stratum CIs to overall; policy margin agreed with product/compliance. |
| TH-5 | Human-review max fabrication rate | Pilot review round in Phase 1 to calibrate; agreed with product. |
| TH-6 | Face-size category boundaries & min face size | Plot ID gain and quality vs face size; boundaries where performance changes materially; min size where ID gain turns negative. |
| TH-7 | Extreme-degradation detection threshold | Plot ID gain vs estimated degradation; set where outputs become unreliable. |
| TH-8 | P95 latency, throughput, memory targets | From consumer SLA needs (P0-T1 Q1) and measured capability on target hardware (P0-T5). |
| TH-9 | Max failure rate / timeout rate | From operational SLO agreed in API phase; validated by load/soak tests. |
| TH-10 | Determinism tolerance | Measure on target hardware; byte-identical if achievable, else small pixel tolerance. |
| TH-11 | Repeated-run count, CI method, α | Fixed in Phase 1 evaluation plan before any candidate results are seen (to avoid tuning thresholds to results). |

**Rule**: thresholds are set and recorded *before* final candidate comparison, based on baselines and noise measurements — not reverse-engineered from the preferred model's results.

## 11. Acceptance Criteria

| # | Criterion | Status |
|---|---|---|
| AC1 | Previous outputs reviewed | ✅ |
| AC2 | Four dimensions A–D defined | ✅ §1 |
| AC3 | Candidate metrics incl. PSNR, SSIM, LPIPS, identity similarity, latency, P95, memory/VRAM, throughput, failure rate | ✅ §3 |
| AC4 | Metric limitations explained | ✅ §4 |
| AC5 | Identity evaluation independent of Feature-Match | ✅ §5 |
| AC6 | Fair comparison methodology | ✅ §7 |
| AC7 | Baselines: degraded input, classical, candidate models | ✅ §6 |
| AC8 | Evidence required for production candidate | ✅ §9 |
| AC9 | No arbitrary final numeric thresholds; TBDs with method | ✅ §10 |
| AC10 | No benchmarks implemented; no models downloaded | ✅ |
