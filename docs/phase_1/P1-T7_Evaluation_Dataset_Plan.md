# P1-T7 — Evaluation Dataset Plan

| Field | Value |
|---|---|
| Phase / Task | Phase 1 / P1-T7 |
| Status | Complete (plan). No training, no benchmarking. |
| Date | 2026-09-28 |
| Depends on | P1-T6 (datasets), P1-T4 (detection/alignment), P1-T5 (test design), P0-T3/T4 |

---

## 1. Selected Portfolio

| Role | Dataset | License status | Rationale |
|---|---|---|---|
| **S-SYN-A** (synthetic, fairness-stratified) | **DS01 FairFace** (official *val* split) | ELIGIBLE (CC BY 4.0) | Only eligible public face set with **demographic balance and labels**. Needed for P0-T3 R-D7 / P0-T4 S-FAIR. Its 224 px resolution limits it to low/moderate-res GT. |
| **S-SYN-B + S-ID** (synthetic, high-res, cross-image identity) | **DS02 Public-domain official portrait set** (to curate) | ELIGIBLE per item | High-res GT (≥ 512 face) for the standard BFR protocol. Multiple portraits per person give **cross-image same-identity** genuine/impostor pairs. |
| **S-REAL + paired real** (gold standard) | **DS03 In-house consented capture** | ELIGIBLE once consent program approved | The only eligible route to **real** degraded faces with **same-identity HQ references**. Needed because synthetic → real domain gap is risk R-5. |
| **S-EDGE** | Constructed from DS01/DS02 (+ DS03) | inherits | No-face (non-face crops from PD/CC0 scenes), multi-face composites, tiny faces, occlusion overlays, pose buckets, grayscale |
| *Conditional* (reviewer approval) | DS05 Casual Conversations v2 · DS12 SCface · DS11 TinyFace · DS10 XQLFW | REVIEW_REQUIRED | Would fill the real-world / paired gap if approved. **Not downloaded.** |

Deliberately **not** used: FFHQ and CelebA. Besides being license-ineligible, they are the training sets of most candidates, so they would produce contaminated, optimistic results (P0-T4 §2).

## 2. Coverage Check

| Requirement | Covered by |
|---|---|
| Low resolution | Synthetic DG1 on DS01/DS02; native in DS03 (distance) |
| Blur (defocus) | Synthetic DG2; DS03 (focus) |
| Motion blur | Synthetic DG3; DS03 (camera/subject motion) |
| Noise | Synthetic DG4; DS03 (low light) |
| Compression | Synthetic DG5; DS03 (messaging-app re-encode) |
| Mixed degradation | Synthetic DG8; DS03 |
| Pose variation | DS01 (in-the-wild poses) stratified by estimated yaw (MediaPipe, P1-T4); DS03 (scripted poses); DS02 mostly frontal (limitation) |
| Lighting variation | Synthetic DG6/DG7; DS03 (controlled lighting) |
| Face sizes | DG1 scale factors → face-size buckets `tiny/small/medium/large` (P0-T2 §2.4) |
| Same-identity quality comparison | R-A synthetic (all), R-B cross-image (DS02, DS03), R-C real paired (DS03) |
| Fairness strata | DS01 labels (aggregate only); DS03 self-reported attributes if consented |

## 3. Evaluation Subsets

| Subset | Source | Content | Primary metrics |
|---|---|---|---|
| **E1 SYN-HR** | DS02 test identities | GT ≥ 512 aligned; full degradation matrix (§4) | PSNR/SSIM/LPIPS, ID gain vs GT, landmark distance |
| **E2 SYN-FAIR** | DS01 val | GT 224 aligned; degradation matrix at ×2/×4/×8 | Same, **stratified** by race/gender/age group |
| **E3 ID-PAIRS** | DS02 (+DS03) | Genuine pairs (degraded A vs clean B of same person) and impostor pairs | ID similarity, TAR@FAR, FAR change, identity-drop rate |
| **E4 REAL-PAIRED** | DS03 | Real LQ vs same-session HQ | Full-reference metrics vs HQ (after alignment), ID similarity, human review |
| **E5 EDGE** | Constructed | No face, multi-face, tiny, occluded, profile, grayscale, extreme degradation | Correct status/warnings (P0-T3 §9 T-Q8…T-Q15) |
| **E6 HQ-PASSTHROUGH** | DS02 clean | Undegraded inputs | Change magnitude; "do no harm" (R-D10) |

## 4. Degradation Matrix (synthetic)

Severity parameters are **starting points** (P0-T3 §2), defined at the 512 aligned-face scale and scaled proportionally for 224 GT. They are to be **calibrated** in Phase 2 against DS03 real samples and the TH-6/TH-7 procedures. Values are not final.

| ID | Degradation | Mild | Moderate | Severe | Extreme |
|---|---|---|---|---|---|
| DG1 | Downscale (bicubic, antialias) → model input size | ×2 | ×4 | ×8 | ×16 |
| DG2 | Gaussian blur σ (px) | 1 | 2 | 4 | 8 |
| DG3 | Linear motion blur, length (px), random angle | 7 | 15 | 25 | 41 |
| DG4 | Gaussian noise σ (0–255); also Poisson and chroma variants | 5 | 15 | 25 | 50 |
| DG5 | JPEG quality (single); double-JPEG variant | 80 | 50 | 30 | 10 |
| DG6 | Under-exposure (gamma γ applied to linearized image) | 1.5 | 2.2 | 3.0 | — |
| DG7 | Contrast scaling α around mean | 0.7 | 0.5 | 0.3 | — |
| DG8a | **Classical BFR chain**: blur(σ) → downscale(r) → noise(σn) → JPEG(q) | σ≤2, r≤2, σn≤5, q≥70 | σ≤4, r≤4, σn≤10, q≥50 | σ≤8, r≤8, σn≤20, q≥30 | r=16 |
| DG8b | **Second-order realistic chain** (two rounds of blur/resize/noise/JPEG, Real-ESRGAN-style) | preset | preset | preset | — |

Cells: 7 single degradations × 4 levels + 2 chains × up to 4 levels, plus clean (E6). Single-degradation cells isolate each failure mode. DG8 represents realistic inputs (P0-T3 DG8 "core").

## 5. Synthetic Degradation Plan (conceptual; implemented in Phase 2)

1. **Input:** aligned clean GT (from the fixed alignment of P1-T5 §5.1), stored as lossless PNG.
2. **Operators** (OpenCV/NumPy only, Apache/BSD, no learned models):
   - downsampling: `cv2.resize`, `INTER_AREA`, plus bicubic variants;
   - Gaussian and anisotropic blur kernels;
   - motion blur from a line kernel, with random trajectories as a variant;
   - Gaussian/Poisson/chroma noise;
   - JPEG via `cv2.imencode` at a fixed libjpeg version;
   - gamma/contrast in linear space.
3. **Randomness:** per-sample seed = `hash(dataset_id, image_id, cell_id, pipeline_version)`. Every parameter drawn is logged.
4. **Output:** degraded image (PNG, to avoid unintended re-compression, except for the JPEG operator output itself), plus a JSON sidecar (`source_id`, `gt_sha256`, `cell`, exact params, `seed`, `pipeline_version`).
5. **Versioning:** the pipeline is semantically versioned. Any change → new version → datasets regenerated, never overwritten.
6. **Upsampling for face models:** degraded LR is bicubic-upsampled to the model input size inside the candidate adapter, not in the dataset (P1-T5 §5.1).

## 6. Identity-Preservation Test Design

| Test | Design | Output |
|---|---|---|
| **ID-1 Self-consistency** (E1/E2) | sim(restored, GT) vs sim(degraded→bicubic, GT), per sample | ID gain distribution; mean, median, 5th percentile; identity-drop rate (δ = TH-1) |
| **ID-2 Cross-image** (E3) | Genuine: sim(restored(A_degraded), B_clean) with A≠B photos of the same person. Impostor: vs other identities. | Genuine/impostor score distributions; TAR@FAR; **FAR change** (hallucination toward generic faces) |
| **ID-3 Real** (E4) | sim(restored(real LQ), same-session HQ) | Real-world ID gain |
| **ID-4 Fairness** (E2) | ID-1 per demographic stratum | Stratum CIs vs overall (TH-4) |
| **ID-5 Human** | Blind "same person?" on sampled (GT, output) pairs, especially metric failures | Agreement rate; inter-rater κ |

Evaluators: E01 SFace (ELIGIBLE) plus a second evaluator pending reviewer (P1-T4 §5 gap). **Evaluators are never used for model selection tuning** (validation split only), to avoid overfitting to one evaluator.

## 7. Split Strategy

| Dataset | Split rule | Leakage control |
|---|---|---|
| DS02 | **By identity**: validation 30% of identities / test 70% | No identity in both splits. Impostor pairs drawn within split. |
| DS01 | Official FairFace **train** split → validation pool (sampled); official **val** split → test | Different images. No identity labels, so the residual cross-split identity-overlap risk is accepted and documented. |
| DS03 | **By participant** (validation / test) | Participants never cross splits |
| All | Degradation seeds differ between validation and test | — |

Use:
- **Validation** is for threshold calibration (TH-x), the one allowed tuning per model, and pilot variance.
- **Test** is used once per frozen configuration for headline results.

No training in Phase 1–2 (P0-T1 X8). If fine-tuning happens later, it uses separate data that is disjoint by identity.

## 8. Sample-Size Targets

**TBD with method (no invented finals):**
1. Pilot on validation: ~100 samples per degradation cell (justified estimate: enough to estimate variance of paired LPIPS and ID-gain differences).
2. Power analysis: n per cell so that a pre-registered minimum meaningful difference (TH-3, TH-1) is detectable at α (TH-11) with power 0.8 in paired tests.
3. Fairness strata: at least the same n per stratum as per cell, or strata merged and documented.
4. E3 pairs: enough impostor pairs to estimate FAR at the chosen operating point (≥ 10/FAR as a rule of thumb). Exact numbers are set after the DS02 curation yield is known.
5. DS03: target number of participants **TBD** by the consent program budget. Minimum reportable n is to be set from pilot variance.

## 9. Provenance & Versioning Plan

| Artifact | Tracked where | Content |
|---|---|---|
| `data/registry/datasets.csv` (git) | Git | dataset id, source URL, version/commit, license, license URL, download date, acquirer |
| `data/registry/<dataset>_files.csv` (git) | Git | relative path, SHA-256, size, split, per-item license (DS02), source URL (DS02) |
| Raw data | `datasets/` (git-ignored, local/secure storage) | Immutable once registered |
| Degraded sets | `datasets/derived/<pipeline_version>/` (git-ignored) | Regenerable from GT + seeds |
| Degradation sidecars | JSON per sample; index CSV in git | params, seed, pipeline version |
| Pipeline code | Git (Phase 2) | semantic version tag |
| Evaluation runs | `experiments/<run_id>/` | config, model manifest ids, dataset manifest hashes, environment manifest |

Rule: a benchmark result is valid only if it cites dataset manifest hashes, the pipeline version, and model manifest ids (P1-T9).

## 10. Data-Quality Checks After Acquisition

1. SHA-256 verification against the registry. Decode test for every file.
2. Resolution and color-mode audit. Remove files below the minimum GT size for their subset.
3. Face detection check with the fixed detector: exactly one dominant face for GT sets; flag failures.
4. Near-duplicate detection (perceptual hash) within and **across splits**.
5. Identity-label sanity (DS02/DS03): same-person pairs are verified by a human on a sample.
6. **License audit per item** (DS02): the source page and license are recorded; items without verifiable PD/CC0 are dropped.
7. Demographic distribution report (DS01 aggregate; DS03 if consented).
8. Contamination check: confirm no evaluation image is from FFHQ/CelebA (hash and source check).
9. Privacy check: EXIF/GPS stripped from working copies; access restricted.

## 11. Remaining Data-Access Blockers

| # | Blocker | Impact | Resolution |
|---|---|---|---|
| DB-1 | DS02 must be **curated** (not a ready download) | Delays high-res GT and cross-image identity tests | Phase 2 prep task: scripted collection from official public-domain sources with per-item provenance |
| DB-2 | DS03 needs a **consent program** and legal review (Q-5) | No eligible **real-world** degraded data; S-REAL is mandatory (master §8) | Product/Legal decision; smaller pilot capture possible |
| DB-3 | Reviewer decisions on DS05/DS12/DS11/DS10 | Alternative real-world sources | License reviewer (B-2) |
| DB-4 | FairFace GT is only 224 px | High-res behavior is measured only on DS02 | Accept; documented |

## 12. Acceptance Criteria

| Criterion | Status |
|---|---|
| Concrete evaluation-data strategy | ✅ §1, §3 |
| Same-identity/different-quality testing possible | ✅ §6 (ID-1…ID-3). ID-2/ID-3 depend on DS02 curation / DS03 collection. |
| Real and synthetic degradation both considered | ✅ E1/E2 synthetic; E4 real (DS03). Real is currently **blocked** (DB-2). |
| Provenance and licensing traceable | ✅ §9 |
| Sample sizes TBD with justified method | ✅ §8 |
| No training or benchmarking | ✅ |
