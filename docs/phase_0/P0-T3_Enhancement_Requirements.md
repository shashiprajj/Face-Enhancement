# P0-T3 — Enhancement & Quality Requirements

| Field | Value |
|---|---|
| Phase / Task | Phase 0 / P0-T3 |
| Status | Complete |
| Date | 2026-09-28 |
| Depends on | P0-T1, P0-T2 (reviewed) |
| Nature | Requirements only — no model chosen, no enhancement code |

---

## 1. Key Principle

> **The goal is faithful restoration, not attractiveness.** FES must recover what was plausibly present in the original scene while preserving facial identity information as accurately as possible. When information is genuinely lost, FES should produce a conservative result and say so — not invent a convincing but different face.

Consequences:

- Sharpness, smoothness, or "prettiness" are **not** success criteria on their own.
- A slightly softer output that preserves identity is preferred over a crisp output that changes it.
- FES must be able to **decline or flag** rather than hallucinate (see `ENHANCEMENT_NOT_RECOMMENDED`, P0-T2 §6).

## 2. Degradation Taxonomy

Severity levels are **conceptual** (Mild / Moderate / Severe / Extreme). Numeric parameters for synthetic test generation are proposed as *starting points* and will be calibrated in Phase 1 (P0-T4).

| ID | Degradation | Description | Mild | Moderate | Severe | Extreme | Target? |
|---|---|---|---|---|---|---|---|
| DG1 | **Low resolution** | Face occupies few pixels (inter-ocular distance small). | Face ≈ 96–128 px | ≈ 48–96 px | ≈ 24–48 px | < 24 px | Mild–Severe **core**; Extreme flag only |
| DG2 | **Defocus / Gaussian blur** | Optical softness, out-of-focus. | Slight softness | Pores/fine edges lost | Features' edges smeared | Features indistinct | Mild–Severe core |
| DG3 | **Motion blur** | Directional smear from camera/subject motion. | Short kernel | Visible streak | Double edges | Features unrecognizable | Mild–Moderate core; Severe best-effort |
| DG4 | **Noise** | Sensor/low-light noise (Gaussian, Poisson, chroma). | Faint grain | Visible grain, detail intact | Detail masked | Structure masked | Mild–Severe core |
| DG5 | **JPEG / compression artifacts** | Blocking, ringing, banding, color bleeding; repeated re-compression. | Q ≈ 60–80 | Q ≈ 30–60 | Q ≈ 10–30 | Q < 10 | Mild–Severe core |
| DG6 | **Poor lighting** | Under/over-exposure, harsh shadows, color cast. | Slightly dark/bright | Clipped regions small | Large clipped/shadowed areas | Face mostly clipped | Conditional (P0-T1 S6) |
| DG7 | **Low contrast** | Flat tonal range (haze, cheap sensor, scans). | Slightly flat | Clearly washed out | Features low-visibility | — | Conditional (P0-T1 S6) |
| DG8 | **Combined degradation** | Realistic mixture, e.g., downscale + blur + noise + JPEG. | 2 mild | Mixed mild/moderate | Several severe | — | **Core** — most representative of real inputs |

Additional factors (not degradations to "fix", but affect outcome): occlusion, pose, expression, accessories (glasses, masks, hats), makeup, facial hair, skin tone, age.

## 3. Target Severity Levels

- **Primary target**: Mild → Moderate across DG1–DG5 and DG8. Output should be clearly better *and* identity-faithful.
- **Secondary target**: Severe. Improvement expected; identity preservation must still be measured; warnings may be attached.
- **Not a target**: Extreme. FES should detect likely-extreme inputs and either return a conservative result flagged `extreme_degradation` or decline with `ENHANCEMENT_NOT_RECOMMENDED`. Policy choice (flag vs. decline) is a Phase 1 decision based on evaluation data.
- Lighting/contrast (DG6–DG7): best effort; must never alter skin tone or facial structure in ways that affect identity.

## 4. Desired Behavior

| ID | Requirement |
|---|---|
| R-D1 | Increase perceptual quality and detail for primary-target inputs relative to the input and to classical baselines (P0-T4). |
| R-D2 | Preserve facial geometry: shape and position of eyes, nose, mouth, jaw, ears, hairline. |
| R-D3 | Preserve identity-bearing attributes: face shape, skin tone, eye color, scars, moles, birthmarks, tattoos, wrinkles, facial hair, visible accessories. |
| R-D4 | Preserve apparent age, expression, and gaze direction. |
| R-D5 | Remove artifacts (blocking, ringing, noise) without introducing new ones (halo, checkerboard, waxy skin, texture tiling). |
| R-D6 | Degrade gracefully: as input severity increases, output should become more conservative, not more inventive. |
| R-D7 | Behave consistently across skin tones, genders, ages, and ethnicities (fairness; measured in P0-T4). |
| R-D8 | Be deterministic for identical input, options, and version (P0-T2 §4.2). |
| R-D9 | Seamless paste-back: no visible seams, color mismatch, or misalignment at the face boundary. |
| R-D10 | Mild-quality inputs should be changed little ("do no harm"): if the input is already good, output ≈ input. |

## 5. Unacceptable Behavior

| ID | Unacceptable behavior | Severity |
|---|---|---|
| U1 | **Identity change**: output appears to be a different person, or identity similarity to the ground truth drops relative to the input. | **Critical** |
| U2 | **Fabricated facial details**: inventing features not supported by the input — new moles, removed scars, different eye shape, added/removed facial hair, different teeth. | **Critical** |
| U3 | Attribute shifts: changes to apparent age, gender presentation, ethnicity, or skin tone. | **Critical** |
| U4 | Beautification: skin smoothing that removes real texture/wrinkles/blemishes; makeup-like effects; symmetry "correction". | High |
| U5 | Expression or gaze change (e.g., closed eyes opened, neutral turned to smile). | High |
| U6 | Occluded regions filled with invented content presented as real (e.g., generating a mouth under a mask). | High |
| U7 | New artifacts: halos, grid patterns, color blotches, texture repetition, uncanny "GAN look". | Medium |
| U8 | Silent failure: returning a heavily hallucinated face without any warning when inputs are extreme. | High |
| U9 | Systematically worse behavior for certain demographic groups. | High |
| U10 | Accessory hallucination/removal: glasses removed or distorted, earrings invented. | Medium |

## 6. Identity-Preservation Principle

1. Identity preservation is a **core, gating requirement**, not a nice-to-have. A model that scores best on visual quality but fails identity preservation is **not** eligible for production.
2. Identity preservation is measured with an **independent** face-identity evaluator used **only for testing** (P0-T4). FES never exposes identity outputs (P0-T1 X3).
3. The reference is the **clean ground truth** where available (synthetic degradation) and the **degraded input** otherwise (real-world images): the enhanced output should be *at least as* close to the true identity as the degraded input is.
4. Where models expose a fidelity/quality trade-off parameter, the default must favor fidelity.
5. Visual quality and identity preservation are reported **separately**; they are never combined into a single score that allows one to compensate for the other.

## 7. Quality Requirements vs. Visual Attractiveness

| Dimension | Is a requirement? | How judged |
|---|---|---|
| Fidelity to ground truth (pixel/structural) | Yes | Full-reference metrics (PSNR/SSIM) on synthetic data |
| Perceptual similarity to ground truth | Yes | LPIPS-style metrics |
| Identity preservation | **Yes — gating** | Independent identity evaluator |
| Artifact absence | Yes | Human review + no-reference metrics |
| Naturalness / realism | Yes | No-reference IQA + human review |
| Attractiveness / beauty / "wow" | **No** | Not measured as a goal; beautification is a defect (U4) |
| Sharpness alone | **No** | Only valued when it reflects real detail |

## 8. Edge-Case Expectations

| ID | Edge case | Expected behavior |
|---|---|---|
| E1 | **No face** | Full/auto mode: reject `NO_FACE_DETECTED`. Crop mode: process whole crop, warn `no_landmarks_found`. Never fabricate a face. |
| E2 | **Multiple faces** | Follow face-selection policy (P0-T2 §2.6). Each face processed independently; one face's failure should not fail others in `all` mode (per-face status). |
| E3 | **Extremely small face** | Below min face size (TBD): warn `face_too_small` or reject per policy. Output must be conservative; no invented detail. |
| E4 | **Extreme degradation** | Warn `extreme_degradation` or decline with `ENHANCEMENT_NOT_RECOMMENDED`. Never silently return a hallucinated face (U8). |
| E5 | **Occlusion** (mask, hand, hair, sunglasses) | Restore visible regions; leave occluder in place; do not reconstruct occluded features (U6). Warn `occlusion_suspected` if detectable. |
| E6 | **Unusual pose** (profile, extreme yaw/pitch/roll) | Process if detection/landmarks succeed; warn `unusual_pose`. Many restoration models assume near-frontal faces, so quality may be lower — must be measured. |
| E7 | Partially out-of-frame face | Process visible part; do not complete missing parts. |
| E8 | Non-human / synthetic / drawn faces | Not a target; behavior undefined but must not crash. |
| E9 | Already high-quality input | Minimal change (R-D10). |
| E10 | Grayscale input | Restore in luminance; do not colorize. |
| E11 | Children's faces | Same identity requirements; include in fairness evaluation. |

## 9. Requirements Traceable to Future Test Cases

Each row can become one or more automated or human-review tests in Phase 1+.

| Test ID | Requirement | Test idea | Pass criterion |
|---|---|---|---|
| T-Q1 | R-D1 | Synthetic DG1–DG5, DG8 at Mild/Moderate; compare output vs input and vs classical baseline. | Output better on quality metrics than input and baseline (threshold TBD, P0-T4). |
| T-Q2 | U1, §6 | Identity similarity (output vs ground truth) ≥ (input vs ground truth) across severity levels. | No statistically significant identity drop; per-sample drop rate below TBD. |
| T-Q3 | U1 | Impostor check: output must not become *more* similar to other identities in the test set. | False-match rate not increased vs input (TBD). |
| T-Q4 | U3 | Attribute consistency (age/skin-tone estimate) between ground truth and output — *test-only evaluator*. | Shift within TBD tolerance. |
| T-Q5 | R-D7, U9 | Stratified results by skin tone, gender, age group. | No group significantly worse than overall (TBD). |
| T-Q6 | R-D8 | Run same input twice. | Byte-identical or within tolerance. |
| T-Q7 | R-D10 | Enhance high-quality images. | Change magnitude below TBD; identity unchanged. |
| T-Q8 | E1 | Images with no face. | `NO_FACE_DETECTED`; no image returned. |
| T-Q9 | E2 | Multi-face images with each selection mode. | Correct face(s) processed; counts correct. |
| T-Q10 | E3/E4 | Tiny faces and extreme degradation. | Warning or decline issued; never silent. |
| T-Q11 | E5, U6 | Masked / sunglasses / hand-occluded faces. | Occluder preserved; no invented features (human review). |
| T-Q12 | E6 | Profile and high-yaw faces. | No crash; warning present; identity measured. |
| T-Q13 | R-D9 | Paste-back on varied backgrounds. | No visible seam (human review, blind). |
| T-Q14 | U2, U4, U5, U10 | Blind human review panel on a curated set. | Rate of reported fabrication/beautification below TBD. |
| T-Q15 | E10 | Grayscale inputs. | Output not colorized. |

## 10. Open Questions

| ID | Question | Needed by |
|---|---|---|
| Q3-1 | For Extreme degradation: warn-and-return or decline? | Phase 1 (from evaluation data) |
| Q3-2 | Should lighting/contrast correction be learned, classical, or omitted in v1? | Phase 1 |
| Q3-3 | Should consumers be allowed to adjust fidelity vs. quality, or is it fixed? | Phase 1/2 |
| Q3-4 | Is a human-review panel available for blind evaluation, and how large? | Phase 1 |
| Q3-5 | Which demographic attributes can be used for fairness stratification, given data availability and privacy? | Phase 1 |

## 11. Acceptance Criteria

| # | Criterion | Status |
|---|---|---|
| AC1 | Previous outputs reviewed | ✅ |
| AC2 | Taxonomy covers low-res, blur, motion blur, noise, JPEG, poor lighting, low contrast, combined | ✅ §2 |
| AC3 | Severity levels distinguished | ✅ §2, §3 |
| AC4 | Desired behavior defined | ✅ §4 |
| AC5 | Unacceptable behavior (esp. identity change / fabrication) defined | ✅ §5 |
| AC6 | Quality separated from attractiveness | ✅ §7 |
| AC7 | Edge cases: no face, multiple, tiny, extreme, occlusion, pose | ✅ §8 |
| AC8 | Identity-preservation principle stated | ✅ §6 |
| AC9 | Requirements convertible to test cases | ✅ §9 |
| AC10 | No model chosen; no enhancement code | ✅ |
