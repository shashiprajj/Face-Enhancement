# P3-T7 — Failure Analysis & Robustness Evaluation

| Field | Value |
|---|---|
| Phase / Task | Phase 3 / P3-T7 |
| Status | Complete |
| Date | 2026-10-01 |
| Population | `main` (896 pairs per configuration) |
| Machine-readable | `results/main/failure_flags.csv.gz` (per config × pair: 11 flags + metrics), `failure_stats.csv` (per config × flag, by family), `sample_index.csv` (5 most extreme pairs per config × flag) — `scripts/p3_failure_index.py` |
| Contact sheets | `experiments/phase3/outputs/sheets/main/p3t7_<flag>.jpg` (**git-ignored**; faces never enter git). Regenerate with `scripts/p3_contact_sheet.py --pairs <sample_index pair_ids>` |
| Inputs | P3-T5 (quality), P3-T6 (identity; single evaluator), trial visual review (P3-T4 §3) |

Flags are **descriptive bins** that organise the analysis, defined relative to the same pair's B0, the evaluator noise floor, or the clean reference. They are **not** pass/fail thresholds (TH-1…TH-11 TBD).

---

## 1. Failure Taxonomy

| ID | Failure mode | Measured proxy (flag) | Visual signature (contact sheets) |
|---|---|---|---|
| **F1** | **Identity substitution.** A plausible, sharp face of a *different* person. | `id_drift`, `id_flip`, `id_closer_to_impostor` with low LPIPS | 14-px and combined-severe inputs under A1/A2: age and gender change, an elderly bearded man becomes a younger clean-shaven one, a girl becomes an adult male face |
| **F2** | **Attribute and accessory removal.** Beard, glasses, nose ring, necklace, bindi, tooth gap. | Not captured by a metric. Visual only. | A1/A2 at 14 px remove glasses and jewellery and "repair" a tooth gap. A4 at noise-extreme drops a nose ring. |
| **F3** | **Structural distortion ("melted" face).** Smeared, asymmetric features and hollow dark eyes. | `geo_landmark_shift`, `geo_detect_lost`, `id_collapse` | A3/A4/A5 on 14-px inputs. Upscaling with no face prior invents texture over an implausible geometry. |
| **F4** | **Pipeline collapse to colour noise.** No face remains. | `id_collapse`, `colour_shift_added`, sharpness ratio ≫ 1 | A5 only: 14-px inputs (internal 2×2 / 1×1 grid in FBCNN/NAFNet, trial finding) and dark 224-px inputs (§3 F7) |
| **F5** | **Over-sharpening and synthetic texture.** GAN micro-texture where the reference is smooth or soft. | `texture_oversharp` | A1/A2 skin pores and wrinkles on soft references. A2 leaves a mesh/grid "skin" texture from noise at noise-extreme. |
| **F6** | **Over-smoothing (missing detail).** Waxy, plastic skin, loss of hair and beard texture. | `texture_oversmooth` | A3/A4 on noise and blur, B0/B1/B2 on blur (input itself) |
| **F7** | **Tone and colour cast.** | `colour_shift_added` | A1: purple/blue cast and raised contrast on underexposed inputs (91 pairs). A4: occasional green cast on noise-extreme. B2c: global tone change (CLAHE, 502 pairs). |
| **F8** | **Mild-input regression.** Visible re-synthesis of an already-good image. | `quality_regression` on mild cells | All learned candidates: LPIPS worse than B0 on mild cells (P3-T5 Q1) |
| **F9** | **Paste-back seam.** A visible band where the 512 canvas does not cover the 224 chip (filled from bicubic input). | Excluded by the evaluation mask (P3-T1 §4.4) | A1/A2 bottom or top edge on some references. Not a model failure, but visible in an output. |
| **F10** | **Soft-reference penalty.** The model sharpens a reference that is itself blurry. PSNR and identity drop "against" a soft ground truth. | `id_drift` on mild/photometric cells of specific references | A1 on ff-val-5428 (motion-blurred clean chip): a sharp, different-looking face. It is the largest A1 drop (−0.45). The trial showed the same thing (gaze change). |

## 2. Candidate-Specific Failure Statistics (counts of 896 pairs; `failure_stats.csv`)

| Flag | B1 | B2 | B2c | A1 | A2 | A3 | A4 | A5 |
|---|---|---|---|---|---|---|---|---|
| quality_regression (LPIPS worse than B0) | 181 | 624 | 805 | 497 | 498 | 510 | **452** | 575 |
| id_drift (> 1-px-shift noise p99) | 14 | 561 | 861 | 667 | 714 | 447 | **379** | 622 |
| id_flip (accepted → rejected) | 5 | 13 | 36 | **12** | 13 | 38 | 38 | 72 |
| id_rescue (rejected → accepted) | 1 | 5 | 2 | **16** | 10 | 15 | 15 | 11 |
| id_closer_to_impostor | 10 | 13 | 28 | 16 | 16 | 34 | 25 | **68** |
| id_collapse (cos < 0.2) | 8 | 16 | 35 | 9 | 11 | 23 | 19 | **65** |
| geo_detect_lost | 2 | 11 | 31 | **0** | **0** | 16 | 17 | 49 |
| geo_landmark_shift (> 0.05 IOD) | 2 | 36 | 164 | 38 | 45 | 34 | **17** | 83 |
| texture_oversharp | 0 | 99 | 387 | 396 | 415 | 308 | 294 | 351 |
| texture_oversmooth | 594 | 528 | 221 | 10 | 6 | 208 | 130 | 192 |
| colour_shift_added | 0 | 0 | 502 | 91 | 0 | 1 | 6 | 33 |

B0 has 6 pairs closer to an impostor and 6 with cos < 0.2 (noise-extreme and combined-severe inputs). These are the floor.

## 3. Failure Modes by Candidate (connecting the examples to the measurements)

**A1 RestoreFormer++.**
- Strengths: it never loses detection (0 `geo_detect_lost`), it has the most rescues (16), and at 14 px its identity is at B0 level (0.463 vs 0.454).
- Failures: F1/F2 at extreme low resolution and combined-severe; it confidently generates a different person with no uncertainty signal. F7 colour cast on underexposure. F10 on soft references. Its pooled ΔID is −0.049 because it drifts on **most** pairs (667), mostly in small steps.

**A2 VQFR v1.**
- Shows the same F1/F2 pattern as A1 with slightly more drift (714) and fewer rescues (10).
- F5: residual mesh texture at noise-extreme. Its noise-extreme LPIPS is 0.60, versus 0.20–0.24 for A1/A3/A4.
- A small uniform lightening at low resolution (ΔL* ≈ +1.3 to +1.7 in the central face, all race groups alike; §5).

**A3 Real-ESRGAN.**
- The best noise removal (Δ −0.52 LPIPS, +7.8 dB).
- F3 at 14–28 px, where 22 of its 38 flips are lowres. F6 over-smoothing (208 pairs).

**A4 SwinIR-L.**
- The fewest drifts (379), the lowest landmark shift (17) and the best pooled LPIPS.
- Still F3 at 14 px (19 lowres flips) and F7 green cast in a few noise-extreme outputs. F2 (accessory loss) on denoising.

**A5 FBCNN → NAFNet → Real-ESRGAN.**
- The most failures in every identity flag: 72 flips, 68 impostor-closer and 65 collapses.
- **F7 → F4 on dark inputs.** One very dark reference (ff-val-2699) collapses on 17 of its 32 cells to a coloured grid or blob texture (ΔE up to 53, sharpness ratio up to 123×). Real-ESRGAN alone (A3) handles the same inputs normally. A stage probe attributes the failure to **NAFNet-SIDD** on low-signal images (§4).
- F4 on 14-px inputs (trial finding confirmed on main: lowres-extreme mean cos 0.098).

**Classical baselines.**
- B2 is mostly F6 plus mild identity drift.
- B2c's F7 is a global tone change: the largest identity drift of all configurations (861 pairs).

## 4. Difficult Preprocessing Cases

| Case | Evidence | Note |
|---|---|---|
| Very dark images | A5 F4 on ff-val-2699 (all families); underexposure-severe is A5's worst family by LPIPS p95 (0.77) | No candidate corrects exposure (P3-T5 Q3). Exposure handling before enhancement is an open design question, not decided here. |
| 14-px inputs (inter-ocular distance ≈ 3–4 px) | lowres-extreme: face models synthesise, general SR models distort, A5 collapses | No candidate preserves identity better than bicubic input here. Detector-based alignment on such inputs fails (P2-T4), so production would also lack the oracle-quality alignment used in this benchmark (P3-T1 §12). |
| Soft or blurred **clean** references (FairFace) | F10 | The benchmark's ground truth limits what "better than reference" can mean. This is a Phase 2 data property, recorded here. |
| Off-canvas chips (3 of 28 references with coverage < 100 %) | F9 | Masked out of metrics. Visible in outputs. |

**A5 stage attribution (dark-image collapse).** The A5 components were re-run on 4 failing pairs and 1 control (`scripts/p3_a5_stage_probe.py` → `repro/a5_stage_probe.json`). The input mean RGB on the failing pairs is ≈ 0.11–0.13; the control is 0.54.

| Stage | Failing pairs (dark) | Control |
|---|---|---|
| FBCNN | Max change 0.01–0.06; Laplacian variance 4–13 (still smooth) | Max change 0.11 |
| **NAFNet-SIDD** | **Max change 0.98–1.00; Laplacian variance explodes to 26,000–106,000** | Max change 0.16; variance 35 |
| Real-ESRGAN on the FBCNN output (skipping NAFNet) | PSNR vs reference equal to SR of the raw input: 32.7–32.9 dB, or 13.9 dB for the underexposure-severe pair, whose input is itself dark | — |
| Real-ESRGAN on the NAFNet output | PSNR 9.3–16.8 dB | 32.9 dB |

**The collapse is introduced by NAFNet-SIDD on very dark inputs.** It turns low-signal regions into high-amplitude structured texture. FBCNN is benign. This is a diagnostic of the frozen A5 configuration. No alternative chain was run in Phase 3.

## 5. Colour and Skin-Tone Check (descriptive; 4 identities per race group)

Mean CIE L* shift of the central face region vs the clean reference, on non-photometric families:

| Config | all | range over race groups |
|---|---|---|
| B0 | +0.2 | +0.2 to +0.3 |
| A1 | −1.3 | −3.0 to −0.1 |
| A2 | +0.9 | +0.4 to +1.5 |
| A3 | −0.7 | −1.1 to −0.3 |
| A4 | −0.4 | −0.7 to −0.2 |

At lowres-severe/extreme, A2 lightens by +1.3 to +2.0 L* for all groups alike. A3 darkens by −1.6 to −2.6.

**No group-specific lightening or darkening is detectable at this sample size.** A single A1 example at 14 px *looked* lighter; the population measurement does not support a systematic effect. A fairness assessment needs the full test_core and a second evaluator (G-1, G-2).

## 6. Representative Sample Index

`sample_index.csv` lists, for each of A1–A5 (and B1–B2c) × 11 flags, the 5 most extreme pairs: `pair_id`, cell, input size, LPIPS, ΔLPIPS, id_cos, ΔID, sharpness ratio and ΔE. That is 393 rows.

The contact sheets `p3t7_<flag>.jpg` show the top 2 per learned candidate for every flag, side by side across B0 and A1–A5 with the clean reference. These sheets are the visual evidence for F1–F10.

## 7. What This Analysis Cannot Show

- Real-world failure frequencies. The degradations are synthetic (G-3).
- Whether a human would judge F1 outputs as "the same person". SFace is the only measurement (G-2).
- F2 (accessory and attribute removal) has no metric. It is documented from visual review only, and its frequency is unknown.

## 8. Done-When Check

The major failure modes are documented in a taxonomy (F1–F10). Each is linked to per-pair measurements, candidate-specific statistics, a representative sample index and visual evidence, rather than relying on aggregate scores alone. ✅
