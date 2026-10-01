# P3-T6 — Identity Preservation Evaluation

| Field | Value |
|---|---|
| Phase / Task | Phase 3 / P3-T6 |
| Status | Complete — **single identity evaluator (G-2 open)** |
| Date | 2026-10-01 |
| Evaluator | E01 SFace `models/E01_sface/face_recognition_sface_2021dec.onnx`, SHA-256 `0ba9fbfa…7c34e79`, test-only (P0 C1) |
| Alignment | Reference landmarks (fixed) → `FaceRecognizerSF.alignCrop` → 128-d, L2-normalised, for **every** image, so identity is measured independently of detector failures (P3-T1 §7) |
| Operating point | FAR 1e-2 from 244,650 clean-vs-clean impostor pairs over the 700-ref test impostor pool → threshold **cos 0.3133** (`metrics/main/identity_calibration.json`) |
| Noise floor | `identity_noise_floor.json` (112 dev refs): 1 − cos p99 = 0.0215 under a 1-px shift and 0.0039 for the harness E→F→E round trip |
| Results | `metrics/main/<config>.csv.gz` (`id_cos`, `imp_cos_*`, `id_ge_thr`), `results/main/identity.csv.gz`, `deltas.csv.gz` (`d_id_cos`), `strata.csv.gz`, `failure_flags.csv.gz` |
| Embeddings | Held in memory only, never written to disk (P2-T7 rule) |

---

## 0. Single-Evaluator Limitation (applies to every number below)

Phase 0 requires at least two independent identity evaluators (P0-T4 §5.2). **Only one, SFace, is available and cleared.** No second evaluator was invented, substituted, or approximated.

So every identity statement here means "according to SFace". In particular:
- A model could produce faces that SFace scores as similar while a different recogniser would not, or the reverse.
- A generative model could learn to satisfy features SFace relies on without preserving identity as a human would judge it.

Visual similarity in the contact sheets (P3-T7) is **not** used as evidence of identity preservation. **The two-evaluator requirement remains unmet (G-2).**

## 1. Headline: cos(output, reference) and Paired ΔID vs B0 (all cells)

| Config | mean cos | ΔID vs B0 [95 % CI] | ΔID vs B2 | TAR @ FAR 1e-2 | observed FAR | mean impostor cos |
|---|---|---|---|---|---|---|
| B0 | 0.854 | — | — | 0.959 | 0.0038 | 0.034 |
| A1 RestoreFormer++ | 0.805 | −0.049 [−0.070, −0.032] ▼ | −0.013 ≈ | 0.963 | 0.0042 | **0.046** |
| A2 VQFR v1 | 0.793 | −0.061 [−0.077, −0.047] ▼ | −0.025 ▼ | 0.955 | 0.0046 | **0.044** |
| A3 Real-ESRGAN | 0.834 | −0.021 [−0.026, −0.014] ▼ | +0.015 ▲ | 0.933 | 0.0036 | 0.040 |
| A4 SwinIR-L | 0.846 | −0.008 [−0.014, −0.001] ▼ | +0.028 ▲ | 0.933 | 0.0033 | 0.040 |
| A5 chain | 0.774 | −0.080 [−0.119, −0.056] ▼ | −0.044 ▼ | **0.891** | 0.0038 | 0.038 |
| B2 classical | 0.818 | −0.036 ▼ | — | 0.950 | 0.0036 | 0.036 |

**Finding I1. No learned candidate improves identity similarity over the unenhanced input in aggregate.** All five are ▼ vs B0. A4 is closest to neutral (−0.008). The face models A1 and A2 lose more identity similarity than the general SR models A3 and A4, yet keep the highest TAR. Their losses are many small drifts, not threshold crossings (§4).

**Finding I2. Face models increase impostor similarity.** The mean cos to *other* people rises from 0.034 (B0) to 0.046 (A1) and 0.044 (A2): faces become more alike. The observed FAR stays below the nominal 1e-2 on this 28-identity population, but the direction is a homogenisation signal worth testing at scale.

## 2. By Severity

ΔID vs B0, mean [95 % CI]:

| Config | mild | moderate | severe | extreme |
|---|---|---|---|---|
| A1 | −0.085 [−0.108, −0.065] ▼ | −0.066 ▼ | −0.051 ▼ | **+0.047 [0.021, 0.068] ▲** |
| A2 | −0.082 ▼ | −0.077 ▼ | −0.066 ▼ | +0.016 [−0.005, 0.035] ≈ |
| A3 | −0.024 ▼ | −0.029 ▼ | −0.038 ▼ | **+0.032 [0.015, 0.051] ▲** |
| A4 | −0.023 ▼ | −0.020 ▼ | −0.022 ▼ | **+0.069 [0.054, 0.086] ▲** |
| A5 | −0.062 ▼ | −0.082 ▼ | −0.126 ▼ | −0.027 [−0.059, −0.001] ▼ |

TAR @ FAR 1e-2:

| Config | mild | moderate | severe | extreme |
|---|---|---|---|---|
| B0 | 1.000 | 1.000 | 0.889 | 0.936 |
| A1 | 1.000 | 1.000 | 0.885 | 0.971 |
| A2 | 1.000 | 1.000 | 0.865 | 0.957 |
| A3 | 1.000 | 1.000 | 0.849 | 0.843 |
| A4 | 1.000 | 1.000 | 0.837 | 0.864 |
| A5 | 0.980 | 0.976 | 0.774 | 0.786 |

**Finding I3.** Identity gains exist only at **extreme** severity, for A1, A3 and A4. At mild to severe, every candidate costs identity similarity that the input already had.

## 3. By Family (mean cos; ΔID vs B0 in parentheses)

| Config | lowres | gaussblur | motionblur | noise | jpeg | underexp. | lowcontrast | combined | realistic2 |
|---|---|---|---|---|---|---|---|---|---|
| B0 | 0.817 | 0.959 | 0.890 | 0.757 | 0.942 | 0.976 | 0.947 | 0.680 | 0.686 |
| A1 | 0.775 (−0.042 ▼) | 0.882 (−0.077 ▼) | 0.835 (−0.054 ▼) | 0.828 (**+0.071 ▲**) | 0.875 (−0.067 ▼) | 0.869 (−0.108 ▼) | 0.847 (−0.100 ▼) | 0.637 (−0.043 ▼) | 0.639 (−0.047 ▼) |
| A2 | 0.742 (−0.076 ▼) | 0.885 (−0.074 ▼) | 0.824 (−0.066 ▼) | 0.806 (+0.048 ▲) | 0.876 (−0.066 ▼) | 0.885 (−0.092 ▼) | 0.824 (−0.124 ▼) | 0.625 (−0.055 ▼) | 0.618 (−0.068 ▼) |
| A3 | 0.677 (**−0.140 ▼**) | 0.937 (−0.022 ▼) | 0.868 (−0.021 ▼) | 0.907 (**+0.149 ▲**) | 0.934 (−0.008 ▼) | 0.948 (−0.029 ▼) | 0.910 (−0.038 ▼) | 0.638 (−0.042 ▼) | 0.632 (−0.054 ▼) |
| A4 | 0.720 (−0.097 ▼) | 0.962 (+0.003 ≈) | 0.878 (−0.011 ▼) | 0.915 (**+0.157 ▲**) | 0.939 (−0.003 ≈) | 0.949 (−0.027 ▼) | 0.914 (−0.033 ▼) | 0.644 (−0.036 ▼) | 0.634 (−0.052 ▼) |
| A5 | 0.573 (**−0.245 ▼**) | 0.885 (−0.074 ▼) | 0.823 (−0.067 ▼) | 0.845 (+0.088 ▲) | 0.908 (−0.033 ▼) | 0.817 (−0.159 ▼) | 0.839 (−0.109 ▼) | 0.612 (−0.068 ▼) | 0.608 (−0.078 ▼) |

**Finding I4. Noise is the only family where every candidate preserves identity better than the input.** At noise-extreme, mean cos rises from 0.391 (B0) to 0.815 (A4), 0.789 (A3) and 0.701 (A1).

**Finding I5. Low resolution is where identity is most at risk.** At **14 px** (lowres-extreme), the general SR models produce perceptually "better" images (LPIPS 0.38 vs 0.50) but identity collapses: mean cos 0.224 (A3), 0.274 (A4) and 0.098 (A5) vs 0.454 (B0). The face models stay near B0 (A1 0.463, A2 0.446). **Perceptual improvement and identity preservation diverge here.**

**Finding I6. Mixed degradations** (combined, realistic2) improve perceptually for every candidate (P3-T5 Q4) **and** lose identity for every candidate (−0.04 to −0.08).

## 4. Pair-Level Identity Risk (descriptive bins; `failure_stats.csv`)

| Count of 896 | B0 | B2 | A1 | A2 | A3 | A4 | A5 |
|---|---|---|---|---|---|---|---|
| Drift beyond the 1-px-shift noise p99 (ΔID < −0.0215) | 0 | 561 | 667 | 714 | 447 | **379** | 622 |
| **Flip**: B0 accepted → output rejected | — | 13 | 12 | 13 | 38 | 38 | **72** |
| **Rescue**: B0 rejected → output accepted | — | 5 | **16** | 10 | 15 | 15 | 11 |
| Output closer to another person than to self | 6 | 13 | 16 | 16 | 34 | 25 | **68** |
| cos < 0.2 (identity essentially absent) | 6 | 16 | 9 | 11 | 23 | 19 | **65** |

Where the flips happen: A3 has 22 of 38 on lowres, A4 19 of 38 on lowres plus 14 on realistic2, and A5 30 of 72 on lowres with the rest spread across all families. Rescues come mostly from noise (6–9 per candidate).

**Finding I7.** Only A1 rescues more pairs than it flips (16 vs 12). For A3, A4 and A5, flips outnumber rescues by about 2.5× to 6.5×. **A5 places 68 pairs closer to a different person than to the true identity.** This is the most direct misidentification risk observed.

## 5. Strata (descriptive only; 2 identities per race × gender stratum)

The stratum mean ΔID ranges from −0.18 to −0.02 (A1), −0.17 to −0.02 (A2), −0.04 to +0.00 (A3), −0.02 to +0.01 (A4), and −0.31 to −0.04 (A5). With two identities per stratum, these spreads are driven by single references: for A5 that is ff-val-2699 (P3-T7 F7), and for A1 ff-val-5428 on photometric cells. **No demographic conclusion can be drawn.** A fairness evaluation needs the full test_core (700 refs) on the GPU track (G-1).

## 6. Achieved Precision

The median 95 % CI half-width of ΔID is 0.015 pooled, 0.019 per family, 0.018 per severity, and 0.023 per cell (max 0.12). Smaller identity differences are not resolvable with 28 references.

## 7. What Is Not Claimed

- No identity "pass": TH-2/TH-4 are TBD, and the FAR 1e-2 point is an operating point, not a requirement.
- No two-evaluator agreement (G-2).
- No real-world identity behaviour (G-3).

## 8. Done-When Check

Identity results are reproducible from fixed inputs, one recorded evaluator, and a fixed alignment. They are reported separately from quality, by family and severity, with pair-level failure and outlier analysis and an explicit single-evaluator limitation. ✅
