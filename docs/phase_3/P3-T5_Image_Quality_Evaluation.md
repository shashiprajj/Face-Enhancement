# P3-T5 — Image Quality Evaluation

| Field | Value |
|---|---|
| Phase / Task | Phase 3 / P3-T5 |
| Status | Complete |
| Date | 2026-10-01 |
| Population | `main` (28 test references × 32 cells = 896 pairs per configuration; all 9 configurations 896/896 OK) |
| Metrics (P3-T1 §7) | PSNR (RGB, masked), SSIM (BT.601 luma, σ 1.5, masked), LPIPS (AlexNet v0.1, masked). Descriptive: sharpness ratio, colour shift ΔE76. |
| Per-pair results | `data/manifests/phase3/metrics/main/<config>.csv.gz` |
| Aggregates | `results/main/cells.csv.gz` (per cell: mean, median, p5, p95), `deltas.csv.gz` (paired Δ vs B0 and vs B2, 95 % cluster-bootstrap CI over references, B = 2000, seed 20260928), `tables.md` (all tables, including the per-cell LPIPS distributions) |
| Failure-case metadata | `results/main/failure_flags.csv.gz`, `failure_stats.csv`, `sample_index.csv` (`scripts/p3_failure_index.py`; see P3-T7) |

**Image quality is reported here only.** Identity is in P3-T6. A better quality score is **not** evidence of identity preservation (§5).

---

## 1. Headline: Paired Difference vs B0 (all 32 cells pooled)

▲ better, ▼ worse, ≈ CI contains 0 (no detectable difference).

| Config | ΔLPIPS ↓ | ΔPSNR (dB) ↑ | ΔSSIM ↑ |
|---|---|---|---|
| A1 RestoreFormer++ | −0.048 [−0.079, −0.014] ▲ | −4.85 [−5.58, −4.23] ▼ | −0.060 [−0.072, −0.048] ▼ |
| A2 VQFR v1 | −0.033 [−0.062, −0.002] ▲ | −4.16 [−4.63, −3.74] ▼ | −0.048 [−0.055, −0.040] ▼ |
| A3 Real-ESRGAN | −0.062 [−0.074, −0.049] ▲ | −1.77 [−2.20, −1.35] ▼ | +0.030 [0.024, 0.035] ▲ |
| A4 SwinIR-L | −0.078 [−0.097, −0.058] ▲ | −1.71 [−2.21, −1.23] ▼ | +0.033 [0.028, 0.037] ▲ |
| A5 FBCNN → NAFNet → Real-ESRGAN | −0.003 [−0.029, 0.029] ≈ | −3.16 [−4.09, −2.41] ▼ | +0.000 [−0.024, 0.015] ≈ |
| B2 classical | +0.030 [0.024, 0.036] ▼ | −1.29 [−1.39, −1.20] ▼ | — |

Against the classical restoration B2 (pooled):

| Config | ΔLPIPS | ΔPSNR | ΔSSIM |
|---|---|---|---|
| A1 | −0.078 ▲ | −3.56 ▼ | −0.043 ▼ |
| A2 | −0.063 ▲ | −2.87 ▼ | −0.031 ▼ |
| A3 | −0.092 ▲ | −0.47 ▼ | +0.047 ▲ |
| A4 | −0.108 ▲ | −0.41 ≈ | +0.049 ▲ |
| A5 | −0.033 ▲ | −1.86 ▼ | +0.017 ≈ |

Pooled numbers average over very different regimes. The by-severity and by-family results below are the meaningful view.

## 2. By Severity — the Dominant Pattern

ΔLPIPS vs B0, mean [95 % CI]:

| Config | mild | moderate | severe | extreme |
|---|---|---|---|---|
| A1 | +0.104 ▼ | −0.014 ≈ | −0.106 ▲ | −0.278 ▲ |
| A2 | +0.094 ▼ | −0.011 ≈ | −0.089 ▲ | −0.200 ▲ |
| A3 | +0.056 ▼ | −0.033 ▲ | −0.116 ▲ | −0.229 ▲ |
| A4 | +0.055 ▼ | −0.046 ▲ | −0.132 ▲ | −0.276 ▲ |
| A5 | +0.105 ▼ | +0.020 ≈ | −0.044 ▲ | −0.165 ▲ |

ΔPSNR (dB): every learned candidate is ▼ on mild, moderate and severe. On **extreme**, only A3 (+1.91 ▲) and A4 (+1.69 ▲) improve; A5 is ≈ and A1/A2 remain ▼.

**Finding Q1.** Every learned candidate **degrades** mildly degraded inputs (perceptually and in pixel fidelity) and improves severely degraded ones perceptually. The crossover lies around *moderate*. For mild inputs the best "enhancement" on this benchmark is none (B0).

## 3. By Family

ΔLPIPS vs B0, mean, with the verdict:

| Config | lowres | gaussblur | motionblur | noise | jpeg | underexp. | lowcontrast | combined | realistic2 |
|---|---|---|---|---|---|---|---|---|---|
| A1 | −0.037 ≈ | +0.034 ≈ | +0.025 ≈ | **−0.476 ▲** | +0.075 ▼ | +0.106 ▼ | +0.064 ▼ | −0.079 ▲ | −0.098 ▲ |
| A2 | −0.033 ≈ | +0.018 ≈ | +0.017 ≈ | −0.322 ▲ | +0.074 ▼ | +0.083 ▼ | +0.091 ▼ | −0.088 ▲ | −0.108 ▲ |
| A3 | −0.020 ≈ | +0.033 ▼ | +0.039 ▼ | **−0.517 ▲** | +0.034 ▼ | +0.063 ▼ | +0.091 ▼ | −0.114 ▲ | −0.124 ▲ |
| A4 | −0.041 ▲ | −0.021 ≈ | +0.003 ≈ | −0.512 ▲ | +0.028 ▼ | +0.068 ▼ | +0.079 ▼ | −0.123 ▲ | −0.131 ▲ |
| A5 | +0.051 ▼ | +0.083 ▼ | +0.083 ▼ | −0.445 ▲ | +0.080 ▼ | +0.164 ▼ | +0.169 ▼ | −0.082 ▲ | −0.085 ▲ |

**Findings:**
- **Q2. Noise** is where every learned candidate helps most (−0.32 to −0.52 LPIPS, +2.5 to +7.8 dB PSNR). It is the only family where PSNR also improves for all five.
- **Q3. Photometric families** (underexposure, lowcontrast) and **JPEG** get worse with every learned candidate. None of them corrects global tone, and each adds texture to an image whose structure was intact.
- **Q4. Mixed degradations** (combined, realistic2) improve perceptually for every candidate (−0.08 to −0.13 LPIPS), while PSNR drops (−1.2 to −3.6 dB).
- **Q5. Blur families** show no detectable perceptual gain for any candidate except A4 on gaussblur, which is ≈ overall but has the lowest LPIPS at extreme (median 0.159 vs B0 0.361).
- **Q6. Low resolution:** A4 is the only detectable perceptual gain pooled. At **14 px** (lowres-extreme), the face models have the lowest LPIPS (A1 0.301, A2 0.298 vs B0 0.499). That is a synthesised face, and P3-T6 shows identity is not recovered there.

## 4. Distributions and Outliers (not only averages)

The per-cell **median [p5–p95]** of LPIPS for every configuration is in `tables.md` → "LPIPS median [p5–p95] per cell".

| Observation | Evidence |
|---|---|
| **Face models have a floor.** A1/A2 LPIPS median ≈ 0.14–0.15 on *every* mild cell, even where B0 is 0.005 (gaussblur-mild). | They regenerate the whole face, so the output never equals the input. |
| **A5 has a heavy tail.** p95 LPIPS reaches 0.74 (lowres-extreme) and 0.77 (underexposure-severe). Its pooled ΔLPIPS CI is the widest. | 33 pairs lose > 0.5 identity cos. 17 of them belong to one reference (ff-val-2699), mostly on 224-px inputs, with ΔE up to 53 and a sharpness ratio up to 123× (P3-T7 F7). |
| **CLAHE (B2c) is not a quality enhancer on this data.** It is ▼ on every family except underexposure PSNR. | P3-T3 §4 |
| **PSNR and LPIPS disagree systematically for face models.** At extreme severity, A1 is ▲ on LPIPS (−0.278) and ▼ on PSNR (−1.11 dB). | Generative priors trade pixel fidelity for plausible texture. Neither metric alone describes the change. |

Machine-readable outliers: `failure_flags.csv.gz` (per pair: `quality_regression` = LPIPS worse than B0; `texture_oversharp`; `texture_oversmooth`; `colour_shift_added`) and `sample_index.csv` (the most extreme pairs per configuration × flag).

| Flag (of 896 pairs) | A1 | A2 | A3 | A4 | A5 | B2 |
|---|---|---|---|---|---|---|
| LPIPS worse than B0 | 497 | 498 | 510 | 452 | 575 | 624 |
| Over-sharpened (sharpness > 1.5× reference, LPIPS worse than B0) | 396 | 415 | 308 | 294 | 351 | 99 |
| Over-smoothed (sharpness < 0.5× reference) | 10 | 6 | 208 | 130 | 192 | 528 |
| Colour shift added (ΔE more than B0's + 5) | 91 | 0 | 1 | 6 | 33 | 0 |

These are descriptive bins, not thresholds.

## 5. What These Quality Results Do NOT Show

- **Identity.** Several cells with the clearest LPIPS gains (lowres-extreme, combined-severe, realistic2-severe) show **identity loss** for the same candidates (P3-T6 §3). Perceptual closeness to the reference is not the same person.
- **Real-world quality.** These are synthetic `fes-syn-v1` degradations on FairFace chips. No real-world or no-reference evaluation was possible (G-3).
- **Significance against a minimum meaningful difference.** TH-1/TH-3 are TBD, so "▲/▼" means only that the 95 % CI excludes 0.

## 6. Achieved Precision (in place of a power analysis; P3-T1 §5)

| Metric | Median 95 % CI half-width: all cells | per family | per cell (max) |
|---|---|---|---|
| ΔLPIPS | 0.029 | 0.035 | 0.034 (0.074) |
| ΔPSNR (dB) | 0.49 | 0.59 | 0.70 (2.25) |

Differences smaller than about 0.03 LPIPS or 0.5 dB **pooled**, and about 0.07 LPIPS or 2 dB **per cell**, cannot be resolved with 28 references.

## 7. Done-When Check

Every runnable candidate and every baseline has reproducible, machine-readable quality measurements on the identical population. They are aggregated by family and severity with distributions, paired CIs, and outlier metadata. ✅
