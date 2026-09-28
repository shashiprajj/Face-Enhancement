# P2-T4 — Controlled Degradation Generator

| Field | Value |
|---|---|
| Phase / Task | Phase 2 / P2-T4 |
| Status | Complete |
| Date | 2026-09-28 |
| Code | `src/fes_data/degrade.py` (`degrade-1.0.0`), runner `scripts/p2_degrade.py` |
| Pilot config | `configs/phase2/degrade_pilot.yaml` |
| Pilot outputs | `datasets/benchmark/fes-syn-pilot-v1/degraded/<profile>-<severity>/<ref_id>.png` (1,280) |
| Paired manifest | `data/manifests/degrade_pilot_v1.csv.gz` |
| Statistics | `data/manifests/degrade_pilot_v1_report.json` |

---

## 1. Design Rules

1. **The clean reference is never modified.** It is opened read-only, and its SHA-256 is asserted against the P2-T3 manifest before every use.
2. **Deterministic.** Each pair's random stream is `numpy PCG64(seed)` with seed = first 8 bytes of `sha256(global_seed | ref_id | profile | severity | degrade_version)`. The same inputs give the same pixels, in any order and with any number of workers.
3. **Named profiles and severities.** Every exact parameter drawn is stored per pair (`params_json`).
4. **Model-compatible outputs.**
   - 8-bit sRGB PNG, lossless, so no unintended extra compression.
   - Same-size profiles stay 224×224.
   - Low-resolution and chain profiles output the true LR image (112/56/28/14 px). The benchmark adapters handle any upsampling (P1-T5 §5.1).
5. **No intentional identity change.** Only global, spatially uniform operators are used. No local edits, warps of facial regions, or learned models.

## 2. Profiles and Severities

Spatial parameters are defined at the 224-px reference scale: P1-T7's 512-scale values, halved.

| Profile (P0 DG) | Mild | Moderate | Severe | Extreme | Notes |
|---|---|---|---|---|---|
| `lowres` (DG1) | ×2 → 112 | ×4 → 56 | ×8 → 28 | ×16 → 14 | `INTER_AREA` (anti-aliased) |
| `gaussblur` (DG2) | σ 0.5 | σ 1.0 | σ 2.0 | σ 4.0 | kernel 2⌈3σ⌉+1, reflect border |
| `motionblur` (DG3) | 4 px | 8 px | 13 px | 21 px | line kernel, angle ~ U[0,180) per image, 8× supersampled raster |
| `noise` (DG4) | σ 5 | σ 15 | σ 25 | σ 50 | i.i.d. Gaussian per channel (0–255 scale) |
| `jpeg` (DG5) | Q 60 | Q 40 | Q 20 | Q 10 | **Below the FairFace Q≈75 floor** (P2-T2). P1-T7's mild Q80 would not degrade these sources. |
| `underexposure` (DG6) | −1 EV | −2 EV | −3 EV | — | exposure scaled in linear light, then back to sRGB (replaces P1-T7's ad-hoc gamma; physically interpretable) |
| `lowcontrast` (DG7) | α 0.7 | α 0.5 | α 0.3 | — | deviations from mean luma scaled by α. Included because P0-T3 lists low contrast; justified as a separate failure mode from exposure. |
| `combined` (DG8a) | σ∈[0.2,1], ×2, n∈[0,5], Q∈[50,70] | σ∈[0.5,2], ×4, n∈[0,10], Q∈[35,60] | σ∈[1,4], ×8, n∈[0,20], Q∈[20,45] | — | blur → downscale → noise → JPEG; ranges sampled per image |
| `realistic2` (DG8b) | ×2 | ×4 | ×8 | — | two rounds of blur/resize/noise/JPEG (second-order, Real-ESRGAN-style); per-round parameters sampled |

In total: **32 cells** (9 profiles).

**Deviations from P1-T7 (explicit):**
- spatial values halved for 224 px;
- JPEG levels moved below the Q75 floor;
- exposure expressed in EV;
- chain downscale factors fixed per level (so output sizes are integers), with the other parameters sampled.

Values remain **starting points**, to be calibrated against real-world data when available (P2-T5, G-3).

## 3. Pilot Run & Statistics

Pilot: 40 seeded dev-pool references × 32 cells = **1,280 pairs**. Runtime ≈ 25 s in the container. The full per-cell table is in the report.

| Cell | Out | PSNR | SSIM | YuNet detects |
|---|---|---|---|---|
| lowres mild / mod / severe / extreme | 112 / 56 / 28 / 14 | 42.5 / 35.0 / 28.4 / 23.6 | 0.99 / 0.94 / 0.82 / 0.68 | 100 / 100 / 100 / 95% |
| gaussblur m / mo / s / x | 224 | 50.2 / 40.1 / 33.2 / 27.6 | 1.00 / 0.98 / 0.93 / 0.82 | 100% |
| motionblur m / mo / s / x | 224 | 39.3 / 33.2 / 29.6 / 26.3 | 0.98 / 0.93 / 0.87 / 0.78 | 100% |
| noise m / mo / s / x | 224 | 34.2 / 24.8 / 20.5 / 15.0 | 0.92 / 0.62 / 0.42 / 0.20 | 100 / 100 / 80 / **0%** |
| jpeg m / mo / s / x | 224 | 40.8 / 36.9 / 33.5 / 29.8 | 0.99 / 0.97 / 0.94 / 0.87 | 100% |
| underexposure m / mo / s | 224 | 17.9 / 13.1 / 10.7 | 0.92 / 0.72 / 0.50 | 100% (mean luma 109 → 78 / 55 / 38) |
| lowcontrast m / mo / s | 224 | 24.0 / 19.5 / 16.6 | 0.94 / 0.86 / 0.74 | 100 / 100 / 97.5% |
| combined m / mo / s | 112 / 56 / 28 | 33.4 / 28.1 / 23.2 | 0.93 / 0.82 / 0.68 | 100 / 100 / 90% |
| realistic2 m / mo / s | 112 / 56 / 28 | 32.4 / 27.4 / 23.2 | 0.92 / 0.80 / 0.69 | 100 / 100 / 90% |

How to read these:
- PSNR and SSIM are measured against the reference. LR outputs are bicubic-upsampled **for measurement only**.
- They are **input-quality statistics, not benchmark results**.
- Quality falls monotonically with severity in every family, as designed.
- The YuNet detection rate shows where preprocessing starts to fail (noise-extreme: 0%). That behavior must be handled by P2-T6 (`NO_FACE_DETECTED`), never by inventing a face.
- Photometric profiles (exposure, contrast) have low PSNR by construction, while structure is kept (SSIM, detection). PSNR alone must not be used to rank their severity (P0-T4 §4).

## 4. Reproducibility Evidence

| Check | Result |
|---|---|
| `--verify-only`: regenerate all 1,280 pairs in memory; compare pixel SHA-256 and `params_json` | **1,280 / 1,280 match** |
| Delete all pilot outputs, rerun the generator, compare manifest SHA-256 | **identical** (`c8bafe78fabfebb1…`) |
| Existing-file guard | Generator asserts on any differing existing output (`non-reproducible degradation`) |

The environment is the pinned container `fes-dev:cpu-p2` (OpenCV 5.0.0, NumPy 2.4.6). Bit-exactness across **other** OpenCV/NumPy builds is not claimed. The manifest records the versions (`run`).

## 5. Paired Manifest Fields

`pair_id` (`<ref_id>.<profile>-<severity>`), `ref_id`, `sample_id`, `pool`, `profile`, `severity`, `ref_path`, `ref_sha256`, `degraded_path`, `degraded_sha256`, `degraded_pixel_sha256`, `out_w`, `out_h`, `seed`, `params_json` (every parameter, including sampled values and chain order), `degrade_version`, `psnr_vs_ref`, `ssim_vs_ref`, `deg_lap_var`, `deg_mean_luma`, `deg_rms_contrast`, `yunet_face_found`, `yunet_score`.

## 6. Limitations

- Synthetic degradations are **not** a substitute for real-world data (Phase 2 README; P2-T5).
- References already carry Q≈75 JPEG artifacts.
- Gaussian noise is signal-independent. Poisson/chroma noise variants are not in `degrade-1.0.0`; they can be added as new profiles with a version bump.
- Degradation strength relative to real captures is uncalibrated until DS03 exists.

## 7. Done-When Check

Every degraded image is reproducible from its clean source and recorded profile, which the verification above demonstrates. ✅
