# P3-T3 — No-Enhancement / Classical Baseline

| Field | Value |
|---|---|
| Phase / Task | Phase 3 / P3-T3 |
| Status | Complete |
| Date | 2026-10-01 |
| Population | `main` — 28 test references × 32 cells = 896 pairs (P3-T1 §3) |
| Configurations | B0 (degraded input), B1 (Lanczos-4), B2 (Lanczos → NLM → unsharp), B2c (B2 + CLAHE) — P3-T1 §6 |
| Output manifests | `data/manifests/phase3/runs/main/{B0,B1,B2,B2c}.csv.gz` (+ `_run.json`) |
| Metric files | `data/manifests/phase3/metrics/main/{B0,B1,B2,B2c}.csv.gz` |
| Aggregates | `data/manifests/phase3/results/main/{cells,deltas,identity,strata}.csv.gz`, `tables.md`, `summary.json` |
| Code | `src/fes_bench/adapters.py` (`B0Identity`, `B1Lanczos`, `B2Classical`), `scripts/p3_run_inference.py`, `scripts/p3_evaluate.py`, `scripts/p3_aggregate.py` |

---

## 1. What the Baselines Are For

- **B0** is the degraded input brought to the 224-px evaluation frame with no enhancement (bicubic for low-resolution inputs, identity for same-size inputs). Every learned candidate is compared against B0 with a **paired** difference per pair (P3-T1 §8).
- **B1, B2, B2c** answer whether a learned model does anything that cheap classical processing does not.
- No baseline was tuned. The parameters are fixed a priori (P3-T1 §6).

## 2. Validation of the Evaluation Pipeline (before any candidate result was read)

| Check | Result |
|---|---|
| Trial population (4 dev references, 128 pairs) run and scored for every configuration before the main run | ✅ All 9 configurations, 128/128. The 9 missing A2 trial pairs were completed on 2026-10-01 (P3-T4 §4). |
| Trial metric files re-scored | ✅ Byte-identical on re-run (same SHA-256), so scoring is deterministic |
| Trial aggregation and table rendering end to end | ✅ `results/trial/` |
| Inputs re-hashed before inference | ✅ Every degraded file is checked against the population manifest SHA-256 by the runner |
| Outputs verified before scoring | ✅ `p3_evaluate.py` asserts each output's SHA-256 against the run manifest before computing any metric |
| Same evaluation mask for all configurations | ✅ Computed from the reference only (P3-T1 §4.4) |
| B1 ≡ B0 on same-size cells | ✅ Expected by construction (Lanczos at scale 1 is the identity). ΔPSNR, ΔLPIPS and ΔID are exactly 0 on every same-size family. This is a built-in sanity check of the pairing. |

## 3. Main-Population Baseline Results

All 4 baselines: **896/896 pairs OK, 0 failed, 0 missing.**

### 3.1 Image quality (means per family; the full per-cell distributions are in `tables.md`)

| Config | lowres | gaussblur | motionblur | noise | jpeg | underexp. | lowcontrast | combined | realistic2 |
|---|---|---|---|---|---|---|---|---|---|
| **LPIPS ↓** B0 | 0.244 | 0.150 | 0.170 | 0.667 | 0.106 | 0.125 | 0.155 | 0.344 | 0.359 |
| B1 | 0.246 | 0.150 | 0.170 | 0.667 | 0.106 | 0.125 | 0.155 | 0.347 | 0.361 |
| B2 | 0.255 | 0.185 | 0.207 | 0.724 | 0.113 | 0.192 | 0.283 | 0.307 | 0.323 |
| B2c | 0.315 | 0.240 | 0.251 | 1.009 | 0.213 | 0.175 | 0.221 | 0.387 | 0.386 |
| **PSNR ↑** B0 | 33.00 | 38.34 | 32.54 | 23.61 | 35.59 | 14.52 | 20.77 | 28.77 | 28.38 |
| B1 | 33.33 | 38.34 | 32.54 | 23.61 | 35.59 | 14.52 | 20.77 | 28.76 | 28.36 |
| B2 | 31.11 | 33.40 | 31.08 | 22.82 | 34.37 | 14.43 | 20.56 | 28.92 | 28.47 |
| B2c | 19.94 | 20.37 | 20.23 | 14.33 | 20.07 | 18.38 | 19.53 | 19.24 | 19.60 |

Paired differences vs B0, as mean [95 % cluster-bootstrap CI], all cells pooled:

| Config | ΔLPIPS | ΔPSNR (dB) | ΔID cos |
|---|---|---|---|
| B1 | +0.001 [0.000, 0.001] worse | +0.037 [0.032, 0.042] better | −0.001, worse |
| B2 | +0.030 [0.024, 0.036] worse | −1.29 [−1.39, −1.20] worse | −0.036, worse |
| B2c | +0.104 [0.096, 0.112] worse | −9.99 [−10.79, −9.24] worse | −0.108, worse |

### 3.2 Identity (SFace, single evaluator; see P3-T6 for the full treatment)

| Config | mean cos | TAR @ FAR 1e-2 | mild | moderate | severe | extreme |
|---|---|---|---|---|---|---|
| B0 | 0.854 | 0.959 | 0.984 | 0.920 | 0.770 | 0.652 |
| B1 | 0.853 | 0.954 | 0.984 | 0.919 | 0.769 | 0.652 |
| B2 | 0.818 | 0.950 | 0.954 | 0.893 | 0.723 | 0.608 |
| B2c | 0.746 | 0.921 | 0.869 | 0.809 | 0.663 | 0.560 |

(Severity columns are the mean cos(output, reference).)

On B0, **37 of the 896 pairs** fall below the operating threshold (cos 0.313 at FAR 1e-2). They are concentrated in noise-extreme and combined/realistic2-severe.

## 4. Observations (descriptive; no thresholds applied)

1. **The degraded input is already a strong baseline on mild and moderate cells.** For example, gaussblur-mild has a B0 LPIPS median of 0.005 and jpeg-mild 0.030. Any model that re-synthesises the face adds error there (P3-T5 §4).
2. **B1 adds almost nothing.** It differs from B0 only on low-resolution inputs, where the Lanczos-vs-bicubic upscale gains +0.32 dB but slightly worsens LPIPS (+0.002).
3. **B2 (denoise + sharpen)** is worse than B0 on every single-factor family. It is better only on `combined` and `realistic2`, by about −0.04 LPIPS, because NLM removes part of the mixed noise. It lowers identity similarity on every family where the difference is detectable.
4. **B2c (CLAHE)** changes global tone. This costs about 10 dB PSNR overall but gains +3.9 dB on `underexposure`, the one family it is designed for. It has the largest identity loss of any baseline (−0.108). Contrast equalisation is therefore not identity-neutral under SFace on this data.
5. The classical baselines never improve identity similarity over B0 in aggregate. **B0 is the reference bar for identity**, and B2 the reference bar for "cheap restoration".

## 5. Reproducibility

- Each baseline is deterministic and has no weights. Output pixel SHA-256s are recorded per pair, and a re-run sample is part of P3-T10.
- Metric-file SHA-256 values (main): B0 `9fe300d9…`, B1 `7ec0f6fe…`, B2 `c29d55ee…`, B2c `79a9fee1…` (`results/main/summary.json`).

## 6. Done-When Check

A reproducible baseline exists for the degraded inputs. It uses the same population, mask, metrics and statistics as every candidate, and was validated on the trial population before use. ✅
