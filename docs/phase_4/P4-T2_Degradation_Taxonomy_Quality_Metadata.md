# P4-T2 — Degradation Taxonomy & Quality Metadata

| Field | Value |
|---|---|
| Phase / Task | Phase 4 / P4-T2 |
| Date | 2026-10-03 |
| Status | **Complete for synthetic populations.** The real-world arm uses the same code once a frozen `rw-v1` exists (G-3). |
| Taxonomy | `configs/phase4/taxonomy_v1.yaml` (`p4-taxonomy-1.0.0`) |
| Features | `src/fes_bench/quality.py` (`p4-quality-1.0.0`) · `scripts/p4_quality_features.py` |
| Tests | `tests/test_p4_quality.py` (8 passed, `fes-dev:cpu-p4`) |
| Evidence | `data/manifests/phase4/quality/{p4dev,main}.csv.gz` + `_report.json` (feature + taxonomy + detector hashes) |

---

## 1. Taxonomy (versioned)

**Families:**
- LOWRES, BLUR_DEF, BLUR_MOT, NOISE, COMPRESS (resolution, blur, noise, compression);
- UNDEREXP, OVEREXP, LOWCON (photometric);
- OCCLUSION, POSE, CROP (face and framing);
- MIXED and CLEAN.

**Severities:** none / mild / moderate / severe / extreme. Phase 0 makes mild–moderate the primary target, severe secondary, and extreme flag-or-decline.

**Every label carries a `label_source`.** That is what keeps observed real degradation distinguishable from synthetic degradation:

| label_source | Meaning | `data_kind` |
|---|---|---|
| SYNTHETIC_GENERATOR | fes-syn-v1 cell (known operator + parameters) | `SYNTHETIC` |
| CAPTURE_PROTOCOL | rw-spec condition/level recorded at capture. The level→severity map is a **prior only**. | `REAL_WORLD` |
| ANNOTATOR | Human label (P4-T5); none collected | — |
| ESTIMATED | Rule-based estimate from features, calibrated on DEV only (P4-T6) | either |

`mixed_flag` is true when the label lists MIXED or ≥ 2 families. The fes-syn-v1 `combined` and `realistic2` chains are MIXED (blur + lowres + noise + compression). The rw-spec LCDV and LOWL conditions are multi-family by nature.

## 2. Quality Metadata (reference-free, runtime-computable)

Each feature uses only the input and the FES detection contract (`preprocess(..., input_mode="face_crop")`: YuNet, pad-and-relax retry, min IOD 12 px). It never uses the clean reference or the degradation label. The C-frame is the input resized to 112 × 112, which normalises for resolution.

| Requirement (P4-T2) | Feature(s) | Valid when |
|---|---|---|
| image dimensions | `in_min_side` | always |
| face dimensions | `iod_px` (detected inter-ocular distance, native px) | face detected |
| blur / sharpness | `lap_var_c`, `lap_norm_c` (÷ contrast²), `hf_ratio_c` (spectral power > 0.25 cyc/px) | always |
| noise | `noise_sigma` (Immerkær 1996, native luma) | side ≥ 3. Biased upward on small native images (edges dominate). |
| brightness / exposure | `luma_mean`, `luma_p05`, `luma_p95`, `dark_clip`, `bright_clip` | always |
| contrast | `rms_contrast`, `dyn_range` | always |
| compression | `blockiness` (8×8 boundary/interior gradient ratio) | side ≥ 32 |
| pose | `roll_deg`, `yaw_proxy` (landmark proxies) | face detected and landmarks plausible |
| occlusion | `occlusion` | **Never.** No eligible model (P2-T6), so it is recorded as NaN rather than invented. |
| detector confidence | `det_status`, `det_score`, `det_retry` | always / when detected |
| alignment quality | `align_resid` (RMS landmark residual after a similarity fit to the FFHQ template ÷ template IOD) | face detected |
| family / severity / mixed | taxonomy label + `label_source` | always |

Invalid values are written as empty (NaN). **They are never imputed.**

## 3. Behaviour on the Synthetic Dev Population (`p4dev`, 28 refs × 32 cells, medians)

| Cell | lap_norm_c | hf_ratio_c | noise_sigma | blockiness | luma_mean | rms_contrast | iod_px |
|---|---|---|---|---|---|---|---|
| clean | 0.117 | 0.006 | 0.40 | 1.31 | 96.8 | 44.7 | 83.3 |
| lowres mild→extreme | 0.117 → 0.002 | 0.006 → 0.000 | 0.9 → 7.3 | 1.00 → n/a | ≈ 97 | 44 → 40 | 41 → 12 → n/a |
| gaussblur mild→extreme | 0.098 → 0.005 | 0.005 → 0.000 | ≈ 0.3 | ≈ 1.0 | ≈ 97 | 44 → 40 | ≈ 82 |
| motionblur mild→extreme | 0.062 → 0.016 | 0.003 → 0.001 | ≈ 0.3 | ≈ 1.0 | ≈ 97 | 44 → 41 | ≈ 81 |
| noise mild→extreme | 0.145 → 2.547 | 0.009 → 0.145 | 3.4 → 30.3 | ≈ 1.0 | 97 → 99 | 45 → 52 | 82 → n/a |
| jpeg mild→extreme | ≈ 0.13–0.18 | ≈ 0.01 | ≈ 0.3–0.5 | 1.27 → 3.41 | ≈ 97 | ≈ 45 | ≈ 83 |
| underexposure mild→severe | ≈ 0.12 | ≈ 0.006 | ≈ 0.4 | ≈ 1.3 | 69 → 33 | 34 → 19 | ≈ 82 |
| lowcontrast mild→severe | ≈ 0.12 | ≈ 0.006 | ≈ 0.4 | ≈ 1.3 | ≈ 97 | 31 → 13 | ≈ 81 |
| combined mild→severe | 0.088 → 0.005 | 0.005 → 0.000 | 0.8 → 2.1 | ≈ 1.3 → n/a | ≈ 97 | 44 → 41 | 41 → 11 |
| realistic2 mild→severe | 0.054 → 0.004 | 0.003 → 0.000 | 0.6 → 1.6 | ≈ 1.3 → n/a | ≈ 97 | 43 → 40 | 41 → 11 |

Every family moves its target feature monotonically with severity. The features are therefore separable enough for a gate and family-routing study (P4-T6/T7). Observations carried forward:

1. **Detection is a quality feature in its own right.** The detection-contract success rate is 0 % on lowres-extreme (14 px) and noise-extreme. It is 29 % on lowres-severe, 14 % on combined-severe, 11 % on realistic2-severe, and 89 % on lowcontrast-severe. These are the inputs where Phase 3 saw identity substitution (F1) and A5 collapse. Phase 3 hid this with oracle alignment; a production pipeline cannot.
2. **Noise inflates sharpness.** `lap_norm_c` *rises* with noise. A sharpness-only gate would call noisy faces "sharp", so the gate must use noise and sharpness jointly.
3. **Clean FairFace chips are not blockiness-free (1.31)**, because of the Q≈75 JPEG floor (P2-T2). The JPEG severity scale starts above that floor.
4. **Small native inputs bias `noise_sigma` upward** (0.9 → 7.3 for lowres with no added noise). Noise must be read jointly with `in_min_side`.

Main-population (test) features are computed with the same frozen code and stored for P4-T6 evaluation. **They were not used for any calibration.**

## 4. Versioning and Reproducibility

- Each `_report.json` records the quality version, taxonomy version + SHA-256, detector id + SHA-256, the population manifest SHA-256 and the output manifest SHA-256.
- Features are deterministic: the tests assert identical output on repeat.

| Population | Output manifest SHA-256 (prefix) |
|---|---|
| p4dev | `d2c74fd1…` |
| main | `8da7583e…` |

## 5. Limits

- No real-world observations (G-3), so there are no CAPTURE_PROTOCOL labels and no observed-severity re-estimation yet.
- Pose is proxy-only, and fes-syn-v1 is mostly frontal, so POSE and CROP are unrepresented.
- Occlusion is not measurable.
- Synthetic-calibrated thresholds are explicitly **not valid** for real inputs until re-calibrated on the real dev split (taxonomy `calibration` block).
