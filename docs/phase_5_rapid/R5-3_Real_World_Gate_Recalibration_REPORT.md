# R5-3 — Real-World Gate Re-calibration

| Field | Value |
|---|---|
| Phase / task | 5 Rapid Validation / R5-3 |
| Date | 2026-10-04 |
| Builds on | `92fe16e` (R5-1 service), `c5ce406` + `db43fc0` (model comparison, CRLF fix; both preserved unchanged) |
| New gate | `p5-rwgate-1.0.0`, `configs/phase5/r5_rwgate.json` (seal `78837a8d…`), code `src/fes_service/rwgate.py` |
| Frozen Phase 4 policy | `configs/phase4/t6_gate.json` / `t7_routing.json` **unchanged**, still selectable (`gate="t6"`) |
| Evidence | `data/manifests/phase5_rapid/{gap, rwgate}/` |
| Tests | `tests/test_r5_rwgate.py` 26 passed · full repository suite in `fes-dev:cpu-p5`: **182 passed, 1 skipped** (opt-in real-A4 test), 20 m 31 s. The uncommitted, superseded 5.1 baseline test (`tests/test_p5_baseline.py`, pins code hashes) was excluded. |

> **Scope of the conclusion.** The 200 real photos have **no same-person references**, so this task establishes gate *behaviour* only. **Nothing here shows that the new gate improves identity matching on real images.** The A4 benefit boundary is still evidenced on synthetic dev data only.

## 1. Executive summary

- **Why only 1/200 reached A4.** Synthetic artefacts in the old rules blocked real photos, **not the photos' actual quality**:
  - the T6 blur rules required noise σ ≤ 0.28, and synthetic blur is noise-free while real blur always carries grain;
  - the T7 "LOWRES" rule sent every face under 71 px IOD to the original.
- **New gate.** It replaces both with:
  - a **grain-robust blur measure** (`lapd1_c`: Laplacian after a σ=1 pre-filter). Adding σ=4 grain changes it by ≤ 4 %; the old measure changed 6×;
  - **explicit face-size zones** taken from a small dev-only gap check. That check shows A4's benefit on strong blur and strong noise **persists down to about 35 px IOD** (168-px chips: 34 helped / 0 harmed; 112-px chips: 32 / 1).
- **Real photos.**
  - **3/200 → A4** (was 1/200), and the reasons are now evidence-based.
  - **88** are degraded but outside A4's evidenced zone, mainly mild scan noise plus a small face. Synthetic evidence says A4 *harms* identity there (noise-mild: 0 helped / 11–12 harmed), so they stay original and are now labelled as degraded rather than "acceptable".
  - **85** are protected: 80 unsafe or uncertain (tiny, unrecognisable, low confidence, no face) plus 5 photometric or compression cases.
- **Safety.**
  - tiny, unrecognisable, extreme and uncertain inputs are blocked;
  - greyscale stays greyscale;
  - A4 failures fall back to the original;
  - only A4 can be routed.
- **Synthetic regression (test set, never used for fitting).**
  - Identity gain is kept under both evaluators (+0.0036 ▲ / +0.0003 ▲) with **0 flips**.
  - **However, 3 mild inputs are now enhanced** (Phase 4: 0). All three come from one identity whose *clean reference photo is itself strongly blurred* (§8). This is reported, not tuned away, since tuning on test is not allowed.

**Answer to the task question.** The gate now correctly recognises real *strongly* degraded but recognisable faces as A4 candidates, and grain no longer hides blur. It protects tiny, unrecognisable and uncertain inputs. It deliberately does **not** admit the large group of mildly or moderately degraded vintage faces, because A4 has no identity evidence there; the synthetic evidence is negative. On this photo set, the evidence-led boundary admits few photos.

## 2. Previous gate behaviour (frozen T6/T7 on the 200 real photos)

| Decision | Count | Detail |
|---|---|---|
| UNSAFE_OR_UNCERTAIN | 75 | no face 25, face < 12 px 22, detector score < 0.85 17, pose 11 |
| DO_NOT_ENHANCE | 120 | no T6 rule fired |
| ENHANCE_CANDIDATE | 5 | 4 → LOWRES bucket → original, **1 → A4** (img147) |

## 3. Root cause of the 1/200 result

| Mechanism | Effect on the 125 safe real photos |
|---|---|
| T6 rules 2/3 = "blur **and noise σ ≤ 0.28**". Synthetic blur is mathematically noise-free (dev blur families: σ 0.29–0.31). | 13 of 15 blur-like photos were blocked **only** by the noise condition (real noise median 1.1, p10 0.43) |
| The T6/T7 sharpness feature `lap_norm_c` measures grain as "detail" | Extreme blur + σ=4 grain: 0.0047 → 0.027, which reads as a sharp image |
| T7 LOWRES bucket = IOD < 71.4 px (2 % quantile of 224-px FairFace chips, where the face fills the frame) | 86/125 real faces fall under it, and LOWRES takes precedence over NOISE/BLUR; 4 of the 5 gated photos were lost here |
| T6 rule0 noise σ ≥ 9.84 | A genuine level (synthetic σ=15 noise); only 3/125 real photos reach it |

The only real photo routed by T6 (img147) is in fact blurrier than anything A4 was ever measured on (`lapd1` 0.0005). The new gate blocks it as *beyond evidence*.

## 4. New gate logic (decision order; everything except ENHANCE_A4 returns the original)

| # | Outcome | Rule | Category |
|---|---|---|---|
| 1 | UNSAFE (original + escalate) | T6 safety, **unchanged**: no valid face / det score < 0.85 / \|yaw\| > 0.25 / NaN pose | uncertain_or_no_face |
| 2 | UNSAFE | IOD < `iod_min_recognizable` | tiny |
| 3 | UNSAFE | `lapd1_c` < `lapd1_floor` (blurrier than any measured A4 case) | unrecognizable |
| 4 | DO_NOT_ENHANCE | luma < `luma_mean_lo` or contrast < `rms_contrast_lo` (A4 has no identity benefit: P4-T4 P7) | photometric |
| 5 | DO_NOT_ENHANCE | blockiness > `blockiness_hi` (no A4 evidence) | compression |
| 6 | **ENHANCE → A4** | noise σ ≥ `noise_min` **and** IOD ≥ `iod_min_a4` | degraded_recognizable (strong_noise) |
| 7 | **ENHANCE → A4** | `lapd1_c` ≤ `blur_max` **and** IOD ≥ `iod_min_a4` | degraded_recognizable (strong_blur) |
| 8 | DO_NOT_ENHANCE | rule 6/7 degradation but IOD < `iod_min_a4` | degraded_small |
| 9 | DO_NOT_ENHANCE | outside the clean-reference range (blur/noise/size) but not in A4's zone | degraded_no_a4_evidence |
| 10 | DO_NOT_ENHANCE | looks like a clean reference | acceptable |

A NaN feature never enhances. Only `A4` or the original can be routed (property-tested over 2,000 random feature vectors).

## 5. Exact rule / threshold changes

| Threshold | Value | Source (DEV only) | Replaces |
|---|---|---|---|
| `noise_min` | 9.837 | T6 rule0, unchanged | — |
| `blur_max` (`lapd1_c`) | **0.002617** | Scan of U = benefit − 3·harm over 224-px dev pairs: peak 14 helped / 1 harmed; U turns negative beyond about 0.0042, where motion/severe blur enters | T6 rules 1–3 (`hf_ratio_c`, `lap_norm_c ≤ 0.0198 ∧ noise ≤ 0.28`, `blockiness ∧ noise ≤ 0.28`) |
| `lapd1_floor` | 0.000974 | 0.9 × the blurriest dev blur pair A4 was measured on | new |
| `iod_min_a4` | **35.3 px** | 2 % IOD quantile of the smallest contiguous size tier with U > 0 and harm ≤ benefit/3 (§5.1) | T7 LOWRES (71.4 px) as an A4 blocker |
| `iod_min_recognizable` | 16 px | Between the about-11 px tier (harm-only for every model) and the about-22 px tier (harm-dominated) | (T6 safety < 12 px kept) |
| `luma_mean_lo`, `rms_contrast_lo`, `blockiness_hi` | 47.1, 25.4, 1.523 | copied from frozen T7 (clean dev-reference quantiles, no outcome data) | unchanged semantics |
| `clean_lapd1_lo`, `clean_noise_hi`, `clean_iod_lo` | 0.00558, 1.084, 71.4 | clean dev references (labels only; no routing effect) | — |

**5.1 Gap check (dev only, 224 pairs).**
- **Setup.** The 28 p4dev references, at 168 px and 112 px, with the four A4-favourable cells re-created at that size: Gaussian blur σ scaled with size, noise σ per pixel unchanged.
- **Scoring.** A4 and B0 were scored by SFace + dlib (`scripts/r5_gap_population.py`).

| Tier (chip / median IOD) | routed | helped | harmed | U |
|---|---|---|---|---|
| 224 / 83 px (Phase 4) | 55 | 31 | 1 | 28 |
| 168 / 66 px | 43 | 34 | 0 | 34 |
| 112 / 45 px | 39 | 32 | 1 | 29 |

## 6. 200-image before / after (full detector → gate → router → A4/original, real A4)

| | Frozen T6/T7 | R5-3 rw gate |
|---|---|---|
| **→ A4** | **1** | **3** (img72 strong noise · img154 strong blur · img165 strong noise). 0 A4 failures; 2 greyscale preserved. |
| → original, unsafe/uncertain | 75 | 80 |
| → original, do-not-enhance | 124 | 117 |
| transitions | — | B0→A4: 3 · A4→B0: 1 (img147, beyond evidence) · B0→B0: 196 |

## 7. Category breakdown (rw gate, 200 photos)

| Category | n | Reasons |
|---|---|---|
| No face / detection failure before the gate | 32 | NO_FACE_DETECTED 25, landmark/size failures |
| Uncertain (safety rules) | 43 | low detector confidence 17, alignment unreliable 11, FACE_TOO_SMALL 22 (part counted above) |
| Tiny | 2 | insufficient_face_size (img10, img89) |
| Unrecognizable / beyond evidence | 3 | extreme blur (img147, img148, img153) |
| Photometric / compression | 5 | photometric 4, compression 1 |
| **Degraded, recognizable → A4** | **3** | strong_noise 2, strong_blur 1 |
| Degraded, no A4 evidence → original | 88 | noise+lowres 47, lowres 22, noise 11, blur(+lowres/noise) 8 |
| Acceptable → original | 24 | — |

`human_recognizable` is recorded as **NOT_REVIEWED** for every photo. The protocol does not accept the model's own visual judgement as human evidence; the contact sheets (§11) are prepared for a human reviewer.

## 8. Safety analysis

| Requirement | Result | Evidence |
|---|---|---|
| Good images not enhanced | 24 acceptable real photos left untouched. Synthetic clean references → DO_NOT_ENHANCE | tests, real run |
| Recognizable degraded images can reach A4 | yes, including with real grain (σ=3 grain on an extreme-blur chip: T6 → original, rw → A4) | `test_recognizable_blurred_face_reaches_a4_and_grain_does_not_block` |
| Tiny faces protected | IOD < 16 → UNSAFE; IOD < 35 → no A4 | tests, img10/img89 |
| Unrecognizable faces protected | `lapd1` below the floor → UNSAFE (σ=12 smear test; img147/148/153) | tests, real run |
| Uncertain → original | T6 safety rules unchanged; no-face input → original + escalate | tests |
| Greyscale stays greyscale | channels identical on output; colour inputs untouched | `test_grayscale_in_grayscale_out`; img154, img165 |
| A4 failure → original | MODEL_FAILURE fallback under the rw gate (real isolated worker) | `test_a4_failure_falls_back_under_rw_gate` + R5-1 failure tests |
| No A1/A2/A5 (or GFPGAN/CodeFormer) routing | `decide` can only return A4 or the original; the pipeline only receives the A4 adapter | property test + config assert |

**Safety regression found (synthetic test, reported, not tuned away).** The rw gate enhances **3 of 252 mild** test inputs (Phase 4: 0). All three are confirmed harms, and all belong to **one identity (ff-val-365)**:
- Its *clean reference* is itself strongly blurred: `lapd1` 0.00238, below `blur_max`. The next-softest of the 28 test references is 0.0060.
- The gate correctly sees a blurred face. A4 sharpens it **away from an equally blurred reference**, which the evaluators count as harm.
- A reference-free gate cannot tell "always-soft photo" from "degraded photo".
- Changing thresholds because of this test identity would be tuning on test, so it was not done. On dev, the mild-enhanced count is 0.

## 9. Greyscale handling

- `rwgate.is_gray`: all three channels are equal within 2 grey levels at the 99th percentile, so greyscale scans saved as RGB/JPEG count as grey.
- For a greyscale input chip, the A4 output is converted to luma and replicated to 3 channels, with the warning `grayscale_preserved`.
- A4 still runs in RGB internally; only the returned image is constrained.
- Colour inputs are never converted.
- ORIGINAL returns are the untouched input bytes anyway.

## 10. Synthetic regression (A4 outputs already scored in Phase 3/4; no new inference)

| Population | Policy | A4 routed | ΔID SFace | ΔID dlib | benefit / harm | flips | mild enhanced |
|---|---|---|---|---|---|---|---|
| `main` (TEST) | T6/T7 frozen | 63 | +0.0044 ▲ | +0.0005 ▲ | 38 / 3 | 0/0 | 0 |
| `main` (TEST) | **rw** | 50 | **+0.0036 ▲** | **+0.0003 ▲** | 29 / 5 | **0/0** | **3** (one identity, §8) |
| `p4dev` (DEV) | T6/T7 frozen | 73 | +0.0049 ▲ | +0.0006 ▲ | 38 / 6 | 0/0 | 0 |
| `p4dev` (DEV) | **rw** | 55 | +0.0044 ▲ | +0.0004 ▲ | 31 / 1 | 0/0 | 0 |

**What changed on synthetic data, stated explicitly.** On `main`, 27 routes changed:
- 10 gaussblur-extreme benefits are no longer routed. They sit in the 0.0026–0.004 `lapd1` band, which the U-optimal threshold excludes.
- The few motion-blur cases T6 routed are dropped.
- 6 are added: 2 benefits and the 3 mild harms above.

The rw gate is somewhat less effective than T6/T7 on the synthetic generator T6 was fitted to, which is expected since T6 exploited generator artefacts. It still shows a positive identity effect with no flips. The frozen T6/T7 policy itself is unchanged: R5-1's 896/896 decision equivalence test still passes.

## 11. Representative examples (`experiments/phase5_rapid/outputs/rwgate/`, git-ignored)

| Sheet | Content |
|---|---|
| `admitted_A4_in_out.jpg` | img72 (portrait grain → smoothed, face intact) · img154 (real motion blur/ghosting; A4 changes little) · img165 (newspaper **halftone** pattern measured as noise) |
| `blocked_tiny_or_unrecognizable.jpg` | img10, img89 (tiny; upscaled chips) · img147, img148, img153 (blurrier than any A4 evidence) |
| `left_original_degraded_no_a4_evidence.jpg` | the typical vintage faces: scan grain + small/medium size; A4 evidence negative at this noise level |
| `left_original_acceptable.jpg`, `blocked_photometric_or_compression.jpg`, `unsafe_uncertain_or_no_face.jpg` | the remaining groups |

These descriptions are the model's own visual notes, not human review.

## 12. Known limitations

1. **No identity evidence on real data.** The 200 photos have no references, so the A4 zone is synthetic-evidenced (now at 3 face sizes).
2. **Blur type is not distinguished.** `lapd1` measures blur strength, not kind. Real motion blur/ghosting (img154) can be admitted, although synthetic motion blur was neutral-to-harmful for A4.
3. **Halftone or print-screen patterns read as "noise"** (img165). The noise estimator was not designed for them, and A4 behaviour on halftones is unmeasured.
4. **Always-soft source photos** are indistinguishable from degraded ones (§8). This is the cause of the 3 mild synthetic admissions.
5. **The large mild-noise and small-face group (88 photos) is not enhanced.** That is evidence-based (negative synthetic evidence), but it means A4, as validated, will rarely act on typical vintage photos.
6. **Thresholds are fitted on 28 dev identities.** Fold-level stability was not re-estimated in this rapid task; the blur scan shows a clear but narrow optimum (0.0026–0.0030).
7. **Gap-check sizes were constructed** by downscaling clean references; real small faces carry different degradation.

## 13. What remains unproven

- Any **identity or matching improvement on real images**, for the new gate or for A4.
- Whether real motion blur, real scan grain and halftone respond to A4 as the synthetic families do.
- Human-perceived recognizability of the routed/blocked groups (`human_recognizable` = NOT_REVIEWED).
- Behaviour on the user's own target imagery (CCTV/phone/ID-style), which differs from vintage photos.
- Feature-Match behaviour (not integrated; out of scope).

## 14. Recommended next step

1. **Real identity-labelled evaluation (R5-2)**, with the rw gate frozen as of this commit. In order of value:
   - **(a)** the user's own real low-quality/reference pairs plus Feature-Match evaluation access;
   - **(b)** LFW-Test (real degraded faces of LFW identities with other photos) as a research proxy, only if instructed.

   Measure A4 vs the original *inside* the admitted zone **and** in the "degraded, no A4 evidence" zone. The latter decides whether the conservative boundary loses useful cases.
2. **Human review** of the contact sheets: fill `human_recognizable`.
3. **Only if (1) shows harm on real motion blur or halftone:** add a targeted blur-type or halftone guard. Do not change thresholds without identity evidence.

## Reproduce

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/r5_gap_population.py                     # dev gap population
# A4 + B0 on r5gap, SFace + dlib scoring (fes-dev:cpu-p5): see scripts/r5_gap_population.py docstring / p3_* scripts
PYTHONPATH=src .venv/Scripts/python.exe scripts/r5_rwgate_calibrate.py                   # → configs/phase5/r5_rwgate.json
PYTHONPATH=src .venv/Scripts/python.exe scripts/r5_rwgate_synthetic.py                   # synthetic regression
FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_realphotos_rwgate.py          # 200 photos, real A4
FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python -m pytest -q tests/test_r5_rwgate.py tests/test_r5_service.py
```
