# R5-8: Local-Patch Artifact Detector — Report (final enhancement-safety experiment)

Date: 2026-10-08 · Task: `Tasks/Phase-8/R5-8_Local_Patch_Artifact_Detector_PROMPT.txt` · Baseline `33922fe` (R5-6 live)

## 1. Executive summary

**Final question:** can a small deterministic local-patch artifact detector make the R5-7 NAFNet candidate safe enough to retain its better full_image benefit?

**Answer under the task's stated standard: NO.** The R5-8 detector (`p5-nafnet-local-artifact-1.0.0`, a single "dottedness" signal):

- **Blocks all 8 artifact escapes R5-7 returned on archive TEST** while keeping the full R5-7 benefit (full_image ΔSFace +0.113 vs +0.053 live; whole-policy +0.049 vs +0.016).
- Costs only ~0.1% of outputs, runs in 2.8 ms per chip, and keeps ztest and real photos identical to R5-6.
- **Lets some localized artifacts through:**
  - On DEV it misses 3 of 17 labelled localized artifacts: two faint, coarse (≈ 6–8 px) grids and one borderline dotted case.
  - On TEST at least **1 clearly visible** (test/image_89280, a dotted cheek patch) plus 2 faint ones were returned.
  - No simple deterministic signal I tested separates the coarse-grid mode without blocking thousands of good outputs.
- **Verdict: RED.** Per the task's RED path, `src/`, `configs/` and `tests/` are left in the **R5-6 state**. The candidate is preserved as a patch.

> ⚠ **Material safety finding about the live R5-6 baseline.**
>
> The same scan shows that **R5-6 itself returns localized dotted/grid artifacts on archive TEST**, roughly **10× more than the R5-8 candidate**:
>
> | | R5-6 live | R5-8 candidate |
> |---|---|---|
> | face_chip | ≈ 4 | 1 faint |
> | full_image | ≈ 25 | 1 clear + ≈ 1–2 faint |
>
> Most R5-6 full_image cases are dotted streaks along the right or bottom chip edge, caused by NAFNet's internal zero padding.
>
> R5-6 and R5-7 never detected them because their checks used global chroma p99.9 and the GT chroma signal, which do not see this pattern. "Retain R5-6" is therefore **not** the artifact-free option it was assumed to be. **This needs an explicit owner decision** (§20). R5-6 stays live/default as the task requires; nothing was switched.

## 2. R5-6 baseline (frozen, re-verified)

| Item | Value |
|---|---|
| Baseline commit | `33922fe` |
| Live route | `p5-nafnet-route-1.0.0`, seal `eb6fe101…` (config file sha256 `b2d0a2f2…`, unchanged) |
| NAFNet | `models/M28_nafnet/NAFNet-GoPro-width64.pth`, sha256 `329d3ab4…`, MIT, production licence REVIEW_REQUIRED |
| Image | `fes-dev:cpu-p5` (`sha256:403e0a63ea01`) |
| Smoke at start | 91 passed |
| Regression at end (restored state) | 253 passed, 5 skipped |

R5-6 control results (from R5-6) are in §14.

## 3. R5-7 candidate reconstruction (research only, opt-in)

The R5-7 changes were rebuilt as an **opt-in profile** (`FES_NAFNET_PROFILE=r5_8`), with the default byte-identical to R5-6:

- the R5-7 global guard values: chroma p99 ≤ 20.4, p99.9 ≤ 36.4;
- out-of-image reflect fill (NAFNet route only; the A4 route and the returned chip are unchanged);
- a reflect-pad16 NAFNet factory (`nafnet_factory_pad16`);
- a local-guard hook that runs after the global guard; any failure returns ORIGINAL with fallback `LOCAL_ARTIFACT_GUARD`.

**Benefit reproduced on DEV before adding the detector:** face_chip +0.157 and full_image +0.117 ΔSFace (R5-7 TEST: +0.154 and +0.113).

## 4. DEV dataset and labels (archive `val` only)

- **Corpus:** the R5-7 candidate with no local guard (`scripts/r5_8_dev_corpus.py`) on all **19,290 val pairs × 2 modes**, giving **16,545 NAFNet outputs** (face_chip 8,065, full_image 8,480). Every output was saved, plus identity vs GT.
- **Review:** zoomed visual review of every output with local-signal value ≥ 8 (face_chip) or ≥ 30 (full_image), random 12 per lower band (seed 0), the tails of every other signal, p99.9 > 20, and 60 random per mode. **477 globally-passing outputs** were labelled zoomed.

| Label | face_chip | full_image |
|---|---|---|
| LEGITIMATE | 215 | 243 |
| LOCALIZED_ARTIFACT | 13 (9 pass the global guard) | 8 |
| DESTRUCTIVE (all already blocked by the global guard) | 11 | 0 |
| UNCERTAIN | 2 (kept as uncertain, not clean) | 0 |

**Artifact types seen:**
1. Colourful dotted/rainbow patches (most common).
2. Faint coarse grids, low-colour, ~6–8 px period (full_image: 68258, 77307).
3. Dotted streaks along chip edges.

## 5. Local features investigated (deterministic, chip-level)

| Signal | Description | Result |
|---|---|---|
| windowed mean chroma change (max, excess over median) | 8×8 box mean of the output-vs-input chroma change | does not separate |
| HF chroma growth (max, p99.9) | output's high-frequency colour energy minus the input's, 8×8 max | face_chip ok; full_image legit up to 83 vs artifact 46: no |
| HF chroma / luma ratio | local ratio of high-frequency colour to luma energy | no |
| connected high-chroma regions (> 20 / > 30) | largest connected area of large chroma change | no |
| **"dottedness" in chroma (`l_dot_c_max`)** | **second-difference energy present in both x and y, output minus input, 8×8 max** | **best: separates 15/17** |
| dottedness in luma | same, on the luma channel | no |
| multi-scale dottedness (step 2–4, 8/16-px windows) | coarser second differences, aimed at 6–8 px grids | catching the coarse grids would block 2,500–5,600 full_image outputs |
| residual gradient periodicity (global and 32-px windows) | periodicity of the output-minus-input gradient profile | artifacts score lower than legitimate outputs: no |

No ML classifier was built, because the task prefers the smallest deterministic detector. The coarse-grid mode remains undetected.

## 6. Final detector

- **Version:** `p5-nafnet-local-artifact-1.0.0`.
- **Rule:** reject if `l_dot_c_max` > **24.5** (8×8 window, reflect borders).
- **Properties:**
  - deterministic, bounded (box filters + 3-tap filters), 2.8 ms per chip;
  - no model, no network;
  - greyscale-safe (no chroma, so no trigger);
  - invalid input or a detector exception → ORIGINAL;
  - accepted pixels are never modified;
  - auditable reason `ARTIFACT_LOCAL_PATCH:l_dot_c_max`.
- **Threshold rule, fixed before computing:** 0.9 × the smallest DEV artifact value that the signal separates from every DEV legitimate value of the same mode.

## 7. DEV calibration (`data/manifests/phase5_rapid/r5_8/local_calibration.json`)

| | face_chip | full_image |
|---|---|---|
| NAFNet outputs / global-guard pass | 8,065 / 8,045 | 8,480 / 8,459 |
| Localized-artifact recall | **8/9** (miss 77722, overlaps a legit output at 24.8) | **6/8** (misses: coarse grids 68258, 77307) |
| Labelled legit blocked | 1/215 | 0/243 |
| Population block (fallback) rate | 0.12% | 0.13% |
| ΔSFace before → after local guard | +0.153 → +0.153 | +0.114 → +0.114 |
| Runtime | 2.8 ms | 2.8 ms |

## 8. Tests

- 20 new candidate tests in `tests/test_r5_8_candidate.py`:
  - legitimate accepted; localized artifact blocked (a patch the global guard misses); destructive blocked;
  - greyscale accepted; detector exception → ORIGINAL; invalid outputs → ORIGINAL;
  - deterministic, with no pixel modification;
  - profile isolation (default = R5-6, app default factory); reflect-pad16 (real-model bit-identity at multiples of 16);
  - full_image fill with the black border kept; the live profile never fills.
- **With the candidate code:** 276 passed, 2 skipped (`FES_RUN_NAFNET=1`); UI 21/21.
- **After restoring R5-6:** 253 passed, 5 skipped. The candidate tests ship inside the patch.

## 9. Frozen configuration and hashes (`data/manifests/phase5_rapid/r5_8/frozen_hashes.txt`)

- **Candidate config:** `r5_8_nafnet_candidate.json`, seal `631acd73ce084f2d…` (route 1.1.0 + local 1.0.0; file sha256 `388d8fe0…`).
- **Live config:** unchanged (`b2d0a2f2…`).
- **Commit at freeze:** `33922fe` plus the candidate working tree, preserved in `Planning/R5-8_candidate/r5_8_candidate.patch` (applies cleanly to HEAD).
- **TEST observations:** none were used to change anything.

## 10. Archive TEST results (one final run; unchanged R5-6 harness, 3 replicas, `FES_NAFNET_PROFILE=r5_8`)

| Run | N | HTTP 200 / valid | NAFNet out | Global blocks | Local blocks | ΔSFace (enhanced) | Δdlib | Policy ΔSFace |
|---|---|---|---|---|---|---|---|---|
| face_chip | 19,290 | 100% / 100% | 8,120 | 20 | 7 | +0.154 [0.151, 0.157] | +0.019 | +0.065 |
| full_image | 19,290 | 100% / 100% | 8,429 | 31 | 10 | +0.113 [0.110, 0.115] | +0.016 | +0.049 |

- Clearly improved / harmed: face_chip 3,267 / 7; full_image 2,718 / 20.
- 0 timeouts, 0 OOMs, 0 worker restarts on all replicas.

## 11. ztest results

Both modes: **0 NAFNet attempts** and 0 fallbacks; all outputs are ORIGINAL and byte-identical to the input. Verification: SFace/dlib 0 flips, 0 new false accepts. Containment is unchanged.

## 12. 200 real photos (full_image, auto)

Identical to R5-6: **4 NAFNet** (all pass both guards), **2 A4** (strong_noise), **194 ORIGINAL**, 0 fallbacks, 0 failed requests.

## 13. Artifact inspection (automated scan of every returned output: `scripts/r5_8_test_scan.py`)

- **R5-7 escapes:** all 8 (face_chip 95636, 88924; full_image 8964, 96161, 97266, 90856, 85126, 96475) were **blocked** by the local guard (`r57_escapes_fate.json`).
- **R5-8 returned outputs:** the top 12 per signal per mode (dottedness, chroma p99.9, HF growth, luma dottedness) were inspected zoomed:
  - full_image **89280**: a clearly visible dotted cheek patch (dottedness 15.4 < 24.5), escaped;
  - 91965 (edge specks) and one face_chip case (small nose streak, dottedness 14): faint, escaped;
  - all others looked legitimate.
- **R5-6 control** (the same scan over R5-6's returned outputs):
  - **face_chip:** 4 returned outputs are above the R5-8 limit (dottedness up to 84). All 4 are visible artifacts: 94918, 98715, 99442, 93965.
  - **full_image:** 25 returned outputs are above the limit (dottedness up to 183). Nearly all show dotted streaks along the right or bottom edge, and some show face patches or a full-face grid: 88572, 90814, 83106, 8650, 97456, 95481, 98424, 91662, 89324, 89666, 84289 and others.

## 14. R5-6 vs R5-8

| Metric | R5-6 (live) | R5-8 candidate | Change |
|---|---|---|---|
| NAFNet eligible, face_chip / full_image | 8,147 / 5,769 | 8,147 / 8,470 | +2,701 (full_image) |
| NAFNet output returned | 7,393 / 5,654 | 8,120 / 8,429 | +727 / +2,775 |
| Global guard blocks | 754 / 115 | 20 / 31 | ↓ |
| Local guard blocks | — | 7 / 10 | new |
| ΔSFace (enhanced), face_chip / full_image | +0.149 / +0.053 | +0.154 / **+0.113** | +0.005 / **+0.060** |
| Policy ΔSFace | +0.057 / +0.016 | +0.065 / **+0.049** | +0.008 / **+0.033** |
| Clearly improved / harmed, full_image | 806 / 28 | 2,718 / 20 | ▲ / ▼ |
| **Returned localized artifacts (visual, archive TEST)** | **≈ 29** | **≈ 2–3** (1 clear) | **≈ −90%** |
| ztest NAFNet attempts / flips | 0 / 0 | 0 / 0 | = |
| Real photos NAFNet / A4 / ORIGINAL | 4 / 2 / 194 | 4 / 2 / 194 | = |
| Failures / timeouts / restarts | 0 | 0 | = |
| NAFNet infer median (128 px) | 0.51–0.57 s | 0.54–0.68 s (CPU shared with scoring) | ≈ |

## 15. Safety analysis

- **Against the task's GREEN criteria:** "no destructive/localized artifact returned" fails. DEV has 3 known misses (2 faint coarse grids, 1 borderline), and TEST has ≥ 1 clear escape (89280). **RED.**
- **What holds:**
  - ztest containment, real-photo behaviour, service stability and determinism;
  - every destructive output is blocked (global guard);
  - every R5-7 escape is blocked;
  - returned localized artifacts drop by roughly 90% relative to the live R5-6.
- **The baseline issue:** R5-6's own safety evidence did not cover this artifact class. Its "0 artifacts returned" (R5-6 report) rested on chroma-based checks that miss dotted/grid patches. By the standard applied to R5-8 here, **R5-6 would also be RED**.

## 16. Identity-benefit analysis

These are SFace/dlib proxies vs the sharp GT of the same photo on synthetic blur (not Feature-Match).

- The local guard costs essentially no benefit (DEV and TEST ΔSFace unchanged to 3 decimals).
- The candidate keeps the R5-7 gains: full_image enhanced ΔSFace about 2.1× R5-6, whole-policy about 3.1× R5-6.

## 17. Runtime and service stability

| Item | Value |
|---|---|
| Local guard | 2.8 ms per chip |
| Replicas | 3 × ~1 GB |
| Throughput | 8–11 requests/s with WelOrbit stopped, ~5–9 with it running |
| Failures | 0 restarts, timeouts, OOMs or failed requests across 39,080 requests |
| Network | none (offline) |

## 18. Licensing status

- NAFNet-GoPro is MIT, but production use is **REVIEW_REQUIRED** (no licence reviewer named).
- The archive dataset's provenance and licence are unknown; it was used for evaluation only.
- No production or commercial approval is claimed.

## 19. Verdict: **RED**

Known material localized artifacts escape the detector (DEV and TEST). Following the task's RED path:

- **`src/`, `configs/`, `tests/` = R5-6 state** (verified: no diff vs `33922fe`; regression 253 passed, 5 skipped).
- **Evidence committed:** scripts, DEV corpus manifests, visual labels, calibration, TEST scans and this report.
- **Rejected candidate preserved:** `Planning/R5-8_candidate/r5_8_candidate.patch` (+ `r5_8_nafnet_candidate_REJECTED.json`).
- **No further artifact-detector or model-search task is started.**

## 20. Adopt vs retain-R5-6 recommendation

- **Task-rule answer: NO.** Retain R5-6 as live/default, stop enhancement experimentation, and move to Feature-Match with R5-6 when access and data exist.
- **However:** the evidence shows R5-6 returns about 10× more of the same artifact class than the rejected candidate. Retaining R5-6 is the *rule-compliant* choice, not the *safer* one. **I recommend the owner decides explicitly between:**
  1. **Keep R5-6 live as-is.** Rule-compliant; accepts ≈ 29 localized artifacts per ~13k enhanced archive outputs (≈ 0.2%).
  2. **Adopt the R5-8 candidate** (apply the patch; it is already opt-in through `FES_NAFNET_PROFILE=r5_8`). Fewer artifacts (≈ 0.02%) and about 3× the full_image benefit, but formally RED.
  3. **Disable NAFNet enhancement for production use** (`FES_ROUTE_MODE=a4`, or ORIGINAL-only) until a human-review gate or a better detector exists.
- **Feature-Match:** whichever option is chosen, Feature-Match validation (≥ 20 people, real model and threshold) is the next task. Enhancement experimentation stops here unless the owner explicitly approves more.
