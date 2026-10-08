# WelOrbit R5-10 — NAFNet Blur-Eligibility Recalibration (Moderate-Blur Routing + Safety Verification)

Date: 2026-10-08 · Baseline commit `e636c58` · Model unchanged: NAFNet-GoPro width64 (`329d3ab4…`)

## 1. Executive summary

- **Question:** can NAFNet-GoPro safely and usefully run on moderately blurred, still-recognisable faces, or is the strong-blur-only boundary justified?
- **Answer:**
  - On synthetic DEV and TEST data, **yes**. NAFNet's identity benefit does not fall off above the current cutoff, and harm stays near zero.
  - On real photos, **not demonstrated safe**. The 200-real-photo check gave R5-10 two new NAFNet attempts. One of them, a grainy vintage print, came back with visible colour mottling that both output guards passed.
- **Verdict: RED.** This follows the task's explicit RED rule "real-photo safety regresses". Nothing changes in default or live behaviour. R5-10 remains an opt-in research profile that is **not enabled** anywhere. Threshold experimentation stops here.

What was done:
1. **Found that the old boundary was never NAFNet's own.** `blur_max = 0.002617` was inherited unchanged from the A4 route. It was never calibrated for NAFNet.
2. **Pre-registered the selection rules** (`protocol.json`) before any result was seen.
3. **Collected DEV evidence:**
   - archive `val`: 19,290 pairs × 2 modes;
   - sharp archive GT faces, for clean protection;
   - p4dev + r5gap, for other degradations.
4. **Selected the boundary under the rules: `blur_max` 0.002617 → 0.005.**
   - The next grid point, 0.005579, fails on two counts:
     - it admits 2.87% of sharp faces;
     - it admits jpeg / low-contrast / under-exposed inputs with negative Δ.
5. **Froze and sealed the config** (`cc2347ac…`) and passed 191 + 32 regression tests.
6. **Ran the existing R5-6 service harness once** on the frozen candidate.

Results:

| | R5-8 | R5-10 |
|---|---|---|
| TEST face_chip NAFNet outputs | 8,120 | **11,777** (+45%) |
| TEST full_image NAFNet outputs | 8,429 | **11,968** (+42%) |
| ΔSFace of newly admitted outputs (fc / fi) | — | **+0.155 / +0.119** (= strong-blur zone) |
| Clearly harmed, newly admitted (fc / fi) | — | 0.14% / 0.17% |
| Whole-policy ΔSFace face_chip | +0.065 | **+0.094** |
| Whole-policy ΔSFace full_image | +0.049 | **+0.071** |
| ztest | contained (0 enhanced) | contained (0 enhanced) |
| Returned localized artifacts in new TEST outputs | — | 1 clear in 7,196 (≈ R5-8's rate) |
| **200 real photos: visible artifacts returned** | **0** | **1** (img118, a newly admitted grainy vintage print) |
| Service failures / restarts / OOM / timeouts | 0 | 0 |

## 2. R5-6 / R5-8 baseline (frozen before any change)

| Item | Value |
|---|---|
| Commit | `e636c58c46c8ff271e9e5b8883e3838bb6d84ef4` |
| R5-6 live route | `configs/phase5/r5_nafnet_route.json`, `p5-nafnet-route-1.0.0`, seal `eb6fe101…` (file sha `b2d0a2f2…`) — **unchanged** |
| R5-8 candidate | `configs/phase5/r5_8_nafnet_candidate.json`, `p5-nafnet-route-1.1.0`, seal `631acd73…` (file sha `388d8fe0…`) — **unchanged** |
| NAFNet checkpoint | `models/M28_nafnet/NAFNet-GoPro-width64.pth` sha256 `329d3ab4…196f` |
| R5-9 UI | running (`fes-ui`, `/ready` = ready, profiles original / r5_6 / r5_8) |
| Baseline regression | 125 passed, 5 skipped (R5 NAFNet / R5-8 / R5-9 / rwgate suites) + 31/31 UI JS tests |

## 3. Current blur thresholds

| Zone | lapd1_c | R5-6 / R5-8 decision |
|---|---|---|
| strong low-noise blur | ≤ 0.002617 | NAFNET_ELIGIBLE_STRONG_LOW_NOISE_BLUR |
| mild blur | 0.002617 < x < 0.005579 | NAFNET_REJECT_MILD_BLUR |
| clean / not blur | ≥ 0.005579 | NAFNET_REJECT_NOT_BLUR |

`blur_max = 0.002617` was **never calibrated for NAFNet**:
- It is the A4 route's strong-blur boundary: `p5-rwgate-1.0.0` `blur_scan_best`, chosen for *A4* identity benefit on p4dev + r5gap.
- R5-5 reused it unchanged when it built the NAFNet route.

`clean_lapd1_lo = 0.005579` is the 2% lapd1 quantile of clean DEV references.

## 4. DEV dataset and methodology

**DEV only.** Nothing from archive TEST, ztest, the 200 real photos or the owner's images was used to choose anything.

| DEV set | Size | Use |
|---|---|---|
| Archive `val` split, face_chip + full_image | 19,290 pairs × 2 modes | identity benefit and harm by blur band (paired sharp GT) |
| Archive `val` **sharp GT** images used as inputs | 19,290 | clean-face protection: which sharp faces a wider boundary admits, and what NAFNet does to them |
| p4dev + r5gap | 896 + 224 pairs | other degradation families (jpeg, noise, low-res, low-contrast, under-exposure) and larger / smaller faces |

**Path.** The R5-8 candidate exactly:
- sealed config, reflect-pad16 factory, reflect fill, global guard, local-patch guard;
- the only difference: the two blur rules are switched off **in memory**, so NAFNet also ran on every input that all *other* gates admit.

**Scoring:**
- Outcomes are scored **after both output guards**. A blocked output counts as ORIGINAL with Δ = 0.
- Identity is SFace (E01) and dlib (E07) cosine to the GT or reference, as Δ versus the untouched input.
- "Clearly improved / harmed" means both Δ > +0.02 / < −0.02 (the R5-6 definition).

**Tools:** `scripts/r5_10_dev_collect.py`, `scripts/r5_10_dev_analyze.py`.

**Pre-registration.** The selection rules were written to `data/manifests/phase5_rapid/r5_10/protocol.json` (sha `234d042c…`). This happened while the DEV collection was still running, before any R5-10 DEV result was inspected.
- **Grid:** {0.002617, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.005579}.
- **Rules for every marginal sub-band, in both modes:**
  - R1: at least 100 attempts.
  - R2: ΔSFace bootstrap lo95 > 0 and mean Δdlib > 0.
  - R3: harm rate ≤ current-zone rate + 1 pp.
  - R4: improved ≥ 3 × harmed.
  - R5: guard block rate ≤ current-zone rate + 2 pp.
- **Rules applied cumulatively at the boundary t:**
  - R6: at most 2% of sharp GT faces admitted.
  - R7: newly admitted non-blur p4dev inputs are not net-harmful.
- **Selection:** the largest contiguous passing grid point. Stop at the first failure; no skipping.

## 5. Blur-score distribution (archive val; inputs that pass every other gate; outcomes after both guards)

| Band (lapd1_c) | fc attempts | fc ΔSFace (lo95) | fc harm % | fi attempts | fi ΔSFace (lo95) | fi harm % |
|---|---|---|---|---|---|---|
| A very strong ≤ 0.0015 | 3,816 | +0.147 (0.143) | 0.16 | 4,244 | +0.109 (0.106) | 0.38 |
| B strong (0.0015, 0.002617] — current zone | 4,249 | +0.158 (0.155) | 0.02 | 4,236 | +0.118 (0.114) | 0.19 |
| C moderate (0.002617, 0.004] | 2,742 | +0.153 (0.149) | 0.15 | 2,665 | +0.117 (0.113) | 0.19 |
| D mild (0.004, 0.005579) | 1,332 | +0.144 (0.138) | 0.00 | 1,297 | +0.119 (0.112) | 0.23 |
| E "not blur" ≥ 0.005579 (diagnostic only) | 944 | +0.133 (0.127) | 0.00 | 898 | +0.116 (0.109) | 0.00 |

- On archive data the benefit does **not** decay with milder blur. This is expected: every archive input is genuinely blurred, so even the "not blur" band is still blurred.
- So the archive cannot locate where *clean* images begin. That limit comes from the clean-GT and p4dev checks (§8).
- Sharp-GT lapd1 quantiles: 0.5% 0.0043 · 1% 0.0047 · 2% 0.0053 · 5% 0.0062 · 50% 0.0121.

## 6. Face-size interaction

Buckets:
- 35.3 px = `iod_min`;
- 71.4 px = rwgate `clean_iod_lo`;
- 40 and 45 px are descriptive splits of the archive range only, not thresholds.

| Zone × IOD | fc n | fc ΔSFace (lo95) | fc harm | fi n | fi ΔSFace (lo95) | fi harm |
|---|---|---|---|---|---|---|
| current [0, 0.002617] · 35–40 | 3,945 | +0.158 (0.154) | 4 | 3,458 | +0.111 (0.107) | 8 |
| current · 40–45 | 2,393 | +0.153 (0.148) | 1 | 2,376 | +0.118 (0.113) | 4 |
| current · 45–71 | 1,727 | +0.140 (0.135) | 2 | 1,308 | +0.116 (0.110) | 4 |
| moderate (0.002617, 0.004] · 35–40 | 1,178 | +0.159 (0.152) | 2 | 886 | +0.111 (0.104) | 1 |
| moderate · 40–45 | 844 | +0.153 (0.145) | 1 | 821 | +0.121 (0.113) | 1 |
| moderate · 45–71 | 719 | +0.144 (0.135) | 1 | 713 | +0.118 (0.110) | 3 |
| mild (0.004, 0.005579) · 35–40 | 540 | +0.150 (0.140) | 0 | 406 | +0.126 (0.115) | 2 |
| mild · 40–45 | 419 | +0.142 (0.132) | 0 | 387 | +0.120 (0.108) | 1 |
| mild · 45–71 | 373 | +0.137 (0.125) | 0 | 432 | +0.109 (0.100) | 0 |

- Moderate-blur benefit **survives at the smallest admitted faces** (35–40 px IOD). No bucket with ≥ 100 attempts fails R2 or R3, so **no size condition was added**.
- Faces below 35.3 px remain rejected by the unchanged `iod_min` gate.
- p4dev at large faces (IOD ≈ 74–85 px) shows the same sign: newly admitted gaussblur gives ΔSFace +0.024, motionblur +0.12.
- The archive has almost no faces ≥ 71 px (1 in the moderate zone).

## 7. Quality-feature interaction

- **Other gates stay authoritative and run first.** Unit tests show noise, compression, pixelation, halftone, photometric, size, missing-feature and T6 safety rejections still block a moderate-blur input.
- **Archive val (face_chip):**
  - newly admitted inputs: noise σ median 0.38 (p95 0.55), blockiness median 1.06;
  - current zone: noise σ median 0.35 (p95 0.46), blockiness median 1.07;
  - so there is no shift toward compression or noise; 1 newly admitted grayscale input.
- **p4dev / r5gap.** Up to t = 0.005 every newly admitted input is gaussblur or motionblur. At 0.005579, jpeg (1), low-contrast (2) and under-exposure (2) start to be admitted, all with negative ΔSFace (mean −0.036). This is why R7 fails there.
- **Gap that DEV could not cover.**
  - No DEV input combines moderate blur with real film or print grain near the noise gate (noise σ ≈ 1.0, gate 1.0835).
  - Archive blur is synthetic and grain-free.
  - The p4dev noise cells lie above the gate.
  - The real-photo escape (§15) sits exactly in this gap.

## 8. Candidate thresholds considered

| t | fc eligible % of inputs | fi eligible % | fc / fi sub-band (prev, t] R1–R5 | sharp GT admitted (R6 ≤ 2%) | p4 non-blur newly admitted (R7) |
|---|---|---|---|---|---|
| 0.002617 (current) | 41.8 | 44.4 | — | 1 (0.02%) | — |
| 0.003 | 47.0 | 49.4 | ✓ / ✓ | 1 (0.02%) ✓ | 0 ✓ |
| 0.0035 | 52.1 | 54.6 | ✓ / ✓ | 5 (0.12%) ✓ | 0 ✓ |
| 0.004 | 56.0 | 58.4 | ✓ / ✓ | 14 (0.33%) ✓ | 0 ✓ |
| 0.0045 | 58.8 | 61.1 | ✓ / ✓ | 28 (0.67%) ✓ | 0 ✓ |
| **0.005** | **61.0** | **63.3** | ✓ / ✓ | **61 (1.46%) ✓** | 0 ✓ |
| 0.005579 | 62.9 | 65.2 | ✓ / ✓ | 120 (2.87%) **✗** | 5, mean ΔSFace −0.036 **✗** |

## 9. DEV benefit / harm table (marginal sub-bands, outcomes after both guards)

| Sub-band | Mode | Attempts | Returned | Mean / median ΔSFace (lo95) | Mean / median Δdlib | Improved | Harmed | Harm % | ≤ 0 % | Global / local blocks |
|---|---|---|---|---|---|---|---|---|---|---|
| reference: current zone | fc | 8,065 | 8,035 | 0.153 / 0.131 (0.150) | 0.019 / 0.016 | 3,260 | 7 | 0.09 | 6.0 | 20 / 10 |
| (0.002617, 0.003] | fc | 992 | 991 | 0.154 / 0.130 (0.147) | 0.021 / 0.017 | 436 | 3 | 0.30 | 4.3 | 0 / 1 |
| (0.003, 0.0035] | fc | 994 | 990 | 0.159 / 0.137 (0.151) | 0.021 / 0.017 | 429 | 1 | 0.10 | 4.7 | 2 / 2 |
| (0.0035, 0.004] | fc | 756 | 754 | 0.145 / 0.119 (0.137) | 0.020 / 0.017 | 308 | 0 | 0.00 | 4.5 | 2 / 0 |
| (0.004, 0.0045] | fc | 544 | 540 | 0.146 / 0.123 (0.136) | 0.021 / 0.016 | 228 | 0 | 0.00 | 5.2 | 3 / 1 |
| (0.0045, 0.005] | fc | 420 | 420 | 0.136 / 0.109 (0.126) | 0.019 / 0.015 | 163 | 0 | 0.00 | 5.0 | 0 / 0 |
| (0.005, 0.005579) | fc | 368 | 367 | 0.151 / 0.128 (0.138) | 0.020 / 0.016 | 153 | 0 | 0.00 | 3.5 | 1 / 0 |
| reference: current zone | fi | 8,480 | 8,448 | 0.113 / 0.093 (0.111) | 0.016 / 0.013 | 2,744 | 24 | 0.28 | 11.3 | 21 / 11 |
| (0.002617, 0.003] | fi | 940 | 939 | 0.119 / 0.099 (0.113) | 0.018 / 0.014 | 317 | 1 | 0.11 | 6.8 | 1 / 0 |
| (0.003, 0.0035] | fi | 1,004 | 998 | 0.114 / 0.091 (0.108) | 0.017 / 0.013 | 331 | 3 | 0.30 | 7.4 | 4 / 2 |
| (0.0035, 0.004] | fi | 721 | 719 | 0.120 / 0.091 (0.112) | 0.017 / 0.013 | 257 | 1 | 0.14 | 6.0 | 0 / 2 |
| (0.004, 0.0045] | fi | 514 | 511 | 0.120 / 0.100 (0.110) | 0.018 / 0.014 | 188 | 2 | 0.39 | 6.6 | 2 / 1 |
| (0.0045, 0.005] | fi | 431 | 426 | 0.114 / 0.090 (0.104) | 0.018 / 0.014 | 160 | 1 | 0.23 | 10.0 | 3 / 2 |
| (0.005, 0.005579) | fi | 352 | 351 | 0.123 / 0.094 (0.112) | 0.018 / 0.013 | 125 | 0 | 0.00 | 6.3 | 0 / 1 |

Whole newly admitted DEV zone (0.002617, 0.005]:
- **fc:** 3,706 attempts, ΔSFace +0.150 (lo95 0.147), Δdlib +0.021, 1,564 improved, 4 harmed (0.11%), block rate 0.30%.
- **fi:** 3,610 attempts, ΔSFace +0.117 (lo95 0.114), Δdlib +0.017, 1,253 improved, 8 harmed (0.22%), block rate 0.47%.

**Clean protection at t = 0.005.**
- 61 of 4,188 sharp GT faces (1.46%) would be attempted. All 61 pass both guards.
- On those 61: mean ΔSFace −0.018, Δdlib −0.003, **0 clearly harmed** (worst single ΔSFace −0.21).
- NAFNet on an already sharp face costs a little similarity; it does not cause gross damage.
- The current boundary admits 1 sharp face (0.02%).

**Visual DEV review.** I inspected the top-16 `l_dot_c_max` and top-8 chroma-p99.9 newly admitted, returned DEV outputs:
- they are legitimate deblurs;
- there is no rainbow, dotted or grid artifact class beyond those R5-8 already knew.

## 10. Selected threshold and rationale

- **Old `blur_max`: 0.002617** (inherited from A4).
- **New `blur_max`: 0.005**, selected mechanically by the pre-registered rule.
  - Every sub-band from 0.002617 to 0.005 passes R1–R5 in both modes.
  - R6 and R7 hold at 0.005.
  - 0.005579 fails both R6 (2.87% of sharp faces) and R7 (non-blur families admitted with negative Δ).
- **Unchanged:** `clean_lapd1_lo` (0.005579), so the NOT_BLUR zone is untouched.
- **Expected coverage (archive DEV):** eligible share 41.8% → 61.0% (fc) and 44.4% → 63.3% (fi), roughly +46% / +43% relative.
- **Known trade-offs:**
  - about 1.5% of sharp faces (lapd1 just under 0.005) become NAFNet-eligible, at about −0.02 SFace each;
  - the DEV population has no grainy-print moderate blur (§7).
- No owner image, TEST, ztest or real photo was used. The R5-9 moderate-blur example played no role.

## 11. Candidate route / config / hash

| Item | Value |
|---|---|
| Profile | `r5_10` (opt-in only: `FES_NAFNET_PROFILE=r5_10`, or per request with `FES_EXPERIMENTAL_PROFILES=r5_10`) |
| Config | `configs/phase5/r5_10_nafnet_candidate.json`, file sha `827c7e01…`, seal **`cc2347ac22c15a3ddbcbdef0e461233a24ef02685df9dd36ae2937125eac2b8d`** |
| Route version | `p5-nafnet-route-1.2.0` |
| Thresholds | the R5-8 candidate's thresholds, except `blur_max` = **0.005** |
| Derived from | R5-8 seal `631acd73…`, live R5-6 seal `eb6fe101…`; `blur_recalibration` records the protocol sha `234d042c…` and the DEV analysis sha `b8c0c9d9…` |
| Feature versions | `p5-nafnet-feat-1.0.0`, `p5-rwfeat-1.0.0` (rwgate seal `78837a8d…`) |
| Global artifact guard | R5-8 values, unchanged (chroma p99 ≤ 20.4, p99.9 ≤ 36.4, hf-colour, gray, channel) |
| Local artifact detector | `p5-nafnet-local-artifact-1.0.0`, `l_dot_c_max` ≤ 24.45, unchanged |
| Model | NAFNet-GoPro width64 `329d3ab4…`, reflect-pad16 factory (same as R5-8) |
| Freeze record | `data/manifests/phase5_rapid/r5_10/frozen_hashes.txt` (written before any TEST request) |

Code changes:
1. **`nafnet_route.py`:** accepts the `p5-nafnet-route-1.2.0` version string. `decide()`, the guards and the features are unchanged.
2. **`pipeline.py`:** adds a per-profile config map and allows `r5_10` alongside `r5_8`.
3. **`app.py`:** `FES_EXPERIMENTAL_PROFILES` accepts `r5_8,r5_10`. Both profiles share **one** pad16 NAFNet worker, because the model is identical and only the config differs.
4. **UI (`compare.html` / `compare.js` / `fes_compare.js`):**
   - adds an R5-10 panel, "NAFNet R5-10 — Recalibrated Blur Experimental", labelled "EXPERIMENTAL — NOT PRODUCTION-APPROVED";
   - each panel shows the blur score (lapd1_c);
   - the panel appears only when the service enables r5_10.

## 12. Unit / regression results

**Before TEST (frozen candidate):**
- `pytest` (fes-dev:cpu-p5, `FES_RUN_NAFNET=1`) across the R5-10, R5-9, R5-8, R5-NAFNet, rwgate, service and UI suites: **191 passed, 3 skipped, 0 failed**. The 3 skips are environment-dependent, e.g. the node suite inside Docker.
- `node --test` on the UI suites: **32/32**.

New file `tests/test_r5_10_blur_recalibration.py` (19 tests, including real NAFNet) covers:
- **Configs:** the R5-6 and R5-8 configs are byte-identical to baseline; R5-10 differs from R5-8 *only* in `blur_max`; the seal detects tampering.
- **Blur zones under R5-6 / R5-8 / R5-10:**
  - old strong blur stays eligible;
  - the moderate zone is eligible only in R5-10;
  - mild blur above 0.005 and clean inputs stay rejected.
- **Other gates:** noise, compression, pixelation, halftone, photometric, size, missing feature and T6 safety each still block a moderate-blur input.
- **Requests and output safety:**
  - default requests are byte-identical with R5-10 enabled;
  - R5-10 is refused unless enabled;
  - both output guards still return ORIGINAL;
  - model failure falls back to ORIGINAL;
  - process-level and per-request R5-10 give identical results;
  - the API shares one worker.
- **Real NAFNet:** a real NAFNet run on a moderate-blur DEV face.

## 13. Archive TEST results

The R5-6 harness ran unchanged over HTTP (`scripts/r5_10_validate*.sh`), against 3 replicas plus a 4th added mid-run on idle CPU. All 4 replicas had their R5-10 seal verified.

| | R5-8 face_chip | **R5-10 face_chip** | R5-8 full_image | **R5-10 full_image** |
|---|---|---|---|---|
| Requests (HTTP 200) | 19,290 (100%) | 19,290 (100%) | 19,290 (100%) | 19,290 (100%) |
| NAFNet eligible = attempts | 8,147 | **11,818** | 8,470 | **12,028** |
| NAFNet outputs returned | 8,120 | **11,777** | 8,429 | **11,968** |
| ORIGINAL / fallback | 11,170 / 27 | 7,513 / 41 | 10,861 / 41 | 7,322 / 60 |
| Global / local guard blocks | 20 / 7 | 30 / 11 | 31 / 10 | 42 / 18 |
| Block rate (of attempts) | 0.33% | 0.35% | 0.48% | 0.50% |
| ΔSFace of returned outputs | +0.154 [0.151, 0.157] | +0.154 [0.152, 0.157] | +0.113 [0.110, 0.115] | +0.115 [0.113, 0.117] |
| Δdlib | +0.019 | +0.020 | +0.016 | +0.016 |
| Clearly improved / harmed | 3,267 / 7 | 4,836 / 12 | 2,718 / 20 | 3,951 / 26 |
| **Whole-policy ΔSFace (all 19,290)** | +0.065 | **+0.094** | +0.049 | **+0.071** |
| Whole-policy Δdlib | +0.0080 | **+0.0120** | +0.0069 | **+0.0101** |

**Newly admitted outputs only.** These are inputs where R5-8 returned ORIGINAL.

| | n | ΔSFace (95% CI) | median | Δdlib | Improved | Harmed | ≤ 0 |
|---|---|---|---|---|---|---|---|
| face_chip | 3,657 | **+0.155** [0.151, 0.159] | 0.129 | +0.021 | 1,569 (42.9%) | 5 (0.14%) | 4.0% |
| full_image | 3,539 | **+0.119** [0.116, 0.123] | 0.096 | +0.017 | 1,233 (34.8%) | 6 (0.17%) | 7.1% |

- **No losses:** no input that R5-8 enhanced was dropped by R5-10 (0 in both modes).
- **Agreement:** every R5-8-eligible input received the same decision.
- **Matches DEV:** the TEST figures reproduce the DEV figures, so there is no DEV-to-TEST drift.

**Artifact scan** (`scripts/r5_10_test_scan.py`, observation only):

| Signal | p99.9 / max, new outputs (fc) | Shared outputs (fc) | New outputs (fi) | Shared outputs (fi) |
|---|---|---|---|---|
| chroma p99.9 | 26.6 / 30.4 | 26.0 / 31.9 | 23.7 / 34.2 | 23.5 / 28.0 |
| `l_dot_c_max` | 15.4 / 23.9 | 13.5 / 18.4 | 16.4 / 19.5 | 13.7 / 20.8 |

**Zoomed visual review of new outputs.** I reviewed the top 12 per signal, per mode:
- **1 clear localized artifact**: `image_84257`, face_chip. It shows dotted colour speckle along the cheek edge (`l_dot_c_max` 16.8). This is the same "faint coarse grid" class that R5-8 documented as a known miss.
- **1 uncertain**: `image_89236`. Motion-blurred hair was rendered as curly strokes over the face.
- Everything else is legitimate deblurring, often dramatic.
- So the new outputs show **≈ 1 clear escape in 7,196 (≈ 0.014%)**, the same order as R5-8's own TEST rate (about 2–3 in 16.5k).

## 14. ztest results

| | R5-8 fc | R5-10 fc | R5-8 fi | R5-10 fi |
|---|---|---|---|---|
| Enhanced | 0 / 100 | **0 / 100** | 0 / 100 | **0 / 100** |
| TAR @ FAR 1e-2 (SFace / dlib) | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| New false accepts / flips | 0 / 0 | **0 / 0** | 0 / 0 | **0 / 0** |

**Containment is unchanged.** Every ztest input remains rejected by the noise / other gates before the blur rule is reached.

## 15. 200-real-photo results (full_image, auto)

| | R5-6 | R5-8 | **R5-10** |
|---|---|---|---|
| NAFNet / A4 / ORIGINAL | 4 / 2 / 194 | 4 / 2 / 194 | **6** / 2 / 192 |
| NAFNet attempts | 4 | 4 | 6 |
| Guard blocks / failures | 0 / 0 | 0 / 0 | 0 / 0 |
| **Visible artifacts returned** | 0 | 0 | **1** |

The two newly admitted photos:
- **`img123`** (lapd1 0.00491, noise σ 0.31): mild, plausible sharpening. Acceptable.
- **`img118`** (lapd1 0.00328, noise σ **1.03** against a noise gate of 1.08; a sepia-toned vintage print, `is_gray` False):
  - **Visible blue/yellow colour mottling** across the forehead, cheek and beard.
  - Both guards passed it:
    - global chroma p99 is 9.5, under R5-8's 20.4 and even under R5-6's 10;
    - local `l_dot_c_max` is 2.3 against a limit of 24.45.
  - This is the R5-ARCHIVE failure mode ("printed / grainy real photos"), now reached through the widened blur boundary.
  - No threshold was changed after seeing it.

## 16. UI manual results

**Not performed, by rule.**
- Step 12 of the task allows the R5-10 profile into the R5-9 UI **only after it passes service-level validation**. It did not pass: see §15 and the verdict.
- The UI code is ready and tested: an R5-10 panel, a blur-score row, export fields and a "not enabled" message. The `fes-ui` container was **not** restarted with R5-10.
- Moderate-blur attempts are confirmed programmatically instead:
  - `test_moderate_blur_attempted_only_by_r5_10`;
  - the real-NAFNet test;
  - TEST: 3,657 + 3,539 newly attempted moderate-blur inputs, plus img118 / img123 on real photos.
- If the owner still wants to compare R5-10 by eye, restart the UI service with `FES_EXPERIMENTAL_PROFILES=r5_8,r5_10`. The owner's moderate-blur example is a **manual validation case** and cannot change the frozen boundary.

## 17. R5-8 vs R5-10 comparison

| Criterion | R5-8 | R5-10 | Change |
|---|---|---|---|
| NAFNet coverage, TEST fc / fi | 8,120 / 8,429 | 11,777 / 11,968 | **+45% / +42%** |
| Whole-policy ΔSFace fc / fi | +0.065 / +0.049 | +0.094 / +0.071 | **+45% / +44%** |
| Per-output ΔSFace, harm rate | +0.154 / 0.09%; +0.113 / 0.24% | +0.154 / 0.10%; +0.115 / 0.22% | = |
| Guard block rate | 0.33% / 0.48% | 0.35% / 0.50% | ≈ |
| TEST localized artifacts returned | ≈ 2–3 (1 clear) | + ≈ 1 clear in the new outputs | ≈ same rate, more outputs |
| ztest | contained | contained | = |
| Real photos: NAFNet outputs / visible artifacts | 4 / 0 | 6 / **1** | **regression** |
| Stability | 0 failures | 0 failures, 0 restarts | = |

## 18. Safety analysis

- **Input gates.** Unchanged and still first. Tests prove every non-blur gate blocks moderate-blur inputs.
- **Clean protection.** NOT_BLUR is unchanged.
  - About 1.5% of sharp faces fall in the new zone, with a small SFace cost and no gross harm (DEV).
  - At 0.005579 this would have been 2.9%, plus degraded non-blur inputs. That boundary was rejected.
- **Output guards.** Unchanged. The block rate barely moves, and TEST artifact leakage per output is the same order as R5-8.
- **The real-world gap.**
  - The widened zone admits real *grainy* moderately blurred prints with noise σ just under the noise gate.
  - DEV had no such inputs: archive blur is grain-free, and p4dev noise sits above the gate.
  - On these, NAFNet can produce low-amplitude colour mottling, which neither guard is designed to see: it is neither high chroma nor dotted.
  - One of the two newly admitted real photos showed it. n = 2 is far too small to estimate a rate, but a 1-in-2 observation on exactly the class R5-ARCHIVE flagged cannot be called "no material regression".
- **Fail-safety.** Model failure, timeout, guard failure and invalid output all return ORIGINAL (tests). There were 0 service failures, restarts, OOMs or timeouts in validation.

## 19. Identity-benefit analysis

- **NAFNet's identity benefit extends to moderate and mild blur.** The marginal sub-bands up to 0.005 give the same ΔSFace as the strong-blur zone:
  - face_chip: +0.14 to +0.16;
  - full_image: +0.11 to +0.12;
  - harm 0–0.4%.
- **TEST confirms it.** Newly admitted outputs give +0.155 (fc) and +0.119 (fi), the same as or slightly better than the existing zone.
- **The widened boundary raises whole-policy SFace benefit by about 45%** on the archive.
- Measured with the SFace / dlib proxies against paired synthetic GT. These are not Feature-Match.

## 20. Runtime / memory

| | Value |
|---|---|
| Latency p50 / p95, TEST fc (all requests) | R5-8 0.52 / 1.75 s → R5-10 0.74 / 1.96 s (more NAFNet runs) |
| Latency p50 / p95, TEST fi | 0.58 / 2.00 s → 1.11 / 2.36 s |
| NAFNet inference p50 | 0.46–0.88 s (CPU, 4 threads, replicas loaded) |
| Memory per replica | 1.1–1.3 GB (unchanged; R5-10 adds no model) |
| Shared worker | with `FES_EXPERIMENTAL_PROFILES=r5_8,r5_10`, both use **one** extra NAFNet worker |
| Throughput | 9.5–10 req/s (fc) and 6.5 req/s (fi) on 3–4 replicas, with WelOrbit stopped |
| Failures / restarts / OOM / timeouts | 0 / 0 / 0 / 0 (`/ready` counters on all 4 replicas) |

**Infrastructure incidents.** These are recorded for transparency; none changed the candidate.
1. **Replica s3 failed its first start** with `Errno 12 Cannot allocate memory` while loading NAFNet (memory pressure, WelOrbit running).
   - It answered `NOT_READY`, and those records were invalid.
   - The partial run was moved to `experiments/phase5_rapid/outputs/r5_6/_r510_aborted_s3_oom/` (kept as evidence).
   - The start script now aborts unless every replica reports `ready: true` with the R5-10 seal.
   - All runs were then repeated from scratch.
2. **A 4th replica was added mid-run** (idle CPU). The harness resumes from records, so no request was sent twice.
3. **The laptop slept for about 95 minutes** during full_image. The run resumed and lost nothing.

## 21. Limitations

- **Identity proxies:** SFace and dlib, not Feature-Match. Feature-Match access is still unavailable (R5-4).
- **DEV and TEST are synthetic paired blur** (archive) at about 128 px, IOD 35–60.
  - They contain no grain or print texture, so they cannot reveal the real-photo failure class.
  - Large faces (IOD > 71) are almost absent.
- **The real-photo set is small and vintage-heavy**: 200 CC0 photos, no identity references. Only 2 photos exercise the new zone, so its real-world artifact rate is unknown.
- **The visual review covered the signal tails.** Faint artifacts outside those tails may remain, as in R5-8.
- **No manual UI evaluation.** The owner's images were not used, by rule.

## 22. Licensing status

- NAFNet-GoPro weights and code: MIT (`models/M28_nafnet/LICENSE`), unchanged.
- Archive dataset (used for DEV and TEST): licence unknown; research evaluation only, as in R5-ARCHIVE, R5-6 and R5-8.
- 200 real photos: a Kaggle CC0 upload.
- No new model, weight or dataset was introduced in R5-10.

## 23. Final verdict — **RED**

GREEN requires all 13 criteria. Status of each:

| # | Criterion | Status |
|---|---|---|
| 1 | DEV supports a wider boundary | ✔ |
| 2 | Coverage materially increases | ✔ (+42–45%) |
| 3 | Positive identity benefit in the new zone | ✔ |
| 4 | Harm does not increase beyond the limit | ✔ (≤ +1 pp rule; observed ≈ +0.05 pp) |
| 5 | Clean faces protected | ✔ (1.46% admitted, 0 harmed; NOT_BLUR unchanged) |
| 6 | Other gates unchanged | ✔ |
| 7 | Guards unchanged | ✔ |
| 8 | ztest contained | ✔ |
| 9 | **Real-photo safety does not materially regress** | **✘ — 1 visible artifact returned (img118, newly admitted), 0 before** |
| 10 | No service failures | ✔ |
| 11 | Regression tests pass | ✔ |
| 12 | Manual UI confirms attempts | ✘ not run (gated by #9); confirmed programmatically |
| 13 | No tuning after TEST | ✔ (frozen hashes written before the first TEST request) |

**RED.** The task's RED list includes "real-photo safety regresses", and it did. The synthetic evidence for the boundary itself is strong. The blocker is a real-world input class (grainy prints with moderate blur) that the DEV data could not represent and the guards cannot see.

Actions under RED:
- **Default and live behaviour:** unchanged, never modified. Requests without `profile` are byte-identical (tests).
- **R5-10:** remains an opt-in research profile, not enabled in `fes-ui` or anywhere else. Evidence and code are preserved.
- **Threshold experimentation:** stopped. There will be no R5-10A/B/C.

## 24. Explicit production recommendation

1. **Do not promote R5-10, and do not enable it by default.** Production stays at the current owner decision: R5-6 live, R5-8 a research comparison, NAFNet not production-approved.
2. **The answer to the task's question:**
   - The strong-blur-only boundary is **not** justified by identity evidence, because NAFNet helps equally on moderate and mild blur.
   - It **is** currently justified by real-world safety, because widening it reaches grainy real prints where NAFNet produces guard-invisible colour mottling.
3. **Is NAFNet still worth Feature-Match testing?**
   - Yes, with the **safest available experimental profile (R5-8)** as the candidate.
   - R5-10's evidence shows the benefit is not limited to extreme blur, so a future route could widen eligibility once the grain issue is handled. That would be a new task with new DEV data: real grainy or printed blurred faces, and possibly a grain-aware input rule. It must not be another threshold move.
4. **If the owner wants a visual look at R5-10 anyway** (for example on the moderate-blur male example from R5-9):
   - restart the UI service with `FES_EXPERIMENTAL_PROFILES=r5_8,r5_10`;
   - the panel is labelled "EXPERIMENTAL — NOT PRODUCTION-APPROVED";
   - this is evaluation only and changes nothing above.
